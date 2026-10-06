import json
import math
import subprocess
import uuid
import wave
from pathlib import Path

import torch
import nodes
import folder_paths
import comfy.model_management as mm
import comfy.utils
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3AddGuide, align_frame_count
from comfy_extras.nodes_lt_audio import LTXVAudioVAEEncode
from comfy_extras.nodes_lt import LTXVConcatAVLatent, LTXVSeparateAVLatent


def result(value):
    return value['result'] if isinstance(value, dict) else value


def segment_plan(samples, sample_rate, seconds, fps=24):
    total = math.ceil(samples * fps / sample_rate)
    step = max(1, round(seconds * fps))
    return [(start, min(step, total-start)) for start in range(0, total, step)]


def lock_audio_mask(latent):
    # fit_audio pads short audio with mask=1. Lock padding too, after fitting.
    video, audio = latent['samples'].unbind()
    latent['noise_mask'] = NestedTensor((torch.ones_like(video), torch.zeros_like(audio)))
    return latent


def run_checked(args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


class H3SegmentedAvatar:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('MODEL',), 'clip': ('CLIP',), 'vae': ('VAE',), 'audio_vae': ('VAE',),
            'image': ('IMAGE',), 'audio': ('AUDIO',), 'width': ('INT', {'default': 1056, 'min': 32, 'step': 32}),
            'height': ('INT', {'default': 1888, 'min': 32, 'step': 32}),
            'mode': (['固定镜头数字人', 'Qwen推理分镜'],),
            'idea': ('STRING', {'multiline': True, 'default': '图中的人物自然讲话，保持人物身份、服装与环境一致。'}),
            'segment_seconds': ('FLOAT', {'default': 15, 'min': 3, 'max': 15, 'step': 1}),
            'continuity': ('BOOLEAN', {'default': True}),
            'seed': ('INT', {'default': 1071168533130136, 'min': 0, 'max': 0xffffffffffffffff}),
            'steps': ('INT', {'default': 6, 'min': 2, 'max': 50}),
            'transition_step': ('INT', {'default': 4, 'min': 1, 'max': 49}),
            'lowres_scale': ('FLOAT', {'default': 0.5, 'min': 0.25, 'max': 1, 'step': 0.05}),
            'upscaler_model': (folder_paths.get_filename_list('latent_upscale_models'),),
            'highres_tiling': ('BOOLEAN', {'default': True}),
            'ref_image_size': (['max', 'match'],),
            'llm_model': ('STRING', {'default': 'Qwen3.8-9B-distill-heretic_nvfp4_q8_0.gguf'}),
            'vision_model': ('STRING', {'default': 'mmproj-BF16.gguf'}),
            'filename_prefix': ('STRING', {'default': 'SelfLift分段数字人'}),
        }}

    RETURN_TYPES = ('STRING', 'STRING')
    RETURN_NAMES = ('成片路径', '分段记录')
    FUNCTION = 'generate'
    CATEGORY = 'MiniMax H3/分段数字人'
    OUTPUT_NODE = True

    def generate(self, model, clip, vae, audio_vae, image, audio, width, height, mode, idea,
                 segment_seconds, continuity, seed, steps, transition_step, lowres_scale,
                 upscaler_model, highres_tiling, ref_image_size, llm_model, vision_model, filename_prefix):
        if transition_step >= steps:
            raise ValueError('transition_step 必须小于 steps，以保留高清采样阶段。')
        if not isinstance(continuity, bool) or not isinstance(highres_tiling, bool):
            raise ValueError('工作流开关参数错位，请重新导入修正版工作流。')
        if 'h3' not in upscaler_model.lower():
            raise ValueError('请选择 MiniMax H3 latent upscaler，不能使用 LTX 放大模型。')
        if audio['waveform'].shape[0] != 1 or audio['waveform'].shape[-1] == 0:
            raise ValueError('请提供一条非空音频。')
        # Reuse the installed VHS ffmpeg selection, including launcher-specific binaries.
        import importlib
        vhs_cls = nodes.NODE_CLASS_MAPPINGS['VHS_VideoCombine']
        ffmpeg = importlib.import_module(vhs_cls.__module__.rsplit('.', 1)[0] + '.utils').ffmpeg_path
        if not ffmpeg:
            raise RuntimeError('VideoHelperSuite 未找到 ffmpeg。')
        sr = audio['sample_rate']
        waveform = audio['waveform'].detach().cpu()
        duration = waveform.shape[-1] / sr
        plan = segment_plan(waveform.shape[-1], sr, segment_seconds)
        safe_prefix = ''.join(c if c.isalnum() or c in '-_' else '_' for c in filename_prefix)[:80] or 'avatar'
        outdir = Path(folder_paths.get_output_directory()) / (safe_prefix + '_' + uuid.uuid4().hex[:10])
        outdir.mkdir(parents=True)
        wavpath = outdir / 'original_audio.wav'
        pcm = (waveform[0].clamp(-1, 1).T * 32767).round().to(torch.int16).numpy().astype('<i2')
        with wave.open(str(wavpath), 'wb') as wav:
            wav.setnchannels(waveform.shape[1]); wav.setsampwidth(2); wav.setframerate(sr); wav.writeframes(pcm.tobytes())
        del pcm
        tail = None
        previous_prompt = ''
        records = []
        paths = []
        progress = comfy.utils.ProgressBar(len(plan))
        sampler_cls = nodes.NODE_CLASS_MAPPINGS['SelfLiftAvatarH3Sampler']
        for index, (start, keep) in enumerate(plan):
            mm.throw_exception_if_processing_interrupted()
            prefix = 22 if continuity and tail is not None else 0
            begin = start - prefix
            length = align_frame_count(keep + prefix)
            a0, a1 = round(begin * sr / 24), min(waveform.shape[-1], round((start + keep) * sr / 24))
            chunk_audio = {'waveform': waveform[..., a0:a1], 'sample_rate': sr}
            if mode == '固定镜头数字人':
                prompt = ('Use <Picture 1> as the exact identity, clothing, setting and framing reference. '
                          'A single continuous locked-off camera shot. Keep the same framing, background, lighting and scale. '
                          'Accurate lip synchronization to the supplied speech or singing audio, subtle facial microexpressions, '
                          'occasional gentle blinking, slight breathing, small head and hand movements. '
                          'No cuts, zooms, pans, scene changes or exaggerated gestures. Continue the opening guide motion smoothly. '
                          'Creative brief: ' + idea)
            else:
                # Generate each local timeline with the original local Qwen node and its skill.
                qwen = nodes.NODE_CLASS_MAPPINGS['QwenH3PromptLocal']
                brief = (f'全片创意：{idea}\n全片音频长{duration:.3f}秒。当前第{index+1}/{len(plan)}段，'
                         f'全片时间{start/24:.3f}至{(start+keep)/24:.3f}秒。只为本段设计局部时间线，'
                         f'本段生成长度{length/24:.3f}秒，其中开头{prefix/24:.3f}秒为上一段续接上下文。'
                         '保持原图人物身份、衣服与环境；根据创意推理合理分镜，镜头切换落在自然动作或停顿处。'
                         '使用传入音频驱动口型，不编造对白，不增加背景音乐。'
                         f'上一段提示词供连续性参考：{previous_prompt}')
                prompt = result(qwen.execute(prompt=brief, skill='h3-prompt-writing', duration=length/24,
                    llm_model=llm_model, vision_model=vision_model, think_mode=False, reasoning_effort='medium',
                    seed=(seed+index) % (2**64), max_tokens=8192, video_sample_frames_per_sec=2,
                    force_unload_model=True, reference_images={'reference_image_0': image}))[0]
            positive, av = MiniMaxH3ReferenceToVideo.execute(clip=clip, prompt=prompt, width=width,
                height=height, length=length, ref_image_size=ref_image_size, vae=vae,
                audio_vae=audio_vae, ref_images={'ref_image_0': image})
            if prefix:
                positive = MiniMaxH3AddGuide.execute(positive=positive, latent=av, frame_idx=0,
                    vae=vae, image=tail)[0]
            negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
            audio_latent = LTXVAudioVAEEncode.execute(chunk_audio, audio_vae)[0]
            audio_latent['noise_mask'] = torch.zeros_like(audio_latent['samples'])
            latent = lock_audio_mask(LTXVConcatAVLatent.execute(av, audio_latent)[0])
            sigmas = nodes.NODE_CLASS_MAPPINGS['BasicScheduler'].execute(model, 'simple', steps, 1.0)[0]
            sampler = nodes.NODE_CLASS_MAPPINGS['KSamplerSelect'].execute('euler')[0]
            sampled = result(sampler_cls().sample(model=model, positive=positive, negative=negative,
                vae=vae, latent_image=latent, sampler=sampler, sigmas=sigmas, seed=(seed+index) % (2**64),
                cfg=1.0, transition_step=transition_step, lowres_scale=lowres_scale, rho=0,
                w_min=0.5, w_max=1, upscaler_model=upscaler_model, highres_tiling=highres_tiling))[0]
            video = LTXVSeparateAVLatent.execute(sampled)[0]
            frames = nodes.VAEDecode().decode(vae, video)[0]
            if frames.shape[0] < prefix+keep:
                raise RuntimeError(f'第{index+1}段解码帧数不足：{frames.shape[0]} < {prefix+keep}')
            retained = frames[prefix:prefix+keep]
            tail = retained[-22:].detach().cpu().clone() if continuity else None
            if tail is not None and tail.shape[0] < 22:
                tail = None
            part = outdir / f'part_{index+1:04d}.mp4'
            logpath = outdir / f'part_{index+1:04d}_ffmpeg.log'
            with logpath.open('wb') as log:
                proc = subprocess.Popen([ffmpeg, '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s',
                    f'{width}x{height}', '-r', '24', '-i', '-', '-an', '-c:v', 'libx264', '-preset',
                    'medium', '-crf', '12', '-pix_fmt', 'yuv420p', str(part)], stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=log)
                try:
                    for frame in retained:
                        mm.throw_exception_if_processing_interrupted()
                        proc.stdin.write((frame.clamp(0, 1)*255).round().to(torch.uint8).cpu().numpy().tobytes())
                    proc.stdin.close()
                    if proc.wait() != 0:
                        raise RuntimeError(f'ffmpeg 编码失败，查看 {logpath}')
                finally:
                    if proc.poll() is None:
                        proc.kill(); proc.wait()
            paths.append(part)
            records.append({'segment': index+1, 'start': start/24, 'kept_seconds': keep/24,
                            'context_frames': prefix, 'generated_frames': length, 'prompt': prompt})
            (outdir / 'segments.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
            previous_prompt = prompt
            del frame, frames, retained, video, sampled, latent, audio_latent, positive, negative, av, sigmas, chunk_audio
            mm.soft_empty_cache()
            progress.update(1)
        listing = outdir / 'concat.txt'
        listing.write_text('\n'.join("file '" + p.name + "'" for p in paths), encoding='utf-8')
        final = outdir / 'final.mp4'
        run_checked([ffmpeg, '-y', '-f', 'concat', '-safe', '0', '-i', str(listing), '-i', str(wavpath),
            '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '320k',
            '-t', str(duration), '-movflags', '+faststart', str(final)])
        summary = f'{mode} | {duration:.3f}秒 | {len(plan)}段 | {width}×{height} | 24fps\n{final}'
        return {'ui': {'text': [summary]}, 'result': (str(final), json.dumps(records, ensure_ascii=False, indent=2))}


NODE_CLASS_MAPPINGS = {'H3SegmentedAvatar': H3SegmentedAvatar}
NODE_DISPLAY_NAME_MAPPINGS = {'H3SegmentedAvatar': 'H3 高清分段数字人 · 固定镜头 / Qwen分镜'}
