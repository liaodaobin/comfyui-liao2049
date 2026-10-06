"""Standalone local YuE2 studio using embedded helpers and ComfyUI core."""
import importlib.util
import json
from pathlib import Path
import re
import sys
import math
import threading
import time

ROOT = Path(__file__).parent
PRESETS = json.loads((ROOT / 'prompt_templates/yue2_styles.json').read_text(encoding='utf-8'))
_modules = {}
_model_cache = {}
_model_cache_lock = threading.Lock()


def cached_model(kind, path, loader):
    """Keep one current model per kind; ComfyUI still manages GPU offloading."""
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _model_cache_lock:
        entry = _model_cache.get(kind)
        if entry is not None and entry[0] == key:
            return entry[1]
        _model_cache.pop(kind, None)
        value = loader()
        _model_cache[kind] = (key, value)
        return value


def cached_pipeline(checkpoint):
    import folder_paths
    path = folder_paths.get_full_path_or_raise('checkpoints', checkpoint)
    return cached_model('yue2', path,
                        lambda: module('comfy_native').YuE2NativePipelineLoader().load(checkpoint, 'none', 0, True)[0])


def cached_encoder(name):
    import folder_paths
    import comfy.utils
    from comfy.audio_encoders.audio_encoders import load_audio_encoder_from_sd
    path = folder_paths.get_full_path_or_raise('audio_encoders', name)
    return cached_model('sheetsage', path,
                        lambda: load_audio_encoder_from_sd(comfy.utils.load_torch_file(path, safe_load=True)))


def module(name):
    if name not in _modules:
        base = ROOT / 'yue2_embedded'
        package = '_liao_yue2_embedded'
        if package not in sys.modules:
            import types
            p = types.ModuleType(package)
            p.__path__ = [str(base)]
            sys.modules[package] = p
        spec = importlib.util.spec_from_file_location(package + '.' + name, base / (name + '.py'))
        if spec is None or spec.loader is None:
            raise RuntimeError('liao2049内置YuE2工具文件缺失，请恢复yue2_embedded目录')
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        spec.loader.exec_module(m)
        _modules[name] = m
    return _modules[name]

def text(default=''):
    return ('STRING', {'default': default, 'multiline': True, 'dynamicPrompts': False})

def normalize_lyrics(value):
    aliases = {'主歌':'verse','副歌':'chorus','桥段':'bridge','前奏':'intro','尾奏':'outro','尾声':'outro','预副歌':'pre-chorus','最终副歌':'final chorus'}
    value = re.sub(r'^\s*#{1,6}\s*(?=\[)', '', value, flags=re.M)
    for cn, en in aliases.items():
        value = value.replace('[' + cn + ']', '[' + en + ']')
    if value.strip() and not re.search(r'^\s*\[.+?\]', value, re.M):
        value = '[verse]\n' + value.strip()
    return value.strip()

def format_recognized_lyrics(value):
    """Group recognized lines without inventing missing words or section roles."""
    value = re.sub(r'^\s*\[(?:verse|chorus|bridge|intro|outro)[^\]]*\]\s*$', '', value, flags=re.M | re.I)
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        raise ValueError('未识别到歌词，请选择有人声的片段再试。')
    blocks = [lines[i:i + 4] for i in range(0, len(lines), 4)]
    keys = ['\n'.join(block) for block in blocks]
    return '\n\n'.join(('[' + ('chorus' if keys.count(key) > 1 else 'verse') + ']\n' + key) for key in keys)

def recognize_uploaded_lyrics(filename, start_seconds=0, length_seconds=30, language='中文', model_path=''):
    import folder_paths
    import torch
    from comfy_extras.nodes_audio import load
    if not filename or not isinstance(filename, str):
        raise ValueError('请先上传原歌曲。')
    start_seconds, length_seconds = float(start_seconds), float(length_seconds)
    if not math.isfinite(start_seconds) or not math.isfinite(length_seconds) or start_seconds < 0 or not 1 <= length_seconds <= 900:
        raise ValueError('参考起点需大于等于0，参考长度需在1至900秒之间。')
    if language not in ('中文', 'English', '日本語', '自动'):
        raise ValueError('不支持的语言选项。')
    base = Path(folder_paths.get_input_directory()).resolve()
    path = (base / filename).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError('原曲文件不存在，或不在ComfyUI input目录中。')
    with torch.inference_mode():
        wave, sr = load(str(path))
        start = round(start_seconds * sr)
        wave = wave[..., start:start + round(length_seconds * sr)]
        if wave.shape[-1] < sr:
            raise ValueError('所选片段不足1秒，请调整参考起点和长度。')
        audio = {'waveform': wave.unsqueeze(0), 'sample_rate': sr}
        recognized = LiaoYuE2Studio.transcribe(audio, model_path, language)
    return {'lyrics': format_recognized_lyrics(recognized), 'seconds': round(wave.shape[-1] / sr, 2),
            'note': '已按每4行分段，重复的完整段落标为副歌；请校对识别文本和段落标签。'}

def parse_melody_only(value):
    # Older saved workflows can serialize BOOLEAN widgets as 0/1 or text.
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ('true', '1', 'yes', 'on'):
            return True
        if value in ('false', '0', 'no', 'off'):
            return False
    raise ValueError('旋律模式参数错误，请重新勾选专业设置中的“自由重编伴奏”。')


def recognize_uploaded_abc(filename, start_seconds=0, length_seconds=90, encoder='none', melody_only=True):
    import folder_paths
    import torch
    import uuid
    from comfy_execution.utils import CurrentNodeContext
    from comfy_extras.nodes_audio import load
    from comfy.audio_encoders.audio_encoders import load_audio_encoder_from_sd
    import comfy.utils
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError('请先上传原歌曲。')
    start_seconds, length_seconds = float(start_seconds), float(length_seconds)
    if not math.isfinite(start_seconds) or not math.isfinite(length_seconds) or start_seconds < 0 or not 1 <= length_seconds <= 900:
        raise ValueError('所选片段起点需大于等于0，长度需在1至900秒之间。')
    base = Path(folder_paths.get_input_directory()).resolve()
    path = (base / filename).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError('原曲文件不存在，或不在ComfyUI input目录中。')
    if encoder not in folder_paths.get_filename_list('audio_encoders') or 'sheetsage' not in encoder.lower():
        raise ValueError('请在专业设置中选择已有的SheetSage旋律识别模型。')
    melody_only = parse_melody_only(melody_only)
    with CurrentNodeContext(prompt_id="liao-abc-" + uuid.uuid4().hex, node_id="liao-abc-button"), torch.inference_mode():
        wave, sr = load(str(path))
        wave = wave[..., round(start_seconds * sr):round((start_seconds + length_seconds) * sr)]
        if wave.shape[-1] < sr:
            raise ValueError('所选片段不足1秒，请调整波形选段。')
        model = cached_encoder(encoder)
        abc = model.generate_abc(wave.unsqueeze(0), sr, melody_only=melody_only)
        if isinstance(abc, list):
            if len(abc) != 1:
                raise ValueError('旋律识别返回多个乐谱，请使用单首音频。')
            abc = abc[0]
        if not str(abc).strip():
            raise ValueError('未提取到ABC旋律，请换一个片段。')
    return {'abc': str(abc)}

class LiaoYuE2Studio:
    CATEGORY = 'liao2049/音乐'
    FUNCTION = 'run'
    RETURN_TYPES = ('AUDIO',)
    RETURN_NAMES = ('歌曲',)
    OUTPUT_NODE = True
    DESCRIPTION = '先提取/规划，编辑歌词与ABC后再生成。复用本地原生YuE2模型，默认INT8。'

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        import hashlib
        import folder_paths
        filename = kwargs.get('原曲文件', '').strip()
        if kwargs.get('操作') != '原曲提取ABC与歌词' or not filename:
            return ''
        base = Path(folder_paths.get_input_directory()).resolve()
        path = (base / filename).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            return 'missing'
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        ckpts = [x for x in folder_paths.get_filename_list('checkpoints') if 'yue2' in x.lower()]
        encoders = [x for x in folder_paths.get_filename_list('audio_encoders') if 'sheetsage' in x.lower()]
        return {'required': {
            '操作': (['写歌并生成', '只规划ABC', '原曲提取ABC与歌词', '按编辑后的ABC改歌', '直接生成不规划'],),
            '风格模板': (list(PRESETS),),
            '风格提示词': text(next(iter(PRESETS.values()))),
            '人声': (['沿用提示词', '温暖女声', '清亮女声', '磁性男声', '沙哑男声', '男女对唱'],),
            '语言': (['中文', 'English', '日本語', '自动'], {'default': '自动'}),
            '歌词': text("[Verse]\nMorning light across the window\nCity waking down below\nI can hear the streets are calling\nFeels like somewhere we should go\n\n[Chorus]\nRun with me into the sunlight\nLeave the shadows far behind\nWe don't need to know tomorrow\nTonight the whole world feels alive\n\n[Verse]\nRadio playing through the open door\nLaughing like we did before\nEvery mile becomes a memory\nAnd I just want a little more\n\n[Chorus]\nRun with me into the sunlight\nLeave the shadows far behind\nWe don't need to know tomorrow\nTonight the whole world feels alive"),
            'ABC乐谱': text(),
            '乐谱模式': (['full', 'melody'], {'default': 'full'}),
            '模型': (ckpts or ['未找到YuE2模型'], {'default': next((x for x in ckpts if 'int8' in x), ckpts[0] if ckpts else '未找到YuE2模型')}),
            '旋律识别模型': (['none'] + encoders, {'default': encoders[0] if encoders else 'none'}),
            '种子': ('INT', {'default': 42, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': True}),
            '最长秒数': ('FLOAT', {'default': 180, 'min': 1, 'max': 900}),
            '采样步数': ('INT', {'default': 32, 'min': 4, 'max': 100}),
            '速度倍率': ('FLOAT', {'default': 1, 'min': .25, 'max': 4, 'step': .01}),
            '升降调半音': ('INT', {'default': 0, 'min': -24, 'max': 24}),
            '自由重编伴奏': ('BOOLEAN', {'default': True}),
            '自动时长预算': ('BOOLEAN', {'default': True}),
            '歌词识别': (['本地Whisper', '跳过使用已填歌词'],),
            'Whisper模型路径': ('STRING', {'default': '', 'tooltip': '本地faster-whisper模型目录；空白使用models/whisper/faster-whisper-base，不自动下载'}),
            '参考起点秒': ('FLOAT', {'default': 0, 'min': 0, 'max': 36000}),
            '参考长度秒': ('FLOAT', {'default': 90, 'min': 1, 'max': 900}),
            '规划温度': ('FLOAT', {'default': .7, 'min': .05, 'max': 2, 'step': .05}),
            '音乐温度': ('FLOAT', {'default': 1, 'min': .05, 'max': 2, 'step': .05}),
            '原曲文件': ('STRING', {'default': '', 'tooltip': '上传后自动填写，或填写input下的相对文件名'}),
            '目标BPM': ('FLOAT', {'default': 0, 'min': 0, 'max': 300, 'tooltip': '0沿用乐谱速度；非0优先于速度倍率，按四分音符每分钟拍数设置'}),
            '拍号': (['自动/沿用', '4/4', '3/4', '6/8', '2/4', '5/4', '7/8'],),
            '调性根音': (['自动/沿用', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'], {'tooltip': '移到指定根音，保留原谱的大调/小调性质；非自动时优先于升降调半音'}),
        }}

    def run(self, **k):
        import torch
        import comfy.model_management as mm
        native = module('comfy_native')
        tools = module('abc_tools')
        timings=[]
        mark=time.perf_counter()
        def record(label):
            nonlocal mark
            now=time.perf_counter();timings.append(f'{label}：{now-mark:.2f}秒');mark=now
        action = k['操作']
        lyrics = normalize_lyrics(k['歌词'])
        abc = k['ABC乐谱'].strip()
        recovered_lyrics = False
        if not lyrics and re.search(r'^\s*\[(?:verse|chorus|bridge|主歌|副歌)', abc, re.M | re.I) and not re.search(r'^(?:X:|K:|V:)', abc, re.M):
            lyrics = normalize_lyrics(abc)
            abc = ''
            recovered_lyrics = True
        style = k['风格提示词'].strip() or PRESETS[k['风格模板']]
        voices = {'温暖女声':'warm expressive female lead vocals','清亮女声':'clear bright female lead vocals','磁性男声':'warm resonant male lead vocals','沙哑男声':'raspy expressive male lead vocals','男女对唱':'alternating male and female duet vocals'}
        if k['人声'] in voices:
            style = re.sub(r'\b(?:warm |expressive |deep |resonant |bright |clear |intimate |soft |powerful |raspy )*(?:female|male)\s+(?:lead\s+)?vocals?\b', '', style, flags=re.I)
            style += '\nVocal direction: ' + voices[k['人声']] + '.'
        if k['语言'] != '自动':
            style += '\nSung language: ' + {'中文':'Mandarin Chinese','日本語':'Japanese'}.get(k['语言'], k['语言']) + '.'
        if k.get('目标BPM', 0):
            style += f"\nTarget tempo: {k['目标BPM']:g} BPM (quarter-note beats)."
        if k.get('拍号', '自动/沿用') != '自动/沿用':
            style += '\nTime signature: ' + k['拍号'] + '.'
        if k.get('调性根音', '自动/沿用') != '自动/沿用':
            style += '\nTonal center: ' + k['调性根音'] + '.'
        report = []
        if recovered_lyrics:
            report.append('已修复旧版字段错位：将ABC框中的歌词恢复为演唱歌词。')
        audio = None
        extracting = action == '原曲提取ABC与歌词'
        if action == '按编辑后的ABC改歌' and not abc:
            raise ValueError('请先提取原曲ABC，或粘贴已编辑的ABC乐谱。')
        if not extracting and not lyrics:
            raise ValueError('请填写要演唱的歌词。YuE2依据歌词谱曲，不会自动写歌词。')
        pipe = k.get('原生管线')
        if extracting:
            source = k.get('原曲')
            if source is None and k.get('原曲文件', '').strip():
                import folder_paths
                from comfy_extras.nodes_audio import load
                base = Path(folder_paths.get_input_directory()).resolve()
                path = (base / k['原曲文件'].strip()).resolve()
                if not path.is_relative_to(base):
                    raise ValueError('原曲文件必须在ComfyUI input目录内。')
                wave, sr = load(str(path))
                source = {'waveform': wave.unsqueeze(0), 'sample_rate': sr}
            if source is None:
                raise ValueError('请连接加载音频节点的AUDIO到原曲输入。')
            sr = source['sample_rate']
            start = round(k['参考起点秒'] * sr)
            end = start + round(k['参考长度秒'] * sr)
            wave = source['waveform'][..., start:end]
            if wave.shape[0] != 1 or wave.shape[-1] < sr:
                raise ValueError('只支持单首参考音频，所选片段需至少1秒。')
            audio = {'waveform': wave, 'sample_rate': sr}
            # Only load SheetSage2 for extraction; do not load the 4GB song checkpoint.
            encoder = pipe.get('audio_encoder') if pipe else None
            if encoder is None:
                import folder_paths
                import comfy.utils
                from comfy.audio_encoders.audio_encoders import load_audio_encoder_from_sd
                if k['旋律识别模型'] == 'none':
                    raise ValueError('请选择sheetsage2旋律识别模型。')
                encoder = cached_encoder(k['旋律识别模型'])
            scores = encoder.generate_abc(wave, sr, melody_only=k['自由重编伴奏'])
            if isinstance(scores, list):
                if len(scores) != 1:
                    raise ValueError('旋律识别返回多个乐谱，请改用单首音频。')
                scores = scores[0]
            abc = str(scores)
            if k['歌词识别'] == '本地Whisper':
                try:
                    lyrics = self.transcribe(audio, k['Whisper模型路径'], k['语言'])
                    report.append('Whisper歌词需人工校对，尤其是伴奏较响、合唱或长音；已回填编辑框。')
                except Exception as e:
                    report.append('歌词识别未完成：' + str(e) + '。ABC已保留，请填歌词或歌词文件内容。')
            report.append('已提取参考片段；返回的音频为原曲片段预听。编辑后切换“按编辑后的ABC改歌”。')
        else:
            if pipe is None:
                pipe = cached_pipeline(k['模型'])
            record('模型准备（缓存复用）')
            if action in ('写歌并生成', '只规划ABC'):
                abc = native.YuE2NativeGenerateABC().generate(pipe, style, lyrics, k['种子'], k['乐谱模式'], temperature=k['规划温度'])[0]
                record('ABC谱曲')
            if action == '直接生成不规划':
                abc = ''
        mode = 'melody' if action == '按编辑后的ABC改歌' and k['自由重编伴奏'] else k['乐谱模式']
        if abc:
            if not extracting:
                target_bpm = float(k.get('目标BPM', 0))
                if target_bpm:
                    abc, tempo_report = tools.ABCModifier().modify(abc, 'override', target_bpm, 1, 'none', 0, False, '')
                    # BPM control uses quarter-note beats consistently.
                    abc = re.sub(r'^Q:.*$', f'Q:1/4={target_bpm:g}', abc, flags=re.M)
                    abc = re.sub(r'\[Q:[^\]]*\]', f'[Q:1/4={target_bpm:g}]', abc)
                    report.append(tempo_report)
                tonic = k.get('调性根音', '自动/沿用')
                if tonic != '自动/沿用':
                    _, _, source_key, _ = tools.analyze_abc(abc)
                    root, _, _ = tools._key_parts(source_key)
                    shift = (tools.KEY_PC[tonic] - tools.KEY_PC[root] + 6) % 12 - 6
                    abc, key_report = tools.ABCModifier().modify(abc, 'keep', 120, 1, 'whole_score', shift, False, '')
                    report.append('目标调性根音：' + tonic + '，保留原谱大小调。' + key_report)
                meter = k.get('拍号', '自动/沿用')
                actual = re.search(r'^M:\s*(\S+)', abc, re.M)
                if meter != '自动/沿用' and (not actual or actual.group(1) != meter):
                    raise ValueError('乐谱拍号未匹配目标' + meter + '。已有旋律不能仅改拍号文字；请切换AI写歌重新谱曲，或选择“自动/沿用”。')
            if not extracting and action != '只规划ABC':
                abc, edit_report = tools.ABCModifier().modify(abc, 'keep' if k.get('目标BPM', 0) or k['速度倍率'] == 1 else 'multiplier', 120, k['速度倍率'], 'whole_score', 0 if k.get('调性根音', '自动/沿用') != '自动/沿用' else k['升降调半音'], mode == 'melody' and k['自由重编伴奏'], '')
                report.append(edit_report)
            if k['语言'] == '中文':
                count = len(re.findall(r'[\u3400-\u9fff]', re.sub(r'\[.*?\]', '', lyrics)))
                notes = sum(1 for voice, _, _ in tools.score_notes(abc) if 'vocal' in voice.lower())
                report.append(f'中文歌词约{count}字，主唱旋律{notes}个音符。长音与一字多音会影响对应关系，请试听校对。')
            else:
                report.append(tools.LyricsMelodyFit().analyze(abc, lyrics, 'English' if k['语言'] == 'English' else 'generic')[0])
        if not extracting and action != '只规划ABC':
            duration = k['最长秒数']
            if abc and k['自动时长预算']:
                _, budget, _, budget_report = tools.calculate_duration_budget(abc)
                duration = min(duration, budget)
                report.append(budget_report)
            model, vae, cond, seconds, limited, info = native.YuE2NativeGenerateMusic().generate(pipe, style, lyrics, abc, k['种子'], mode, duration, temperature=k['音乐温度'])
            record('乐谱处理与音乐语义生成')
            from comfy_extras.nodes_yue2 import FRAMES_PER_SECOND
            import nodes
            latent = {'samples': torch.zeros((1, 64, max(1, round(seconds * FRAMES_PER_SECOND))), device=mm.intermediate_device(), dtype=mm.intermediate_dtype()), 'type': 'audio', 'downscale_ratio_temporal': 1920}
            # Match stock ConditioningZeroOut while preserving YuE2 metadata.
            negative = nodes.ConditioningZeroOut().zero_out(cond)[0]
            samples = nodes.common_ksampler(model, k['种子'], k['采样步数'], 1., 'dpm_2', 'sgm_uniform', cond, negative, latent)[0]
            record('音频采样')
            from comfy_extras.nodes_audio import vae_decode_audio
            audio = vae_decode_audio(vae, samples)
            record('VAE解码')
            report.append(info)
        elif audio is None:
            audio = {'waveform': torch.zeros((1, 2, 4800)), 'sample_rate': 48000}
            report.append('只规划ABC：音频输出为0.1秒静音占位，尚未生成歌曲。')
        if timings:report.append('阶段耗时（保存文件另计）：\n'+'\n'.join(timings))
        return {'ui': {'liao_yue2': [{'abc': abc, 'lyrics': lyrics, 'style': style, 'report': '\n'.join(report), 'action': action}]}, 'result': (audio,)}

    @staticmethod
    def transcribe(audio, model_path, language):
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise ValueError('未安装歌词识别依赖faster-whisper；请手动填写歌词。写歌、改歌和旋律识别仍可使用，不会自动下载。') from error
        import comfy.audio
        import torch
        import folder_paths
        model_path = str(model_path or '').strip().strip('"')
        default_dirs = [Path(folder_paths.models_dir) / 'whisper/faster-whisper-base',
                        ROOT.parents[1] / 'models/whisper/faster-whisper-base']
        candidates = []
        if model_path:
            selected = Path(model_path).expanduser()
            if selected.is_file():
                selected = selected.parent
            candidates.append(selected)
            if not selected.is_absolute():
                candidates.extend([Path(folder_paths.models_dir) / selected, ROOT.parents[1] / selected])
        candidates.extend(default_dirs)
        model_dir = next((p for p in candidates if (p / 'model.bin').is_file()), None)
        if model_dir is None:
            raise ValueError('未找到歌词识别模型，已检查目录：' + '；'.join(str(p) for p in candidates))
        missing = [name for name in ('config.json', 'tokenizer.json', 'vocabulary.txt') if not (model_dir / name).is_file()]
        if missing:
            raise ValueError('本地Whisper模型不完整，缺少：' + '、'.join(missing) + '。请手动填写歌词，或指定完整模型目录。')
        model = WhisperModel(str(model_dir), device='cpu', compute_type='int8', local_files_only=True)
        wave = comfy.audio.resample(audio['waveform'].mean(dim=1), audio['sample_rate'], 16000)[0].detach().float().cpu().numpy()
        segments, _ = model.transcribe(wave, language={'中文':'zh','English':'en','日本語':'ja'}.get(language), condition_on_previous_text=False, vad_filter=False)
        lines = [s.text.strip() for s in segments if s.text.strip()]
        del model
        return normalize_lyrics('\n'.join(lines))

NODE_CLASS_MAPPINGS = {'LiaoYuE2Studio': LiaoYuE2Studio}
NODE_DISPLAY_NAME_MAPPINGS = {'LiaoYuE2Studio': 'YuE2 超级写歌·改歌工作室 | B站：liao_2049'}

def register_lyrics_route():
    import asyncio
    from aiohttp import web
    from server import PromptServer
    if getattr(PromptServer, 'instance', None) is None:
        return
    gate = asyncio.Semaphore(1)

    @PromptServer.instance.routes.post('/liao_yue2/recognize_abc')
    async def recognize_abc(request):
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError('请求格式错误。')
            async with gate:
                result = await asyncio.to_thread(recognize_uploaded_abc,
                    payload.get('filename', ''), payload.get('start_seconds', 0),
                    payload.get('length_seconds', 90), payload.get('encoder', 'none'),
                    payload.get('melody_only', True))
            return web.json_response(result)
        except (ValueError, TypeError) as error:
            return web.json_response({'error': str(error)}, status=400)
        except Exception as error:
            return web.json_response({'error': 'ABC识别失败：' + str(error)}, status=500)

    @PromptServer.instance.routes.post('/liao_yue2/recognize_lyrics')
    async def recognize_lyrics(request):
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError('请求格式错误。')
            async with gate:
                result = await asyncio.to_thread(recognize_uploaded_lyrics,
                    payload.get('filename', ''), payload.get('start_seconds', 0),
                    payload.get('length_seconds', 30), payload.get('language', '中文'),
                    payload.get('model_path', ''))
            return web.json_response(result)
        except (ValueError, TypeError) as error:
            return web.json_response({'error': str(error)}, status=400)
        except Exception as error:
            return web.json_response({'error': '歌词识别失败：' + str(error)}, status=500)

register_lyrics_route()
