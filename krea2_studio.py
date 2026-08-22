"""Compact Krea2 text/image/style studio built from the verified two-stage workflow."""

import base64
import math
import mimetypes
import os
import re
from collections import OrderedDict

import folder_paths
from comfy_execution.graph_utils import GraphBuilder

from .krea2_prompt_instruct import (
    KREA2_IMAGE_WASH_SYSTEM,
    KREA2_STYLE_SYSTEM,
    KREA2_TEXT_SYSTEM,
    _build_image_wash_prompt,
    _build_style_prompt,
    _build_text_prompt,
    _enforce_wash_no_text,
)
from .minimax_h3_unified import _WenWuEmbeddedLlama, _build_messages


MODES = ("文生图", "洗图", "风格参考")
RATIOS = {"方形 1:1": (1, 1), "竖图 3:4": (3, 4), "横图 4:3": (4, 3), "竖屏 9:16": (9, 16), "横屏 16:9": (16, 9)}
PIXELS = {"标准 1MP": 1_048_576, "高清 1.5MP": 1_572_864, "超清 2MP": 2_097_152}


def _filter(names, *tokens):
    result = [name for name in names if any(token in name.lower() for token in tokens)]
    return result or list(names) or ["未找到模型"]


def _pick(names, *tokens):
    return next((name for name in names if all(token in name.lower() for token in tokens)), names[0])


def _dimensions(ratio_name, resolution_name):
    rw, rh = RATIOS.get(ratio_name, (1, 1))
    pixels = PIXELS.get(resolution_name, 1_048_576)
    width = max(256, round(math.sqrt(pixels * rw / rh) / 16) * 16)
    height = max(256, round(width * rh / rw / 16) * 16)
    return width, height


def _image_data_url(filename):
    path = folder_paths.get_annotated_filepath(filename)
    if not path or not os.path.isfile(path):
        raise ValueError(f"找不到参考图片：{filename}")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as handle:
        return f"data:{mime};base64," + base64.b64encode(handle.read()).decode("ascii")


def _clean_llm_output(text):
    value = str(text or "").strip()
    value = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", value, flags=re.I | re.S).strip()
    value = re.sub(r"^(?:final prompt|prompt)\s*:\s*", "", value, flags=re.I).strip()
    return value


class LiaoKrea2Studio:
    @classmethod
    def INPUT_TYPES(cls):
        diffusion = _filter(folder_paths.get_filename_list("diffusion_models"), "krea2")
        encoders = _filter(folder_paths.get_filename_list("text_encoders"), "krea2", "qwen3vl")
        vaes = _filter(folder_paths.get_filename_list("vae"), "qwen_image", "qwen-image")
        try:
            if "LLM" not in folder_paths.folder_names_and_paths:
                folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))
            ggufs = [name for name in folder_paths.get_filename_list("LLM") if name.lower().endswith(".gguf")]
        except Exception:
            ggufs = []
        llms = [name for name in ggufs if "mmproj" not in name.lower()]
        mmprojs = [name for name in ggufs if "mmproj" in name.lower()]
        files = sorted(folder_paths.filter_files_content_types(os.listdir(folder_paths.get_input_directory()), ["image"]))
        required = OrderedDict([
            ("模式", (list(MODES), {"default": "文生图"})),
            ("提示词", ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})),
            ("参考图片", (["未选择"] + files, {"default": "未选择", "image_upload": True})),
            ("画面比例", (list(RATIOS), {"default": "方形 1:1"})),
            ("输出分辨率", (list(PIXELS), {"default": "标准 1MP"})),
            ("随机种子", ("INT", {"default": 40931725392963, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})),
            ("Raw模型", (diffusion, {"default": _pick(diffusion, "raw")})),
            ("Turbo模型", (diffusion, {"default": _pick(diffusion, "turbo")})),
            ("文本编码器", (encoders, {"default": encoders[0]})),
            ("VAE", (vaes, {"default": vaes[0]})),
            ("智能改写", ("BOOLEAN", {"default": True})),
            ("Llama模型", ("STRING", {"default": llms[0] if llms else ""})),
            ("视觉模型", ("STRING", {"default": mmprojs[0] if mmprojs else ""})),
            ("Llama上下文", ("INT", {"default": 8192, "min": 1024, "max": 131072, "step": 1024})),
        ])
        return {"required": required}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("图像", "最终提示词")
    FUNCTION = "generate"
    CATEGORY = "Liao2049/Krea2"

    def generate(self, 模式, 提示词, 参考图片, 画面比例, 输出分辨率, 随机种子,
                 Raw模型, Turbo模型, 文本编码器, VAE, 智能改写,
                 Llama模型, 视觉模型, Llama上下文):
        mode = 模式 if 模式 in MODES else "文生图"
        source = str(提示词 or "").strip()
        needs_image = mode in {"洗图", "风格参考"}
        if needs_image and (not 参考图片 or 参考图片 == "未选择"):
            raise ValueError(f"{mode}模式需要上传一张参考图片。")
        if not source and mode != "洗图":
            raise ValueError("请输入希望生成的画面内容。")

        final_prompt = source
        if 智能改写 or needs_image:
            if not Llama模型:
                raise ValueError("当前模式需要提示词推理，请把 GGUF 放入 models/LLM 并选择 Llama 模型。")
            if needs_image and not 视觉模型:
                raise ValueError("洗图和风格参考需要 mmproj 视觉模型，请放入 models/LLM。")
            if mode == "洗图":
                system, request = KREA2_IMAGE_WASH_SYSTEM, _build_image_wash_prompt(source)
            elif mode == "风格参考":
                system, request = KREA2_STYLE_SYSTEM, _build_style_prompt(source)
            else:
                system, request = KREA2_TEXT_SYSTEM, _build_text_prompt(source)
            urls = [_image_data_url(参考图片)] if needs_image else []
            final_prompt = _clean_llm_output(_WenWuEmbeddedLlama.invoke(
                Llama模型, int(Llama上下文), "全部GPU",
                _build_messages(system, request, urls, image_detail="high"),
                vision_model=视觉模型 if needs_image else "",
                temperature=0.2 if mode == "洗图" else 0.55,
                top_p=0.8, max_tokens=2300 if mode == "洗图" else 1400,
                repeat_penalty=1.08,
            ))
            if mode == "洗图":
                final_prompt = _enforce_wash_no_text(final_prompt)

        width, height = _dimensions(画面比例, 输出分辨率)
        g = GraphBuilder()
        raw = g.node("UNETLoader", unet_name=Raw模型, weight_dtype="default")
        raw_shift = g.node("ModelSamplingAuraFlow", model=raw.out(0), shift=3.0)
        turbo = g.node("UNETLoader", unet_name=Turbo模型, weight_dtype="default")
        turbo_shift = g.node("ModelSamplingAuraFlow", model=turbo.out(0), shift=3.0)
        clip = g.node("CLIPLoader", clip_name=文本编码器, type="krea2", device="cpu")
        vae = g.node("VAELoader", vae_name=VAE)
        positive = g.node("CLIPTextEncode", clip=clip.out(0), text=final_prompt)
        negative = g.node("ConditioningZeroOut", conditioning=positive.out(0))
        latent = g.node("EmptyLatentImage", width=width, height=height, batch_size=1)
        first = g.node(
            "KSamplerAdvanced", model=raw_shift.out(0), add_noise="enable", noise_seed=int(随机种子),
            steps=12, cfg=3.5, sampler_name="euler", scheduler="simple",
            positive=positive.out(0), negative=negative.out(0), latent_image=latent.out(0),
            start_at_step=0, end_at_step=3, return_with_leftover_noise="enable",
        )
        second = g.node(
            "KSamplerAdvanced", model=turbo_shift.out(0), add_noise="disable", noise_seed=int(随机种子),
            steps=12, cfg=1.0, sampler_name="euler", scheduler="simple",
            positive=positive.out(0), negative=negative.out(0), latent_image=first.out(0),
            start_at_step=3, end_at_step=10000, return_with_leftover_noise="disable",
        )
        image = g.node("VAEDecode", samples=second.out(0), vae=vae.out(0))
        return {"result": (image.out(0), final_prompt), "expand": g.finalize()}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Portable string GGUF fields must never trigger ComfyUI's missing-model
        # preflight when text mode runs with smart rewriting disabled.
        return True


NODE_CLASS_MAPPINGS = {"LiaoKrea2Studio": LiaoKrea2Studio}
NODE_DISPLAY_NAME_MAPPINGS = {"LiaoKrea2Studio": "Liao-Krea2 智能生图台"}
