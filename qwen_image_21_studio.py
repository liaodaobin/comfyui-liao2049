"""Minimal all-in-one Qwen-Image 2.1 generation and editing node."""

import base64
import asyncio
import json
import math
import mimetypes
import os
import re
from collections import OrderedDict
from io import BytesIO

import folder_paths
from comfy_execution.graph_utils import GraphBuilder
from PIL import Image, ImageOps

from .minimax_h3_unified import _WenWuEmbeddedLlama, _build_messages


MODES = ("文生图", "单图编辑", "多图编辑")
RATIOS = {
    "方形 1:1": (1, 1), "竖图 3:4": (3, 4), "横图 4:3": (4, 3),
    "竖屏 9:16": (9, 16), "横屏 16:9": (16, 9), "海报 2:3": (2, 3), "摄影 3:2": (3, 2),
}
RESOLUTIONS = {"标准 1K": 1024, "高清 1.5K": 1536, "原生 2K": 2048}
_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "prompt_templates", "qwen_image_21_rewrite.md")
_EDIT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "prompt_templates", "qwen_image_21_edit.md")


def _names(folder, predicate):
    values = list(folder_paths.get_filename_list(folder))
    # Put compatible models first without hiding other installed files.
    return sorted(values, key=lambda name: (not predicate(name.lower()), name.lower())) or ["未找到模型"]


def _pick(values, *tokens):
    return next((v for v in values if all(t in v.lower() for t in tokens)), values[0])


def _preferred(values, predicate, *tokens):
    matches = [name for name in values if predicate(name.lower())]
    return _pick(matches, *tokens) if matches else values[0]


def _compact_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _is_qwen21_diffusion(name):
    return "qwenimage21" in _compact_name(name)


def _is_qwen21_clip(name):
    return "qwen3vl8b" in _compact_name(name)


def _is_qwen21_vae(name):
    compact = _compact_name(name)
    return "qwenimage21" in compact and "vae" in compact


def _ensure_llm_folder():
    if "LLM" not in folder_paths.folder_names_and_paths:
        folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))


def _image_files():
    return sorted(folder_paths.filter_files_content_types(os.listdir(folder_paths.get_input_directory()), ["image"]))


def _data_url(filename):
    path = folder_paths.get_annotated_filepath(filename)
    if not path or not os.path.isfile(path):
        raise ValueError(f"找不到参考图片：{filename}")
    with Image.open(path) as source:
        if max(source.size) > 1024:
            image = ImageOps.exif_transpose(source)
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                image = image.convert("RGBA")
                background = Image.new("RGB", image.size, "white")
                background.paste(image, mask=image.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=90, subsampling=0, optimize=True)
            return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as handle:
        return f"data:{mime};base64," + base64.b64encode(handle.read()).decode("ascii")


def _clean(text):
    value = str(text or "").strip()
    # Some GGUF chat templates leak internal reasoning, sometimes even after
    # an initial draft answer. Only the answer after the last closing tag is
    # eligible to become a generation prompt.
    if re.search(r"</think\s*>", value, flags=re.I):
        value = re.split(r"</think\s*>", value, flags=re.I)[-1].strip()
    value = re.sub(r"^```(?:text|markdown|json)?\s*|\s*```$", "", value, flags=re.I | re.S).strip()
    value = re.sub(r"^(?:final prompt|rewritten prompt|prompt|最终提示词|提示词)\s*[:：]\s*", "", value, flags=re.I).strip()
    # A line-wrapped final answer is still one usable image prompt.
    return re.sub(r"\s*\r?\n\s*", " ", value.strip('"').strip()).strip()


def _valid_rewrite(mode, text):
    if not text or re.search(r"</?think\b|<\|im_(?:start|end)\|>", text, flags=re.I):
        return False
    if len(text) > (1200 if mode != "文生图" else 5000):
        return False
    return not re.search(r"\b(?:let me|the user said|thinking process|final answer|draft)\b", text, flags=re.I)


def _is_chinese_prose(text):
    prose = re.sub(r'"[^"\n]*"|<image\d+>', '', text, flags=re.I)
    han = len(re.findall(r'[\u4e00-\u9fff]', prose))
    latin_words = len(re.findall(r'[A-Za-z]+', prose))
    return han >= 6 and han >= latin_words * 2


def _extract_t2i_prompt(text):
    """PE-T2I normally returns a JSON object; its ratio is advisory only."""
    value = _clean(text)
    try:
        payload = json.loads(value)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict):
        value = _clean(payload.get("rewritten_prompt", ""))
    elif value.startswith("{"):
        return ""
    return value if _valid_rewrite("文生图", value) and not _is_chinese_prose(value) else ""


def _dimensions(ratio_name, resolution_name):
    rw, rh = RATIOS.get(ratio_name, (1, 1))
    edge = RESOLUTIONS.get(resolution_name, 1024)
    pixels = edge * edge
    width = max(256, round(math.sqrt(pixels * rw / rh) / 32) * 32)
    height = max(256, round(math.sqrt(pixels * rh / rw) / 32) * 32)
    return width, height


def _load_image_node(graph, filename):
    return graph.node("LoadImage", image=filename).out(0)


def rewrite_prompt(mode, source, refs, t2i_model, i2i_model, mmproj, context):
    with open(_TEMPLATE_PATH if mode == "文生图" else _EDIT_TEMPLATE_PATH, "r", encoding="utf-8") as handle:
        system = handle.read()
    if mode == "文生图":
        model_name, vision_name, urls = t2i_model, "", []
        request = source
    else:
        model_name, vision_name = i2i_model, mmproj
        urls = [_data_url(name) for name in refs]
        roles = "The sole image is the base edit target; do not use an image tag." if mode == "单图编辑" else (
            "Decide whether this is a local edit or a new scene/composite. "
            "For a local edit, <image1> is the base canvas. For a new scene featuring subjects "
            "from multiple references, every image is an identity or asset source; do not force "
            "the old <image1> setting or composition into the new scene. Describe each image's "
            "distinct role. If the user wants subjects from two images together, keep both "
            "subjects present and separate; never replace one with the other."
        )
        request = f"Mode: {mode}.\nReference binding: {roles}\nUser edit request: {source}"
    if not model_name:
        raise ValueError("没有找到对应的 GGUF 反推模型。")
    if refs and not vision_name:
        raise ValueError("图像反推需要 mmproj 视觉投影模型。")
    if mode == "文生图":
        result = _extract_t2i_prompt(_WenWuEmbeddedLlama.invoke(
            model_name, int(context), "全部GPU",
            _build_messages(system, request, [], image_detail="high"),
            temperature=0.25, top_p=0.9, max_tokens=1800,
            repeat_penalty=1.05,
        ))
        if result:
            return result
        result = _extract_t2i_prompt(_WenWuEmbeddedLlama.invoke(
            model_name, int(context), "全部GPU",
            _build_messages(system + "\nReturn the final JSON answer immediately, without analysis.", request, [], image_detail="high"),
            temperature=0.1, top_p=0.8, max_tokens=1500,
            repeat_penalty=1.05,
        ))
        if not result:
            raise ValueError("文生图 LLM 未返回有效的英文提示词，请重试或更换文生图 LLM 模型。")
        return result
    result = _clean(_WenWuEmbeddedLlama.invoke(
        model_name, int(context), "全部GPU",
        _build_messages(system, request, urls, image_detail="high"),
        vision_model=vision_name, temperature=0.25, top_p=0.8,
        max_tokens=2600, repeat_penalty=1.08,
    ))
    if not _valid_rewrite(mode, result):
        retry_system = system + (
            "\nReturn the final image prompt immediately. No analysis, reasoning, drafts, "
            "self-correction, <think> tags, or line breaks. For editing, use at most "
            "400 Chinese characters and include only the requested change, reference roles, "
            "essential preservation, and new setting."
        )
        result = _clean(_WenWuEmbeddedLlama.invoke(
            model_name, int(context), "全部GPU",
            _build_messages(retry_system, request, urls, image_detail="high"),
            vision_model=vision_name, temperature=0.1, top_p=0.8,
            max_tokens=1600, repeat_penalty=1.08,
        ))
        if not _valid_rewrite(mode, result):
            raise ValueError("反推模型返回了思考过程或过长内容，请重试或更换 LLM 模型。")
    if not _is_chinese_prose(result):
        correction = (
            "将下列图像生成提示词改写为简体中文。描述性内容必须全部用中文；"
            "保留 <image1> 等图片标签、专有名称，以及双引号内用户指定在图像中显示的原文。"
            "不得改变人物数量、图像对应关系、动作或场景。只输出最终中文提示词。"
        )
        result = _clean(_WenWuEmbeddedLlama.invoke(
            model_name, int(context), "全部GPU",
            _build_messages(correction, result, [], image_detail="high"),
            vision_model=vision_name, temperature=0.1, top_p=0.8,
            max_tokens=3000, repeat_penalty=1.0,
        ))
        if not _valid_rewrite(mode, result) or not _is_chinese_prose(result):
            raise ValueError("反推模型未按要求输出中文提示词，请检查所选 LLM 模型。")
    return result


def _register_preview_route():
    try:
        from aiohttp import web
        from server import PromptServer
        @PromptServer.instance.routes.post("/liao_qwen21/rewrite")
        async def liao_qwen21_rewrite(request):
            try:
                data = await request.json()
                mode = data.get("mode", "文生图")
                if mode not in MODES:
                    raise ValueError("无效模式。")
                source = str(data.get("prompt", "")).strip()
                refs = list(data.get("images") or [])[:10]
                if not source:
                    raise ValueError("请先输入画面描述或编辑要求。")
                if mode == "单图编辑" and len(refs) < 1 or mode == "多图编辑" and len(refs) < 2:
                    raise ValueError("当前编辑模式缺少参考图。")
                if mode == "单图编辑":
                    refs = refs[:1]
                result = await asyncio.to_thread(
                    rewrite_prompt, mode, source, refs,
                    data.get("t2i_model", ""), data.get("i2i_model", ""),
                    data.get("mmproj", ""), int(data.get("context", 8192)),
                )
                return web.json_response({"prompt": result})
            except Exception as exc:
                return web.json_response({"error": str(exc)}, status=400)
    except (ImportError, AttributeError):
        pass


_register_preview_route()


class LiaoQwenImage21Studio:
    @classmethod
    def INPUT_TYPES(cls):
        diffusion = _names("diffusion_models", _is_qwen21_diffusion)
        encoders = _names("text_encoders", _is_qwen21_clip)
        vaes = _names("vae", _is_qwen21_vae)
        _ensure_llm_folder()
        ggufs = [n for n in folder_paths.get_filename_list("LLM") if n.lower().endswith(".gguf")]
        llms = [n for n in ggufs if "mmproj" not in n.lower()]
        mmprojs = [n for n in ggufs if "mmproj" in n.lower() and ("qwen-image-2.1" in n.lower() or "qwenimage2.1" in n.lower())]
        loras = ["不使用"] + list(folder_paths.get_filename_list("loras"))
        images = ["未选择"] + _image_files()
        required = OrderedDict([
            ("模式", (list(MODES), {"default": "文生图"})),
            ("提示词", ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})),
            ("参考图1", (images, {"default": "未选择"})),
            ("参考图2", (images, {"default": "未选择"})),
            ("参考图3", (images, {"default": "未选择"})),
            ("参考图4", (images, {"default": "未选择"})),
            ("参考图5", (images, {"default": "未选择"})),
            ("画面比例", (list(RATIOS), {"default": "方形 1:1"})),
            ("输出分辨率", (list(RESOLUTIONS), {"default": "标准 1K"})),
            ("随机种子", ("INT", {"default": 40931725392963, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})),
            ("智能反推改写", ("BOOLEAN", {"default": False})),
            ("负面提示词", ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})),
            ("采样步数", ("INT", {"default": 25, "min": 1, "max": 100, "step": 1})),
            ("CFG", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1})),
            ("扩散模型", (diffusion, {"default": _preferred(diffusion, _is_qwen21_diffusion, "int8")})),
            ("文本编码器", (encoders, {"default": _preferred(encoders, _is_qwen21_clip, "int8")})),
            ("VAE", (vaes, {"default": _preferred(vaes, _is_qwen21_vae, "2.1")})),
            ("文生图反推模型", (llms or [""], {"default": _pick(llms, "qwen-image-2.1", "t2i") if llms else ""})),
            ("图像反推模型", (llms or [""], {"default": _pick(llms, "qwen-image-2.1", "i2i") if llms else ""})),
            ("视觉投影模型", (mmprojs or [""], {"default": (mmprojs or [""])[0]})),
            ("Llama上下文", ("INT", {"default": 8192, "min": 2048, "max": 131072, "step": 1024})),
        ])
        for index in range(6, 11):
            required[f"参考图{index}"] = (images, {"default": "未选择"})
        for index in (1, 2):
            required[f"LoRA{index}"] = (loras, {"default": "不使用"})
            required[f"LoRA{index}强度"] = ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05})
        required["已采用增强提示词"] = ("STRING", {"default": "", "multiline": True})
        return {"required": required, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "generate"
    CATEGORY = "Liao2049/Qwen Image 2.1"

    def generate(self, 模式, 提示词, 参考图1, 参考图2, 参考图3, 参考图4, 参考图5,
                 画面比例, 输出分辨率, 随机种子, 智能反推改写, 负面提示词, 采样步数, CFG,
                 扩散模型, 文本编码器, VAE, 文生图反推模型, 图像反推模型, 视觉投影模型, Llama上下文,
                 参考图6="未选择", 参考图7="未选择", 参考图8="未选择", 参考图9="未选择", 参考图10="未选择",
                 LoRA1="不使用", LoRA1强度=1.0, LoRA2="不使用", LoRA2强度=1.0,
                 已采用增强提示词="", unique_id=None):
        mode = 模式 if 模式 in MODES else "文生图"
        source = str(提示词 or "").strip()
        refs = [x for x in (参考图1, 参考图2, 参考图3, 参考图4, 参考图5,
                            参考图6, 参考图7, 参考图8, 参考图9, 参考图10) if x and x != "未选择"]
        if mode == "文生图":
            refs = []
        if not source:
            raise ValueError("请输入生成或编辑要求。")
        if mode == "单图编辑" and len(refs) < 1:
            raise ValueError("单图编辑需要参考图1。")
        if mode == "多图编辑" and len(refs) < 2:
            raise ValueError("多图编辑至少需要参考图1和参考图2；参考图1是基础画布。")
        if mode == "单图编辑":
            refs = refs[:1]

        final_prompt = str(已采用增强提示词 or "").strip() or source
        if 智能反推改写 and not str(已采用增强提示词 or "").strip():
            final_prompt = rewrite_prompt(mode, source, refs, 文生图反推模型,
                                          图像反推模型, 视觉投影模型, Llama上下文)

        width, height = _dimensions(画面比例, 输出分辨率)
        graph = GraphBuilder()
        model = graph.node("UNETLoader", unet_name=扩散模型, weight_dtype="default")
        clip = graph.node("CLIPLoader", clip_name=文本编码器, type="qwen_image", device="default")
        model_out, clip_out = model.out(0), clip.out(0)
        for lora_name, strength in ((LoRA1, LoRA1强度), (LoRA2, LoRA2强度)):
            if lora_name and lora_name != "不使用" and float(strength) != 0:
                applied = graph.node("LoraLoader", model=model_out, clip=clip_out,
                                     lora_name=lora_name, strength_model=float(strength), strength_clip=float(strength))
                model_out, clip_out = applied.out(0), applied.out(1)
        vae = graph.node("VAELoader", vae_name=VAE)

        if refs:
            model_out = graph.node("QwenImage21Cache", model=model_out,
                                   device="auto", dtype="default").out(0)
            loaded = [_load_image_node(graph, name) for name in refs]
            # TextEncodeQwenImage21 takes its output latent aspect ratio from
            # image_1. Match that base canvas to the selected ratio first;
            # otherwise an uploaded portrait silently overrides "方形 1:1".
            loaded[0] = graph.node(
                "ImageScale", image=loaded[0], upscale_method="lanczos",
                width=width, height=height, crop="center",
            ).out(0)
            encode_args = {
                "clip": clip_out, "prompt": final_prompt, "negative_prompt": "",
                # Match the original edit workflow: only image_1 sets the
                # canvas; other references keep their own native size.
                "vae": vae.out(0), "resolution": 0,
            }
            for index, image in enumerate(loaded, 1):
                encode_args[f"images.image_{index}"] = image
            encoded = graph.node("TextEncodeQwenImage21", **encode_args)
            latent = encoded.out(2)
        else:
            encoded = graph.node(
                "TextEncodeQwenImage21", clip=clip_out, prompt=final_prompt,
                negative_prompt="", vae=vae.out(0), resolution=RESOLUTIONS.get(输出分辨率, 1024),
            )
            latent = graph.node("EmptyLatentImage", width=width, height=height, batch_size=1).out(0)

        sampled = graph.node(
            "KSampler", model=model_out, seed=int(随机种子), steps=int(采样步数), cfg=1.0,
            sampler_name="euler", scheduler="simple", positive=encoded.out(0), negative=encoded.out(1), latent_image=latent,
        )
        image = graph.node("VAEDecode", samples=sampled.out(0), vae=vae.out(0))
        if unique_id is not None:
            try:
                from server import PromptServer
                PromptServer.instance.send_sync("liao_qwen21_generation_start", {
                    "node_id": str(unique_id), "width": width, "height": height,
                    "ratio": 画面比例, "resolution": 输出分辨率,
                })
            except (ImportError, AttributeError):
                pass
        return {"result": (image.out(0),), "expand": graph.finalize()}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True


NODE_CLASS_MAPPINGS = {"LiaoQwenImage21Studio": LiaoQwenImage21Studio}
NODE_DISPLAY_NAME_MAPPINGS = {"LiaoQwenImage21Studio": "Liao-QwenImage2.1 智能生图台"}
