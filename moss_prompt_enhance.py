"""MOSS-specific caption enhancement; layout and typography remain code-owned."""
import asyncio
import hashlib
import importlib
import json
from pathlib import Path
import re
import secrets

from aiohttp import web
import folder_paths
import comfy.model_management
from server import PromptServer

from .minimax_h3_unified import _WenWuEmbeddedLlama, _build_messages
from .qwen_image_21_studio import _ensure_llm_folder
from .flux3_layout import layout_elements


TEMPLATE = Path(__file__).parent / "prompt_templates" / "moss_image_enhance.md"


def llm_models():
    _ensure_llm_folder()
    return [n for n in folder_paths.get_filename_list("LLM")
            if n.lower().endswith(".gguf") and "mmproj" not in n.lower()]


def vision_models():
    _ensure_llm_folder()
    return [n for n in folder_paths.get_filename_list("LLM") if n.lower().endswith(".gguf") and "mmproj" in n.lower()]


def enhancement_key(mode, source, layout="[]", negative=""):
    body = [mode, source.strip(), layout if mode == "构图生图" else "[]", negative.strip()]
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def parse_caption(raw, source, mode):
    body = re.split(r"</think\s*>", str(raw), flags=re.I)[-1].strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", body, flags=re.I).strip()
    try:
        data = json.loads(body)
    except ValueError as exc:
        raise ValueError("增强模型没有返回有效的 caption JSON，请重试或换用通用指令模型。") from exc
    if not isinstance(data, dict) or set(data) != {"caption"} or not isinstance(data["caption"], str) or not data["caption"].strip():
        raise ValueError("增强结果应只包含非空 caption，不能返回构图坐标或其他字段。")
    caption = data["caption"].strip()
    if mode == "文生图":
        for match in re.finditer(r'[“"]([^”"]+)[”"]', source):
            prefix = source[max(0, match.start()-24):match.start()]
            if re.search(r'写着|文字|标题|标语|字样|reads?\b|says?\b|text\b|wording\b', prefix, re.I):
                if match.group(1) not in caption:
                    raise ValueError("增强结果改动或遗漏了指定文字，请重试；原始提示词没有被覆盖。")
    return caption


def rewrite_moss(mode, source, layout, model, context=8192, negative="", vision_model="无", provider="本地 Llama", cloud_model="deepseek-flash", token="", api_key=""):
    provider = str(provider or "本地 Llama").strip() or "本地 Llama"
    if mode not in {"文生图", "构图生图"}:
        raise ValueError("无效的 MOSS 模式。")
    if not source.strip():
        raise ValueError("请先输入画面描述。")
    if provider != "本地 Llama":
        if provider != "DeepSeek API":
            raise ValueError("无效的增强服务。")
        from .image_cloud_enhance import cloud_invoke
        canvas = []
        if mode == "构图生图":
            layout_elements(layout)
            canvas = json.loads(layout)
        system = "Expand the user's" + TEMPLATE.read_text(encoding="utf-8").split("Expand the user's", 1)[1]
        request = json.dumps({"mode":mode,"image_idea":source,"locked_canvas":canvas,"excluded_content":negative},ensure_ascii=False)
        raw = cloud_invoke(cloud_model, _build_messages(system,request,[]), token, api_key)
        return parse_caption(raw,source,mode)
    if model not in llm_models():
        raise ValueError("请选择已安装的 GGUF 增强模型；不需要 mmproj。")
    if not 2048 <= int(context) <= 131072:
        raise ValueError("增强上下文范围为 2048–131072。")
    vision_model = "" if vision_model in {"", "无"} else vision_model
    if vision_model and vision_model not in vision_models():
        raise ValueError("请选择已安装且与增强模型配套的视觉模型。")
    canvas = []
    if mode == "构图生图":
        layout_elements(layout)
        # These are context, never accepted back as new geometry or text fields.
        canvas = json.loads(layout)
    request = json.dumps({"mode": mode, "image_idea": source, "locked_canvas": canvas, "excluded_content": negative}, ensure_ascii=False)
    system = TEMPLATE.read_text(encoding="utf-8").split("Expand the user's", 1)[1]
    system = "Expand the user's" + system
    # Release llama.cpp allocations before diffusion sampling needs the GPU.
    with _WenWuEmbeddedLlama._lock:
        try:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            raw = _WenWuEmbeddedLlama.invoke(
                model, int(context), "全部GPU", _build_messages(system, request, [], image_detail="high"),
                vision_model=vision_model, temperature=.25, top_p=.9, max_tokens=1800, repeat_penalty=1.05,
                seed=secrets.randbelow(2_147_483_647),
            )
        finally:
            _WenWuEmbeddedLlama.unload()
    return parse_caption(raw, source, mode)


def register_routes():
    try:
        routes = PromptServer.instance.routes
    except AttributeError:
        return

    @routes.get("/liao_moss/enhance/status")
    async def status(request):
        try:
            importlib.import_module("llama_cpp")
            available, message = True, "本地 Llama 可用；文字增强不需要视觉投影模型。"
        except (ImportError, OSError) as exc:
            available, message = False, f"本地 Llama 依赖未就绪：{exc}"
        return web.json_response({"available": available, "models": llm_models(),
                                  "message": message})

    @routes.post("/liao_moss/enhance")
    async def enhance(request):
        try:
            data = await request.json()
            running, _ = PromptServer.instance.prompt_queue.get_current_queue_volatile()
            if running and data.get("provider", "本地 Llama") == "本地 Llama":
                raise ValueError("当前正在生成图像，请等任务结束后再手动增强；也可使用运行时自动增强。")
            mode, source, layout = str(data.get("mode", "文生图")), str(data.get("prompt", "")), str(data.get("layout", "[]"))
            result = await asyncio.to_thread(rewrite_moss, mode, source, layout,
                str(data.get("model", "")), int(data.get("context", 8192)), str(data.get("negative", "")), str(data.get("vision_model", "无")), str(data.get("provider", "本地 Llama")), str(data.get("cloud_model", "deepseek-flash")), str(data.get("token", "")), str(data.get("api_key", "")))
            return web.json_response({"prompt": result, "key": enhancement_key(mode, source, layout, str(data.get("negative", "")))})
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)


register_routes()
