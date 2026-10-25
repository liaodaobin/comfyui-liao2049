"""Cloud credentials stay in process memory; workflows carry only opaque handles."""
import secrets
import threading
from aiohttp import web
from server import PromptServer
from .minimax_h3_unified import _cloud_prompt_invoke
_KEYS = {}
_LOCK = threading.Lock()
def cloud_key(token):
    with _LOCK:
        key = _KEYS.get(str(token or ""), "")
    if not key:
        raise ValueError("请在当前浏览器填写 DeepSeek Key，等待本机凭据就绪后重试。")
    return key

def cloud_invoke(model, messages, token="", api_key=""):
    return _cloud_prompt_invoke("DeepSeek API", api_key or cloud_key(token), model or "deepseek-flash", messages)

@PromptServer.instance.routes.post("/liao/image/cloud-session")
async def cloud_session(request):
    data = await request.json()
    key = str(data.get("api_key", "")).strip()
    if not key:
        return web.json_response({"error":"请填写 DeepSeek API Key。"}, status=400)
    with _LOCK:
        token = next((t for t, value in _KEYS.items() if value == key), None)
        if token is None:
            if len(_KEYS) >= 256:
                return web.json_response({"error":"本机凭据会话已满，请重启 ComfyUI。"}, status=429)
            token = secrets.token_urlsafe(32)
            _KEYS[token] = key
    return web.json_response({"token":token})
