"""Dependency diagnostics and opt-in GPU installer for Liao-H3."""
from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import traceback
import urllib.error
import urllib.request

try:
    import server
    from aiohttp import web
except Exception:
    server = None
    web = None

_LOCK = threading.RLock()
_STATE = {"running": False, "stage": "idle", "ok": None, "log": []}
_PYPI_MIRRORS = (
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple",
    "https://mirrors.cloud.tencent.com/pypi/simple",
)
_CUDA_WHEEL_TAGS = ("cu131", "cu130", "cu128", "cu126", "cu125", "cu124")
_JAMEPENG_RELEASES_API = "https://api.github.com/repos/JamePeng/llama-cpp-python/releases?per_page=100"


def _runtime_environment():
    info = {
        "platform": platform.platform(), "machine": platform.machine(),
        "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "torch_cuda": "", "nvidia": False, "cuda_wheel_tag": "",
    }
    try:
        import torch
        info["torch_cuda"] = str(torch.version.cuda or "")
        info["nvidia"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        parts = info["torch_cuda"].split(".")
        requested = int(parts[0]) * 10 + int(parts[1])
        supported = sorted((int(tag[2:]), tag) for tag in _CUDA_WHEEL_TAGS)
        compatible = [item for item in supported if item[0] <= requested]
        info["cuda_wheel_tag"] = (compatible[-1] if compatible else supported[0])[1]
    except Exception:
        pass
    return info


def _cuda_indexes(tag):
    origin = f"https://abetlen.github.io/llama-cpp-python/whl/{tag}"
    return (origin, f"https://ghfast.top/{origin}", f"https://ghproxy.net/{origin}")


def _compatible_cuda_tags(torch_cuda):
    try:
        major, minor = (int(x) for x in str(torch_cuda).split(".")[:2])
        requested = major * 10 + minor
    except Exception:
        requested = 124
    ranked = [tag for tag in _CUDA_WHEEL_TAGS if int(tag[2:]) <= requested]
    return ranked or ["cu124"]


def _github_json(url, timeout=25):
    urls = (url, f"https://ghfast.top/{url}", f"https://ghproxy.net/{url}")
    last_error = None
    for candidate in urls:
        try:
            request = urllib.request.Request(candidate, headers={
                "Accept": "application/vnd.github+json", "User-Agent": "Liao-H3/1.0"
            })
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            _append_log(f"发布信息地址不可用，尝试下一个：{exc}")
    raise RuntimeError(f"无法读取 JamePeng wheel 发布信息：{last_error}")


def _jamepeng_wheel(runtime):
    releases = _github_json(_JAMEPENG_RELEASES_API, timeout=30)
    python_tag = str(runtime.get("python_tag") or "").lower()
    cuda_tags = _compatible_cuda_tags(runtime.get("torch_cuda"))
    candidates = []
    for release_order, release in enumerate(releases if isinstance(releases, list) else []):
        tag_name = str(release.get("tag_name") or "").lower()
        cuda_tag = next((tag for tag in cuda_tags if tag in tag_name), "")
        if not cuda_tag:
            continue
        version_match = re.search(r"(?:^|v)(\d+)\.(\d+)\.(\d+)", tag_name)
        version = tuple(int(x) for x in version_match.groups()) if version_match else (0, 0, 0)
        if version < (0, 3, 35):
            continue
        for asset in release.get("assets") or []:
            name = str(asset.get("name") or "")
            lowered = name.lower()
            if not (lowered.endswith(".whl") and "llama_cpp_python" in lowered and
                    python_tag in lowered and "win_amd64" in lowered):
                continue
            # New dynamic-backend wheels are preferred; on older releases Basic is
            # portable across more CPUs than AVX-specific builds.
            portable_rank = 2 if version >= (0, 3, 39) else (1 if "basic" in lowered else 0)
            candidates.append((version, portable_rank, -cuda_tags.index(cuda_tag), -release_order, {
                "name": name, "url": str(asset.get("browser_download_url") or ""),
                "cuda_tag": cuda_tag, "release": str(release.get("tag_name") or ""),
            }))
    if not candidates:
        raise RuntimeError(
            f"JamePeng 发布页没有找到适配 {python_tag} / Windows x64 / "
            f"CUDA {runtime.get('torch_cuda') or '未知'} 的 Qwen3.5 GPU wheel。"
        )
    return max(candidates, key=lambda item: item[:4])[4]


def _download_wheel(asset, destination):
    origin = asset["url"]
    urls = (origin, f"https://ghfast.top/{origin}", f"https://ghproxy.net/{origin}")
    last_error = None
    for url in urls:
        try:
            _append_log(f"下载 {asset['name']}：{url}")
            request = urllib.request.Request(url, headers={"User-Agent": "Liao-H3/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, open(destination, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if os.path.getsize(destination) < 1024 * 1024:
                raise RuntimeError("下载结果不是有效的 wheel 文件")
            return
        except Exception as exc:
            last_error = exc
            _append_log(f"该下载地址失败，尝试下一个：{exc}")
            try:
                os.remove(destination)
            except OSError:
                pass
    raise RuntimeError(f"所有 wheel 下载地址均失败：{last_error}")


def _http_json(url, api_key, *, payload=None, timeout=25):
    headers = {"Accept": "application/json", "User-Agent": "Liao-H3/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data, method = None, "GET"
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"], method = "application/json", "POST"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")[:1500]
        try:
            detail = json.loads(raw)
        except Exception:
            detail = {"message": raw or str(exc)}
        return exc.code, detail


def _openai_compatible_base(value):
    base = str(value or "").strip().rstrip("/")
    if not base:
        raise ValueError("请填写 OpenAI 兼容 API 地址。")
    if not re.match(r"^https?://", base, flags=re.IGNORECASE):
        raise ValueError("OpenAI 兼容 API 地址必须以 http:// 或 https:// 开头。")
    if base.lower().endswith("/chat/completions"):
        return base[:-len("/chat/completions")]
    if base.lower().endswith("/v1") or re.search(r"/api/v\d+$", base, flags=re.IGNORECASE):
        return base
    return base + "/v1"


def _api_model_check(provider, api_key, requested_model="", compatible_base_url=""):
    provider, api_key, requested_model = map(lambda x: str(x or "").strip(), (provider, api_key, requested_model))
    if provider != "OpenAI 兼容 API" and not api_key:
        raise ValueError("请先填写云端 API Key。")
    if provider == "MiniMax API":
        base = "https://api.minimaxi.com/v1"
        status, result = _http_json(f"{base}/models", api_key)
        if status != 200:
            return {"ok": False, "provider": provider, "http_status": status, "error": result}
        models = [str(x.get("id") or "").strip() for x in result.get("data", []) if isinstance(x, dict)]
        preferred = ("MiniMax-M2.7", "MiniMax-M2.7-highspeed", "MiniMax-M2.5", "MiniMax-M2.5-highspeed")
        selected = requested_model if requested_model in models else next((x for x in preferred if x in models), models[0] if models else "")
        if not selected:
            return {"ok": False, "provider": provider, "http_status": 200, "error": {"message": "接口未返回可用模型。"}}
        probe_status, probe = _http_json(f"{base}/chat/completions", api_key, payload={
            "model": selected, "messages": [{"role": "user", "content": "Reply OK."}],
            "stream": False, "temperature": 0, "max_completion_tokens": 1,
        })
        return {"ok": probe_status == 200, "provider": provider, "http_status": probe_status,
                "models": models, "selected": selected, "error": None if probe_status == 200 else probe}
    if provider == "Kimi API":
        errors = []
        for base in ("https://api.moonshot.cn/v1", "https://api.kimi.com/coding/v1"):
            status, result = _http_json(f"{base}/models", api_key)
            if status == 200:
                models = [str(x.get("id") or "").strip() for x in result.get("data", []) if isinstance(x, dict)]
                if models:
                    selected = requested_model if requested_model in models else next((x for x in models if x.lower() == "k3"), models[0])
                    return {"ok": True, "provider": provider, "http_status": 200, "models": models, "selected": selected}
            errors.append({"endpoint": base, "http_status": status, "detail": result})
        return {"ok": False, "provider": provider, "http_status": errors[0]["http_status"] if errors else 502, "error": errors}
    if provider == "OpenAI 兼容 API":
        base = _openai_compatible_base(compatible_base_url)
        status, result = _http_json(f"{base}/models", api_key)
        if status != 200:
            return {"ok": False, "provider": provider, "http_status": status, "error": result}
        models = [str(x.get("id") or "").strip() for x in result.get("data", []) if isinstance(x, dict)]
        selected = requested_model if requested_model in models else (models[0] if models else requested_model)
        if not selected:
            return {"ok": False, "provider": provider, "http_status": 200, "error": {"message": "接口未返回模型，请手动填写模型名称。"}}
        return {"ok": True, "provider": provider, "http_status": 200, "models": models, "selected": selected}
    raise ValueError("请选择 MiniMax API、Kimi API 或 OpenAI 兼容 API。")


def _diagnose():
    result = {"python": sys.version.split()[0], "executable": sys.executable,
              "llama_cpp": False, "version": "", "qwen35_handler": False,
              "gpu_offload": False, "complete": False, "runtime": _runtime_environment()}
    try:
        import llama_cpp
        result["llama_cpp"] = True
        result["version"] = str(getattr(llama_cpp, "__version__", "unknown"))
        from llama_cpp.llama_chat_format import Qwen35ChatHandler  # noqa: F401
        result["qwen35_handler"] = True
        probe = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        result["gpu_offload"] = bool(probe()) if callable(probe) else False
    except Exception as exc:
        result["error"] = str(exc)
    result["complete"] = bool(result["llama_cpp"] and result["qwen35_handler"] and result["gpu_offload"])
    return result


def _append_log(text):
    with _LOCK:
        _STATE["log"] = (_STATE["log"] + str(text or "").replace("\r", "").splitlines())[-100:]


def _run_step(stage, args, env):
    with _LOCK:
        _STATE["stage"] = stage
    _append_log(f"[{stage}] {' '.join(args)}")
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", env=env,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    for line in process.stdout or ():
        _append_log(line.rstrip())
    code = process.wait()
    if code:
        raise RuntimeError(f"{stage}失败，退出码 {code}")


def _install_worker():
    try:
        runtime = _runtime_environment()
        if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
            raise RuntimeError("自动安装目前只支持 Windows x64；未执行安装。")
        if not runtime["nvidia"] or not runtime["torch_cuda"]:
            raise RuntimeError("未检测到 NVIDIA CUDA PyTorch；为避免误装 CPU 版，已停止安装。")
        tag = runtime["cuda_wheel_tag"]
        if not tag:
            raise RuntimeError(f"无法为 PyTorch CUDA {runtime['torch_cuda']} 选择 GPU wheel。")
        env = os.environ.copy()
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        _append_log(f"环境：Windows x64 / {runtime['python_tag']} / PyTorch CUDA {runtime['torch_cuda']}；选择 {tag}。")
        asset = _jamepeng_wheel(runtime)
        _append_log(
            f"匹配到 {asset['release']} / {asset['cuda_tag']} / {asset['name']}"
        )
        with tempfile.TemporaryDirectory(prefix="liao_h3_llama_") as temp_dir:
            wheel_path = os.path.join(temp_dir, asset["name"])
            _download_wheel(asset, wheel_path)
            _run_step("安装 JamePeng Qwen3.5 CUDA wheel", [
                sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
                "--no-cache-dir", "--no-deps", wheel_path,
            ], env)
        status = _diagnose()
        if not status["complete"]:
            detail = status.get("error") or "未知导入错误"
            raise RuntimeError(f"wheel 已安装，但 GPU offload 或 Qwen3.5 多模态检测未通过：{detail}")
        _append_log(f"完成：llama_cpp {status['version']}，GPU offload 与 Qwen3.5 多模态可用。")
        with _LOCK:
            _STATE.update({"ok": True, "stage": "complete"})
    except Exception as exc:
        _append_log(str(exc)); _append_log(traceback.format_exc())
        with _LOCK:
            _STATE.update({"ok": False, "stage": "failed"})
    finally:
        with _LOCK:
            _STATE["running"] = False


def _response_payload():
    with _LOCK:
        state = dict(_STATE); state["log"] = list(_STATE["log"])
    state["diagnostics"] = _diagnose()
    tag = state["diagnostics"]["runtime"].get("cuda_wheel_tag") or "cu124"
    state["mirrors"] = list(_PYPI_MIRRORS)
    state["cuda_indexes"] = list(_cuda_indexes(tag))
    state["wheel_source"] = "JamePeng/llama-cpp-python releases"
    return state


if server is not None and web is not None:
    @server.PromptServer.instance.routes.get("/liao_h3/models/llm")
    async def liao_h3_llm_models(_request):
        try:
            import folder_paths
            llm_dir = os.path.join(folder_paths.models_dir, "LLM")
            os.makedirs(llm_dir, exist_ok=True)
            if "LLM" not in folder_paths.folder_names_and_paths:
                folder_paths.add_model_folder_path("LLM", llm_dir)
            models = [
                str(name) for name in folder_paths.get_filename_list("LLM")
                if str(name).lower().endswith(".gguf")
            ]
            vision = [name for name in models if "mmproj" in name.lower()]
            text_models = [name for name in models if name not in vision]
            return web.json_response({"ok": True, "models": text_models, "vision": vision})
        except Exception as exc:
            return web.json_response({"ok": False, "models": [], "vision": [], "error": str(exc)}, status=500)

    @server.PromptServer.instance.routes.get("/liao_h3/dependencies")
    async def liao_h3_dependency_status(_request):
        return web.json_response(_response_payload())

    @server.PromptServer.instance.routes.post("/liao_h3/dependencies/install")
    async def liao_h3_dependency_install(request):
        data = await request.json()
        if data.get("confirm") != "INSTALL_LLAMA_GPU":
            return web.json_response({"error": "缺少安装确认。"}, status=400)
        current = _diagnose()
        with _LOCK:
            if _STATE["running"] or current["complete"]:
                return web.json_response(_response_payload())
            _STATE.update({"running": True, "stage": "starting", "ok": None, "log": []})
        threading.Thread(target=_install_worker, name="liao-h3-dependency-installer", daemon=True).start()
        return web.json_response(_response_payload())

    @server.PromptServer.instance.routes.post("/liao_h3/api/models/check")
    async def liao_h3_api_models_check(request):
        try:
            data = await request.json()
            result = await asyncio.to_thread(
                _api_model_check,
                data.get("provider"), data.get("api_key"), data.get("model"), data.get("base_url"),
            )
            return web.json_response(result, status=200 if result.get("ok") else 400)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"API 检测失败：{exc}"}, status=500)
