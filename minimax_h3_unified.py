"""文武 H3 多参视频生成：把官方 Ref2VA 工作流封装为单节点。"""
from __future__ import annotations

import os
import re
import math
import json
import base64
import io
import importlib
import threading
import urllib.error
import urllib.request
from collections import OrderedDict

from .h3_latent_upscaler_embedded import LiaoH3EmbeddedLatentUpscaler3D

CATEGORY = "Liao2049/MiniMax H3"


def _build_messages(system_prompt, user_text, image_urls, image_detail="auto"):
    """Build OpenAI-compatible multimodal messages without legacy-node imports."""
    parts = []
    for index, url in enumerate(image_urls or []):
        parts.append({"type": "text", "text": f"\n[Image {index}]:"})
        parts.append({"type": "image_url", "image_url": {"url": url, "detail": image_detail}})
    if user_text:
        parts.append({"type": "text", "text": user_text})
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": parts}]
FPS = 24
DEFAULT_MODEL_OPTIONS = [
    "H3/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "H3/minimax_h3_ref2va_int8_convrot.safetensors",
    "H3/minimax_h3_ref2va_bf16.safetensors",
]
FL2VA_MODEL_NAME = "minimax_h3_fl2va_int8_convrot.safetensors"
# 与 Downloads/video_minimax_h3_r2v.json 官方本地工作流保持一致。
DEFAULT_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
SAGE_MODES = [
    "disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda",
    "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda",
    "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean",
]

PREFERRED_H3_TURBO_LORA = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16"
PREFERRED_H3_BALANCED_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16"


def _pick_minimax_h3_turbo_lora(loras):
    """Prefer the recommended H3 Turbo LoRA, then rank compatible installed H3 LoRAs."""
    ranked = []
    for index, filename in enumerate(loras or []):
        name = str(filename or "")
        lower_name = name.replace("\\", "/").lower()
        compact = re.sub(r"[^a-z0-9]", "", lower_name)
        if PREFERRED_H3_TURBO_LORA in lower_name:
            return name
        if "minimaxh3" not in compact:
            continue
        score = 0
        score += 50 if ("fl2v" in compact or "fl2va" in compact) else 0
        score += 40 if "turbo" in compact else 0
        score += 30 if "4step" in compact else 0
        score += 10 if "comfyui" in compact else 0
        score += 5 if "bf16" in compact else 0
        score -= 15 if "lightx2v" in compact else 0
        ranked.append((score, -index, name))
    return max(ranked, default=(0, 0, None))[2]


def _pick_minimax_h3_balanced_lora(loras):
    """Prefer the supplied 8-step H3 LoRA and never fall back to a 4-step LoRA."""
    ranked = []
    for index, filename in enumerate(loras or []):
        name = str(filename or "")
        lower_name = name.replace("\\", "/").lower()
        compact = re.sub(r"[^a-z0-9]", "", lower_name)
        if PREFERRED_H3_BALANCED_LORA in lower_name:
            return name
        if "minimaxh3" not in compact or "8step" not in compact or "4step" in compact:
            continue
        score = 0
        score += 50 if ("fl2v" in compact or "fl2va" in compact) else 0
        score += 40 if "turbo" in compact else 0
        score += 10 if "comfyui" in compact else 0
        score += 5 if "bf16" in compact else 0
        ranked.append((score, -index, name))
    return max(ranked, default=(0, 0, None))[2]

RATIOS = {
    "1:1 (Square)": (1, 1), "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2), "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3), "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9), "21:9 (Ultrawide)": (21, 9),
}
MEGAPIXELS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98, 1.0, 1.2, 1.5, 1.8, 2.0]


def _is_minimax_h3_video_model(filename: str) -> bool:
    """Accept only MiniMax H3 FL2VA/Ref2VA diffusion models from any publisher."""
    compact = re.sub(r"[^a-z0-9]", "", str(filename or "").lower())
    return "minimaxh3" in compact and ("fl2va" in compact or "ref2va" in compact)


class _WenWuEmbeddedLlama:
    """Liao-H3 llama.cpp runtime with an optional invisible llama-cpp-vlm backend."""

    _lock = threading.RLock()
    _model = None
    _chat_handler = None
    _config = None
    _external_storage = None

    @classmethod
    def _find_external_storage(cls):
        """Find the registered ComfyUI-llama-cpp_vlm runtime without importing by path."""
        try:
            comfy_nodes = importlib.import_module("nodes")
            mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})
            loader_cls = mappings.get("llama_cpp_model_loader")
            instruct_cls = mappings.get("llama_cpp_instruct_adv")
            if loader_cls is None or instruct_cls is None:
                return None
            module = importlib.import_module(loader_cls.__module__)
            storage = getattr(module, "LLAMA_CPP_STORAGE", None)
            if storage is None or not callable(getattr(storage, "load_model", None)):
                return None
            return storage
        except Exception:
            return None

    @classmethod
    def _invoke_external(cls, model_name, vision_name, n_ctx, messages, params):
        storage = cls._find_external_storage()
        if storage is None:
            return None
        config = {
            "model": model_name,
            "mmproj": vision_name or "None",
            "chat_handler": "Qwen3.5" if vision_name else "None",
            "n_ctx": int(n_ctx),
            "vram_limit": -1,
            "image_max_tokens": 0,
            "image_min_tokens": 0,
            "load_mtp": False,
        }
        try:
            with cls._lock:
                if not getattr(storage, "llm", None) or getattr(storage, "current_config", None) != config:
                    print(f"[Liao-H3] 后台复用 ComfyUI-llama-cpp_vlm：{model_name}")
                    storage.load_model(config)
                result = storage.llm.create_chat_completion(messages=messages, stream=False, **params)
                content = result["choices"][0]["message"]["content"]
                try:
                    storage.llm.n_tokens = 0
                    storage.llm._ctx.memory_clear(True)
                except Exception:
                    pass
                cls._external_storage = storage
                return str(content or "").strip()
        except Exception as exc:
            # This backend is optional. Upstream private-API changes must not break Liao-H3.
            print(f"[Liao-H3] llama-cpp-vlm 后台不可用，回退内置推理：{exc}")
            try:
                storage.clean(all=True)
            except Exception:
                pass
            cls._external_storage = None
            return None

    @classmethod
    def invoke(cls, model_name, n_ctx, gpu_mode, messages, vision_model="", **params):
        try:
            import folder_paths
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "WenWu 内置 Llama 需要 Python 底层库 llama_cpp；不需要安装 ComfyUI-llama-cpp 插件。"
            ) from exc

        model_path = folder_paths.get_full_path("LLM", model_name)
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError(f"找不到 Llama GGUF 模型：{model_name}")
        vision_name = str(vision_model or "").strip()
        if vision_name in {"无", "未找到视觉模型"}:
            vision_name = ""
        vision_path = folder_paths.get_full_path("LLM", vision_name) if vision_name else ""
        if vision_name and (not vision_path or not os.path.isfile(vision_path)):
            raise FileNotFoundError(f"找不到视觉识别模型：{vision_name}")

        external_result = cls._invoke_external(
            model_name, vision_name, n_ctx, messages, dict(params)
        )
        if external_result is not None:
            return external_result
        # llama_cpp expects an integer here. -1 means offload every possible layer to GPU.
        # Keep gpu_mode in the signature only for old-workflow positional compatibility.
        n_gpu_layers = -1
        config = (os.path.normcase(model_path), os.path.normcase(vision_path or ""), int(n_ctx), n_gpu_layers)
        with cls._lock:
            if cls._model is None or cls._config != config:
                cls.unload()
                print(f"[WenWu H3] 加载内置 Llama：{model_name}")
                chat_handler = None
                if vision_path:
                    from llama_cpp.llama_chat_format import Qwen35ChatHandler
                    chat_handler = Qwen35ChatHandler(
                        clip_model_path=vision_path,
                        enable_thinking=False,
                        add_vision_id=True,
                        use_gpu=True,
                        verbose=False,
                    )
                cls._model = Llama(
                    model_path=model_path,
                    n_ctx=int(n_ctx),
                    n_gpu_layers=n_gpu_layers,
                    use_mmap=True,
                    chat_handler=chat_handler,
                    verbose=False,
                )
                cls._chat_handler = chat_handler
                cls._config = config
            result = cls._model.create_chat_completion(messages=messages, stream=False, **params)
            try:
                content = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"Llama 返回格式异常：{result}") from exc
            try:
                cls._model.n_tokens = 0
                cls._model._ctx.memory_clear(True)
            except Exception:
                pass
            return str(content or "").strip()

    @classmethod
    def unload(cls):
        if cls._external_storage is not None:
            try:
                cls._external_storage.clean(all=True)
            except Exception:
                pass
            cls._external_storage = None
        if cls._model is not None:
            try:
                cls._model.close()
            except Exception:
                pass
        if cls._chat_handler is not None:
            try:
                cls._chat_handler.close()
            except Exception:
                pass
        cls._model = None
        cls._chat_handler = None
        cls._config = None

    @classmethod
    def is_loaded(cls):
        return cls._model is not None or (
            cls._external_storage is not None
            and getattr(cls._external_storage, "llm", None) is not None
        )


def _resolve_kimi_endpoint_model(api_key: str, requested_model: str) -> tuple[str, str]:
    """Resolve the model against the catalog exposed by the supplied Kimi key."""
    requested = str(requested_model or "").strip()
    auto = not requested or requested.lower() in {
        "auto", "自动", "自动匹配", "自动匹配（优先k3）", "kimi-k2.5",
    } or requested.lower().startswith("minimax-")
    catalogs = (
        "https://api.moonshot.cn/v1",
        "https://api.kimi.com/coding/v1",
    )
    discovered: list[tuple[str, list[str]]] = []
    for base_url in catalogs:
        request = urllib.request.Request(f"{base_url}/models", method="GET", headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "Liao-H3/1.0",
        })
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            model_ids = [str(item.get("id") or "").strip() for item in payload.get("data", []) if isinstance(item, dict)]
            model_ids = [item for item in model_ids if item]
        except Exception:
            continue
        if not model_ids:
            continue
        discovered.append((base_url, model_ids))
        if not auto and requested in model_ids:
            return f"{base_url}/chat/completions", requested

    if not auto:
        # A manually entered model is authoritative. Use the catalog that accepts
        # this key, or retain the original Moonshot-compatible endpoint.
        base_url = discovered[0][0] if discovered else catalogs[0]
        return f"{base_url}/chat/completions", requested

    def rank(model_id: str) -> tuple[int, int, str]:
        value = model_id.lower()
        if value == "k3":
            return (0, 0, value)
        if "k3" in value:
            return (1, 0, value)
        if not (value.startswith("kimi") or value.startswith("k")):
            return (9, 0, value)
        numbers = [int(part) for part in re.findall(r"\d+", value)]
        version = numbers[0] * 100 + (numbers[1] if len(numbers) > 1 else 0) if numbers else 0
        return (2, -version, value)

    candidates = [(rank(model_id), base_url, model_id) for base_url, model_ids in discovered for model_id in model_ids]
    if candidates:
        best_rank, base_url, chosen = min(candidates, key=lambda item: item[0])
        if best_rank[0] < 9:
            print(f"[Liao-H3] Kimi API 自动匹配模型：{chosen} ({base_url})")
            return f"{base_url}/chat/completions", chosen
    # Current official Kimi Code fallback when /models is temporarily unavailable.
    return "https://api.kimi.com/coding/v1/chat/completions", "k3"


def _normalize_openai_compatible_endpoint(base_url: str) -> str:
    """Accept a host, an API base, or a full chat-completions URL."""
    value = str(base_url or "").strip().rstrip("/")
    if not value:
        raise ValueError("请填写 OpenAI 兼容 API 地址。")
    if not re.match(r"^https?://", value, flags=re.IGNORECASE):
        raise ValueError("OpenAI 兼容 API 地址必须以 http:// 或 https:// 开头。")
    if value.lower().endswith("/chat/completions"):
        return value
    if value.lower().endswith("/v1") or re.search(r"/api/v\d+$", value, flags=re.IGNORECASE):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def _cloud_prompt_invoke(
    provider: str, api_key: str, model: str, messages: list[dict], compatible_base_url: str = "",
) -> str:
    """Call an OpenAI-compatible cloud endpoint using only Python's standard library."""
    provider = str(provider or "").strip()
    if provider == "Kimi API":
        api_key = str(api_key or os.environ.get("MOONSHOT_API_KEY", "") or os.environ.get("KIMI_API_KEY", "")).strip()
        if not api_key:
            raise ValueError("请填写 Kimi API Key，或在启动环境设置 MOONSHOT_API_KEY / KIMI_API_KEY。")
        endpoint, model = _resolve_kimi_endpoint_model(api_key, model)
    elif provider == "MiniMax API":
        endpoint = "https://api.minimaxi.com/v1/chat/completions"
        api_key = str(api_key or os.environ.get("MINIMAX_API_KEY", "")).strip()
        model = str(model or "MiniMax-M2.7").strip()
    elif provider == "OpenAI 兼容 API":
        endpoint = _normalize_openai_compatible_endpoint(compatible_base_url)
        api_key = str(api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        model = str(model or "").strip()
        if not model:
            raise ValueError("请填写 OpenAI 兼容服务的模型名称。")
    else:
        raise ValueError(f"不支持的云端提示词服务：{provider}")
    if provider != "OpenAI 兼容 API" and not api_key:
        env_name = "MOONSHOT_API_KEY" if provider == "Kimi API" else "MINIMAX_API_KEY"
        raise ValueError(f"请填写 {provider} Key，或在启动环境设置 {env_name}。")
    request_body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.2,
        "top_p": 0.84,
    }
    request_body["max_completion_tokens" if provider == "MiniMax API" else "max_tokens"] = 2800
    payload = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": "Liao-H3/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(endpoint, data=payload, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"{provider} 请求失败（HTTP {exc.code}）：{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 {provider}：{exc.reason}") from exc
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{provider} 返回格式异常：{result}") from exc
    # Reasoning models may include an internal thinking block in content.
    return re.sub(r"^\s*<think>[\s\S]*?</think>\s*", "", str(content or "")).strip()


def resolution_from_megapixels(aspect_ratio: str, megapixels: float, multiple: int = 32) -> tuple[int, int]:
    w_ratio, h_ratio = RATIOS[aspect_ratio]
    scale = math.sqrt(float(megapixels) * 1024 * 1024 / (w_ratio * h_ratio))
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple
    return max(multiple, width), max(multiple, height)


def duration_to_frames(seconds: float) -> int:
    """H3 要求帧数满足 17n+5；与原工作流公式完全一致。"""
    frames = max(5, round(float(seconds) * FPS))
    return frames + (5 - frames % 17) % 17


def _result(value):
    return value.result if hasattr(value, "result") else value


def _clean_filename(value: str) -> str:
    value = str(value or "").strip()
    return "" if value in {"未选择", "None", "null"} else value


_MEDIA_EXTENSIONS = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"},
    "video": {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"},
    "audio": {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"},
}


def _media_kind_from_filename(filename: str, fallback: str) -> str:
    plain_name = str(filename or "").split("[")[0].strip()
    extension = os.path.splitext(plain_name)[1].lower()
    for kind, extensions in _MEDIA_EXTENSIONS.items():
        if extension in extensions:
            return kind
    return fallback


def _rebucket_media_names(image_names, video_names, audio_names):
    """Repair workflows where a video/audio was serialized into an image slot."""
    limits = {"image": 9, "video": 3, "audio": 3}
    buckets = {kind: [] for kind in limits}
    for declared_kind, names in (("image", image_names), ("video", video_names), ("audio", audio_names)):
        for name in names:
            if not name:
                continue
            actual_kind = _media_kind_from_filename(name, declared_kind)
            if len(buckets[actual_kind]) < limits[actual_kind]:
                buckets[actual_kind].append(name)
    return tuple(
        buckets[kind] + [""] * (limits[kind] - len(buckets[kind]))
        for kind in ("image", "video", "audio")
    )


def _audio_duration_seconds(filename: str) -> float:
    """Read duration with ComfyUI's own audio decoder; no extra runtime is introduced."""
    import folder_paths
    from comfy_extras.nodes_audio import load

    path = folder_paths.get_annotated_filepath(filename)
    waveform, sample_rate = load(path)
    if not sample_rate or getattr(waveform, "ndim", 0) < 2:
        raise ValueError("无法读取连续数字人的驱动音频长度。")
    return float(waveform.shape[-1]) / float(sample_rate)


H3_OFFICIAL_COMPACT_COMMON = """You write production-ready prompts for MiniMax H3 video generation.
Return only the final prompt in English. Preserve supplied dialogue, lyrics and visible text exactly in their original language.
Use only reference tags that are explicitly listed by the user: <Subject N>, <Picture N>, <Video N>, <Audio N>. Never invent a tag, transcript, character, object or timestamp.
Runtime model names such as Qwythos, Llama, Qwen or GGUF filenames are never subjects and must never appear in the final prompt. A subject tag always uses an integer, for example <Subject 1>, and its identity comes from the correspondingly listed picture.
Describe visible subject identity, action progression, environment, composition, lighting and camera motion concretely. Describe sound effects, ambience, dialogue and music separately where relevant. Keep continuity, anatomy, screen direction, contact and scale stable. Match the requested duration and avoid redundant adjectives or unsupported scene changes.
Dialogue rule: if the user supplies exact spoken words, preserve them verbatim. If the user explicitly requests speech, shouting, arguing, asking, answering, narration or singing but supplies no exact words, create one short, natural, context-specific utterance instead of writing a placeholder such as "speaks", "talks about things" or "starts shouting". Identify each speaker consistently as (S1), (S2), etc., and put only the audible words in <d>[Chinese]...</d> or the correct original-language tag. Fit the line to its available screen time and describe delivery and lip synchronization outside the <d> block. Do not invent dialogue when the user requests no vocal act.
"""

H3_OFFICIAL_MODE_RULES = {
    "T2VA": """Mode: T2VA (text-to-video with audio). No reference tag is available. Write these ordered sections: integrated_multimodal_description, overall_soundscape, non_diegetic_music. Build one coherent audiovisual scene from the user's idea.""",
    "I2VA": """Mode: I2VA (image-to-video with audio). Treat <Picture 1> as the opening visual anchor. Preserve its subject identity and scene facts, then describe only plausible temporal motion. Write these ordered sections: integrated_multimodal_description, overall_soundscape, non_diegetic_music.""",
    "FL2VA": """Mode: FL2VA (first/last-frame video with audio). The first and last images are passed through dedicated frame inputs, so do not invent reference tags. Describe a continuous, physically plausible transition from the supplied first frame to the supplied last frame. Write these ordered sections: integrated_multimodal_description, overall_soundscape, non_diegetic_music.""",
    "Ref2VA": """Mode: Ref2VA (reference-based generation/editing with audio). Write these ordered sections: subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music. In retention_analysis assign each used visual reference one of fully_preserved, partially_preserved, attribute_transfer, weak_reference; assign each used audio reference one of fully_copy, partially_copy, reference, weak_reference. Bind every tag to one explicit role and never blend separate identities. For video editing, preserve all unrelated source-video content and change only what the user requests. For a digital human, fully_copy the driving audio unless the user explicitly asks otherwise, preserving voice, wording, rhythm and timing.""",
}

H3_OUTPUT_FIELDS = {
    "T2VA": ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"),
    "I2VA": ("alignment_instruction", "integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"),
    "FL2VA": ("alignment_instruction", "integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"),
    "L2VA": ("alignment_instruction", "integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"),
    "Ref2VA": (
        "subject_definitions", "summary", "retention_analysis",
        "detailed_description", "overall_soundscape", "non_diegetic_music",
    ),
}

H3_FIELD_MIN_LENGTHS = {
    "alignment_instruction": 60,
    "integrated_multimodal_description": 160,
    "subject_definitions": 40,
    "summary": 40,
    "retention_analysis": 30,
    "detailed_description": 180,
    "overall_soundscape": 30,
    "non_diegetic_music": 4,
}

PROMPT_TEMPLATE_AUTO = "自动匹配（按生成模式）"
PROMPT_TEMPLATE_OFFICIAL = "官方 MiniMax H3 Skill"
PROMPT_TEMPLATE_OFFICIAL_LEGACY = "官方 MiniMax H3"
PROMPT_TEMPLATE_STORYBOARD = "Liao 分镜模板（最长15秒）"
PROMPT_TEMPLATE_STORYBOARD_LEGACY = "WenWu 15S 分镜引擎"
PROMPT_TEMPLATE_STORYBOARD_LEGACY_2 = "WenWu 分镜引擎（最长15秒）"
PROMPT_TEMPLATE_STORYBOARD_LEGACY_3 = "WenWu 图生分镜（最长15秒）"
# 仅用于迁移旧工作流；不再显示，也不再启用私有视频编辑模板。
PROMPT_TEMPLATE_VIDEO_EDIT_LEGACY = "Liao 视频编辑模板"
STORYBOARD_15S_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "prompt_templates", "minimax_h3_storyboard_15s.md"
)
OFFICIAL_H3_SKILL_PATH = os.path.join(
    os.path.dirname(__file__), "prompt_templates", "official_h3_skill_compiled.md"
)


def _storyboard_15s_profile(target_duration: float, shot_count: int = 3) -> str:
    """Load the bundled user-authored 15-second storyboard engine."""
    try:
        with open(STORYBOARD_15S_TEMPLATE_PATH, "r", encoding="utf-8-sig") as handle:
            template = handle.read().strip()
    except OSError as exc:
        raise RuntimeError("WenWu 15S 分镜模板缺失，请重新复制完整插件目录。") from exc
    duration = max(1.0, min(15.0, float(target_duration)))
    shots = max(1, min(12, int(shot_count)))
    return template + f"""

## 节点输出协议（优先级最高）

15秒只是本引擎允许的最大时长，绝不是固定时长。本次节点选择的目标时长是 {duration:.1f} 秒，目标分镜数是 {shots} 个。必须严格输出 {shots} 个连续分镜，并按照目标时长重新计算每镜持续时间、动作密度和声音节奏；第一镜从00.0秒开始，末镜必须精确结束于 {duration:.1f} 秒。不能把15秒方案机械截断、压缩或继续写到目标时长之外。

final_prompt 必须把秒级时间轴直接写在正文里，不能只返回一段连续的画面描述。严格使用“镜头01【00.0—01.2秒】：……”这种可见格式并输出恰好 {shots} 个镜头；下一镜起点必须等于上一镜终点，时间段不得缺失、重叠或超出目标时长。即使画面内容很简单，也不得省略镜号和起止秒数。

对白规则：用户提供了具体原话时必须逐字保留。用户明确要求人物说、喊、嚷、争吵、询问、回答、旁白或唱歌但没有给出原话时，必须根据当前人物关系和场景补写一句简短、自然、可在对应时段说完的具体内容，使用“人物(S1)说：<d>[Chinese]具体台词</d>”格式。禁止只写“开始说话”“低声诉说琐事”“大声嚷嚷”等没有可听内容的占位描述。说话语气、停顿、口型和说后反应写在标签外；只有真正听到的字句放进 <d> 标签。用户没有要求发声时不得凭空添加对白。

你只能返回一个 JSON 对象，且只能包含 final_prompt 字段。final_prompt 的值必须是上述规则要求的、可直接提交给 MiniMax H3 的中文成片提示词正文。不要输出分析、推理、说明、Markdown 代码围栏或额外字段。引用素材时只能使用用户实际列出的标签。"""


def _storyboard_response_format() -> dict:
    return {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {"final_prompt": {"type": "string", "minLength": 220}},
            "required": ["final_prompt"],
            "additionalProperties": False,
        },
    }


def _storyboard_time_ranges(value: str) -> list[tuple[float, float]]:
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:秒|s)?\s*[-—–~至到]\s*"
        r"(\d+(?:\.\d+)?)\s*(?:秒|s)",
        re.IGNORECASE,
    )
    return [(float(start), float(end)) for start, end in pattern.findall(value)]


def _format_storyboard_output(raw: str, target_duration: float, source: str = "", shot_count: int = 3) -> str:
    try:
        data = json.loads(str(raw or "").strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("模型没有返回有效的 15S 分镜提示词，请重新增强或更换指令模型。") from exc
    value = str(data.get("final_prompt") if isinstance(data, dict) else "").strip()
    if len(value) < 220:
        raise RuntimeError("15S 分镜提示词为空或过短，请重新增强或更换指令模型。")
    contamination = ("the user wants", "i need to", "i should", "system prompt", "我需要先", "分析如下")
    if any(marker in value.lower() for marker in contamination):
        raise RuntimeError("模型返回了分析过程而不是 15S 分镜提示词，请换用指令型 GGUF 模型。")
    ranges = _storyboard_time_ranges(value)
    requested_shots = max(1, min(12, int(shot_count)))
    if len(ranges) != requested_shots:
        # 分镜数为“自动”时，前端会把按时长估算出的数字送入后端；
        # 它只是生成建议，不能因为指令模型少写/合并一镜就阻断整个工作流。
        print(f"[Liao2049 H3] 分镜数量提示：建议 {requested_shots} 个，实际 {len(ranges)} 个；继续使用模型结果。")
        return value
    duration = max(1.0, min(15.0, float(target_duration)))
    tolerance = 0.16
    if abs(ranges[0][0]) > tolerance or abs(ranges[-1][1] - duration) > tolerance:
        print(f"[Liao2049 H3] 分镜时间轴未完全覆盖 0.0-{duration:.1f}s；继续使用模型结果。")
        return value
    previous_end = ranges[0][0]
    for start, end in ranges:
        if end <= start or abs(start - previous_end) > tolerance or end > duration + tolerance:
            print("[Liao2049 H3] 分镜时间段存在空档、重叠或倒序；继续使用模型结果。")
            return value
        previous_end = end
    return value


def _video_edit_profile(source: str, image_count: int) -> str:
    """Liao-H3 native reference-guided video-edit contract."""
    references = (
        f"The user supplied {image_count} reference image(s), addressed only as <Picture 1> through <Picture {image_count}>."
        if image_count else "The user supplied no reference image."
    )
    return f"""You write one precise MiniMax H3 video-edit instruction.

The source video is <Video 1>. {references}
The user's edit request is: {source}

Rules with highest priority:
- Preserve <Video 1>'s duration, timing, action, pose, body motion, camera movement, framing, background, lighting, occlusion, reflections, clothing and all content that the user did not explicitly ask to change.
- A reference image supplies only the requested identity, object, material or visual attribute. Never inherit its background, pose, action, framing, lighting or clothing unless explicitly requested.
- For person replacement, bind the new identity to the requested <Picture N> while retaining the source person's performance and wardrobe unless the user explicitly asks to replace clothing.
- Start with one direct imperative edit command. Do not create a new story, location, shot list or unrelated detail.
- The sampled visual attachments are observation frames from <Video 1>; they are never additional Picture tags.
- Use only tags that actually exist. Never turn a model filename, GGUF name or attachment number into a subject.
- Return only a JSON object with one string field named rewritten_text. No analysis, headings, Markdown or extra fields."""


def _video_edit_response_format() -> dict:
    return {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {"rewritten_text": {"type": "string", "minLength": 80}},
            "required": ["rewritten_text"],
            "additionalProperties": False,
        },
    }


def _format_video_edit_output(raw: str) -> str:
    try:
        data = json.loads(str(raw or "").strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Liao 视频编辑模板没有返回有效提示词，请重新增强。") from exc
    value = str(data.get("rewritten_text") if isinstance(data, dict) else "").strip()
    if len(value) < 80:
        raise RuntimeError("Liao 视频编辑提示词为空或过短，请重新增强。")
    return value


def _h3_response_format(mode: str) -> dict:
    """llama.cpp JSON schema that prevents reasoning prose outside H3 fields."""
    fields = H3_OUTPUT_FIELDS.get(mode, H3_OUTPUT_FIELDS["Ref2VA"])
    return {
        "type": "json_object",
        "schema": {
            "type": "object",
            "properties": {
                name: {"type": "string", "minLength": H3_FIELD_MIN_LENGTHS[name]}
                for name in fields
            },
            "required": list(fields),
            "additionalProperties": False,
        },
    }


def _format_h3_structured_output(raw: str, mode: str) -> str:
    fields = H3_OUTPUT_FIELDS.get(mode, H3_OUTPUT_FIELDS["Ref2VA"])
    try:
        data = json.loads(str(raw or "").strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("本地模型没有返回有效的 H3 结构化提示词，请更换指令模型后重试。") from exc
    if not isinstance(data, dict):
        raise RuntimeError("本地模型返回的 H3 提示词结构无效。")
    values = []
    contamination = (
        "the user wants", "i need to", "i should", "system prompt",
        "the instruction says", "now combine", "i'll assume", "i will assume",
    )
    for name in fields:
        value = str(data.get(name) or "").strip()
        if len(value) < H3_FIELD_MIN_LENGTHS[name]:
            raise RuntimeError(f"本地模型返回的 {name} 内容为空或过短，请重新增强或更换指令型 GGUF 模型。")
        lowered = value.lower()
        if any(marker in lowered for marker in contamination):
            raise RuntimeError("本地模型返回了分析过程而不是最终提示词，请换用指令型 GGUF 模型后重试。")
        # Official keyframe modes require their alignment instruction as the
        # first unlabelled line, before the three shared H3 fields.
        values.append(value if name == "alignment_instruction" else f"{name}:\n{value}")
    return "\n\n".join(values)


def _validate_h3_reference_tags(text: str, image_count: int, video_count: int, audio_count: int,
                                allow_video_frame_aliases: bool = False) -> str:
    """Reject invented H3 tags and repair aliases caused by video-frame vision attachments."""
    result = str(text or "").strip()
    limits = {
        "Picture": max(0, int(image_count)),
        "Video": max(0, int(video_count)),
        "Audio": max(0, int(audio_count)),
        # A Ref2VA subject identity may come from either a picture or a video.
        "Subject": max(0, int(image_count) + int(video_count)),
    }

    def replace(match):
        kind, raw_id = match.group(1), match.group(2).strip()
        if kind == "Subject" and not raw_id.isdigit() and limits["Subject"] == 1:
            return "<Subject 1>"
        if not raw_id.isdigit():
            raise RuntimeError(f"提示词生成了非法标签 <{kind} {raw_id}>；主体只能使用编号标签。")
        index = int(raw_id)
        if index < 1 or index > limits[kind]:
# During Liao-H3 video editing the vision model receives several
            # sampled frames from <Video 1> followed by the real reference
            # pictures. Some VLMs number those frame attachments as pictures
            # despite the explicit mapping. They are observations of the same
            # source video, not additional H3 picture references.
            if kind == "Picture" and allow_video_frame_aliases and limits["Video"] >= 1:
                return "<Video 1>"
            raise RuntimeError(f"提示词引用了不存在的标签 <{kind} {index}>。")
        return f"<{kind} {index}>"

    return re.sub(r"<(Subject|Picture|Video|Audio)\s+([^>]+)>", replace, result)


def _official_h3_profile(mode: str, include_director_layer: bool = False,
                         target_duration: float = 5.0, shot_count: int = 1) -> str:
    """Compile the bundled official H3 Skill for one mode and runtime context.

    This deliberately follows a Skill-style router/audit pipeline instead of
    sending the small output-field hint that older liao2049 builds used.  The
    bundled file is self-contained so copied plugins do not depend on Codex or
    on a separately installed skill.
    """
    try:
        with open(OFFICIAL_H3_SKILL_PATH, "r", encoding="utf-8-sig") as handle:
            skill = handle.read().strip()
    except OSError as exc:
        raise RuntimeError("官方 MiniMax H3 Skill 文件缺失，请重新复制完整的 comfyui-liao2049 插件。") from exc
    duration = max(1.0, min(15.0, float(target_duration)))
    shots = max(1, min(12, int(shot_count)))
    profile = (skill
               .replace("{{MODE}}", mode)
               .replace("{{DURATION}}", f"{duration:.2f}")
               .replace("{{SHOT_COUNT}}", str(shots)))
    profile += "\n\n" + H3_OFFICIAL_COMPACT_COMMON
    if include_director_layer:
        profile += """

Liao director layer: strengthen the visible cause-and-effect, performance beats and motivated camera language without changing the official section order. Prefer a readable action arc over decorative prose. A video-edit instruction remains minimal and must not introduce new shots, styles or transitions unless requested."""
    return profile


def _resolve_storyboard_count(value, target_duration: float) -> int:
    """Resolve the UI's automatic shot count from duration, or accept 1..12."""
    if str(value or "自动").strip() == "自动":
        return max(1, min(5, math.ceil(max(1.0, min(15.0, float(target_duration))) / 3.0)))
    try:
        return max(1, min(12, int(value)))
    except (TypeError, ValueError):
        return max(1, min(5, math.ceil(max(1.0, min(15.0, float(target_duration))) / 3.0)))


def _extract_explicit_storyboard_count(source: str) -> int | None:
    """Prefer a shot count explicitly written by the user over the UI hint."""
    text = str(source or "")
    chinese_numbers = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
        "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
    }
    count_match = re.search(
        r"(?:需要|要求|生成|制作|做成|分成|包含|共|总共|请)?\s*"
        r"(1[0-2]|[1-9]|十二|十一|十|九|八|七|六|五|四|三|二|两|一)\s*"
        r"(?:个|段)?\s*(?:分镜头|分镜|镜头)",
        text,
        flags=re.IGNORECASE,
    )
    if count_match:
        token = count_match.group(1)
        value = int(token) if token.isdigit() else chinese_numbers.get(token)
        if value is not None:
            return max(1, min(12, value))
    # Re-enhancing an existing storyboard should preserve its visible highest
    # shot number even if the user does not repeat "N个镜头" in prose.
    numbered = [int(value) for value in re.findall(r"镜头\s*0?([1-9]|1[0-2])", text)]
    if numbered:
        return max(1, min(12, max(numbered)))
    return None


def _official_shot_timeline_profile(mode: str, target_duration: float, shot_count: int) -> str:
    """Require a detailed, machine-checkable storyboard inside official H3 fields."""
    duration = max(1.0, min(15.0, float(target_duration)))
    shots = max(1, min(12, int(shot_count)))
    timeline_field = "detailed_description" if mode == "Ref2VA" else "integrated_multimodal_description"
    return f"""

Official detailed storyboard requirement (highest priority):
- Inside {timeline_field}, output exactly {shots} consecutive shots covering exactly 0.0 to {duration:.1f} seconds.
- Start every shot on its own line using this exact machine-readable form: Shot 01 [0.0-1.7s]: ...
- The first shot starts at 0.0s; each next start equals the previous end; the final shot ends at {duration:.1f}s. No gap, overlap or extra shot is allowed.
- Every shot must concretely describe subject identity/continuity, visible action and body mechanics, environment and spatial change, framing/camera motion, lighting, and synchronized sound event where applicable.
- Do not use a one-paragraph summary as a substitute for the shot timeline. Keep all other official H3 fields and their prescribed order unchanged.
"""


def _validate_official_shot_timeline(value: str, target_duration: float, shot_count: int) -> str:
    """Reject official prompts that omit the requested detailed shot timeline."""
    text = str(value or "").strip()
    pattern = re.compile(
        r"Shot\s*0?\d+\s*\[\s*(\d+(?:\.\d+)?)\s*[-—–~]\s*"
        r"(\d+(?:\.\d+)?)\s*s\s*\]\s*:",
        re.IGNORECASE,
    )
    ranges = [(float(start), float(end)) for start, end in pattern.findall(text)]
    requested = max(1, min(12, int(shot_count)))
    duration = max(1.0, min(15.0, float(target_duration)))
    if len(ranges) != requested:
        # 自动分镜数是建议值，不是执行前置条件。不同 LLM 可能自然合并
        # 相邻镜头；保留有效增强文本比让整个视频生成失败更合理。
        print(f"[Liao2049 H3] 官方分镜数量提示：建议 {requested} 个，实际 {len(ranges)} 个；继续使用增强结果。")
        return text
    tolerance = 0.16
    if abs(ranges[0][0]) > tolerance or abs(ranges[-1][1] - duration) > tolerance:
        print(f"[Liao2049 H3] 官方分镜时间轴未完全覆盖 0.0-{duration:.1f}s；继续使用增强结果。")
        return text
    previous_end = ranges[0][0]
    for start, end in ranges:
        if end <= start or abs(start - previous_end) > tolerance or end > duration + tolerance:
            print("[Liao2049 H3] 官方分镜存在空档、重叠、倒序或越界；继续使用增强结果。")
            return text
        previous_end = end
    return text


def _multi_reference_adaptive_profile() -> str:
    """Assign each Ref2VA image a semantic role instead of assuming a person."""
    return r"""

Liao multi-reference adaptive-binding override (highest priority):
- This is reference generation, NOT image-to-video and NOT keyframe continuation.
- First determine the role requested for every <Picture N> from the user's wording. Explicit user wording wins. Only when the wording is ambiguous may vision inspection classify the image.
- Allowed roles are: character/animal identity; object/product identity or structure; scene/environment; wardrobe/prop/material; visual style/lighting/composition. One image may supply several roles only when the user explicitly requests them.
- For a character, animal, object or product role, create <Subject N> with <Picture N> as its identity/form source. Preserve only the intrinsic identity, anatomy, shape, markings or product structure needed for recognition.
- A <Video N> may likewise define a reusable character, animal, object, product, motion or performance reference. When it supplies identity, define a subject whose source is that video; do not treat the video as a new target scene unless the user requests its scene.
- For a scene/environment role, keep <Picture N> as a picture reference and describe only the requested geography, architecture, spatial layout or atmosphere. Do not invent a <Subject N> for the scene.
- For wardrobe, prop, material, style, lighting or composition roles, use attribute_transfer or weak_reference and transfer only the named attribute; do not transfer unrelated content.
- Do not inherit background, pose, action, framing, crop, composition, camera, lighting, color grade or clothing unless that exact aspect is the selected role or the user explicitly asks to retain it.
- The user's text has absolute priority for the target location, action, wardrobe, props, shot design, lighting and atmosphere.
- In retention_analysis state both the chosen role and its exact preservation scope. Never mark a whole picture fully_preserved unless the user explicitly requests full scene retention.
- Never describe unrequested source-image content merely because it is visible to the vision model. The summary and detailed_description must describe the requested target result from its first frame.
- Preserve the user's requested action, location and event literally. Never replace an unfamiliar, colloquial or sensitive-but-allowed action with a safer unrelated action. For example, "在蹲坑" means using/squatting over a squat toilet; it must never become walking on a city street.
- Example semantic rule: if the user asks for the woman from <Picture 1> skiing on a snowy mountain, preserve only her identity; place her skiing on the snowy mountain and discard the source pose, room, background and composition.
- Example semantic rule: if the user asks for a subject inside the snowy mountain scene from <Picture 1>, use <Picture 1> as a scene/environment reference and do not turn the mountain into <Subject 1>.
"""


def _h3_reference_image_urls(image_names: list[str], limit: int = 4) -> tuple[list[str], list[str]]:
    """Encode selected ComfyUI input images for the local Qwen3.5 vision handler."""
    import folder_paths
    from PIL import Image

    urls, mappings = [], []
    for slot, name in enumerate(image_names, 1):
        if not name or len(urls) >= limit:
            continue
        try:
            path = folder_paths.get_annotated_filepath(name)
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=90, optimize=True)
            urls.append("data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
            mappings.append(f"Image {len(urls) - 1} = <Picture {len(urls)}>")
        except Exception as exc:
            print(f"[WenWu H3] 视觉识别跳过图片{slot}：{exc}")
    return urls, mappings


def _h3_source_video_frame_urls(video_name: str, sample_count: int = 3) -> tuple[list[str], list[str]]:
    """Sample source-video frames for Liao-H3 visual edit analysis using existing OpenCV."""
    import cv2
    import folder_paths

    if not video_name:
        return [], []
    path = folder_paths.get_annotated_filepath(video_name)
    capture = cv2.VideoCapture(path)
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not capture.isOpened() or frame_count < 1:
            raise RuntimeError("无法读取待编辑视频画面。")
        indices = [round(i * (frame_count - 1) / max(1, sample_count - 1)) for i in range(sample_count)]
        urls, mappings = [], []
        for order, frame_index in enumerate(indices, 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, 1024.0 / max(width, height))
            if scale < 1.0:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            encoded, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if not encoded:
                continue
            urls.append("data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii"))
            mappings.append(f"Attachment {len(urls) - 1} = <Video 1> source frame {order}/{sample_count}")
        if not urls:
            raise RuntimeError("无法从待编辑视频抽取视觉帧。")
        return urls, mappings
    finally:
        capture.release()


def build_native_prompt(prompt: str, image_count: int, videos_with_audio: list[bool], audio_count: int) -> str:
    """将固定槽位别名转换为 H3 按实际呈现顺序生成的原生标签。"""
    mapping = {}
    for index in range(1, image_count + 1):
        mapping[f"@图片{index}"] = f"<Picture {index}>"
    audio_ordinal = 1
    for index, has_soundtrack in enumerate(videos_with_audio, 1):
        if has_soundtrack:
            mapping[f"@视频音频{index}"] = f"<Audio {audio_ordinal}>"
            audio_ordinal += 1
        mapping[f"@视频{index}"] = f"<Video {index}>"
    for index in range(1, audio_count + 1):
        mapping[f"@音频{index}"] = f"<Audio {audio_ordinal}>"
        audio_ordinal += 1

    text = str(prompt or "").strip()
    # Users naturally write 图二/图片2/第二张图. MiniMax H3 only understands
    # the native <Picture N> contract once more than one image is attached.
    chinese_numbers = {
        1: r"(?:1|一|壹)", 2: r"(?:2|二|两|贰)", 3: r"(?:3|三|叁)",
        4: r"(?:4|四|肆)", 5: r"(?:5|五|伍)", 6: r"(?:6|六|陆)",
        7: r"(?:7|七|柒)", 8: r"(?:8|八|捌)", 9: r"(?:9|九|玖)",
    }
    for index in range(1, image_count + 1):
        number = chinese_numbers[index]
        natural_alias = rf"(?<!@)(?:第\s*{number}\s*张\s*(?:图片|图)|图片\s*{number}|图\s*{number})"
        text = re.sub(natural_alias, f"<Picture {index}>", text, flags=re.IGNORECASE)
    # An optional reference is not an execution requirement.  Older UI builds
    # inserted clauses such as "如有@图片2，则……" even when only one image was
    # uploaded.  Drop only those explicitly conditional clauses before strict
    # validation, while keeping ordinary missing @ references as useful errors.
    present_aliases = set(mapping)
    referenced_aliases = {
        f"@{kind}{number}" for kind, number in re.findall(r"@(图片|视频音频|视频|音频)(\d+)", text)
    }
    for alias in sorted(referenced_aliases - present_aliases, key=len, reverse=True):
        conditional_clause = (
            rf"(?:如|若|如果)(?:还)?有\s*{re.escape(alias)}\s*"
            rf"(?:，|,)?\s*(?:则|就|可|可以)?[^。；;\n]*(?:[。；;]|$)"
        )
        text = re.sub(conditional_clause, "", text, flags=re.IGNORECASE)
    aliases = set(re.findall(r"@(图片|视频音频|视频|音频)(\d+)", text))
    unknown = sorted(f"@{kind}{number}" for kind, number in aliases if f"@{kind}{number}" not in mapping)
    if unknown:
        raise ValueError("以下@引用没有对应素材：" + "、".join(unknown))
    for source in sorted(mapping, key=len, reverse=True):
        text = text.replace(source, mapping[source])
    # Common two-reference shorthand: picture 1 is the uploaded subject and a
    # later explicitly-numbered picture is the destination scene. Keep this a
    # minimal binding sentence instead of expanding/re-writing the user prompt.
    if image_count >= 2 and "<Picture 1>" not in text and re.search(r"<Picture\s+[2-9]>", text):
        subject = r"(女孩|女生|女人|女性|男孩|男生|男人|男性|人物|角色|主角|模特|老人|儿童|猫|狗|动物|商品|产品|物体|汽车|车辆|机器人|玩偶)"
        scene_ref = r"(<Picture\s+[2-9]>)"
        relation = r"(在|放在|置于|位于|进入|来到)"
        text, replacements = re.subn(
            rf"{subject}\s*{relation}\s*{scene_ref}",
            rf"<Picture 1>中的\1\2\3",
            text,
            count=1,
        )
        if not replacements:
            text = f"<Picture 1>提供主要主体身份；仅将明确指定的后续图片用于对应场景或属性。\n{text}"
    return text


def _load_media(image_names, video_names, audio_names):
    import folder_paths
    import nodes
    from comfy_api.latest import InputImpl
    from comfy_extras.nodes_audio import LoadAudio

    images = []
    for name in image_names:
        if name:
            images.append(nodes.LoadImage().load_image(name)[0])

    videos, video_audios, soundtrack_flags = [], [], []
    for name in video_names:
        if not name:
            continue
        path = folder_paths.get_annotated_filepath(name)
        components = InputImpl.VideoFromFile(path).get_components()
        videos.append(components.images)
        soundtrack = components.audio
        has_audio = soundtrack is not None
        soundtrack_flags.append(has_audio)
        video_audios.append(soundtrack if has_audio else None)

    audios = []
    for name in audio_names:
        if name:
            audios.append(_result(LoadAudio.execute(name))[0])
    return images, videos, video_audios, soundtrack_flags, audios


def _sample_progress_only(noise, guider, sampler, sigmas, latent):
    """等价执行 SamplerCustomAdvanced，但不生成/保留 Latent 预览图和 denoised 副本。"""
    import comfy.sample
    import comfy.model_management
    import comfy.utils

    source = latent
    latent_image = source["samples"]
    out = source.copy()
    latent_image = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher,
        latent_image,
        source.get("downscale_ratio_spacial"),
        source.get("downscale_ratio_temporal"),
    )
    out["samples"] = latent_image
    noise_mask = source.get("noise_mask")
    total = max(0, int(sigmas.shape[-1]) - 1)
    pbar = comfy.utils.ProgressBar(total)

    # 保留 ComfyUI 原生进度/取消链路，只省略 latent_preview.prepare_callback 的图像解码。
    def callback(step, _x0, _x, total_steps):
        pbar.update_absolute(step + 1, total_steps, None)

    samples = guider.sample(
        noise.generate_noise(out), latent_image, sampler, sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=noise.seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())
    result = source.copy()
    result.pop("downscale_ratio_spacial", None)
    result.pop("downscale_ratio_temporal", None)
    result["samples"] = samples
    return result


def _has_official_h3_structure(prompt: str) -> bool:
    lowered = str(prompt or "").lower()
    return (
        "integrated_multimodal_description:" in lowered
        or all(field in lowered for field in (
            "subject_definitions:", "summary:", "retention_analysis:", "detailed_description:"
        ))
    )


def _infer_direct_picture_role(source: str, index: int, image_count: int) -> str:
    """Infer an explicit picture role from text when prompt enhancement is off.

    H3 still sees the picture through MiniMaxH3ReferenceToVideo. This parser only
    supplies the correct preservation strength; it never tries to visually
    classify the image itself.
    """
    text = re.sub(r"\s+", "", str(source or "").lower())
    numbered = rf"(?:@图片{index}|<picture{index}>|图片{index}|图{index}|第{index}张(?:图|图片))"
    generic = r"(?:参考图|图片|图中|图片中)" if image_count == 1 else numbered
    ref = rf"(?:{numbered}|{generic})"
    scene = r"(?:场景|背景|环境|空间|地点|建筑|室内|室外|街道|房间|地貌|风景|氛围)"
    attribute = r"(?:服装|衣服|穿搭|道具|材质|纹理|颜色|色调|光线|灯光|构图|镜头|画风|风格)"
    subject = (
        r"(?:人物|人脸|脸|身份|角色|主角|女人|女孩|女生|男性|男人|男孩|男生|模特|老人|儿童|"
        r"动物|猫|狗|宠物|商品|产品|物体|物品|汽车|车辆|摩托|机器人|玩偶|雕像)"
    )
    # Require an explicit relation to the reference. A target phrase such as
    # "the woman skis on a snowy mountain" must not turn "snowy mountain" into
    # the reference role merely because a scene noun appears in the request.
    if re.search(rf"{ref}.{{0,12}}{scene}|{scene}.{{0,12}}(?:参考|使用|采用|来自|按照){ref}", text):
        return "scene"
    if index > 1 and re.search(rf"(?:在|到|进入|来到|放在|置于|位于){ref}(?:中|里|内|的|场景|环境)", text):
        return "scene"
    if re.search(rf"{ref}.{{0,12}}{attribute}|{attribute}.{{0,12}}(?:参考|使用|采用|来自|按照){ref}", text):
        return "attribute"
    if re.search(rf"{ref}.{{0,16}}{subject}|{subject}.{{0,12}}(?:来自|参考|使用|采用|按照){ref}", text):
        return "subject"
    return "adaptive"


def _official_direct_prompt(source: str, mode: str, image_count: int, video_count: int,
                            audio_count: int, duration: float) -> str:
    """Build an official-H3-shaped prompt without invoking an LLM.

    The user's creative sentence is intentionally preserved verbatim. The
    deterministic wrapper contributes only mode, reference provenance,
    retention relationships and the official field order.
    """
    creative = str(source or "").strip() or "Generate a coherent audiovisual scene."
    seconds = max(1.0, float(duration))

    if mode == "文生视频":
        return (
            "integrated_multimodal_description: [Shot 1] Follow this creative instruction exactly: "
            f"{creative} Build a coherent visible action arc that ends within {seconds:.2f} seconds.\n\n"
            "overall_soundscape: Use scene-consistent ambience, physical action sounds and non-verbal human sounds; "
            "do not invent dialogue or lyrics.\n\n"
            "non_diegetic_music: N/A"
        )

    if mode == "图生视频":
        return (
            "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            "integrated_multimodal_description: [Shot 1] Begin from <Picture 1>, preserving its subject identity, clothing, "
            "composition, environment, lighting and spatial relationships. Follow this creative instruction exactly: "
            f"{creative} Develop only plausible continuous motion from the first frame and finish within {seconds:.2f} seconds.\n\n"
            "overall_soundscape: Use scene-consistent ambience and synchronized physical action sounds; do not invent dialogue or lyrics.\n\n"
            "non_diegetic_music: N/A"
        )

    if mode == "首尾帧":
        return (
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second "
            f"mark of the target video; Picture 2 (from Shot 1) aligns with the {seconds:.2f}-second mark of the target video.\n\n"
            "integrated_multimodal_description: [Shot 1] Begin exactly from <Picture 1> and follow this creative instruction: "
            f"{creative} Use one continuous, physically plausible motion path that progressively reaches the subject state, "
            f"object state, framing, lighting and composition of <Picture 2> at {seconds:.2f} seconds.\n\n"
            "overall_soundscape: Use continuous scene-consistent ambience and synchronized physical action sounds; do not invent dialogue or lyrics.\n\n"
            "non_diegetic_music: N/A"
        )

    picture_roles = {
        i: _infer_direct_picture_role(creative, i, image_count)
        for i in range(1, image_count + 1)
    } if mode == "多参考" else {}
    if mode == "多参考":
        subject_lines = []
        for i, role in picture_roles.items():
            if role == "subject":
                subject_lines.append(
                    f"<Subject {i}> is the character, animal, object or product identity defined by <Picture {i}>; "
                    "preserve its intrinsic recognizable identity/form, not the source scene."
                )
            elif role == "scene":
                subject_lines.append(
                    f"<Picture {i}> is a scene/environment reference only; it defines the requested location or spatial atmosphere, not a subject identity."
                )
            elif role == "attribute":
                subject_lines.append(
                    f"<Picture {i}> is an attribute reference only; transfer only the explicitly requested wardrobe, prop, material, color, lighting, composition or style attribute."
                )
            else:
                subject_lines.append(
                    f"<Picture {i}> is an adaptive weak visual reference. Use only the role explicitly implied by the creative instruction; "
                    "it is not a keyframe or whole-scene anchor."
                )
    else:
        subject_lines = [
            f"<Subject {i}> is the reusable visible identity and appearance defined by @图片{i}."
            for i in range(1, image_count + 1)
        ]
    video_lines = [
        f"<Video {i}> is a reference video providing temporal motion, camera and scene structure."
        for i in range(1, video_count + 1)
    ]
    audio_lines = [
        f"@音频{i} is a standalone audio reference for the target video."
        for i in range(1, audio_count + 1)
    ]

    task_types = ["reference generation"]
    if mode == "视频编辑":
        task_types = ["video editing"]
        if video_count:
            video_lines[0] = "<Video 1> is the source video for the target video edit."
            audio_lines.insert(0, "@视频音频1 is the synchronized source audio of <Video 1>.")
            task_types.append("audio reuse")
    elif mode in {"单人数字人", "双人数字人"}:
        task_types.append("audio reuse")
        for i in range(1, min(image_count, audio_count) + 1):
            audio_lines[i - 1] = f"@音频{i} is fully reused as the synchronized voice and timing track for <Subject {i}> (S{i})."

    definitions = subject_lines + video_lines + audio_lines
    if not definitions:
        definitions = ["No external reference label is available."]

    retention = []
    if mode == "多参考":
        for i, role in picture_roles.items():
            if role == "subject":
                retention.append(
                    f"<Subject {i}>: fully_preserved - preserve the referenced subject identity and intrinsic form from <Picture {i}>; "
                    "replace the source pose, action, background, composition and lighting with the user's requested target scene."
                )
            elif role == "scene":
                retention.append(
                    f"<Picture {i}>: partially_preserved - preserve only the requested scene/environment, spatial layout or atmosphere; "
                    "do not copy unrelated people, objects or actions."
                )
            elif role == "attribute":
                retention.append(
                    f"<Picture {i}>: attribute_transfer - transfer only the explicitly requested attribute; ignore all unrelated source content."
                )
            else:
                retention.append(
                    f"<Picture {i}>: weak_reference - use only the role clearly implied by the creative instruction; "
                    "do not retain unrelated identity, pose, action, clothing, background, lighting, camera, crop or composition."
                )
    else:
        retention.extend(
            f"<Subject {i}>: fully_preserved - preserve the identity and intrinsic appearance defined by @图片{i}."
            for i in range(1, image_count + 1)
        )
    for i in range(1, video_count + 1):
        marker = "fully_preserved" if mode == "视频编辑" and i == 1 else "weak_reference"
        detail = ("preserve all source-video content except the user's explicitly requested edit."
                  if marker == "fully_preserved" else "use only the requested motion, camera or temporal characteristics.")
        retention.append(f"<Video {i}>: {marker} - {detail}")
    if mode == "视频编辑" and video_count:
        retention.append("@视频音频1: fully_copy - reuse the source video's complete synchronized audio unchanged.")
    retention.extend(
        f"@音频{i}: fully_copy - reuse this audio signal with its wording, voice, rhythm and timing unchanged."
        for i in range(1, audio_count + 1)
    )

    if mode == "视频编辑":
        summary = (
            f"[{' + '.join(task_types)}] The target video is an edited version of <Video 1>. "
            "Apply only the explicitly requested change and preserve all unrelated source-video content."
        )
        detail_prefix = (
            "Use <Video 1> as the complete temporal and compositional source. Track the edited target consistently through every frame. "
        )
    elif mode in {"单人数字人", "双人数字人"}:
        summary = f"[{' + '.join(task_types)}] Generate synchronized digital-human performance from the defined subjects and audio tracks."
        detail_prefix = "Keep each subject identity stable and synchronize mouth, expression, breathing and gesture to its assigned audio. "
        if mode == "双人数字人":
            detail_prefix += "Keep <Subject 1> and <Subject 2> simultaneously visible, separate and correctly matched to their own audio. "
    elif mode == "多参考":
        summary = (
            f"[{' + '.join(task_types)}] Generate the target result using each picture only for its user-assigned semantic role. "
            "The user's requested setting, action, wardrobe and shot design override all unrelated source-picture content."
        )
        detail_prefix = (
            "Resolve each <Picture N> as a character/animal, object/product, scene/environment or attribute reference from the creative instruction. "
            "A fully_preserved <Subject N> must remain recognizably identical across the whole clip while its action and setting follow the user. "
            "Transfer only the assigned role, start directly with the requested result, and do not recreate unrelated source-image content. "
        )
    else:
        summary = f"[{' + '.join(task_types)}] Generate the target video from the explicitly defined image, video and audio roles."
        detail_prefix = "Keep every reference role separate and do not blend identities, clothing, environments or audio assignments. "

    return (
        "subject_definitions:\n" + "\n".join(definitions) +
        "\n\nsummary:\n" + summary +
        "\n\nretention_analysis:\n" + ("\n".join(retention) if retention else "No reference retention is required.") +
        "\n\ndetailed_description:\n[Shot 1] " + detail_prefix +
        f"Follow this creative instruction exactly: {creative} Complete the requested result within {seconds:.2f} seconds. "
        "Do not invent dialogue, lyrics, labels, people or scene changes that the user did not request.\n\n"
        "overall_soundscape:\nPreserve assigned audio when present; otherwise use only scene-consistent ambience and physical action sounds.\n\n"
        "non_diegetic_music:\nN/A"
    )


class WenWuH3PrepareReferences:
    """子图内部节点：只负责素材读取和H3条件编码，可独立缓存。"""
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "clip": ("CLIP",), "video_vae": ("VAE",), "audio_vae": ("VAE",),
            "prompt": ("STRING", {"multiline": True}),
            "width": ("INT",), "height": ("INT",), "length": ("INT",),
            "reference_size": (["match", "max"],),
        }
        for kind, count in (("image", 9), ("video", 3), ("audio", 3)):
            for i in range(1, count + 1):
                required[f"{kind}_{i}"] = ("STRING", {"default": ""})
        return {"required": required}

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    FUNCTION = "prepare"
    CATEGORY = "Liao2049/Internal"

    def prepare(self, clip, video_vae, audio_vae, prompt, width, height, length, reference_size, **kwargs):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        image_names = [_clean_filename(kwargs.get(f"image_{i}")) for i in range(1, 10)]
        video_names = [_clean_filename(kwargs.get(f"video_{i}")) for i in range(1, 4)]
        audio_names = [_clean_filename(kwargs.get(f"audio_{i}")) for i in range(1, 4)]
        images, videos, video_audios, soundtrack_flags, audios = _load_media(
            [x for x in image_names if x], [x for x in video_names if x], [x for x in audio_names if x]
        )
        native_prompt = build_native_prompt(prompt, len(images), soundtrack_flags, len(audios))
        result = _result(MiniMaxH3ReferenceToVideo.execute(
            clip, video_vae, audio_vae, native_prompt, int(width), int(height), int(length), reference_size,
            {f"ref_image_{i}": v for i, v in enumerate(images)},
            {f"ref_video_{i}": v for i, v in enumerate(videos)},
            {f"ref_video_audio_{i}": v for i, v in enumerate(video_audios) if v is not None},
            {f"ref_audio_{i}": v for i, v in enumerate(audios)},
        ))
        return tuple(result)


class WenWuH3ProgressSampler:
    """子图内部采样节点：保留进度，不制作Latent预览图。"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "noise": ("NOISE",), "guider": ("GUIDER",), "sampler": ("SAMPLER",),
            "sigmas": ("SIGMAS",), "latent_image": ("LATENT",),
        }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Liao2049/Internal"

    def sample(self, noise, guider, sampler, sigmas, latent_image):
        return (_sample_progress_only(noise, guider, sampler, sigmas, latent_image),)


class WenWuH3PhaseMarker:
    """Pass a latent through while reporting an internal two-pass phase."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "phase": ("STRING", {"default": "处理中"}),
            "progress": ("INT", {"default": 0, "min": 0, "max": 100}),
            "span": ("INT", {"default": 0, "min": 0, "max": 100}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "mark"
    CATEGORY = "Liao2049/Internal"
    NOT_IDEMPOTENT = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def mark(self, samples, phase="处理中", progress=0, span=0):
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("liao_h3_phase", {
                "phase": str(phase),
                "progress": int(progress),
                "span": int(span),
            })
        except Exception:
            pass
        print(f"[Liao-H3] 二采阶段: {phase} ({int(progress)}%)")
        return (samples,)


class WenWuH3SigmaTailRefiner:
    """Add a small low-noise finishing step without an external Sigma plugin."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sigmas": ("SIGMAS",),
            "extra_steps": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1}),
            "start_at_sigma": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 20.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "refine"
    CATEGORY = "Liao2049/Internal"

    def refine(self, sigmas, extra_steps=1, start_at_sigma=0.7):
        import torch
        extra_steps = max(0, int(extra_steps))
        if extra_steps == 0 or sigmas.numel() < 3:
            return (sigmas,)
        threshold = float(start_at_sigma)
        matches = torch.nonzero(sigmas <= threshold, as_tuple=False).flatten()
        start = int(matches[0].item()) if matches.numel() else max(1, sigmas.numel() - 2)
        start = min(max(1, start), sigmas.numel() - 2)
        head, tail = sigmas[:start], sigmas[start:]
        source_x = torch.linspace(0.0, 1.0, tail.numel(), device=tail.device, dtype=torch.float32)
        target_x = torch.linspace(0.0, 1.0, tail.numel() + extra_steps, device=tail.device, dtype=torch.float32)
        # Cosine placement concentrates the additional point near the clean end.
        target_x = 0.5 - 0.5 * torch.cos(target_x * math.pi)
        right = torch.searchsorted(source_x, target_x, right=False).clamp(1, tail.numel() - 1)
        left = right - 1
        weight = (target_x - source_x[left]) / (source_x[right] - source_x[left]).clamp_min(1e-8)
        refined = tail[left] + (tail[right] - tail[left]) * weight.to(tail.dtype)
        refined[0], refined[-1] = tail[0], tail[-1]
        return (torch.cat((head, refined)),)


class WenWuH3ReleaseAtStart:
    DEPRECATED = True
    """每次新任务最先执行：清掉上一次H3任务残留的GPU驻留和CUDA缓存。"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": ("STRING", {"default": ""}),
            "clip_name": ("STRING", {"default": ""}),
            "video_vae_name": ("STRING", {"default": ""}),
            "audio_vae_name": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("unet_name", "clip_name", "video_vae_name", "audio_vae_name")
    FUNCTION = "release"
    CATEGORY = "Liao2049/Internal"
    NOT_IDEMPOTENT = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # NOT_IDEMPOTENT在当前ComfyUI中只会把稳定node_id加入缓存键，第二次队列仍可能命中缓存。
        # NaN永不等于自身，强制每次Queue都真正执行“加载前释放”，并使下游Loader不复用旧任务对象。
        return float("NaN")

    def release(self, unet_name, clip_name, video_vae_name, audio_vae_name):
        import gc
        import comfy.model_management as mm
        before = mm.get_free_memory()
        # 提示词增强完成后保留 llama.cpp 权重驻留，方便连续修改和再次增强；
        # 只有用户正式运行 H3 任务、执行到本开始节点时才关闭本地 Llama。
        llama_was_loaded = _WenWuEmbeddedLlama.is_loaded()
        _WenWuEmbeddedLlama.unload()
        # 放在Loader之前而不是采样前：避免第二次任务先构造新patcher，再与旧任务驻留交叠。
        mm.unload_all_models()
        gc.collect()
        mm.cleanup_models()
        mm.cleanup_models_gc()
        mm.soft_empty_cache()
        after = mm.get_free_memory()
        llama_note = "，已卸载本地 Llama" if llama_was_loaded else ""
        print(f"[文武H3] 新任务加载前释放显存{llama_note}：{before / 1024**2:.0f} MiB → {after / 1024**2:.0f} MiB 可用")
        return (unet_name, clip_name, video_vae_name, audio_vae_name)


class WenWuH3ReleaseBeforeDecode:
    DEPRECATED = True
    """采样完成后、双VAE解码前，明确卸载扩散/文本模型的GPU驻留。"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "release"
    CATEGORY = "Liao2049/Internal"
    NOT_IDEMPOTENT = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def release(self, samples):
        import gc
        import comfy.model_management as mm
        before = mm.get_free_memory()
        mm.unload_all_models()
        gc.collect()
        mm.cleanup_models()
        mm.cleanup_models_gc()
        mm.soft_empty_cache()
        after = mm.get_free_memory()
        print(f"[文武H3] VAE解码前释放显存：{before / 1024**2:.0f} MiB → {after / 1024**2:.0f} MiB 可用")
        return (samples,)


class LiaoH3SecondPassModelBarrier:
    """Finish pass one, release its model, then expose the pass-two model name.

    The latent input creates a real graph dependency, preventing ComfyUI from
    loading both large H3 UNets at the same time.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",),
            "unet_name": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "unet_name")
    FUNCTION = "swap"
    CATEGORY = "Liao2049/Internal"
    NOT_IDEMPOTENT = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def swap(self, samples, unet_name):
        import gc
        import comfy.model_management as mm
        mm.unload_all_models()
        gc.collect()
        mm.cleanup_models()
        mm.cleanup_models_gc()
        mm.soft_empty_cache()
        print(f"[Liao-H3] 首采完成并释放模型，准备双模型重绘：{unet_name}")
        return (samples, unet_name)


class WenWuH3ReleaseBeforeConditioning:
    DEPRECATED = True
    """FL2VA条件编码前释放旧GPU驻留，再按需加载CLIP与视频VAE。"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "vae": ("VAE",)}}

    RETURN_TYPES = ("CLIP", "VAE")
    RETURN_NAMES = ("clip", "vae")
    FUNCTION = "release"
    CATEGORY = "Liao2049/Internal"
    NOT_IDEMPOTENT = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def release(self, clip, vae):
        import gc
        import comfy.model_management as mm
        before = mm.get_free_memory()
        # Loader只构造patcher；ImageToVideo会实际执行CLIP和首尾帧VAE编码。
        # 这里清除上轮采样/解码驻留，避免条件编码峰值与旧模型叠加。
        mm.unload_all_models()
        gc.collect()
        mm.cleanup_models()
        mm.cleanup_models_gc()
        mm.soft_empty_cache()
        after = mm.get_free_memory()
        print(f"[文武H3] FL2VA条件编码前释放显存：{before / 1024**2:.0f} MiB → {after / 1024**2:.0f} MiB 可用")
        return (clip, vae)


class WenWuH3AudioDrive:
    DEPRECATED = True
    """Lock source speech into the H3 joint latent without an external custom node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"av_latent": ("LATENT",), "source_audio": ("AUDIO",), "audio_vae": ("VAE",)}}

    RETURN_TYPES = ("LATENT", "AUDIO")
    RETURN_NAMES = ("锁定音频的AV latent", "原始音频")
    FUNCTION = "lock"
    CATEGORY = CATEGORY

    def lock(self, av_latent, source_audio, audio_vae):
        import torch
        import torchaudio
        import comfy.nested_tensor

        if not isinstance(av_latent, dict) or "samples" not in av_latent:
            raise ValueError("数字人模式需要 MiniMax H3 联合音视频 latent。")
        samples = av_latent["samples"]
        if not getattr(samples, "is_nested", False):
            raise ValueError("数字人模式收到的不是 MiniMax H3 联合音视频 latent。")
        video, template_audio = tuple(samples.unbind())
        waveform = source_audio.get("waveform") if isinstance(source_audio, dict) else None
        sample_rate = source_audio.get("sample_rate") if isinstance(source_audio, dict) else None
        if not isinstance(waveform, torch.Tensor) or sample_rate is None or waveform.ndim != 3:
            raise ValueError("数字人模式需要有效的源音频。")
        vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if int(sample_rate) != vae_rate:
            waveform = torchaudio.functional.resample(waveform, int(sample_rate), vae_rate)
        encoded = audio_vae.encode(waveform[:1].movedim(1, -1))
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 4:
            raise ValueError("音频 VAE 没有返回 MiniMax H3 所需的四维音频 latent。")
        if encoded.shape[1:-1] != template_audio.shape[1:-1]:
            raise ValueError("源音频 latent 与 MiniMax H3 音频 latent 结构不匹配。")
        if encoded.shape[-1] > template_audio.shape[-1]:
            encoded = encoded[..., :template_audio.shape[-1]]
        elif encoded.shape[-1] < template_audio.shape[-1]:
            padding = encoded.new_zeros((*encoded.shape[:-1], template_audio.shape[-1] - encoded.shape[-1]))
            encoded = torch.cat((encoded, padding), dim=-1)
        encoded = encoded.to(device=template_audio.device, dtype=template_audio.dtype)
        masks = av_latent.get("noise_mask")
        video_mask = tuple(masks.unbind())[0] if getattr(masks, "is_nested", False) else masks
        if video_mask is None:
            video_mask = torch.ones_like(video)
        output = av_latent.copy()
        output["samples"] = comfy.nested_tensor.NestedTensor((video, encoded))
        output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, torch.zeros_like(encoded)))
        return (output, source_audio)


class WenWuH3AudioLength:
    DEPRECATED = True
    """Convert the final driving audio duration to H3's 17n+5 frame grid."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("H3帧数",)
    FUNCTION = "plan"
    CATEGORY = CATEGORY

    def plan(self, audio):
        waveform = audio.get("waveform") if isinstance(audio, dict) else None
        sample_rate = audio.get("sample_rate") if isinstance(audio, dict) else None
        if waveform is None or not sample_rate or getattr(waveform, "ndim", 0) != 3:
            raise ValueError("数字人模式无法读取驱动音频长度。")
        frames = max(5, round((waveform.shape[-1] / int(sample_rate)) * FPS))
        frames += (5 - (frames % 17)) % 17
        return (int(frames),)


class WenWuH3AudioCrop:
    DEPRECATED = True
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "开始秒": ("FLOAT", {"default": 0.0}), "结束秒": ("FLOAT", {"default": 15.0})}}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "crop"
    CATEGORY = CATEGORY

    def crop(self, audio, 开始秒, 结束秒):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        start = max(0, round(float(开始秒) * sample_rate))
        end = min(waveform.shape[-1], round(float(结束秒) * sample_rate))
        if end <= start:
            raise ValueError("连续数字人音频分段为空。")
        return ({"waveform": waveform[..., start:end], "sample_rate": sample_rate},)


class WenWuH3LastFrame:
    DEPRECATED = True
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "last"
    CATEGORY = CATEGORY

    def last(self, images):
        if images is None or images.shape[0] < 1:
            raise ValueError("连续数字人无法取得上一段尾帧。")
        return (images[-1:].clone(),)


class WenWuH3TrimFramesToAudio:
    DEPRECATED = True
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "audio": ("AUDIO",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "trim"
    CATEGORY = CATEGORY

    def trim(self, images, audio):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        target = max(1, round((waveform.shape[-1] / sample_rate) * FPS))
        return (images[:min(images.shape[0], target)],)


class WenWuH3ModelLoraConfig:
    DEPRECATED = True
    """Separate model/LoRA selection from the creative H3 console."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            diffusion = folder_paths.get_filename_list("diffusion_models")
            encoders = folder_paths.get_filename_list("text_encoders")
            vaes = folder_paths.get_filename_list("vae")
            loras = folder_paths.get_filename_list("loras")
        except Exception:
            diffusion, encoders, vaes, loras = [], [], [], []
        h3_models = [x for x in diffusion if _is_minimax_h3_video_model(x)] or list(DEFAULT_MODEL_OPTIONS)
        h3_encoders = [x for x in encoders if any(tag in x.lower() for tag in ("minimax_h3", "qwen3vl", "qwen3-vl", "qwen3_vl"))] or [DEFAULT_TEXT_ENCODER]
        video_vaes = [x for x in vaes if "minimax_h3_video_vae" in x.lower()] or [DEFAULT_VIDEO_VAE]
        audio_vaes = [x for x in vaes if "minimax_h3_audio_vae" in x.lower()] or [DEFAULT_AUDIO_VAE]
        source_model = next((x for x in h3_models if "fl2va_pruned_w4a8_mixed" in x.lower()), h3_models[0])
        source_video_vae = next((x for x in video_vaes if "int8_convrot" in x.lower()), video_vaes[0])
        turbo_lora = _pick_minimax_h3_turbo_lora(loras)
        required = OrderedDict([
            ("模型", (h3_models, {"default": source_model})),
            ("文本编码器", (h3_encoders, {"default": h3_encoders[0]})),
            ("视频VAE", (video_vaes, {"default": source_video_vae})),
            ("音频VAE", (audio_vaes, {"default": audio_vaes[0]})),
            ("文本编码器类型", (["minimax"], {"default": "minimax"})),
            ("文本编码器设备", (["default", "cpu"], {"default": "default"})),
            ("模型权重精度", (["default", "fp8_e4m3fn", "fp8_e5m2"], {"default": "default"})),
            ("SageAttention", (SAGE_MODES, {"default": "auto"})),
            ("允许编译", ("BOOLEAN", {"default": False})),
            ("加速方案", (["参考工作流加速", "兼容模式"], {"default": "参考工作流加速"})),
            ("视频SigmaShift", ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1})),
            ("音频SigmaShift", ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1})),
        ])
        lora_choices = ["无"] + list(loras)
        for index in range(1, 3):
            default_lora = turbo_lora if index == 1 and turbo_lora else "无"
            default_strength = 0.75 if index == 1 and turbo_lora else 1.0
            required[f"LoRA{index}"] = (lora_choices, {"default": default_lora})
            required[f"LoRA{index}强度"] = ("FLOAT", {"default": default_strength, "min": -4.0, "max": 4.0, "step": 0.05})
        return {"required": required}

    RETURN_TYPES = ("WENWU_H3_MODEL_CONFIG",)
    RETURN_NAMES = ("模型与LoRA配置",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, 模型, 文本编码器, 视频VAE, 音频VAE, 文本编码器类型, 文本编码器设备,
              模型权重精度, SageAttention, 允许编译, 加速方案, 视频SigmaShift, 音频SigmaShift, **kwargs):
        loras = []
        for index in range(1, 3):
            name = str(kwargs.get(f"LoRA{index}") or "无")
            if name != "无":
                try:
                    strength = float(kwargs.get(f"LoRA{index}强度", 1.0))
                    if not math.isfinite(strength):
                        strength = 1.0
                except (TypeError, ValueError):
                    strength = 1.0
                loras.append({"name": name, "strength": strength})
        return ({
            "model": 模型, "text_encoder": 文本编码器,
            "video_vae": 视频VAE, "audio_vae": 音频VAE,
            "text_encoder_type": 文本编码器类型, "text_encoder_device": 文本编码器设备,
            "weight_dtype": 模型权重精度, "sage_attention": SageAttention,
            "allow_compile": bool(允许编译), "loras": loras,
            "accelerated": 加速方案 == "参考工作流加速",
            "shift_video": float(视频SigmaShift), "shift_audio": float(音频SigmaShift),
        },)


VIDEO_EDIT_TOOLS = ("通用编辑", "去除字幕", "动作迁移", "角色替换")


def _video_edit_tool_prompt(source: str, tool: str, image_count: int, duration: float) -> str:
    """Build deterministic Ref2VA instructions for common video-edit tasks."""
    creative = str(source or "").strip() or "Follow the selected video-edit task exactly."
    seconds = max(1.0, min(15.0, float(duration)))
    if tool == "去除字幕":
        return (
            "summary:\nRemove only the visible subtitles and captions from <Video 1>.\n\n"
            "retention_analysis:\n<Video 1> is fully_preserved except for subtitle/caption pixels. Preserve every person, face, body, "
            "garment, action, object, background, camera move, framing, color, lighting and audio.\n\n"
            "integrated_multimodal_description:\nTrack every subtitle or caption region over time and reconstruct only the occluded background "
            "with temporally stable texture. Do not crop, restage, beautify, replace, recolor or redesign any other content. Do not add new text. "
            f"Additional user instruction: {creative} Finish within {seconds:.2f} seconds.\n\n"
            "overall_soundscape:\nFully preserve the synchronized source audio from <Video 1>.\n\n"
            "non_diegetic_music:\nPreserve source music unchanged."
        )
    if tool == "动作迁移":
        scene_rule = (
            "<Picture 2> is the destination scene reference only; preserve its environment and spatial atmosphere."
            if image_count > 1 else
            "The destination scene follows the user's instruction and <Picture 1>; never copy the source video's location or background."
        )
        return (
            "subject_definitions:\n<Subject 1> is the exact target subject defined by <Picture 1>; preserve its recognizable identity, "
            "intrinsic appearance, body proportions, hairstyle and clothing.\n"
            f"{scene_rule}\n\n"
            "<Audio 1> is the synchronized audio track extracted from <Video 1>.\n\n"
            "summary:\nGenerate <Subject 1> performing the complete motion from <Video 1>. Completely exclude the original performer.\n\n"
            "retention_analysis:\n<Picture 1> is fully_preserved for the target subject. <Video 1> is weak_reference only for pose "
            "sequence, choreography, gesture, facial-performance rhythm, body timing, action speed and camera timing. Explicitly exclude "
            "the source performer, face, identity, hair, clothing, background, location, objects, color palette and lighting.\n\n"
            "integrated_multimodal_description:\nUse a single continuous shot. Animate <Subject 1> reproducing <Video 1> frame by frame: "
            "match every pose transition, foot placement, balance change, limb angle, hand gesture, head turn, gaze, expression, pause and beat. "
            "Retarget motion naturally while preserving <Subject 1> in every frame. The source performer must never appear, even briefly. "
            f"Follow this additional creative instruction: {creative} Complete the action within {seconds:.2f} seconds.\n\n"
            "overall_soundscape:\nFully copy <Audio 1> and keep every action synchronized to its original timing.\n\n"
            "non_diegetic_music:\nPreserve source music unchanged when present."
        )
    if tool == "角色替换":
        return (
            "subject_definitions:\n<Subject 1> is the replacement character from <Picture 1>. Fully preserve the exact facial identity, "
            "face shape, eyes, nose, mouth, skin tone, age cues, hair, body proportions, clothing and complete recognizable appearance.\n\n"
            "summary:\nReplace the entire original main character in <Video 1> with <Subject 1>. The original character and original outfit must be completely absent.\n\n"
            "retention_analysis:\n<Picture 1> is fully_preserved for the replacement character's complete identity, hairstyle, body proportions and clothing. "
            "<Video 1> is partially_preserved only for the background environment, objects, composition, camera movement and lighting, and is a "
            "weak_reference for body pose, performance, action timing, expression timing and motion trajectory. Explicitly exclude the source "
            "person's face, identity, hair, body appearance and clothing.\n\n"
            "integrated_multimodal_description:\nPlace <Subject 1> into the source video's scene and make <Subject 1> perform the original character's "
            "complete motion from the first frame to the last. The visible person must match <Picture 1> in face, hair, body and outfit throughout "
            "front, profile and moving views. Retarget the source pose, gaze, expression, occlusion and action timing to <Subject 1>, while retaining "
            "only the source background, objects, camera and lighting. Never reconstruct, blend back or retain any part of the original person. "
            "Do not copy <Picture 1>'s background or static pose. "
            f"Additional user instruction: {creative} Finish within {seconds:.2f} seconds.\n\n"
            "overall_soundscape:\nFully preserve <Video 1> audio.\n\n"
            "non_diegetic_music:\nPreserve source music unchanged."
        )
    return creative


class WenWuMiniMaxH3Unified:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            model_options = folder_paths.get_filename_list("diffusion_models")
            text_encoders = folder_paths.get_filename_list("text_encoders")
            vaes = folder_paths.get_filename_list("vae")
            lora_options = folder_paths.get_filename_list("loras")
        except Exception:
            model_options = list(DEFAULT_MODEL_OPTIONS)
            text_encoders = [DEFAULT_TEXT_ENCODER]
            vaes = [DEFAULT_VIDEO_VAE, DEFAULT_AUDIO_VAE]
            lora_options = []

        h3_models = [x for x in model_options if _is_minimax_h3_video_model(x)] or list(DEFAULT_MODEL_OPTIONS)
        # 用户实际可稳定运行的 F:/video_minimax_h3_r2v (1).json 使用完整INT8版，不是pruned版。
        source_model = next((x for x in h3_models if x.replace("\\", "/").lower().endswith("minimax_h3_ref2va_int8_convrot.safetensors") and "pruned" not in x.lower()), h3_models[0])
        h3_text_encoders = [x for x in text_encoders if any(tag in x.lower() for tag in ("minimax_h3", "qwen3vl", "qwen3-vl", "qwen3_vl"))] or [DEFAULT_TEXT_ENCODER]
        h3_video_vaes = [x for x in vaes if "minimax_h3_video_vae" in x.lower()] or [DEFAULT_VIDEO_VAE]
        h3_audio_vaes = [x for x in vaes if "minimax_h3_audio_vae" in x.lower()] or [DEFAULT_AUDIO_VAE]

        required = OrderedDict([
            ("模型", (h3_models, {"default": source_model})),
            ("文本编码器", (h3_text_encoders, {"default": DEFAULT_TEXT_ENCODER if DEFAULT_TEXT_ENCODER in h3_text_encoders else h3_text_encoders[0]})),
            ("文本编码器类型", (["minimax"], {"default": "minimax"})),
            ("文本编码器设备", (["default", "cpu"], {"default": "default"})),
            ("视频VAE", (h3_video_vaes, {"default": h3_video_vaes[0]})),
            ("音频VAE", (h3_audio_vaes, {"default": h3_audio_vaes[0]})),
            ("模型权重精度", (["default", "fp8_e4m3fn", "fp8_e5m2"], {"default": "default"})),
            # 与 F:/video_minimax_h3_r2v (1).json 一致：KJ SageAttention=auto。
            ("SageAttention", (SAGE_MODES, {"default": "auto"})),
            ("允许编译", ("BOOLEAN", {"default": False})),
            ("画面比例", (list(RATIOS), {"default": "16:9 (Widescreen)"})),
            ("百万像素", (MEGAPIXELS, {"default": 0.4})),
            ("尺寸倍数", ([32], {"default": 32})),
            ("时长秒", ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5})),
            ("提示词", ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""})),
            ("随机种子", ("INT", {"default": 470115107471061, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})),
            ("采样器", (["euler", "res_multistep"], {"default": "euler"})),
            ("调度器", (["simple"], {"default": "simple"})),
            ("采样步数", ("INT", {"default": 6, "min": 1, "max": 100})),
            ("降噪强度", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
            ("参考图尺寸", (["match", "max"], {"default": "match"})),
        ])
        # 文件由前端素材卡上传到 ComfyUI/input；普通下拉作为无 JS 时的兼容回退。
        try:
            files = sorted(os.listdir(folder_paths.get_input_directory()))
        except Exception:
            files = []
        choices = ["未选择"] + files
        for kind, count in (("图片", 9), ("视频", 3), ("音频", 3)):
            for index in range(1, count + 1):
                required[f"{kind}{index}"] = (choices, {"default": "未选择"})
        # 序列化层必须追加在全部旧字段之后，避免升级时把旧工作流的降噪、参考图和素材按位置错写。
        # 自定义DOM仍把它们显示在采样步数正下方。
        required["文生视频"] = ("BOOLEAN", {"default": False})
        required["图生视频"] = ("BOOLEAN", {"default": False})
        required["首尾帧"] = ("BOOLEAN", {"default": False})
        # WenWu owns its Llama model picker and runtime.  Register models/LLM
        # here so no external ComfyUI-llama-cpp custom node is required.
        try:
            import folder_paths
            llm_dir = os.path.join(folder_paths.models_dir, "LLM")
            os.makedirs(llm_dir, exist_ok=True)
            if "LLM" not in folder_paths.folder_names_and_paths:
                folder_paths.add_model_folder_path("LLM", llm_dir)
            all_llm_models = [name for name in folder_paths.get_filename_list("LLM") if name.lower().endswith(".gguf")]
            vision_models = [name for name in all_llm_models if "mmproj" in name.lower()]
            llm_models = [name for name in all_llm_models if name not in vision_models]
        except Exception:
            llm_models, vision_models = [], []
        # Keep GGUF selections as portable strings instead of ComfyUI model
        # combos. The frontend supplies suggestions from our own endpoint.
        # This prevents ComfyUI's client-side missing-model scanner from
        # blocking a workflow saved on another machine before it is queued.
        required["Llama模型"] = ("STRING", {
            # Never serialize a machine-local GGUF into a newly created H3
            # node. Prompt enhancement is optional; the frontend can offer
            # installed models only after the user enables it.
            "default": "",
            "multiline": False,
        })
        required["Llama上下文"] = ("INT", {"default": 8192, "min": 1024, "max": 131072, "step": 1024})
        required["Llama运算设备"] = (["自动", "全部GPU", "仅CPU"], {"default": "自动"})
        required["启用提示词增强"] = ("BOOLEAN", {"default": False})
        # UI-only state stays after all legacy fields so old workflows retain
        # their original generation widget positions.
        required["增强源提示词"] = ("STRING", {"default": "", "multiline": True})
        required["仅增强提示词"] = ("BOOLEAN", {"default": False})
        # 新模式必须放在全部既有字段末尾，旧工作流才能按原位置无损载入。
        required["视频编辑"] = ("BOOLEAN", {"default": False})
        required["数字人"] = ("BOOLEAN", {"default": False})
        required["加速方案"] = (["参考工作流加速", "兼容模式"], {"default": "参考工作流加速"})
        required["视频SigmaShift"] = ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1})
        required["音频SigmaShift"] = ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1})
        lora_choices = ["无"] + list(lora_options)
        turbo_lora = _pick_minimax_h3_turbo_lora(lora_options)
        required["LoRA1"] = (lora_choices, {"default": turbo_lora or "无"})
        required["LoRA1强度"] = ("FLOAT", {"default": 0.75 if turbo_lora else 1.0, "min": -4.0, "max": 4.0, "step": 0.05})
        required["LoRA2"] = (lora_choices, {"default": "无"})
        required["LoRA2强度"] = ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05})
        required["双人数字人"] = ("BOOLEAN", {"default": False})
        required["视频编辑模式"] = (["极速4步", "均衡8步", "质量20步"], {"default": "均衡8步"})
        # Cloud prompt fields are appended to preserve all existing workflow widget positions.
        required["提示词服务"] = (["本地 Llama", "Kimi API", "MiniMax API", "OpenAI 兼容 API"], {"default": "本地 Llama"})
        required["云端APIKey"] = ("STRING", {"default": "", "multiline": False})
        # MiniMax 官方当前 OpenAI 兼容接口默认模型。Kimi 模式仍会在
        # 执行时把该默认值视为自动匹配，不要求用户手动清空。
        required["云端模型"] = ("STRING", {"default": "MiniMax-M2.7", "multiline": False})
        # 每次点击增强由前端递增，避免相同创意命中 ComfyUI 节点缓存。
        # 放在最后以保证所有旧工作流的 widget 位置保持不变。
        required["增强请求序号"] = ("INT", {"default": 0, "min": 0, "max": 2147483646, "step": 1})
        required["视觉识别模型"] = ("STRING", {
            # Keep disabled prompt enhancement fully portable. A saved
            # mmproj filename must not make another PC fail preflight.
            "default": "",
            "multiline": False,
        })
        # 模板选择永远追加在末尾，避免旧工作流按控件位置反序列化时错位。
        required["提示词模板"] = (
            [PROMPT_TEMPLATE_AUTO, PROMPT_TEMPLATE_OFFICIAL, PROMPT_TEMPLATE_STORYBOARD],
            {"default": PROMPT_TEMPLATE_AUTO},
        )
        # Append only: keeps positional loading of every existing workflow intact.
        # The visible "automatic" choice is frontend-only. ComfyUI always
        # receives a validated integer, avoiding unsupported combo values in
        # old workflows or servers that have not restarted yet.
        required["分镜数"] = ("INT", {"default": 2, "min": 1, "max": 12, "step": 1})
        # 前端在用户手动修改模型/LoRA/VAE时开启；追加在末尾以兼容旧工作流位置。
        required["自定义模型配置"] = ("BOOLEAN", {"default": False})
        # Append only: old workflows load widgets positionally.
        required["视频编辑功能"] = (list(VIDEO_EDIT_TOOLS), {"default": "通用编辑"})
        # Append only: optional H3 latent-space upscale.  Keeping this at the
        # very end prevents positional widget drift in existing workflows.
        required["二采放大精修"] = ("BOOLEAN", {"default": False})
        # Append only: old workflows keep every prior positional widget value.
        required["OpenAI兼容地址"] = ("STRING", {"default": "http://127.0.0.1:11434/v1", "multiline": False})
        # Append only: old workflows default to the original latent method.
        required["二采方式"] = (["潜空间二采", "双模型重绘"], {"default": "潜空间二采"})
        # Append-only MV state and extra picture slots. Existing workflows keep
        # every historical widget index, while MV timelines can cover up to
        # five minutes with twenty <=15-second picture sections.
        required["MV数字人"] = ("BOOLEAN", {"default": False})
        for index in range(10, 21):
            required[f"图片{index}"] = (choices, {"default": "未选择"})
        return {"required": required}

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Bypass ComfyUI's machine-local combo-list validation.

        ComfyUI normally rejects a workflow before execution whenever a saved
        model, LoRA, VAE, media or GGUF filename is absent from another PC's
        current combo list. A wildcard validator is intentional: it also covers
        legacy workflow fields and ComfyUI versions whose serialized inputs do
        not exactly match the current schema. Resources and numeric safety are
        resolved/checked by generate() at runtime instead.
        """
        return True

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("图像", "音频")
    FUNCTION = "generate"
    # The aurora UI queues this node by itself when the user clicks prompt
    # enhancement. ComfyUI otherwise rejects that reduced graph as having no
    # output node before generate() can enter its prompt-only branch.
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "本地 MiniMax H3 Ref2VA 一体化生成：9图、3视频、3音频，输出最终图像帧与音频。"

    def generate(self, 模型, 文本编码器, 文本编码器类型, 文本编码器设备, 视频VAE, 音频VAE,
                 模型权重精度, SageAttention, 允许编译, 画面比例, 百万像素, 尺寸倍数, 时长秒, 提示词, 随机种子,
                 采样器, 调度器, 采样步数, 降噪强度, 参考图尺寸="match", 文生视频=False, 图生视频=False, 首尾帧=False, **kwargs):
        时长秒 = float(时长秒)
        if not math.isfinite(时长秒) or not 1.0 <= 时长秒 <= 15.0:
            raise ValueError("Liao-H3 单次生成时长仅支持1至15秒，请将时长调整到15秒以内。")
        image_names = [_clean_filename(kwargs.get(f"图片{i}")) for i in range(1, 21)]
        video_names = [_clean_filename(kwargs.get(f"视频{i}")) for i in range(1, 4)]
        audio_names = [_clean_filename(kwargs.get(f"音频{i}")) for i in range(1, 4)]
        image_names, video_names, audio_names = _rebucket_media_names(image_names, video_names, audio_names)
        model_config = kwargs.get("模型配置")
        loras = []
        for index in range(1, 3):
            lora_name = str(kwargs.get(f"LoRA{index}") or "无")
            if lora_name != "无":
                strength = float(kwargs.get(f"LoRA{index}强度", 1.0))
                loras.append({"name": lora_name, "strength": strength if math.isfinite(strength) else 1.0})
        accelerated = str(kwargs.get("加速方案", "参考工作流加速")) == "参考工作流加速"
        custom_model_config = bool(kwargs.get("自定义模型配置", False))
        shift_video = float(kwargs.get("视频SigmaShift", 10.0))
        shift_audio = float(kwargs.get("音频SigmaShift", 3.0))
        latent_enhance = bool(kwargs.get("二采放大精修", False))
        refine_method = str(kwargs.get("二采方式") or "潜空间二采")
        dual_model_refine = latent_enhance and refine_method == "双模型重绘"
        if isinstance(model_config, dict):
            模型 = model_config.get("model", 模型)
            文本编码器 = model_config.get("text_encoder", 文本编码器)
            视频VAE = model_config.get("video_vae", 视频VAE)
            音频VAE = model_config.get("audio_vae", 音频VAE)
            文本编码器类型 = model_config.get("text_encoder_type", 文本编码器类型)
            文本编码器设备 = model_config.get("text_encoder_device", 文本编码器设备)
            模型权重精度 = model_config.get("weight_dtype", 模型权重精度)
            SageAttention = model_config.get("sage_attention", SageAttention)
            允许编译 = model_config.get("allow_compile", 允许编译)
            loras = list(model_config.get("loras") or [])
            accelerated = bool(model_config.get("accelerated", False))
            shift_video = float(model_config.get("shift_video", 10.0))
            shift_audio = float(model_config.get("shift_audio", 3.0))
            if accelerated and not custom_model_config:
                采样器, 调度器, 采样步数, 降噪强度 = "euler", "simple", 6, 1.0
        if accelerated and not custom_model_config:
            采样器, 调度器, 采样步数, 降噪强度 = "euler", "simple", 6, 1.0
        # “仅增强”只是提示词预处理时的瞬时状态。总开关关闭后，即使旧工作流
        # 残留了 True，也必须进入正常视频生成，不能再次误走增强分支。
        if bool(kwargs.get("仅增强提示词", False)) and bool(kwargs.get("启用提示词增强", False)):
            prompt_service = str(kwargs.get("提示词服务") or "本地 Llama")
            llama_model = str(kwargs.get("Llama模型") or "")
            if prompt_service == "本地 Llama":
                try:
                    import folder_paths
                    installed_ggufs = [
                        name for name in folder_paths.get_filename_list("LLM")
                        if str(name).lower().endswith(".gguf")
                    ]
                except Exception:
                    installed_ggufs = []
                installed_vision = [name for name in installed_ggufs if "mmproj" in name.lower()]
                installed_text = [name for name in installed_ggufs if name not in installed_vision]
                if llama_model not in installed_text:
                    llama_model = installed_text[0] if installed_text else ""
                if not llama_model:
                    raise ValueError("本机未找到可用 GGUF 模型。请把文本 GGUF 放入 ComfyUI/models/LLM 后刷新模型列表。")
                requested_vision = str(kwargs.get("视觉识别模型") or "")
                if requested_vision not in installed_vision:
                    kwargs["视觉识别模型"] = installed_vision[0] if installed_vision else "未找到视觉模型"
            source = str(kwargs.get("增强源提示词") or 提示词 or "").strip()
            if not source:
                raise ValueError("请先输入需要增强的创意或提示词。")

            image_count = sum(bool(name) for name in image_names)
            video_count = sum(bool(name) for name in video_names)
            audio_count = sum(bool(name) for name in audio_names)
            picture_tags = "、".join(f"<Picture {i}>" for i in range(1, image_count + 1)) or "无"
            video_tags = "、".join(f"<Video {i}>" for i in range(1, video_count + 1)) or "无"
            audio_tags = "、".join(f"<Audio {i}>" for i in range(1, audio_count + 1)) or "无"
            subject_options = [
                f"<Subject {i}>（仅当 <Picture {i}> 提供人物、动物、物体或商品身份时使用）"
                for i in range(1, image_count + 1)
            ]
            subject_options.extend(
                f"<Subject {image_count + i}>（仅当 <Video {i}> 提供人物、动物、物体或商品身份时使用）"
                for i in range(1, video_count + 1)
            )
            subject_tags = "、".join(subject_options) or "无"
            mode = ("文生视频" if 文生视频 else "图生视频" if 图生视频 else "首尾帧" if 首尾帧
                    else "视频编辑" if bool(kwargs.get("视频编辑", False))
                    else "MV数字人" if bool(kwargs.get("MV数字人", False))
                    else "双人数字人" if bool(kwargs.get("双人数字人", False))
                    else "单人数字人" if bool(kwargs.get("数字人", False)) else "多参考生成")
            official_mode = ("T2VA" if mode == "文生视频" else "I2VA" if mode == "图生视频"
                             else "FL2VA" if mode == "首尾帧" else "Ref2VA")
            requested_template = str(kwargs.get("提示词模板") or PROMPT_TEMPLATE_AUTO)
            if requested_template == PROMPT_TEMPLATE_OFFICIAL_LEGACY:
                requested_template = PROMPT_TEMPLATE_OFFICIAL
            if requested_template == PROMPT_TEMPLATE_VIDEO_EDIT_LEGACY:
                requested_template = PROMPT_TEMPLATE_OFFICIAL
            if requested_template == PROMPT_TEMPLATE_AUTO:
                prompt_template = (PROMPT_TEMPLATE_STORYBOARD if mode == "图生视频" else
                                   PROMPT_TEMPLATE_OFFICIAL)
            else:
                prompt_template = requested_template
            storyboard_template = prompt_template in {
                PROMPT_TEMPLATE_STORYBOARD, PROMPT_TEMPLATE_STORYBOARD_LEGACY,
                PROMPT_TEMPLATE_STORYBOARD_LEGACY_2, PROMPT_TEMPLATE_STORYBOARD_LEGACY_3,
            }
            explicit_storyboard_count = _extract_explicit_storyboard_count(source)
            storyboard_count = (explicit_storyboard_count if explicit_storyboard_count is not None else
                                _resolve_storyboard_count(kwargs.get("分镜数", "自动"), float(时长秒)))
            if explicit_storyboard_count is not None:
                print(f"[Liao2049 H3] 已采用用户文字中明确指定的 {storyboard_count} 个镜头。")
            video_edit_template = False
            system_prompt = (_storyboard_15s_profile(float(时长秒), storyboard_count) if storyboard_template else
                             _official_h3_profile(official_mode, include_director_layer=False,
                                                  target_duration=float(时长秒), shot_count=storyboard_count))
            if not storyboard_template:
                system_prompt += _official_shot_timeline_profile(official_mode, float(时长秒), storyboard_count)
            if mode == "多参考生成" and not storyboard_template:
                system_prompt += _multi_reference_adaptive_profile()
            vision_model = str(kwargs.get("视觉识别模型") or "")
            image_urls, vision_mappings = ([], [])
            video_frames_attached = False
            if prompt_service == "本地 Llama" and vision_model not in {"", "无", "未找到视觉模型"}:
                if video_names[0] and (video_edit_template or mode in {"多参考生成", "视频编辑"}):
                    image_urls, vision_mappings = _h3_source_video_frame_urls(video_names[0])
                    video_frames_attached = bool(image_urls)
                    reference_urls, _ = _h3_reference_image_urls(image_names)
                    offset = len(image_urls)
                    image_urls.extend(reference_urls)
                    vision_mappings.extend(
                        f"Attachment {offset + index - 1} = <Picture {index}> reference image"
                        for index in range(1, len(reference_urls) + 1)
                    )
                else:
                    image_urls, vision_mappings = _h3_reference_image_urls(image_names)
            vision_note = "、".join(vision_mappings) if vision_mappings else "未启用视觉识别或没有可读取图片"
            if video_frames_attached and vision_mappings:
                vision_note += (
                    "。重要：视频抽帧附件只是 <Video 1> 的观察帧，不是新的 Picture；"
                    "无论看见多少帧，都不得把它们写成 <Picture 2>、<Picture 3> 或 <Picture 4>。"
                )
            output_instruction = (f"请按 WenWu 分镜引擎输出严格为 {float(时长秒):.1f} 秒、可直接提交的中文成片提示词。"
                                  if storyboard_template else
                                  "请按 Liao 视频编辑模板输出仅修改目标、保留源视频其余内容的英文指令。"
                                   if video_edit_template else
                                  "请输出一份可直接用于 MiniMax H3 的英文提示词。")
            request = f"""用户原始创意：
{source}

节点模式：{mode}
官方提示词模式：{official_mode}
目标时长：{float(时长秒):.1f} 秒
目标分镜数：{storyboard_count}
画面比例：{画面比例}
可用图片标签：{picture_tags}
可用视频标签：{video_tags}
可用音频标签：{audio_tags}
允许的主体标签：{subject_tags}
视觉附件映射：{vision_note}
多参考语义：{"逐图按用户文字判定人物/动物、物体/商品、场景/环境或局部属性角色；只迁移指定内容，未指定内容不得继承。场景图不得伪造为 Subject。" if mode == "多参考生成" else "按当前生成模式处理。"}
忠实性硬约束：用户指定的动作、地点和事件必须逐义保留，不得改写成无关动作或场景；“在蹲坑”就是在蹲式厕所如厕，绝不能改成街道行走。

{output_instruction}不要分析任务或复述要求。
返回值必须是 JSON 对象；严格遵守当前所选模板规定的字段与正文格式。
对白、歌词和屏幕文字保持原语言。用户给出原话时逐字保留；用户明确要求说话、喊叫、嚷嚷、询问、回答、旁白或唱歌却未提供原句时，必须补写一句符合场景且能在对应时段说完的简短具体内容，并使用稳定说话人编号与 <d>[Chinese]具体台词</d> 格式。禁止用“开始说话”“诉说琐事”“大声嚷嚷”等空泛动作替代实际可听台词。用户未要求发声时不得添加对白。
只使用上面实际存在的标签，不要解释，不要推理过程，不要输出 Markdown 代码块。"""
            messages = _build_messages(system_prompt, request, image_urls, image_detail="high")
            if prompt_service == "本地 Llama":
                enhanced = (_WenWuEmbeddedLlama.invoke(
                    llama_model,
                    int(kwargs.get("Llama上下文", 8192)),
                    str(kwargs.get("Llama运算设备", "自动")),
                    messages, vision_model=vision_model if image_urls else "",
                    temperature=0.1, top_p=0.8, max_tokens=2800,
                    repeat_penalty=1.12, frequency_penalty=0.1,
                    response_format=(_storyboard_response_format() if storyboard_template else
                                     _video_edit_response_format() if video_edit_template else
                                     _h3_response_format(official_mode)),
                ) or "").strip()
                enhanced = (_format_storyboard_output(enhanced, float(时长秒), source, storyboard_count) if storyboard_template else
                            _format_video_edit_output(enhanced) if video_edit_template else
                            _format_h3_structured_output(enhanced, official_mode))
            else:
                enhanced = _cloud_prompt_invoke(
                    prompt_service,
                    str(kwargs.get("云端APIKey") or ""),
                    str(kwargs.get("云端模型") or ""),
                    messages,
                    str(kwargs.get("OpenAI兼容地址") or ""),
                )
                if storyboard_template:
                    enhanced = _format_storyboard_output(enhanced, float(时长秒), source, storyboard_count)
                elif video_edit_template:
                    enhanced = _format_video_edit_output(enhanced)
                else:
                    enhanced = _format_h3_structured_output(enhanced, official_mode)
            if not enhanced:
                raise RuntimeError(f"{prompt_service} 没有返回增强提示词，请检查模型与连接设置。")
            if not storyboard_template and not video_edit_template:
                enhanced = _validate_official_shot_timeline(enhanced, float(时长秒), storyboard_count)
            enhanced = _validate_h3_reference_tags(
                enhanced, image_count, video_count, audio_count,
                allow_video_frame_aliases=video_frames_attached,
            )
            return {"ui": {"aurora_enhanced_prompt": [enhanced]}, "result": (None, None)}

        from comfy_execution.graph_utils import GraphBuilder
        视频编辑 = bool(kwargs.get("视频编辑", False))
        数字人 = bool(kwargs.get("数字人", False))
        双人数字人 = bool(kwargs.get("双人数字人", False))
        MV数字人 = bool(kwargs.get("MV数字人", False))
        selected_modes = [name for name, enabled in (("文生视频", 文生视频), ("图生视频", 图生视频), ("首尾帧", 首尾帧), ("视频编辑", 视频编辑), ("单人数字人", 数字人), ("双人数字人", 双人数字人), ("MV数字人", MV数字人)) if enabled]
        if len(selected_modes) > 1:
            raise ValueError("生成方式只能开启一个。")
        generation_mode = selected_modes[0] if selected_modes else "多参考"
        # The custom DOM mode selector can outlive ComfyUI's hidden boolean
        # widget state in older workflows. Infer an edit server-side whenever
        # a source video is present and the instruction clearly requests a
        # mutation, so a stale `视频编辑=false` cannot silently route the job to
        # generic multi-reference generation.
        direct_instruction = str(kwargs.get("增强源提示词") or 提示词 or "").strip().lower()
        edit_cues = (
            "修改", "编辑", "替换", "换成", "换为", "换脸", "换人", "更换", "改成", "改为",
            "删除", "去掉", "移除", "擦除", "保留其余", "只改变", "只修改",
            "replace", "swap", "edit", "change", "remove", "erase",
        )
        if generation_mode == "多参考" and any(video_names) and any(cue in direct_instruction for cue in edit_cues):
            generation_mode = "视频编辑"
        # 空素材的默认“多参考”没有可参考的内容，按文生视频安全执行。
        # 明确选择图生、首尾帧、视频编辑或数字人时仍保留严格素材校验。
        if generation_mode == "多参考" and not any(image_names + video_names + audio_names):
            generation_mode = "文生视频"
        # Portable workflows often keep a saved mode while their machine-local
        # upload widget values disappear after copying to another computer.
        # Do not hard-fail such workflows before execution: degrade FL2VA to the
        # closest runnable mode according to the images that are actually
        # available in this session.
        available_image_count = sum(bool(x) for x in image_names)
        if generation_mode == "图生视频" and available_image_count < 1:
            print("[Liao-H3] 图生视频未检测到首帧图片，已自动切换为文生视频。")
            generation_mode = "文生视频"
        elif generation_mode == "首尾帧" and available_image_count < 2:
            if available_image_count == 1:
                print("[Liao-H3] 首尾帧仅检测到1张图片，已自动切换为图生视频。")
                generation_mode = "图生视频"
            else:
                print("[Liao-H3] 首尾帧未检测到图片，已自动切换为文生视频。")
                generation_mode = "文生视频"
        fl_mode = generation_mode if generation_mode in {"文生视频", "图生视频", "首尾帧"} else ""
        reference_edit_mode = generation_mode in {"多参考", "视频编辑", "单人数字人", "双人数字人", "MV数字人"}
        legacy_profiles = {"快速6步": "极速4步", "极速6步": "极速4步", "快速创意编辑6步（弱保留）": "极速4步", "均衡10步": "均衡8步", "精准人物替换20步": "质量20步"}
        video_edit_profile = str(kwargs.get("视频编辑模式", "均衡8步"))
        video_edit_profile = legacy_profiles.get(video_edit_profile, video_edit_profile)
        video_edit_fast = video_edit_profile == "极速4步"
        video_edit_tool = str(kwargs.get("视频编辑功能", "通用编辑"))
        if video_edit_tool not in VIDEO_EDIT_TOOLS:
            video_edit_tool = "通用编辑"
        if reference_edit_mode and not custom_model_config:
            # 多参考、视频编辑和数字人都必须走 Ref2VA；FL2VA Turbo 会弱化或合并独立参考主体。
            accelerated = False
            采样器, 调度器, 采样步数, 降噪强度 = "res_multistep", "simple", 20, 1.0
            loras = []
        # 三档是全局性能预设，对 FL2VA / Ref2VA 以及所有生成模式生效。
        if not custom_model_config and video_edit_profile == "极速4步":
            accelerated = True
            采样器, 调度器, 采样步数, 降噪强度 = "euler", "simple", 4, 1.0
        elif not custom_model_config and video_edit_profile == "均衡8步":
            accelerated = False
            采样器, 调度器, 采样步数, 降噪强度 = "res_multistep", "simple", 8, 1.0
        elif not custom_model_config:
            accelerated = False
            采样器, 调度器, 采样步数, 降噪强度 = "res_multistep", "simple", 20, 1.0
        # FL2VA 模式只读取自己需要的素材；切换模式时保留槽位内容，但不让旧素材进入后端。
        if generation_mode == "文生视频":
            image_names = [""] * 20
            video_names = [""] * 3
            audio_names = [""] * 3
        elif generation_mode == "图生视频":
            image_names = image_names[:1] + [""] * 19
            video_names = [""] * 3
            audio_names = [""] * 3
        elif generation_mode == "首尾帧":
            image_names = image_names[:2] + [""] * 18
            video_names = [""] * 3
            audio_names = [""] * 3
        elif generation_mode == "单人数字人":
            image_names = image_names[:1] + [""] * 19
            video_names = [""] * 3
            audio_names = audio_names[:1] + [""] * 2
        elif generation_mode == "双人数字人":
            image_names = image_names[:2] + [""] * 18
            video_names = [""] * 3
            audio_names = audio_names[:2] + [""]
        elif generation_mode == "MV数字人":
            video_names = [""] * 3
            audio_names = audio_names[:1] + [""] * 2
        if fl_mode and not custom_model_config:
            # FL2VA 的首/尾帧通过专用输入传递，不使用 Ref2VA 的 @素材标签。
            提示词 = re.sub(r"@(图片|视频音频|视频|音频)\d+", "", str(提示词 or "")).strip()
        # 槽位编号必须连续，确保可见 @编号 与 H3 原生编号完全一致。
        for label, values in (("图片", image_names), ("视频", video_names), ("音频", audio_names)):
            seen_gap = False
            for value in values:
                if not value:
                    seen_gap = True
                elif seen_gap:
                    raise ValueError(f"{label}槽位必须从1开始连续添加，不能跳号。")

        image_count = sum(bool(x) for x in image_names)
        if fl_mode == "图生视频" and image_count != 1:
            raise ValueError("图生视频模式必须上传1张图片作为首帧。")
        if fl_mode == "首尾帧" and image_count != 2:
            raise ValueError("首尾帧模式必须上传2张图片：图片1为首帧、图片2为尾帧。")
        if generation_mode == "视频编辑" and not any(video_names):
            raise ValueError("视频编辑模式必须上传至少1个需要编辑的视频。")
        if generation_mode == "视频编辑" and video_edit_tool in {"动作迁移", "角色替换"} and image_count < 1:
            raise ValueError(f"{video_edit_tool}需要至少上传1张目标主体参考图，并上传1段源视频。")
        # 对常见的人物替换短句补全 H3 原生素材标签与保留约束。
        # 否则“把视频中女人换成图中女人”容易被解读为整镜头重绘。
        raw_edit_prompt = str(提示词 or "").strip()
        # Normalize both UI aliases and native H3 labels before classifying the
        # edit. This covers "@视频1中女人替换成@图片1中女人" as well as
        # natural-language and already-expanded <Video>/<Picture> prompts.
        edit_intent = re.sub(
            r"@?视频\s*\d+|<Video\s+\d+>|video\s*\d+", "视频", raw_edit_prompt, flags=re.IGNORECASE
        )
        edit_intent = re.sub(
            r"@?图片\s*\d+|图\s*\d+|照片\s*\d+|<Picture\s+\d+>|picture\s*\d+|image\s*\d+",
            "图片", edit_intent, flags=re.IGNORECASE,
        )
        edit_intent = re.sub(r"\s+", "", edit_intent).lower()
        has_video_source = "视频" in edit_intent
        has_picture_source = any(token in edit_intent for token in (
            "图片", "参考图", "图中", "照片", "参考人物", "referenceimage", "referencepicture"
        ))
        has_person_target = any(token in edit_intent for token in (
            "女人", "女性", "男人", "男性", "人物", "主角", "角色", "人脸", "脸", "身份",
            "person", "woman", "man", "character", "face", "identity"
        ))
        has_replacement_action = any(token in edit_intent for token in (
            "换成", "换为", "替换成", "替换为", "替换", "换脸", "换人", "身份迁移", "人物更换",
            "replace", "swap", "identitytransfer", "faceswap"
        ))
        identity_replace = video_edit_tool == "角色替换" or (
            video_edit_tool == "通用编辑" and has_video_source and has_picture_source and has_person_target and has_replacement_action
        )
        if generation_mode == "视频编辑" and video_edit_tool != "通用编辑":
            提示词 = _video_edit_tool_prompt(raw_edit_prompt, video_edit_tool, image_count, float(时长秒))
        elif generation_mode == "视频编辑" and image_count and identity_replace:
            提示词 = (
                "subject_definitions:\n"
                "<Subject 1> is the exact facial identity, age cues and intrinsic physical appearance shown in <Picture 1>.\n\n"
                "summary:\n"
                "Identity replacement in <Video 1>: replace only the main woman/person with <Subject 1>.\n\n"
                "retention_analysis:\n"
                "<Video 1> is fully_preserved for clothing, hairstyle motion, body shape, pose, performance, action timing, "
                "camera movement, framing, background, lighting, objects and audio. "
                "<Picture 1> is attribute_transfer for identity and face only.\n\n"
                "detailed_description:\n"
                "Track the original target person throughout every frame of <Video 1> and replace that person's identity and face "
                "with <Subject 1>. Keep the source video's original clothing and hair motion; adapt only facial geometry, skin and "
                "identity-defining appearance to the existing head pose, expression, occlusion and illumination. Preserve all scene "
                "content and temporal motion exactly. Do not copy <Picture 1>'s clothing, pose, background, composition or lighting. "
                "Do not redesign, restage or regenerate the scene.\n\n"
                "overall_soundscape:\nFully preserve <Video 1> audio.\n\n"
                "non_diegetic_music:\nPreserve the source video music unchanged."
            )
        if not bool(kwargs.get("启用提示词增强", False)):
            提示词 = str(提示词 or "").strip()
            # A single-reference workflow can rely on the only picture implicitly.
            # With several references H3 needs explicit role/retention bindings or
            # it tends to treat Picture 1 as the entire frame and ignore the later
            # destination scene. Build the deterministic official wrapper only for
            # multi-reference input; no LLM or visual-recognition dependency is used.
            if generation_mode == "多参考" and image_count > 1:
                direct_prompt = build_native_prompt(
                    提示词, image_count, [True for name in video_names if name], sum(bool(x) for x in audio_names)
                )
                提示词 = _official_direct_prompt(
                    direct_prompt, "多参考", image_count,
                    sum(bool(x) for x in video_names), sum(bool(x) for x in audio_names), float(时长秒),
                )
        if generation_mode == "单人数字人" and (image_count != 1 or sum(bool(x) for x in audio_names) != 1):
            raise ValueError("单人数字人模式需要1张人物参考图和1段驱动音频。")
        if generation_mode == "双人数字人" and (image_count != 2 or sum(bool(x) for x in audio_names) != 2):
            raise ValueError("双人数字人模式需要2张人物参考图和2段驱动音频。")
        if generation_mode == "MV数字人":
            if image_count < 1 or sum(bool(x) for x in audio_names) != 1:
                raise ValueError("MV模式需要至少1张图片和1段完整音乐。")
            mv_duration = _audio_duration_seconds(audio_names[0])
            if mv_duration <= 0:
                raise ValueError("MV模式无法读取音乐时长。")
            if mv_duration / image_count > 15.0 + 1e-6:
                required_images = math.ceil(mv_duration / 15.0)
                raise ValueError(f"这段音乐至少需要{required_images}张图片；每个图片轨道最长15秒。")
        if not fl_mode and not any(image_names + video_names + audio_names):
            raise ValueError("请至少拖入一张图片、一个视频或一段音频。")
        width, height = resolution_from_megapixels(画面比例, 百万像素, int(尺寸倍数))
        # Memory-balanced variant of the verified MiniMax H3 latent-upscale
        # workflow: start at 75% of the selected linear resolution, use the
        # learned 3D latent upscaler at its native 2x scale, then sample the
        # low-sigma tail at the enlarged target.  The final dimensions are
        # therefore about 1.5x the selected dimensions (2.25x the pixels).
        condition_width, condition_height = width, height
        if latent_enhance and not dual_model_refine:
            condition_width = max(32, int(round(width * 0.75 / 16)) * 16)
            condition_height = max(32, int(round(height * 0.75 / 16)) * 16)
            width = condition_width * 2
            height = condition_height * 2
            print(
                f"[Liao-H3] 二采放大精修: 首采={condition_width}x{condition_height}, "
                f"最终输出={width}x{height}"
            )
        elif dual_model_refine:
            # The comparison workflow keeps pass one at the selected size,
            # pixel-upscales its decoded frames, then re-encodes and redraws at
            # roughly 2MP with a separate W4A8 model.
            condition_width, condition_height = width, height
            width, height = resolution_from_megapixels(画面比例, 2.0, int(尺寸倍数))
            print(
                f"[Liao-H3] 双模型重绘: 首采={condition_width}x{condition_height}, "
                f"二采输出={width}x{height}"
            )
        length = duration_to_frames(时长秒)
        continuous_duration = 0.0
        continuous_segments = 1
        if generation_mode == "单人数字人":
            continuous_duration = _audio_duration_seconds(audio_names[0])
            continuous_segments = max(1, math.ceil(continuous_duration / 15.0))
            if continuous_segments > 20:
                raise ValueError("连续数字人目前最多自动分20段（约5分钟音频）。请先裁短音频。")
        elif generation_mode == "MV数字人":
            continuous_duration = _audio_duration_seconds(audio_names[0])
            continuous_segments = image_count

        # 关键修复：展开成真实ComfyUI子图。Loader/条件/采样/解码重新成为独立缓存节点，
        # 不再在一个Python函数中手工持有整套模型对象。
        g = GraphBuilder()
        selected_model = 模型
        if fl_mode:
            if "minimax_h3_fl2va" in str(模型).replace("\\", "/").lower():
                selected_model = 模型
                fl_candidates = []
            else:
                fl_candidates = None
            try:
                import folder_paths
                installed = folder_paths.get_filename_list("diffusion_models")
            except Exception:
                installed = []
            if fl_candidates is None:
                fl_candidates = [x for x in installed if "minimax_h3_fl2va" in x.replace("\\", "/").lower()]
                selected_model = next(
                    (x for x in fl_candidates if x.replace("\\", "/").lower().endswith(FL2VA_MODEL_NAME)),
                    next((x for x in fl_candidates if "int8_convrot" in x.lower()), fl_candidates[0] if fl_candidates else FL2VA_MODEL_NAME),
                )
        # Clicking a dedicated edit tool clears stale custom state in the UI and
        # selects the verified recipe. If the user then deliberately changes a
        # model/LoRA/VAE, custom_model_config becomes true again and must win.
        elif reference_edit_mode and not custom_model_config:
            selected_is_ref = "minimax_h3_ref2va" in str(模型).replace("\\", "/").lower()
            try:
                import folder_paths
                installed = folder_paths.get_filename_list("diffusion_models")
            except Exception:
                installed = []
            ref_candidates = [x for x in installed if "minimax_h3_ref2va" in x.replace("\\", "/").lower()]
            fl_candidates = [x for x in installed if "minimax_h3_fl2va" in x.replace("\\", "/").lower()]
            if generation_mode == "多参考":
                if video_edit_profile == "极速4步":
                    # Exact accelerated reference-workflow hybrid: FL2VA W4A8
                    # model + FL2V Turbo LoRA, but references remain connected
                    # through MiniMaxH3ReferenceToVideo below.
                    selected_model = next(
                        (x for x in fl_candidates if "kijai_" in x.lower() and "w4a8_mixed" in x.lower()),
                        next((x for x in fl_candidates if "w4a8" in x.lower()), 模型),
                    )
                elif video_edit_profile == "均衡8步":
                    # 均衡档使用较大的 Ref2VA INT8（优先非 pruned），
                    # 配套 8-step Turbo LoRA 由下方统一选择并以 0.75 加载。
                    selected_model = next(
                        (x for x in ref_candidates if "int8_convrot" in x.lower() and "pruned" not in x.lower()),
                        next((x for x in ref_candidates if "int8_convrot" in x.lower()), 模型),
                    )
                elif video_edit_profile == "质量20步":
                    selected_model = next(
                        (x for x in ref_candidates if "bf16" in x.lower()),
                        next((x for x in ref_candidates if "int8_convrot" in x.lower() and "pruned" not in x.lower()), 模型),
                    )
                else:
                    selected_model = next(
                        (x for x in ref_candidates if "int8_convrot" in x.lower() and "pruned" not in x.lower()),
                        next((x for x in ref_candidates if "int8_convrot" in x.lower()), 模型),
                    )
                selected_is_ref = "minimax_h3_ref2va" in str(selected_model).replace("\\", "/").lower()
                if video_edit_profile == "极速4步":
                    accelerated = True
                    采样器, 调度器, 降噪强度 = "euler", "simple", 1.0
                elif video_edit_profile == "均衡8步":
                    accelerated = False
                    采样器, 调度器, 降噪强度 = "res_multistep", "simple", 1.0
            elif generation_mode == "视频编辑":
                # 速度档位不再强制改模型，FL2VA / Ref2VA 均可手动选择。
                selected_model = 模型
                selected_lower = str(selected_model).replace("\\", "/").lower()
                if video_edit_tool in {"去除字幕", "动作迁移", "角色替换"}:
                    # Dedicated edits must use Ref2VA. FL2VA accepts the graph
                    # but collapses independent picture/video reference roles.
                    if video_edit_tool in {"动作迁移", "角色替换"}:
                        # Exact recipe from the verified action-transfer workflow.
                        selected_model = next(
                            (x for x in ref_candidates if x.replace("\\", "/").lower().endswith("minimax_h3_ref2va_pruned_int8_convrot.safetensors")),
                            next((x for x in ref_candidates if "pruned_int8_convrot" in x.lower()), 模型),
                        )
                        采样步数 = 8
                    elif video_edit_profile == "质量20步":
                        selected_model = next(
                            (x for x in ref_candidates if "bf16" in x.lower()),
                            next((x for x in ref_candidates if "int8_convrot" in x.lower() and "pruned" not in x.lower()), 模型),
                        )
                        采样步数 = 20
                    else:
                        selected_model = next(
                            (x for x in ref_candidates if "int8_convrot" in x.lower() and "pruned" not in x.lower()),
                            next((x for x in ref_candidates if "int8_convrot" in x.lower()), 模型),
                        )
                        采样步数 = 8
                    selected_lower = str(selected_model).replace("\\", "/").lower()
                    accelerated = False
                    采样器, 调度器, 降噪强度 = "res_multistep", "simple", 1.0
                    loras = []
                    if video_edit_tool in {"动作迁移", "角色替换"}:
                        try:
                            installed_loras = folder_paths.get_filename_list("loras")
                        except Exception:
                            installed_loras = []
                        action_lora = next(
                            (x for x in installed_loras if x.replace("\\", "/").lower().endswith("minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors")),
                            _pick_minimax_h3_balanced_lora(installed_loras),
                        )
                        if action_lora:
                            loras = [{"name": action_lora, "strength": 1.0}]
                elif identity_replace:
                    # Four-step FL2VA distillation is too weak for stable
                    # reference-identity transfer. Use the installed Ref2VA
                    # family and keep at least ten native denoising steps.
                    if video_edit_profile == "质量20步":
                        selected_model = next(
                            (x for x in ref_candidates if "bf16" in x.lower()),
                            next((x for x in ref_candidates if "int8_convrot" in x.lower() and "pruned" not in x.lower()), 模型),
                        )
                        采样步数 = 20
                    else:
                        selected_model = next(
                            (x for x in ref_candidates if "pruned_int8_convrot" in x.lower()),
                            next((x for x in ref_candidates if "int8_convrot" in x.lower()), 模型),
                        )
                        采样步数 = 10
                    selected_lower = str(selected_model).replace("\\", "/").lower()
                    accelerated = False
                    采样器, 调度器, 降噪强度 = "res_multistep", "simple", 1.0
                    loras = []
                # FL2VA 极速档可自动使用 Turbo LoRA；Ref2VA 绝不混用该 LoRA。
                if video_edit_fast and "minimax_h3_fl2va" in selected_lower:
                    try:
                        installed_loras = folder_paths.get_filename_list("loras")
                    except Exception:
                        installed_loras = []
                    turbo_lora = _pick_minimax_h3_turbo_lora(installed_loras)
                    if turbo_lora:
                        loras = [{"name": turbo_lora, "strength": 0.75}]
                elif video_edit_profile == "均衡8步" and "minimax_h3_ref2va" in selected_lower and "pruned" not in selected_lower:
                    try:
                        installed_loras = folder_paths.get_filename_list("loras")
                    except Exception:
                        installed_loras = []
                    balanced_lora = _pick_minimax_h3_balanced_lora(installed_loras)
                    if balanced_lora:
                        # 均衡档使用配套的 8-step LoRA。
                        loras = [{"name": balanced_lora, "strength": 0.75}]
            elif not selected_is_ref:
                selected_model = next(
                    (x for x in ref_candidates if "int8_convrot" in x.lower() and "pruned" not in x.lower()),
                    next((x for x in ref_candidates if "int8_convrot" in x.lower()), ref_candidates[0] if ref_candidates else 模型),
                )
        # 加速档统一使用用户指定的 MiniMax H3 FL2V Turbo v1.0 768p BF16 LoRA。
        try:
            import folder_paths
            installed_loras = folder_paths.get_filename_list("loras")
        except Exception:
            installed_loras = []
        dedicated_reference_edit = (
            generation_mode == "视频编辑"
            and video_edit_tool in {"动作迁移", "角色替换"}
        )
        # Dedicated action/character edits selected their verified 8-step LoRA
        # above. Do not erase it here. The old generic reset accidentally removed
        # the role-replacement LoRA, while action transfer happened to add one back.
        if not custom_model_config and not dedicated_reference_edit:
            loras = []
        if (
            not custom_model_config
            and not dedicated_reference_edit
            and video_edit_profile in {"极速4步", "均衡8步"}
            and not identity_replace
        ):
            turbo = (_pick_minimax_h3_balanced_lora(installed_loras)
                     if video_edit_profile == "均衡8步" else _pick_minimax_h3_turbo_lora(installed_loras))
            if turbo:
                loras = [{"name": turbo, "strength": 0.75}]
        # 均衡8步只叠加配套8-step LoRA，不混用4-step蒸馏LoRA。
        print(
            f"[Liao-H3] {generation_mode}/{video_edit_tool}: "
            f"model={selected_model}, steps={int(采样步数)}, "
            f"loras={[(x.get('name'), x.get('strength')) for x in loras]}"
        )
        # 起始释放节点同时作为四个Loader名称的依赖屏障，保证它必定先于任何大模型加载执行。
        start = g.node(
            "WenWuH3ReleaseAtStart", unet_name=selected_model, clip_name=文本编码器,
            video_vae_name=视频VAE, audio_vae_name=音频VAE,
        )
        model = g.node("UNETLoader", unet_name=start.out(0), weight_dtype=模型权重精度)
        if accelerated:
            model = g.node("MiniMaxH3MemoryEfficientSageAttentionPatch", model=model.out(0))
        for lora in loras:
            model = g.node(
                "LoraLoaderModelOnly", model=model.out(0),
                lora_name=lora["name"], strength_model=float(lora["strength"]),
            )
        if accelerated:
            model = g.node("MiniMaxH3SigmaShift", model=model.out(0), shift_video=shift_video, shift_audio=shift_audio)
        elif SageAttention != "disabled":
            # 必须同时送入Guider和Scheduler；这正是用户实际原工作流的连接方式。
            model = g.node("PathchSageAttentionKJ", model=model.out(0), sage_attention=SageAttention, allow_compile=bool(允许编译))
        if reference_edit_mode and not accelerated:
            model = g.node("MiniMaxH3MemoryEfficientSageAttentionPatch", model=model.out(0))
        clip = g.node("CLIPLoader", clip_name=start.out(1), type=文本编码器类型, device=文本编码器设备)
        video_vae = g.node("VAELoader", vae_name=start.out(2))
        audio_vae = g.node("VAELoader", vae_name=start.out(3))
        if generation_mode == "单人数字人" and continuous_segments > 1:
            # 音频决定分段数量；上一段解码尾帧作为下一段额外 Picture 参考，形成严格串行依赖。
            portrait = g.node("LoadImage", image=image_names[0])
            full_audio = g.node("LoadAudio", audio=audio_names[0])
            segment_seconds = continuous_duration / continuous_segments
            previous_tail = None
            output_images = None
            for segment_index in range(continuous_segments):
                begin = segment_index * segment_seconds
                end = continuous_duration if segment_index == continuous_segments - 1 else (segment_index + 1) * segment_seconds
                cropped_audio = g.node("WenWuH3AudioCrop", audio=full_audio.out(0), 开始秒=begin, 结束秒=end)
                segment_prompt = build_native_prompt(提示词, 1, [], 1)
                if previous_tail is not None:
                    segment_prompt = (
                        "连续镜头约束：<Picture 2> 是上一段的最后一帧，本段必须从该时刻自然继续；"
                        "人物身份、服装、发型、背景方位、镜头方向和动作速度保持连续，不得重新开场或跳变。\n"
                        + segment_prompt
                    )
                segment_inputs = {
                    "clip": clip.out(0), "vae": video_vae.out(0), "audio_vae": audio_vae.out(0),
                    "prompt": segment_prompt, "width": width, "height": height,
                    "length": duration_to_frames(end - begin), "ref_image_size": 参考图尺寸,
                    "ref_images.ref_image_0": portrait.out(0),
                    "ref_audios.ref_audio_0": cropped_audio.out(0),
                }
                if previous_tail is not None:
                    segment_inputs["ref_images.ref_image_1"] = previous_tail.out(0)
                segment_condition = g.node("MiniMaxH3ReferenceToVideo", **segment_inputs)
                segment_driven = g.node(
                    "WenWuH3AudioDrive", av_latent=segment_condition.out(1),
                    source_audio=cropped_audio.out(0), audio_vae=audio_vae.out(0),
                )
                segment_noise = g.node("RandomNoise", noise_seed=(int(随机种子) + segment_index) & 0xffffffffffffffff)
                segment_guider = g.node("BasicGuider", model=model.out(0), conditioning=segment_condition.out(0))
                segment_sampler = g.node("KSamplerSelect", sampler_name=采样器)
                segment_sigmas = g.node(
                    "BasicScheduler", model=model.out(0), scheduler=调度器,
                    steps=int(采样步数), denoise=float(降噪强度),
                )
                segment_sampled = g.node(
                    "SamplerCustomAdvanced", noise=segment_noise.out(0), guider=segment_guider.out(0),
                    sampler=segment_sampler.out(0), sigmas=segment_sigmas.out(0), latent_image=segment_driven.out(0),
                )
                segment_released = g.node("WenWuH3ReleaseBeforeDecode", samples=segment_sampled.out(0))
                decoded_frames = g.node("VAEDecode", samples=segment_released.out(0), vae=video_vae.out(0))
                segment_frames = g.node("WenWuH3TrimFramesToAudio", images=decoded_frames.out(0), audio=cropped_audio.out(0))
                output_images = segment_frames if output_images is None else g.node(
                    "ImageBatch", image1=output_images.out(0), image2=segment_frames.out(0)
                )
                previous_tail = g.node("WenWuH3LastFrame", images=segment_frames.out(0))
            return {"result": (output_images.out(0), full_audio.out(0)), "expand": g.finalize()}
        if generation_mode == "MV数字人":
            # Music fixes the total runtime. Each ordered picture owns one
            # equal timeline section (validated to <=15 s), then all generated
            # frame batches are concatenated and paired with the untouched
            # complete source track.
            full_audio = g.node("LoadAudio", audio=audio_names[0])
            segment_seconds = continuous_duration / continuous_segments
            output_images = None
            for segment_index, image_name in enumerate(x for x in image_names if x):
                begin = segment_index * segment_seconds
                end = continuous_duration if segment_index == continuous_segments - 1 else (segment_index + 1) * segment_seconds
                cropped_audio = g.node("WenWuH3AudioCrop", audio=full_audio.out(0), 开始秒=begin, 结束秒=end)
                picture = g.node("LoadImage", image=image_name)
                segment_prompt = (
                    f"MV第{segment_index + 1}/{continuous_segments}段。<Picture 1>定义本段人物、主体、服装、场景与视觉风格；"
                    "根据<Audio 1>的节奏、情绪和音乐变化产生自然表演与镜头运动，不生成字幕、水印或歌词文字。\n"
                    + build_native_prompt(提示词, 1, [], 1)
                )
                segment_condition = g.node(
                    "MiniMaxH3ReferenceToVideo", clip=clip.out(0), vae=video_vae.out(0), audio_vae=audio_vae.out(0),
                    prompt=segment_prompt, width=width, height=height, length=duration_to_frames(end - begin),
                    ref_image_size=参考图尺寸, **{
                        "ref_images.ref_image_0": picture.out(0),
                        "ref_audios.ref_audio_0": cropped_audio.out(0),
                    },
                )
                segment_driven = g.node(
                    "WenWuH3AudioDrive", av_latent=segment_condition.out(1),
                    source_audio=cropped_audio.out(0), audio_vae=audio_vae.out(0),
                )
                segment_noise = g.node("RandomNoise", noise_seed=(int(随机种子) + segment_index) & 0xffffffffffffffff)
                segment_guider = g.node("BasicGuider", model=model.out(0), conditioning=segment_condition.out(0))
                segment_sampler = g.node("KSamplerSelect", sampler_name=采样器)
                segment_sigmas = g.node(
                    "BasicScheduler", model=model.out(0), scheduler=调度器,
                    steps=int(采样步数), denoise=float(降噪强度),
                )
                segment_sampled = g.node(
                    "SamplerCustomAdvanced", noise=segment_noise.out(0), guider=segment_guider.out(0),
                    sampler=segment_sampler.out(0), sigmas=segment_sigmas.out(0), latent_image=segment_driven.out(0),
                )
                segment_released = g.node("WenWuH3ReleaseBeforeDecode", samples=segment_sampled.out(0))
                decoded_frames = g.node("VAEDecode", samples=segment_released.out(0), vae=video_vae.out(0))
                segment_frames = g.node("WenWuH3TrimFramesToAudio", images=decoded_frames.out(0), audio=cropped_audio.out(0))
                output_images = segment_frames if output_images is None else g.node(
                    "ImageBatch", image1=output_images.out(0), image2=segment_frames.out(0)
                )
            return {"result": (output_images.out(0), full_audio.out(0)), "expand": g.finalize()}
        if fl_mode:
            # 1:1复刻 F:/video_minimax_h3_i2v (5).json 的官方原生条件节点。
            conditioning_release = g.node("WenWuH3ReleaseBeforeConditioning", clip=clip.out(0), vae=video_vae.out(0))
            condition_inputs = {
                "clip": conditioning_release.out(0), "vae": conditioning_release.out(1), "prompt": 提示词,
                "width": condition_width, "height": condition_height, "length": length,
            }
            if image_count:
                first = g.node("LoadImage", image=image_names[0])
                condition_inputs["first_frame"] = first.out(0)
            if fl_mode == "首尾帧":
                last = g.node("LoadImage", image=image_names[1])
                condition_inputs["last_frame"] = last.out(0)
            prepared = g.node("MiniMaxH3ImageToVideo", **condition_inputs)
        else:
            # 1:1复刻Ref2VA：素材保持独立原生子节点缓存边界。
            condition_inputs = {
                "clip": clip.out(0), "vae": video_vae.out(0), "audio_vae": audio_vae.out(0),
                "prompt": build_native_prompt(提示词, image_count, [True for x in video_names if x], sum(bool(x) for x in audio_names)), "width": condition_width, "height": condition_height, "length": length,
                "ref_image_size": 参考图尺寸,
            }
            for i, filename in enumerate((x for x in image_names if x)):
                loaded = g.node("LoadImage", image=filename)
                condition_inputs[f"ref_images.ref_image_{i}"] = loaded.out(0)
            for i, filename in enumerate((x for x in video_names if x)):
                loaded = g.node("LoadVideo", file=filename)
                components = g.node("GetVideoComponents", video=loaded.out(0))
                condition_inputs[f"ref_videos.ref_video_{i}"] = components.out(0)
                if generation_mode == "视频编辑" and video_edit_tool == "动作迁移" and i == 0:
                    # The verified action-transfer workflow routes its soundtrack
                    # as Audio 1. Character replacement keeps the source video and
                    # its audio coupled, matching the previously successful path.
                    condition_inputs["ref_audios.ref_audio_0"] = components.out(1)
                else:
                    condition_inputs[f"ref_video_audios.ref_video_audio_{i}"] = components.out(1)
            loaded_audios = []
            audio_offset = 1 if generation_mode == "视频编辑" and video_edit_tool == "动作迁移" and any(video_names) else 0
            for i, filename in enumerate((x for x in audio_names if x)):
                loaded = g.node("LoadAudio", audio=filename)
                loaded_audios.append(loaded)
                condition_inputs[f"ref_audios.ref_audio_{i + audio_offset}"] = loaded.out(0)
            source_audio = None
            if generation_mode in {"单人数字人", "双人数字人"}:
                source_audio = loaded_audios[0]
                if generation_mode == "双人数字人":
                    silence = g.node("EmptyAudio", duration=1.0, sample_rate=44100, channels=2)
                    source_audio = g.node("AudioConcat", audio1=source_audio.out(0), audio2=silence.out(0), direction="after")
                    source_audio = g.node("AudioConcat", audio1=source_audio.out(0), audio2=loaded_audios[1].out(0), direction="after")
                audio_length = g.node("WenWuH3AudioLength", audio=source_audio.out(0))
                condition_inputs["length"] = audio_length.out(0)
            prepared = g.node("MiniMaxH3ReferenceToVideo", **condition_inputs)
            if source_audio is not None:
                driven = g.node("WenWuH3AudioDrive", av_latent=prepared.out(1), source_audio=source_audio.out(0), audio_vae=audio_vae.out(0))
            else:
                driven = None
        noise = g.node("RandomNoise", noise_seed=int(随机种子))
        guider = g.node("BasicGuider", model=model.out(0), conditioning=prepared.out(0))
        sampler = g.node("KSamplerSelect", sampler_name=采样器)
        sigmas = g.node("BasicScheduler", model=model.out(0), scheduler=调度器, steps=int(采样步数), denoise=float(降噪强度))
        initial_latent = driven.out(0) if generation_mode in {"单人数字人", "双人数字人"} else prepared.out(1)
        if not latent_enhance:
            # 使用和用户原始工作流完全相同的原生采样节点，排除自定义采样语义/显存差异。
            sampled = g.node(
                "SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0),
                sampler=sampler.out(0), sigmas=sigmas.out(0), latent_image=initial_latent,
            )
            sampled_for_decode = sampled
        elif dual_model_refine:
            try:
                import folder_paths
                installed_models = list(folder_paths.get_filename_list("diffusion_models"))
            except Exception:
                installed_models = []
            family = "ref2va" if "ref2va" in str(selected_model).lower() else "fl2va"
            second_candidates = [
                name for name in installed_models
                if "minimax_h3" in name.replace("\\", "/").lower()
                and family in name.replace("\\", "/").lower()
                and ("w4a8" in name.lower() or "mixed" in name.lower())
            ]
            second_model_name = next(
                (name for name in second_candidates if "pruned_w4a8_mixed" in name.lower()),
                next((name for name in second_candidates if "w4a8_mixed" in name.lower()), None),
            )
            if not second_model_name:
                raise RuntimeError(
                    f"双模型重绘缺少 MiniMax H3 {family.upper()} W4A8 Mixed 二采模型。"
                    "请先放入 diffusion_models；潜空间二采不受影响。"
                )

            first_input = g.node(
                "WenWuH3PhaseMarker", samples=initial_latent,
                phase="双模型首采生成", progress=6, span=38,
            )
            # Match the compared workflow instead of inheriting the currently
            # selected profile: Euler + beta, ten full-denoise steps.
            dual_first_sampler = g.node("KSamplerSelect", sampler_name="euler")
            dual_first_sigmas = g.node(
                "BasicScheduler", model=model.out(0), scheduler="beta",
                steps=10, denoise=1.0,
            )
            first_sample = g.node(
                "SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0),
                sampler=dual_first_sampler.out(0), sigmas=dual_first_sigmas.out(0), latent_image=first_input.out(0),
            )
            barrier = g.node(
                "LiaoH3SecondPassModelBarrier", samples=first_sample.out(0),
                unet_name=second_model_name,
            )
            first_separated = g.node("LTXVSeparateAVLatent", av_latent=barrier.out(0))
            first_pixels = g.node("VAEDecode", samples=first_separated.out(0), vae=video_vae.out(0))
            # Built-in Lanczos keeps this portable. If KJ's RTX VSR is present,
            # users may still compare it externally without making it a hard dependency.
            upscaled_pixels = g.node(
                "ImageScale", image=first_pixels.out(0), upscale_method="lanczos",
                width=int(width), height=int(height), crop="center",
            )
            video_reencoded = g.node("VAEEncode", pixels=upscaled_pixels.out(0), vae=video_vae.out(0))
            first_audio = g.node("VAEDecodeAudio", samples=first_separated.out(1), vae=audio_vae.out(0))
            audio_reencoded = g.node("VAEEncodeAudio", audio=first_audio.out(0), vae=audio_vae.out(0))
            second_latent = g.node(
                "LTXVConcatAVLatent", video_latent=video_reencoded.out(0),
                audio_latent=audio_reencoded.out(0),
            )
            second_model = g.node("UNETLoader", unet_name=barrier.out(1), weight_dtype=模型权重精度)
            if SageAttention != "disabled":
                second_model = g.node(
                    "PathchSageAttentionKJ", model=second_model.out(0),
                    sage_attention=SageAttention, allow_compile=bool(允许编译),
                )
            second_noise = g.node("RandomNoise", noise_seed=int(随机种子))
            second_guider = g.node("BasicGuider", model=second_model.out(0), conditioning=prepared.out(0))
            second_sampler = g.node("KSamplerSelect", sampler_name="res_multistep")
            second_sigmas = g.node(
                "BasicScheduler", model=second_model.out(0), scheduler="beta",
                steps=3, denoise=0.2,
            )
            second_input = g.node(
                "WenWuH3PhaseMarker", samples=second_latent.out(0),
                phase="双模型低降噪重绘", progress=55, span=38,
            )
            sampled = g.node(
                "SamplerCustomAdvanced", noise=second_noise.out(0), guider=second_guider.out(0),
                sampler=second_sampler.out(0), sigmas=second_sigmas.out(0), latent_image=second_input.out(0),
            )
            sampled_for_decode = g.node(
                "WenWuH3PhaseMarker", samples=sampled.out(0),
                phase="VAE 解码与音视频输出", progress=94, span=0,
            )
        else:
            try:
                import folder_paths
                import nodes as comfy_nodes
                node_mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})
                embedded_upscaler = "LiaoH3EmbeddedLatentUpscaler3D" in node_mappings
                learned_upscaler = embedded_upscaler or "MinimaxH3LatentUpscalerNode3D" in node_mappings
                combined_upscaler = "MiniMaxH3LatentUpscaleCombined" in node_mappings
                if not learned_upscaler and not combined_upscaler:
                    raise RuntimeError(
                        "Liao2049 内置 MiniMax H3 潜空间放大器未注册。"
                        "请重新复制完整的 comfyui-liao2049 文件夹并重启 ComfyUI。"
                    )
                upscale_models = (
                    list(folder_paths.get_filename_list("latent_upscale_models"))
                    if learned_upscaler else []
                )
            except RuntimeError:
                raise
            except Exception as error:
                raise RuntimeError(
                    "Liao2049 内置潜空间增强组件初始化失败。请确认插件文件完整，"
                    "并把模型放入 ComfyUI/models/latent_upscale_models 后重启。"
                ) from error
            upscale_model = next(
                (name for name in upscale_models if "3d_fp16" in name.lower()),
                next((name for name in upscale_models if "3d_bf16" in name.lower()), None),
            ) if learned_upscaler else None
            if learned_upscaler and not upscale_model:
                raise RuntimeError(
                    "未找到 H3 3D 潜空间放大模型。请把 "
                    "minimax_h3_latent_upscaler_3d_fp16.safetensors 或 BF16 版本放入 "
                    "ComfyUI/models/latent_upscale_models。"
                )

            # Mirrors the verified local workflow: refine the low-noise sigma
            # tail, split sampling in half, upscale only the video latent, then
            # finish the remaining sigmas at the selected target resolution.
            refined_sigmas = g.node(
                "WenWuH3SigmaTailRefiner", sigmas=sigmas.out(0), extra_steps=1,
                start_at_sigma=0.7,
            )
            split_step = max(1, int(采样步数) // 2)
            split_sigmas = g.node("SplitSigmas", sigmas=refined_sigmas.out(0), step=split_step)
            first_input = g.node(
                "WenWuH3PhaseMarker", samples=initial_latent,
                phase="首采生成", progress=8, span=40,
            )
            first_sample = g.node(
                "SamplerCustomAdvanced", noise=noise.out(0), guider=guider.out(0),
                sampler=sampler.out(0), sigmas=split_sigmas.out(0), latent_image=first_input.out(0),
            )
            upscale_input = g.node(
                "WenWuH3PhaseMarker", samples=first_sample.out(1),
                phase="潜空间 2 倍放大", progress=50, span=0,
            )
            if learned_upscaler:
                separated = g.node("LTXVSeparateAVLatent", av_latent=upscale_input.out(0))
                if embedded_upscaler:
                    scale_width = float(width) / float(condition_width)
                    scale_height = float(height) / float(condition_height)
                    upscaled_video = g.node(
                        "LiaoH3EmbeddedLatentUpscaler3D", latent=separated.out(0),
                        model_name=upscale_model,
                        scale=math.sqrt(scale_width * scale_height),
                        scale_width=scale_width, scale_height=scale_height,
                        device="cuda", precision="fp16",
                    )
                else:
                    upscaled_video = g.node(
                        "MinimaxH3LatentUpscalerNode3D", latent=separated.out(0),
                        model_name=upscale_model, scale=4.0 / 3.0,
                        device="cuda", precision="fp16",
                    )
                joined = g.node(
                    "LTXVConcatAVLatent", video_latent=upscaled_video.out(0), audio_latent=separated.out(1)
                )
                second_noise = noise
                second_guider = guider
                second_latent = joined
            else:
                # Compatibility with ComfyUI-MiniMaxH3_LatentUpscaler.  This
                # implementation also rescales Ref2VA conditioning metadata and
                # prepares CONST re-noise internally, which is essential for the
                # second pass to actually affect the enlarged canvas.
                combined = g.node(
                    "MiniMaxH3LatentUpscaleCombined", samples=upscale_input.out(0),
                    scale_by=2.0, method="bicubic", model=model.out(0),
                    noise=noise.out(0), sigmas=split_sigmas.out(1), audio_denoise=0.35,
                    positive=prepared.out(0),
                )
                second_noise = g.node("DisableNoise")
                second_guider = g.node("BasicGuider", model=model.out(0), conditioning=combined.out(1))
                second_latent = combined
            second_input = g.node(
                "WenWuH3PhaseMarker", samples=second_latent.out(0),
                phase="放大后二采精修", progress=58, span=34,
            )
            sampled = g.node(
                "SamplerCustomAdvanced", noise=second_noise.out(0), guider=second_guider.out(0),
                sampler=sampler.out(0), sigmas=split_sigmas.out(1), latent_image=second_input.out(0),
            )
            sampled_for_decode = g.node(
                "WenWuH3PhaseMarker", samples=sampled.out(0),
                phase="VAE 解码与音视频输出", progress=94, span=0,
            )
        released = g.node("WenWuH3ReleaseBeforeDecode", samples=sampled_for_decode.out(0))
        image_decode = g.node("VAEDecode", samples=released.out(0), vae=video_vae.out(0))
        if generation_mode in {"单人数字人", "双人数字人"}:
            output_audio = driven.out(1)
        else:
            audio_decode = g.node("VAEDecodeAudio", samples=released.out(0), vae=audio_vae.out(0))
            output_audio = audio_decode.out(0)
        return {"result": (image_decode.out(0), output_audio), "expand": g.finalize()}


NODE_CLASS_MAPPINGS = {
    "WenWuH3AudioDrive": WenWuH3AudioDrive,
    "WenWuH3AudioLength": WenWuH3AudioLength,
    "WenWuH3AudioCrop": WenWuH3AudioCrop,
    "WenWuH3LastFrame": WenWuH3LastFrame,
    "WenWuH3TrimFramesToAudio": WenWuH3TrimFramesToAudio,
    "WenWuH3ModelLoraConfig": WenWuH3ModelLoraConfig,
    "WenWuMiniMaxH3Unified": WenWuMiniMaxH3Unified,
    "WenWuH3ReleaseAtStart": WenWuH3ReleaseAtStart,
    "WenWuH3ReleaseBeforeConditioning": WenWuH3ReleaseBeforeConditioning,
    "WenWuH3ReleaseBeforeDecode": WenWuH3ReleaseBeforeDecode,
    "WenWuH3PhaseMarker": WenWuH3PhaseMarker,
    "LiaoH3EmbeddedLatentUpscaler3D": LiaoH3EmbeddedLatentUpscaler3D,
    "WenWuH3SigmaTailRefiner": WenWuH3SigmaTailRefiner,
    "LiaoH3SecondPassModelBarrier": LiaoH3SecondPassModelBarrier,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WenWuH3AudioDrive": "Liao2049 H3 内部音频锁定",
    "WenWuH3AudioLength": "Liao2049 H3 内部音频时长",
    "WenWuH3AudioCrop": "Liao2049 H3 内部音频分段",
    "WenWuH3LastFrame": "Liao2049 H3 内部尾帧",
    "WenWuH3TrimFramesToAudio": "Liao2049 H3 内部音画对齐",
    "WenWuH3ModelLoraConfig": "Liao2049 H3 内部模型配置",
    "WenWuMiniMaxH3Unified": "Liao-H3 智能创作台　QQ群 38251314　更多工具和教程搜索B站：liao_2049",
    "WenWuH3ReleaseAtStart": "Liao2049 H3 内部显存释放",
    "WenWuH3ReleaseBeforeConditioning": "Liao2049 H3 内部条件前释放",
    "WenWuH3ReleaseBeforeDecode": "Liao2049 H3 内部解码前释放",
    "WenWuH3SigmaTailRefiner": "Liao2049 H3 内部低Sigma精修",
    "LiaoH3SecondPassModelBarrier": "Liao2049 H3 双模型切换屏障",
}


