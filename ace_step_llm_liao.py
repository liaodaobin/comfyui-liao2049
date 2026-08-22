import json
import gc
import os
import re
import sys
import time
import importlib.util

try:
    import server
    from aiohttp import web
except Exception:
    server = None
    web = None


LANGUAGES = [
    "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa",
    "fi", "fr", "he", "hi", "hr", "ht", "hu", "id", "is", "it", "ja", "ko",
    "la", "lt", "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa",
    "sk", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi",
    "yue", "zh", "unknown",
]

COMMON_LANGUAGES = ["zh", "en", "ja", "ko", "yue", "fr", "de", "es", "ru", "unknown"]


class AnyType(str):
    def __ne__(self, value):
        return False


ANY_TYPE = AnyType("*")


KEYSCALES = [
    f"{root} {quality}"
    for quality in ["major", "minor"]
    for root in ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
]


SYSTEM_PROMPT = """你是 ACE Step Audio 1.5 的中文音乐制作总监和作词人。
用户会给一段自然语言需求，你必须把它改写成可用于 TextEncodeAceStepAudio1.5 的结构化音乐方案。

只输出一个 JSON 对象，不要输出 Markdown、解释、代码块或思考过程。JSON 字段必须是：
{
  "tags": "多行中文音乐属性描述",
  "lyrics": "带 [第一节] [副歌] 等段落标记的歌词",
  "bpm": 95,
  "duration": 120,
  "timesignature": "4",
  "language": "zh",
  "keyscale": "E minor"
}

要求：
1. tags 至少包含：演唱/人声、曲风定位、曲风、节奏、乐器、编曲特点；必要时写段落推进。
2. lyrics 必须适合演唱，按用户语言创作；中文歌词每行尽量 4-12 个汉字，避免过长散文句。
3. bpm 根据情绪和曲风判断：摇篮/民谣 60-85，流行抒情 75-100，国风/叙事 80-110，摇滚/电子/燃向 110-150，舞曲 120-140。
4. timesignature 只能是 "2"、"3"、"4"、"6"。默认流行/摇滚/电子用 "4"，圆舞/摇篮感用 "3" 或 "6"。
5. language 必须使用 ACE 支持代码，例如中文 "zh"，粤语 "yue"，英文 "en"，日文 "ja"，韩文 "ko"。
6. keyscale 必须从常见调式中选择，例如 "C major"、"D minor"、"E minor"、"F major"、"G major"、"A minor"、"Bb major"。悲伤、悬疑、武侠、厚重多用 minor；明亮、童趣、庆典多用 major。
7. duration 用用户要求；未给时 120 秒。
8. 不要把 JSON 值写成数组或嵌套对象。
9. 如果用户要求轻音乐、纯音乐、配乐、背景音乐、无人声、没人声、没有人声、无歌词、不要歌词、instrumental、no vocal 或 no vocals：tags 只优化曲风、节奏、乐器、编曲和情绪画面；lyrics 必须严格写成 "没有人声"，不要创作歌词。
"""


STRICT_JSON_PROMPT = """Return ONLY one valid JSON object. Do not output Markdown, bullets, analysis, or explanations.
The first character must be { and the last character must be }.
Use exactly these keys: tags, lyrics, bpm, duration, timesignature, language, keyscale.
All values must be JSON-safe. Escape newlines inside strings as \\n. Do not leave any string unfinished."""

KEYSCALE_GUIDE_PROMPT = """Choose keyscale from the user's style, not from habit:
A major = sunny, pure, bright, energetic pop or countryside.
G major = natural, lively, guitar/folk/pop storytelling.
D major = warm, active, penetrating pop/rock/celtic/happy melody.
E major = bright summer, pastoral, light dance, heroic martial energy only.
F major = soft, stable, warm, children/light music/gentle background.
Bb major = warm, round, swing/R&B/jazz/comfortable pop.
B major = brilliant, powerful, epic climax or strong electronic.
E minor = universal default for guofeng, folk, rock, wuxia tenderness, lonely story.
D minor = desolate, tragic, wuxia, epic, sad atmosphere.
A minor = natural sadness, basic folk/pop melancholy.
B minor = magnificent, fierce, epic wuxia or intense sadness.
Do not overuse E major. If style changes, choose a different matching keyscale."""


class AceStepLLMSongEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "llama_model": ("LLAMACPPMODEL",),
                "description": (
                    "STRING",
                    {
                        "default": "写一首欢快明亮的中文流行歌，主题是周末出游、阳光和好心情，节奏轻快，旋律上口，适合让人开心跟唱。",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "duration": ("FLOAT", {"default": 120.0, "min": 10.0, "max": 2000.0, "step": 0.1}),
                "bpm_override": ("INT", {"default": 0, "min": 0, "max": 300, "tooltip": "0 = let the LLM decide"}),
                "language_hint": (["auto"] + COMMON_LANGUAGES,),
                "keyscale_hint": (["auto"] + KEYSCALES,),
                "generate_audio_codes": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative_hint": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "INT", "FLOAT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "tags", "lyrics", "bpm", "duration", "timesignature", "language", "keyscale", "plan_json")
    FUNCTION = "encode"
    CATEGORY = "AceStep/LLM"

    def encode(
        self,
        clip,
        llama_model,
        description,
        seed,
        duration,
        bpm_override,
        language_hint,
        keyscale_hint,
        generate_audio_codes,
        negative_hint="",
    ):
        plan = self._ask_llama(llama_model, description, seed, duration, bpm_override, language_hint, keyscale_hint, negative_hint)
        plan = self._normalize_plan(plan, description, duration, bpm_override, language_hint, keyscale_hint)

        tokens = clip.tokenize(
            plan["tags"],
            lyrics=plan["lyrics"],
            bpm=plan["bpm"],
            duration=plan["duration"],
            timesignature=int(plan["timesignature"]),
            language=plan["language"],
            keyscale=plan["keyscale"],
            seed=seed,
            generate_audio_codes=generate_audio_codes,
            cfg_scale=2.0,
            temperature=0.85,
            top_p=0.9,
            top_k=0,
            min_p=0,
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        return (
            conditioning,
            plan["tags"],
            plan["lyrics"],
            plan["bpm"],
            plan["duration"],
            plan["timesignature"],
            plan["language"],
            plan["keyscale"],
            plan_json,
        )

    def _ask_llama(self, llama_model, description, seed, duration, bpm_override, language_hint, keyscale_hint, negative_hint):
        storage = _get_llama_storage()
        if storage.llm is None or storage.current_config != llama_model:
            storage.load_model(llama_model)

        hint_lines = [
            f"Local suggested keyscale: {_guess_keyscale(description)}",
            "Use the keyscale guide. Do not repeatedly choose E major unless the style clearly asks for bright summer/pastoral/light dance/heroic martial energy.",
            "Output JSON only. No analysis. No bullet list. Start with { and end with }.",
            "Format tags as separate lines by category: voice, style, rhythm, instruments, arrangement, section progression.",
            "Choose bpm and timesignature from style: lullaby/meditation slow, ballad/guofeng narrative medium-slow, happy children/pop faster, rock/EDM high energy, waltz 3, lullaby/sway 6, march 2.",
            f"用户需求：{description}",
            f"目标时长：{duration} 秒",
        ]
        if bpm_override:
            hint_lines.append(f"用户指定 BPM：{bpm_override}")
        if language_hint != "auto":
            hint_lines.append(f"用户指定语言代码：{language_hint}")
        if keyscale_hint != "auto":
            hint_lines.append(f"用户指定调式：{keyscale_hint}")
        if negative_hint.strip():
            hint_lines.append(f"避免：{negative_hint.strip()}")

        output = storage.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + STRICT_JSON_PROMPT + "\n\n" + KEYSCALE_GUIDE_PROMPT},
                {"role": "user", "content": "\n".join(hint_lines)},
            ],
            seed=seed,
            max_tokens=2200,
            temperature=0.45,
            top_p=0.9,
            repeat_penalty=1.05,
        )
        text = output["choices"][0]["message"]["content"].removeprefix(": ").strip()
        try:
            return _parse_json_object(text)
        except ValueError as exc:
            print(f"[AceStep-LLM-Planner] LLM JSON parse failed, using local fallback. {exc}")
            return {}

    def _normalize_plan(self, plan, description, duration, bpm_override, language_hint, keyscale_hint):
        if not isinstance(plan, dict):
            plan = {}

        tags = _clean_text(plan.get("tags", ""))
        lyrics = _clean_text(plan.get("lyrics", ""))
        instrumental = _is_instrumental_request(description)
        if not tags:
            tags = _fallback_tags(description)
        if instrumental:
            tags = _ensure_instrumental_tags(tags)
            lyrics = "没有人声"
        if not lyrics:
            lyrics = _fallback_lyrics(description)

        tags = _format_tags_lines(tags)

        suggested_bpm = _guess_bpm(description)
        bpm = _safe_int(plan.get("bpm"), suggested_bpm)
        if bpm_override:
            bpm = bpm_override
        elif bpm in {90, 95} and abs(suggested_bpm - bpm) >= 8:
            bpm = suggested_bpm
        bpm = max(10, min(300, bpm))

        out_duration = _safe_float(duration, 0.0)
        if out_duration <= 0:
            out_duration = _safe_float(plan.get("duration"), 120.0)
        out_duration = max(10.0, min(2000.0, out_duration))

        suggested_timesignature = _guess_timesignature(description)
        timesignature = str(plan.get("timesignature", "")).strip()
        if timesignature not in {"2", "3", "4", "6"}:
            timesignature = suggested_timesignature
        elif timesignature == "4" and suggested_timesignature != "4":
            timesignature = suggested_timesignature

        language = str(plan.get("language", "zh")).strip()
        if language_hint != "auto":
            language = language_hint
        if language not in LANGUAGES:
            language = "zh" if _has_cjk(description + lyrics) else "en"

        keyscale = str(plan.get("keyscale", "")).strip()
        suggested_keyscale = _guess_keyscale(description)
        if keyscale_hint != "auto":
            keyscale = keyscale_hint
        elif keyscale not in KEYSCALES:
            keyscale = suggested_keyscale
        elif keyscale in {"E major", "E minor"} and suggested_keyscale != keyscale:
            keyscale = suggested_keyscale

        return {
            "tags": tags,
            "lyrics": lyrics,
            "bpm": bpm,
            "duration": out_duration,
            "timesignature": timesignature,
            "language": language,
            "keyscale": keyscale,
        }


class AceStepLLMSongPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "llama_model": ("LLAMACPPMODEL",),
                "description": (
                    "STRING",
                    {
                        "default": "写一首欢快明亮的中文流行歌，主题是周末出游、阳光和好心情，节奏轻快，旋律上口，适合让人开心跟唱。",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "duration": ("FLOAT", {"default": 120.0, "min": 10.0, "max": 2000.0, "step": 0.1}),
                "language": (["auto"] + COMMON_LANGUAGES, {"default": "auto"}),
                "use_confirmed_prompt": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "approved_plan": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "source_lyrics": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "FLOAT")
    RETURN_NAMES = ("conditioning", "duration")
    FUNCTION = "plan"
    CATEGORY = "AceStep/LLM"
    OUTPUT_NODE = True

    def plan(
        self,
        clip,
        llama_model,
        description,
        duration=120.0,
        language="auto",
        use_confirmed_prompt=False,
        approved_plan="",
        source_lyrics="",
    ):
        helper = AceStepLLMSongEncoder()
        seed = 0
        bpm_override = 0
        language_hint = language
        keyscale_hint = "auto"
        negative_hint = ""
        approved_plan = _clean_approved_plan_text(approved_plan)
        source_lyrics = _clean_text(source_lyrics)
        has_confirmed_lyrics = bool(
            re.search(r"(?im)^\s*\[LYRICS\]\s*$", approved_plan)
        )
        if use_confirmed_prompt and approved_plan and has_confirmed_lyrics:
            plan = _parse_approved_plan_text(approved_plan)
            plan = helper._normalize_plan(plan, description, duration, bpm_override, language_hint, keyscale_hint)
        else:
            plan = helper._ask_llama(llama_model, description, seed, duration, bpm_override, language_hint, keyscale_hint, negative_hint)
            plan = helper._normalize_plan(plan, description, duration, bpm_override, language_hint, keyscale_hint)
            if source_lyrics:
                # Remix mode: the LLM may plan the style and musical parameters,
                # but it must never replace the lyrics transcribed from the source song.
                plan["lyrics"] = source_lyrics
            _cleanup_llama_storage()
        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
        tokens = clip.tokenize(
            plan["tags"],
            lyrics=plan["lyrics"],
            bpm=plan["bpm"],
            duration=plan["duration"],
            timesignature=int(plan["timesignature"]),
            language=plan["language"],
            keyscale=plan["keyscale"],
            seed=seed,
            generate_audio_codes=True,
            cfg_scale=2.0,
            temperature=0.85,
            top_p=0.9,
            top_k=0,
            min_p=0,
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        return {
            "ui": {"text": [_format_preview_text(plan)]},
            "result": (
                conditioning,
                plan["duration"],
            ),
        }


class AceStepSongPlanPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "AceStep/LLM"
    OUTPUT_NODE = True

    def preview(self, plan_json):
        plan = _parse_plan_json(plan_json)
        preview_text = _format_preview_text(plan)
        return {
            "ui": {"text": [preview_text]},
            "result": (),
        }


class AceStepConfirmedSongEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "tags": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "lyrics": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "bpm": ("INT", {"default": 95, "min": 10, "max": 300}),
                "duration": ("FLOAT", {"default": 120.0, "min": 10.0, "max": 2000.0, "step": 0.1}),
                "timesignature": (["2", "3", "4", "6"], {"default": "4"}),
                "language": (COMMON_LANGUAGES, {"default": "zh"}),
                "keyscale": (KEYSCALES, {"default": "E minor"}),
                "generate_audio_codes": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode_confirmed"
    CATEGORY = "AceStep/LLM"

    def encode_confirmed(self, clip, tags, lyrics, seed, bpm, duration, timesignature, language, keyscale, generate_audio_codes):
        tokens = clip.tokenize(
            tags,
            lyrics=lyrics,
            bpm=bpm,
            duration=duration,
            timesignature=int(timesignature),
            language=language,
            keyscale=keyscale,
            seed=seed,
            generate_audio_codes=generate_audio_codes,
            cfg_scale=2.0,
            temperature=0.85,
            top_p=0.9,
            top_k=0,
            min_p=0,
        )
        return (clip.encode_from_tokens_scheduled(tokens),)


class AceStepLlamaMemoryCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (ANY_TYPE,),
            },
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("any",)
    FUNCTION = "cleanup"
    CATEGORY = "AceStep/LLM"

    def cleanup(self, any):
        _cleanup_llama_storage()
        return (any,)


def _get_llama_storage():
    # Prefer the runtime already registered by ComfyUI. This supports both the
    # current ComfyUI-llama-cpp_vlm fork and the legacy ComfyUI-llama-cpp node.
    try:
        import nodes as comfy_nodes
        mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {})
        loader_cls = mappings.get("llama_cpp_model_loader")
        if loader_cls is not None:
            module = sys.modules.get(loader_cls.__module__)
            if module is None:
                module = __import__(loader_cls.__module__, fromlist=["LLAMA_CPP_STORAGE"])
            storage = getattr(module, "LLAMA_CPP_STORAGE", None)
            if storage is not None:
                return storage
    except Exception:
        pass

    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", "") or ""
        normalized = path.replace("\\", "/").lower()
        if "llama-cpp" in normalized and normalized.endswith("/nodes.py") and hasattr(module, "LLAMA_CPP_STORAGE"):
            return module.LLAMA_CPP_STORAGE

    here = os.path.dirname(os.path.abspath(__file__))
    custom_nodes = os.path.dirname(here)
    candidates = [
        os.path.join(custom_nodes, "ComfyUI-llama-cpp_vlm", "nodes.py"),
        os.path.join(custom_nodes, "ComfyUI-llama-cpp", "nodes.py"),
    ]
    llama_nodes = next((path for path in candidates if os.path.exists(path)), None)
    if not llama_nodes:
        raise RuntimeError("未找到 ComfyUI-llama-cpp_vlm/ComfyUI-llama-cpp，请先安装并启用其中一个。")

    spec = importlib.util.spec_from_file_location("comfyui_llama_cpp_nodes_for_ace_planner", llama_nodes)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LLAMA_CPP_STORAGE


def _cleanup_llama_storage():
    storage = _get_llama_storage()
    print("[AceStep-LLM-Liao2049] Unloading llama.cpp model and collecting memory...")
    storage.clean()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"[AceStep-LLM-WENWU] CUDA cache cleanup skipped: {exc}")
    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception as exc:
        print(f"[AceStep-LLM-WENWU] ComfyUI soft cache cleanup skipped: {exc}")


def _parse_json_object(text):
    text = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    candidates = _json_object_candidates(text)
    decoder = json.JSONDecoder(strict=False)
    errors = []
    for candidate in reversed(candidates):
        try:
            value, _ = decoder.raw_decode(candidate)
            if isinstance(value, dict):
                return value
        except Exception as exc:
            errors.append(str(exc))

    for start in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            value, _ = decoder.raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except Exception as exc:
            errors.append(str(exc))

    detail = errors[-1] if errors else "no JSON object found"
    raise ValueError(f"LLM 没有返回有效 JSON，已尝试清理思考内容和提取最后一个 JSON 对象。\n解析错误：{detail}\n原始输出：\n{text}")


def _json_object_candidates(text):
    candidates = []
    stack = []
    start = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if not stack:
                start = index
            stack.append(char)
        elif char == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates


def _clean_text(value):
    return str(value or "").replace("\\n", "\n").strip()


def _clean_approved_plan_text(value):
    text = _clean_text(value)
    return "" if text.lower() in {"true", "false"} else text


def _format_tags_lines(tags):
    text = _clean_text(tags)
    if not text:
        return ""

    labels = [
        "\u6f14\u5531/\u4eba\u58f0",
        "\u66f2\u98ce\u5b9a\u4f4d",
        "\u66f2\u98ce",
        "\u98ce\u683c",
        "\u8282\u594f",
        "\u4e50\u5668",
        "\u7f16\u66f2\u7279\u70b9",
        "\u6bb5\u843d\u63a8\u8fdb",
        "\u60c5\u7eea",
        "\u6c1b\u56f4",
    ]
    for label in labels:
        marker = rf"({re.escape(label)}\s*[:\uff1a])"
        text = re.sub(rf"(?<!^)(?<!\n)[\s,\uff0c;\uff1b\u3002]*{marker}", r"\n\1", text)

    lines = []
    for line in text.splitlines():
        line = _clean_text(line).strip(" \t\u3002")
        if line:
            lines.append(line)
    return "\n".join(lines)


def _safe_int(value, default):
    try:
        return int(float(value))
    except Exception:
        return default


def _safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def _has_cjk(text):
    return re.search(r"[\u3400-\u9fff]", text or "") is not None


def _format_preview_text(plan, plan_json=""):
    return (
        f"[PARAMS] Key={plan.get('keyscale', '')} | BPM={plan.get('bpm', '')} | "
        f"Time={plan.get('timesignature', '')}/4 | Language={plan.get('language', '')} | "
        f"Duration={plan.get('duration', '')}s\n\n"
        f"[TAGS]\n{plan.get('tags', '')}\n\n"
        f"[LYRICS]\n{plan.get('lyrics', '')}"
    )


def _parse_plan_json(plan_json):
    try:
        value = json.loads(plan_json) if str(plan_json or "").strip() else {}
    except Exception:
        value = _parse_json_object(plan_json)
    if not isinstance(value, dict):
        value = {}
    return {
        "tags": _clean_text(value.get("tags", "")),
        "lyrics": _clean_text(value.get("lyrics", "")),
        "bpm": value.get("bpm", ""),
        "duration": value.get("duration", ""),
        "timesignature": value.get("timesignature", ""),
        "language": value.get("language", ""),
        "keyscale": value.get("keyscale", ""),
    }


def _parse_approved_plan_text(text):
    text = _clean_text(text)
    plan = {}

    json_match = re.search(r"\[PLAN JSON\]\s*(\{.*\})\s*$", text, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, dict):
                plan.update(parsed)
        except Exception:
            pass

    tags = _extract_section(text, "TAGS", ["LYRICS", "PLAN JSON"])
    lyrics = _extract_section(text, "LYRICS", ["PLAN JSON"])
    if tags:
        plan["tags"] = tags
    if lyrics:
        plan["lyrics"] = lyrics

    scalar_patterns = {
        "keyscale": r"(?im)\bKey\s*[:=]\s*([^|\n]+)",
        "bpm": r"(?im)\bBPM\s*[:=]\s*(\d+)",
        "timesignature": r"(?im)\bTime(?:\s+Signature)?\s*[:=]\s*(\d+)",
        "language": r"(?im)\bLanguage\s*[:=]\s*([^|\n]+)",
        "duration": r"(?im)\bDuration\s*[:=]\s*([0-9.]+)",
    }
    for key, pattern in scalar_patterns.items():
        match = re.search(pattern, text)
        if match:
            plan[key] = match.group(1).strip()
    return plan


def _extract_section(text, section, end_sections):
    pattern = rf"(?is)\[{re.escape(section)}\]\s*(.*)"
    match = re.search(pattern, text)
    if not match:
        return ""
    value = match.group(1)
    end_positions = []
    for end_section in end_sections:
        end_match = re.search(rf"(?is)\[{re.escape(end_section)}\]", value)
        if end_match:
            end_positions.append(end_match.start())
    if end_positions:
        value = value[: min(end_positions)]
    return value.strip()


def _is_instrumental_request(text):
    lowered = (text or "").lower()
    keywords = [
        "轻音乐", "纯音乐", "配乐", "背景音乐", "无人声", "没人声", "没有人声",
        "无歌词", "不要歌词", "不要人声", "去人声", "伴奏", "instrumental",
        "no vocal", "no vocals", "without vocal", "without vocals", "no lyrics",
    ]
    return any(keyword in lowered for keyword in keywords)


def _ensure_instrumental_tags(tags):
    tags = _clean_text(tags)
    if not tags:
        tags = "曲风定位：轻音乐、纯音乐、无歌词、无人声\n乐器：钢琴、吉他、弦乐、轻打击乐\n编曲特点：旋律清晰，氛围舒展，适合作为背景音乐"
    if not re.search(r"(无人声|没有人声|纯音乐|无歌词|instrumental)", tags, re.IGNORECASE):
        tags = "演唱/人声：没有人声，纯音乐，无歌词\n" + tags
    return tags


def _guess_bpm(text):
    lowered = (text or "").lower()
    if _contains_any(lowered, ["edm", "club", "techno", "house", "dance", "\u821e\u66f2", "\u8fea\u58eb\u79d1", "\u7535\u97f3"]):
        return 128
    if _contains_any(lowered, ["punk", "rock", "\u6447\u6eda", "\u71c3", "\u70ed\u8840", "\u6218\u6597", "\u6fc0\u70c8"]):
        return 132
    if _contains_any(lowered, ["trap", "hip hop", "hip-hop", "rap", "\u8bf4\u5531", "\u9677\u9631"]):
        return 92
    if _contains_any(lowered, ["\u5706\u821e", "\u534e\u5c14\u5179", "waltz"]):
        return 84
    if _contains_any(lowered, ["\u6447\u7bee", "lullaby", "\u7761\u524d", "\u51a5\u60f3", "\u7597\u6108\u7761\u7720"]):
        return 68
    if _contains_any(lowered, ["\u513f\u7ae5", "\u513f\u6b4c", "\u7ae5\u8da3", "\u6b22\u5feb", "\u5f00\u5fc3", "\u6d3b\u6cfc", "\u660e\u5feb", "\u8f7b\u5feb"]):
        return 108
    if _contains_any(lowered, ["\u8f7b\u97f3\u4e50", "\u7eaf\u97f3\u4e50", "\u80cc\u666f\u97f3\u4e50", "ambient", "\u6e29\u6696", "\u8212\u7f13"]):
        return 86
    if _contains_any(lowered, ["\u53f2\u8bd7", "\u5927\u6c14", "\u4ea4\u54cd", "\u9884\u544a\u7247", "cinematic", "epic"]):
        return 96
    if _contains_any(lowered, ["\u56fd\u98ce", "\u53e4\u98ce", "\u6b66\u4fa0", "\u6c5f\u6e56", "\u53d9\u4e8b"]):
        return 82
    if _contains_any(lowered, ["\u60b2\u4f24", "\u54c0\u4f24", "\u5b64\u72ec", "\u6292\u60c5", "\u6c11\u8c23", "ballad"]):
        return 78
    lowered = text.lower()
    if any(x in lowered for x in ["舞曲", "edm", "电子", "club", "techno", "house"]):
        return 128
    if any(x in lowered for x in ["摇滚", "燃", "热血", "战斗", "punk", "rock"]):
        return 132
    if any(x in lowered for x in ["童谣", "儿童", "摇篮", "lullaby"]):
        return 76
    if any(x in lowered for x in ["悲伤", "哀伤", "治愈", "抒情", "民谣", "ballad"]):
        return 82
    return 95


def _guess_timesignature(text):
    lowered = (text or "").lower()
    if _contains_any(lowered, ["\u5706\u821e", "\u534e\u5c14\u5179", "waltz", "3/4"]):
        return "3"
    if _contains_any(lowered, ["6/8", "\u516b\u516d\u62cd", "\u6447\u7bee", "\u6447\u66f3", "\u8239\u6b4c", "\u6d41\u52a8\u611f"]):
        return "6"
    if _contains_any(lowered, ["2/4", "\u56db\u4e8c\u62cd", "\u8fdb\u884c\u66f2", "\u6ce2\u5c14\u5361", "\u8282\u62cd\u5206\u660e"]):
        return "2"
    return "4"


def _contains_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def _guess_keyscale(text):
    if any(x in text for x in ["明亮", "快乐", "可爱", "童趣", "庆典", "希望", "阳光", "热闹"]):
        return "G major"
    if any(x in text for x in ["国风", "武侠", "江湖", "古风"]):
        return "E minor"
    if any(x in text for x in ["悲", "伤", "悬疑", "孤独", "黑暗", "史诗", "厚重"]):
        return "D minor"
    return "E minor"


def _fallback_tags(description):
    bpm = _guess_bpm(description)
    key = _guess_keyscale(description)
    return "\n".join(
        [
            "演唱：情绪充沛的女声，咬字清晰，副歌更有力量",
            f"曲风定位：根据主题生成的叙事流行歌曲，{key}",
            "曲风：中文流行、国风叙事、轻摇滚质感、情绪递进",
            f"节奏：中速，4/4 拍，约 {bpm} BPM，主歌克制，副歌展开",
            "乐器：钢琴、木吉他、弦乐铺底、低鼓、贝斯，副歌加入鼓组和合唱垫",
            "编曲特点：开头留白，主歌讲述，预副歌升温，副歌爆发，结尾渐弱",
        ]
    )


def _fallback_lyrics(description):
    theme = re.sub(r"\s+", "", description)[:16] or "心中的愿望"
    return f"""[第一节]
风从旧路吹来
灯在夜里醒来
我把{theme}
藏进胸口尘埃

[第二节]
人潮慢慢散开
脚步还在等待
一声沉默之后
故事终于展开

[副歌]
我不退让
让风雨穿过肩膀
我还守望
守住心里的光

[结尾]
一生一念
回到最初地方"""


if server is not None and web is not None:
    @server.PromptServer.instance.routes.post("/liao2049/ace_step_llm_planner/preview")
    async def ace_step_llm_planner_preview(request):
        try:
            data = await request.json()
            llama_model = data.get("llama_model")
            if not isinstance(llama_model, dict):
                return web.json_response({"error": "Missing llama_model config."}, status=400)

            description = _clean_text(data.get("description", ""))
            if not description:
                return web.json_response({"error": "Description is empty."}, status=400)

            duration = _safe_float(data.get("duration"), 120.0)
            language = _clean_text(data.get("language", "auto")) or "auto"
            if language not in (["auto"] + LANGUAGES):
                language = "auto"
            seed = _safe_int(data.get("seed"), int(time.time() * 1000) & 0xFFFFFFFF)
            source_lyrics = _clean_text(data.get("source_lyrics", ""))
            helper = AceStepLLMSongEncoder()
            plan = helper._ask_llama(llama_model, description, seed, duration, 0, language, "auto", "")
            plan = helper._normalize_plan(plan, description, duration, 0, language, "auto")
            if source_lyrics:
                # Preview the actual source-song lyrics. Only tags and musical
                # parameters are planned by the LLM.
                plan["lyrics"] = source_lyrics
            _cleanup_llama_storage()
            return web.json_response({
                "text": _format_preview_text(plan),
                "plan": plan,
                "plan_json": json.dumps(plan, ensure_ascii=False, indent=2),
            })
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)


NODE_CLASS_MAPPINGS = {
    "AceStepLLMSongEncoder": AceStepLLMSongEncoder,
    "AceStepLLMSongPlanner": AceStepLLMSongPlanner,
    "AceStepSongPlanPreview": AceStepSongPlanPreview,
    "AceStepConfirmedSongEncoder": AceStepConfirmedSongEncoder,
    "AceStepLlamaMemoryCleanup": AceStepLlamaMemoryCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLLMSongEncoder": "ACE Step 1.5 LLM Song Encoder",
    "AceStepLLMSongPlanner": "ACE Step 1.5 LLM · Liao2049",
    "AceStepSongPlanPreview": "ACE Step Song Plan Preview",
    "AceStepConfirmedSongEncoder": "ACE Step 1.5 Encode Confirmed Song Plan",
    "AceStepLlamaMemoryCleanup": "ACE Step Llama Memory Cleanup",
}
