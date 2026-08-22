import importlib.util
import os
import sys
from pathlib import Path

import folder_paths
from nodes import LoadImage


LLAMA_CPP_ALIAS = "liao2049_krea2_llama_cpp"


def _load_llama_cpp_module():
    try:
        import nodes as comfy_nodes
        loader_cls = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get("llama_cpp_model_loader")
        if loader_cls is not None:
            module = sys.modules.get(loader_cls.__module__)
            if module is None:
                module = __import__(loader_cls.__module__, fromlist=["llama_cpp_instruct_adv"])
            if hasattr(module, "llama_cpp_instruct_adv"):
                return module
    except Exception:
        pass

    custom_nodes = Path(__file__).resolve().parents[1]
    candidates = [custom_nodes / "ComfyUI-llama-cpp_vlm", custom_nodes / "ComfyUI-llama-cpp"]
    llama_cpp_dir = next((path for path in candidates if (path / "nodes.py").exists()), None)
    if llama_cpp_dir is None:
        raise RuntimeError(
            "ComfyUI-llama-cpp_vlm/ComfyUI-llama-cpp was not found. "
            "Install and enable one of them under ComfyUI/custom_nodes first."
        )
    llama_cpp_init = llama_cpp_dir / "__init__.py"
    llama_cpp_nodes = llama_cpp_dir / "nodes.py"

    module_name = f"{LLAMA_CPP_ALIAS}.nodes"
    if module_name in sys.modules:
        return sys.modules[module_name]

    package_spec = importlib.util.spec_from_file_location(
        LLAMA_CPP_ALIAS,
        llama_cpp_init,
        submodule_search_locations=[str(llama_cpp_dir)],
    )
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[LLAMA_CPP_ALIAS] = package

    spec = importlib.util.spec_from_file_location(module_name, llama_cpp_nodes)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_llama_cpp_instruct():
    module = _load_llama_cpp_module()
    return module.llama_cpp_instruct_adv


def _refresh_llama_cache_if_needed(llama_model):
    if not isinstance(llama_model, dict):
        return
    module = _load_llama_cpp_module()
    storage = getattr(module, "LLAMA_CPP_STORAGE", None)
    if storage is None:
        return
    current_config = getattr(storage, "current_config", None)
    current_llm = getattr(storage, "llm", None)
    if current_llm is not None and current_config != llama_model:
        print("[Liao2049 Krea2] llama model config changed, refreshing cached llama-cpp model.")
        storage.clean(all=True)


KREA2_TEXT_SYSTEM = """You are a Krea2 prompt specialist. Convert the user's Chinese or mixed-language image request into one single continuous English prompt for Krea2 text-to-image generation. Output only the final English prompt, with no title, no explanation, no markdown, no bullet points, no parameters, no Chinese, and no extra commentary. The prompt should be detailed, commercially usable, and optimized for Krea2: lock the main subject first, then enrich scene, composition, camera perspective, lighting, material texture, color system, mood, rendering quality, and visual restrictions. Keep the user's core subject and intent unchanged. If the user request is short, intelligently expand it into a rich professional prompt. Avoid text, logos, watermarks, distorted anatomy, extra limbs, blurry details, clutter, overexposure, underexposure, low resolution, pixelation, ugly deformation, and random floating objects."""


KREA2_STYLE_SYSTEM = """You are a Krea2 single-image style transfer prompt specialist. The input image is only a visual style reference, not content to copy. Convert the user's new subject and scene request into one single continuous English prompt for Krea2. Output only the final English prompt, with no title, no explanation, no markdown, no bullet points, no parameters, no Chinese, and no extra commentary. Preserve the user's new subject, new scene, and core creative logic. Extract only transferable visual qualities from the reference image: color palette, saturation, contrast, lighting softness, shadow logic, material texture, lens mood, depth, atmosphere, grain, and emotional tone. Do not copy the reference image's original objects, people, clothing, symbols, text, layout, or story content. If the user's color or style words conflict with the reference image, silently harmonize them toward the reference style. Include concrete visual descriptors, high-definition rendering quality, clean composition, and restrictions against text, logos, watermarks, distorted anatomy, extra limbs, blurry details, clutter, overexposure, underexposure, pixelation, low-resolution artifacts, and readable symbols."""


KREA2_IMAGE_WASH_SYSTEM = """You are a Krea2 faithful image-to-prompt reconstruction specialist. Analyze the attached image and convert it into one single continuous English prompt for Krea2 text-to-image generation. Output only the final English prompt, with no title, no explanation, no markdown, no bullet points, no parameters, no Chinese, and no extra commentary. Treat the image as the exact visual source for subject identity, object count, scene hierarchy, camera distance, and mood, but do not overfit accidental screenshot borders or arbitrary aspect ratio. The prompt should preserve similarity while allowing adaptive framing for the user's final canvas. Reconstruct what is visibly present with high fidelity and enough layered detail: main subject type and count, pose, action, expression, clothing, accessories, object placement, near foreground, close foreground, midground, far background, spatial depth, framing style, crop tightness, camera perspective, visual focus, visual guidance, relative scale, lighting direction, shadow softness, color palette, material texture, surface detail, atmosphere, and photographic or artistic character. The final prompt must read as one continuous Krea2 prompt, but it should follow this internal order: subject anchor, foreground and near-field details, midground details, background and far-field details, composition and visual guidance, lighting atmosphere, material texture, image quality, and restrictions. Preserve the original subject positions, proportions, viewpoint, dominant shapes, background darkness or brightness, and visual hierarchy as closely as text can describe. If a layer has little visible content, describe it as minimal, dark, blurred, shallow-depth, or empty instead of inventing new scenery. If the final image uses a different aspect ratio, adapt by minimally extending or cropping simple background areas while keeping the main subject scale, count, relative placement, and overall visual balance similar. If the image shows one dominant object, say one dominant object; do not generalize it into multiple similar objects. If the background is dark, blurred, or minimal, keep it dark, blurred, or minimal instead of expanding it into a wider busy scene. Do not invent new subjects, props, scenery, styles, dramatic lighting, weather, moods, symbols, or story details that are not clearly visible. ABSOLUTE NO-TEXT RULE: even when the source image visibly contains writing, subtitles, signs, labels, logos, watermarks, signatures, UI overlays, letters, numbers, or garbled pseudo-characters, treat all of them as removable contamination. Never describe or reproduce them. Replace those regions with plausible clean material or background texture. This rule overrides the source image and every user correction. Never mention the attached image, source image, original image, reference, or reconstruction process in the final prompt. The final prompt must be detailed, literal, Krea2-ready English while avoiding text, logos, watermarks, readable characters, symbols, distorted anatomy, extra limbs, overexposure, underexposure, pixelation, and low-resolution artifacts."""


KREA2_WASH_NO_TEXT_SUFFIX = (
    "The finished image is purely visual and contains absolutely no letters, words, numbers, "
    "typography, captions, subtitles, signs, labels, logos, brand marks, watermarks, signatures, "
    "UI elements, stamps, readable characters, or garbled pseudo-text anywhere in the frame; "
    "remove every text-like mark from the source and replace it with clean, natural material or "
    "background texture."
)


def _enforce_wash_no_text(prompt):
    body = str(prompt or "").strip()
    if not body:
        return KREA2_WASH_NO_TEXT_SUFFIX
    return f"{body.rstrip(' .')}. {KREA2_WASH_NO_TEXT_SUFFIX}"


KREA2_STYLE_MODES = ("文生图", "风格参考", "洗图")


DEFAULT_LLAMA_PARAMETERS = {
    "max_tokens": 1024,
    "top_k": 30,
    "top_p": 0.9,
    "min_p": 0.05,
    "typical_p": 1.0,
    "temperature": 0.8,
    "repeat_penalty": 1.0,
    "frequency_penalty": 0.0,
    "present_penalty": 0.0,
    "mirostat_mode": 0,
    "mirostat_eta": 0.1,
    "mirostat_tau": 5.0,
    "state_uid": -1,
}


def _build_text_prompt(user_prompt):
    return "#Krea2 high-end prompt generation\nUser request:\n" + (user_prompt or "").strip()


def _build_style_prompt(user_prompt):
    return (
        "#Krea2 single-image precise style transfer\n"
        "Use the attached image only as a style reference. Generate a Krea2 prompt for this new request:\n"
        + (user_prompt or "").strip()
    )


def _build_image_wash_prompt(user_prompt):
    extra_direction = (user_prompt or "").strip()
    prompt = (
        "#Krea2 similarity-preserving adaptive image wash\n"
        "Use the attached image as the exact visual source. Create one Krea2-ready English prompt that preserves the same visible subject, object count, camera distance, focal hierarchy, and scene mood, while allowing the final canvas ratio to adapt naturally.\n"
        "Begin the final prompt with a similarity anchor sentence in natural English, explicitly stating: main subject count, main subject approximate position, dominant foreground object count and shape, camera distance such as macro close-up or medium shot, background simplicity or complexity, and the overall light-dark relationship.\n"
        "After that, write a detailed single-paragraph English prompt in this order: main subject details; foreground and near-field elements including objects closest to camera, occlusion, droplets, texture, and edge blur; midground elements including the surface or object supporting the subject and its visible structure; far background elements including darkness, blur, color masses, bokeh, or empty negative space; composition and visual guidance such as leading veins, diagonal lines, central weight, subject offset, negative space, and focus path; lighting atmosphere including direction, intensity, highlights, shadows, contrast, and mood; material texture and image quality.\n"
        "Every visible depth layer should be represented. If a layer is mostly absent, describe it as minimal, dark, softly blurred, or shallow-depth rather than adding new objects. Use concrete nouns and visual relationships instead of generic beauty words.\n"
        "Preserve count, subject hierarchy, relative positions, crop tightness, and background character. If the generation aspect ratio differs from the source, adapt by extending or trimming only low-importance background space; do not change the subject count, turn one dominant leaf into many leaves, turn a single subject into a group, turn a dark blurred background into a bright open scene, or turn a tight macro into a general nature photo.\n"
        "ABSOLUTE NO-TEXT RULE: remove every visible letter, word, number, subtitle, sign, label, logo, watermark, signature, UI overlay, stamp, readable character, and garbled pseudo-text from the reconstructed result, replacing those regions with clean natural texture. This rule overrides the source image and any user correction. "
        "Do not add extra objects, people, scenery, dramatic atmosphere, weather, symbolic elements, text, logos, watermarks, captions, UI marks, readable characters, or decorative symbols. If a detail is unclear, describe it neutrally instead of inventing it. Aim for about 750-850 English words so the prompt can fully cover subject, foreground, midground, background, composition, visual guidance, lighting, atmosphere, materials, and image-quality constraints."
    )
    if extra_direction:
        prompt += "\nUser correction to incorporate only if it does not contradict the visible image:\n" + extra_direction
    return prompt


def _normalize_style_mode(value):
    if isinstance(value, bool):
        return "风格参考" if value else "文生图"
    text = str(value or "").strip()
    aliases = {
        "": "文生图",
        "text": "文生图",
        "text-to-image": "文生图",
        "t2i": "文生图",
        "文生图": "文生图",
        "style": "风格参考",
        "style_reference": "风格参考",
        "style reference": "风格参考",
        "风格参考": "风格参考",
        "true": "风格参考",
        "image wash": "洗图",
        "wash": "洗图",
        "rewrite image": "洗图",
        "洗图": "洗图",
    }
    return aliases.get(text.lower(), aliases.get(text, "文生图"))


def _clean_int(value, default, minimum=None, maximum=None):
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _clean_bool(value, default=False):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _clean_float(value, default, minimum=None, maximum=None):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


class WenWuKrea2PromptInstruct:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        files = sorted(files) or [""]

        return {
            "required": {
                "llama_model": ("LLAMACPPMODEL",),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Enter a Krea2 image request, style-transfer subject, or optional wash direction.",
                }),
                "style": (list(KREA2_STYLE_MODES), {"default": "文生图", "label": "模式"}),
                "style_image": (files, {"image_upload": True, "label": "参考/洗图图像"}),
                "max_frames": ("INT", {"default": 24, "min": 2, "max": 1024, "step": 1}),
                "max_size": ("INT", {"default": 768, "min": 128, "max": 16384, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "save_states": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "queue_handler": ("*",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("output", "output_list", "state_uid")
    OUTPUT_IS_LIST = (False, True, False)
    FUNCTION = "process"
    CATEGORY = "Liao2049/Krea2"

    def process(
        self,
        llama_model,
        user_prompt,
        style,
        style_image,
        max_frames,
        max_size,
        seed,
        force_offload,
        save_states,
        unique_id,
        queue_handler=None,
        parameters=None,
        images=None,
        **kwargs,
    ):
        _refresh_llama_cache_if_needed(llama_model)
        instruct_cls = _load_llama_cpp_instruct()
        instruct = instruct_cls()

        style_mode = _normalize_style_mode(style)
        use_image = style_mode in {"风格参考", "洗图"}
        if style_mode == "风格参考":
            system_prompt = KREA2_STYLE_SYSTEM
            custom_prompt = _build_style_prompt(user_prompt)
        elif style_mode == "洗图":
            system_prompt = KREA2_IMAGE_WASH_SYSTEM
            custom_prompt = _build_image_wash_prompt(user_prompt)
        else:
            system_prompt = KREA2_TEXT_SYSTEM
            custom_prompt = _build_text_prompt(user_prompt)
        max_frames = _clean_int(max_frames, 24, 2, 1024)
        max_size = _clean_int(max_size, 768, 128, 16384)
        if style_mode == "洗图":
            max_size = max(max_size, 768)
        seed = _clean_int(seed, 0, 0, 0xffffffffffffffff)
        force_offload = _clean_bool(force_offload, True)
        save_states = _clean_bool(save_states, False)
        embedded_image = LoadImage().load_image(style_image)[0] if use_image and style_image else None
        merged_parameters = dict(DEFAULT_LLAMA_PARAMETERS)
        if isinstance(parameters, dict):
            merged_parameters.update(parameters)
        if style_mode == "洗图":
            merged_parameters["temperature"] = _clean_float(merged_parameters.get("temperature"), 0.8, maximum=0.25)
            merged_parameters["top_p"] = _clean_float(merged_parameters.get("top_p"), 0.9, maximum=0.75)
            merged_parameters["repeat_penalty"] = _clean_float(merged_parameters.get("repeat_penalty"), 1.0, minimum=1.05)
            merged_parameters["max_tokens"] = _clean_int(merged_parameters.get("max_tokens"), 1024, minimum=2048, maximum=32768)

        result = instruct.process(
            llama_model=llama_model,
            preset_prompt="Normal - Describe",
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            inference_mode="one by one",
            max_frames=max_frames,
            max_size=max_size,
            seed=seed,
            force_offload=force_offload,
            save_states=save_states,
            unique_id=unique_id,
            parameters=merged_parameters,
            images=embedded_image if use_image else None,
            queue_handler=queue_handler,
        )
        if style_mode != "洗图":
            return result

        output, output_list, state_uid = result
        clean_output = _enforce_wash_no_text(output)
        clean_list = [_enforce_wash_no_text(item) for item in (output_list or [output])]
        return (clean_output, clean_list, state_uid)

    @classmethod
    def VALIDATE_INPUTS(cls, style, style_image, **kwargs):
        style_mode = _normalize_style_mode(style)
        if style_mode == "文生图":
            return True
        if not style_image:
            return f"{style_mode} mode is enabled, but no image is selected."
        if not folder_paths.exists_annotated_filepath(style_image):
            return f"Invalid style image file: {style_image}"
        return True
