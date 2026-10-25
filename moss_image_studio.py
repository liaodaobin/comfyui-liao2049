"""MOSS workbench preserving the supplied Base/SFT, and layout graphs."""
from collections import OrderedDict
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import folder_paths
import comfy.samplers
from comfy_execution.graph_utils import GraphBuilder
from server import PromptServer

from .qwen_image_21_studio import _pick, RATIOS
from .flux3_layout import layout_elements
from .moss_typography import AUTO_FONT, font_files, resolve_font, render_text
from .moss_prompt_enhance import llm_models, vision_models, enhancement_key, rewrite_moss


PROFILES = ["标准50步", "标准25步", "极速8步"]


class LiaoMossImageStudio:
    @classmethod
    def INPUT_TYPES(cls):
        models = []
        for name in folder_paths.get_filename_list("diffusion_models"):
            normalized = name.replace("\\", "/").lower()
            filename = normalized.rsplit("/", 1)[-1]
            if ("moss_image" in normalized or "mossimage" in normalized.split("/")[:-1]
                    or filename.startswith(("moss-", "moss_"))):
                models.append(name)
        models = models or ["未找到MOSS模型"]
        clips = [n for n in folder_paths.get_filename_list("text_encoders") if "qwen3.5_9b" in n.lower()] or ["未找到MOSS编码器"]
        vaes = [n for n in folder_paths.get_filename_list("vae") if "flux2" in n.lower()] or ["未找到Flux2 VAE"]
        loras = ["不使用"] + list(folder_paths.get_filename_list("loras"))
        required = OrderedDict([
            ("模式", (["文生图", "构图生图"], {"default": "文生图"})),
            ("生成方案", (PROFILES, {"default": PROFILES[0]})),
            ("提示词", ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})),
            ("负面提示词", ("STRING", {"default": "", "multiline": True})),
            ("画面比例", (list(RATIOS), {"default": "竖图 3:4"})),
            ("百万像素", ("FLOAT", {"default": 2., "min": .1, "max": 16., "step": .1})),
            ("采样步数", ("INT", {"default": 50, "min": 1, "max": 1000})),
            ("CFG", ("FLOAT", {"default": 3.5, "min": 0., "max": 20., "step": .1})),
            ("随机种子", ("INT", {"default": 666, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})),
            ("采样器", (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"})),
            ("调度器", (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"})),
            ("模型", (models, {"default": _pick(models, "moss_image_sft")})),
            ("文本编码器", (clips, {"default": _pick(clips, "qwen3.5_9b_int8")})),
            ("VAE", (vaes, {"default": _pick(vaes, "flux2-vae")})),
            ("极速LoRA", (loras, {"default": next((n for n in loras if "moss" in n.lower() and "lora_latest" in n.lower()), "不使用")})),
            ("极速LoRA强度", ("FLOAT", {"default": 1., "min": -4., "max": 4., "step": .05})),
            ("SLG", ("BOOLEAN", {"default": True})),
            ("SLG双流层", ("STRING", {"default": ""})),
            ("SLG单流层", ("STRING", {"default": "22"})),
            ("SLG开始", ("FLOAT", {"default": 0., "min": 0., "max": 1., "step": .01})),
            ("SLG结束", ("FLOAT", {"default": .5, "min": 0., "max": 1., "step": .01})),
            ("构图数据", ("STRING", {"default": "[]", "multiline": True})),
        ])
        for i in (1, 2):
            required[f"LoRA{i}"] = (loras, {"default": "不使用"})
            required[f"LoRA{i}强度"] = ("FLOAT", {"default": 1., "min": -4., "max": 4., "step": .05})
        optional = OrderedDict([
            ("文字绘制", (["字体精确排字", "模型生成文字"], {"default": "字体精确排字"})),
            ("排字字体", ([AUTO_FONT] + list(font_files(folder_paths.models_dir, Path(__file__).parent)), {"default": AUTO_FONT})),
            ("清除框内原字", ("BOOLEAN", {"default": True})),
        ])
        llms = llm_models()
        optional.update([
            ("提示词增强", ("BOOLEAN", {"default": False})),
            ("增强模型", (llms or ["未选择"], {"default": next((n for n in llms if "ornith" in n.lower()), llms[0] if llms else "未选择")})),
            ("增强上下文", ("INT", {"default": 8192, "min": 2048, "max": 131072, "step": 1024})),
            ("每次推理", ("BOOLEAN", {"default": False})),
            ("增强提示词", ("STRING", {"default": "", "multiline": True})),
            ("增强来源", ("STRING", {"default": ""})),
            ("增强视觉模型", (["无"] + vision_models(), {"default": "无"})),
        ])
        for i in (3, 4, 5):
            optional[f"LoRA{i}"] = (loras, {"default": "不使用"})
            optional[f"LoRA{i}强度"] = ("FLOAT", {"default": 1., "min": -4., "max": 4., "step": .05})
        optional.update({"增强服务": (["本地 Llama", "DeepSeek API"], {"default":"本地 Llama"}), "云端增强模型": ("STRING", {"default":"deepseek-flash"}), "云端凭据会话": ("STRING", {"default":""})})
        return {"required": required, "optional": optional, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "generate"
    CATEGORY = "Liao2049/MOSS Image"

    @classmethod
    def IS_CHANGED(cls, **v):
        return float("nan") if v.get("提示词增强", False) and v.get("每次推理", False) else ""

    def generate(self, **v):
        if not v["提示词"].strip():
            raise ValueError("请填写画面描述。")
        g = GraphBuilder()
        clip = g.node("CLIPLoader", clip_name=v["文本编码器"], type="moss_image", device="default")
        vae = g.node("VAELoader", vae_name=v["VAE"])
        rw, rh = RATIOS[v["画面比例"]]
        scale = math.sqrt(float(v["百万像素"]) * 1024 * 1024 / (rw * rh))
        width, height = max(16, round(rw*scale/16)*16), max(16, round(rh*scale/16)*16)
        text = v["提示词"]
        if v.get("提示词增强", False):
            key = enhancement_key(v["模式"], text, v["构图数据"], v["负面提示词"])
            chosen = str(v.get("增强提示词", "")).strip()
            if chosen and v.get("增强来源") == key and not v.get("每次推理", False):
                text = chosen
            else:
                text = rewrite_moss(v["模式"], text, v["构图数据"], v.get("增强模型", ""), int(v.get("增强上下文", 8192)), v["负面提示词"], v.get("增强视觉模型", "无"), v.get("增强服务", "本地 Llama"), v.get("云端增强模型", "deepseek-flash"), v.get("云端凭据会话", ""))
                PromptServer.instance.send_sync("liao_moss_prompt_enhanced", {"node": v.get("unique_id", ""), "prompt": text, "key": key})
        exact = v["模式"] == "构图生图" and v.get("文字绘制", "字体精确排字") == "字体精确排字"
        if v["模式"] == "构图生图":
            moss_elements(v["构图数据"])
            if exact:
                fonts = font_files(folder_paths.models_dir, Path(__file__).parent)
                for box in json.loads(v["构图数据"]):
                    if box.get("type") == "text":
                        resolve_font(v.get("排字字体", AUTO_FONT), box.get("desc", ""), fonts)
                text += "\nRender only the scene and blank areas for later typography. No writing, letters, captions or watermarks."
            elements = g.node("LiaoMossLayoutElements", layout=v["构图数据"], omit_text=exact)
            text = g.node("MOSSImageLayoutPrompt", prompt=text, element=elements.out(0),
                          character_boxes=elements.out(1), width=width, height=height).out(0)
        positive = g.node("CLIPTextEncode", clip=clip.out(0), text=text)
        negative = g.node("CLIPTextEncode", clip=clip.out(0), text=v["负面提示词"])
        latent = g.node("EmptyLatentImage", width=width, height=height, batch_size=1)
        fast = v["生成方案"] == "极速8步"
        if fast and v["极速LoRA"] == "不使用":
            raise ValueError("极速8步需要 lora_latest.safetensors，请选择极速LoRA。")

        def sample(model_name):
            model = g.node("UNETLoader", unet_name=model_name, weight_dtype="default")
            if fast:
                model = g.node("LoraLoaderModelOnly", model=model.out(0), lora_name=v["极速LoRA"], strength_model=float(v["极速LoRA强度"]))
            for i in range(1, 6):
                name, strength = v.get(f"LoRA{i}", "不使用"), float(v.get(f"LoRA{i}强度", 1.))
                if name != "不使用" and strength != 0:
                    model = g.node("LoraLoaderModelOnly", model=model.out(0), lora_name=name, strength_model=strength)
            if v["SLG"]:
                model = g.node("SkipLayerGuidanceDiTSimple", model=model.out(0), double_layers=v["SLG双流层"],
                               single_layers=v["SLG单流层"], start_percent=float(v["SLG开始"]), end_percent=float(v["SLG结束"]))
            sampled = g.node("KSampler", model=model.out(0), positive=positive.out(0), negative=negative.out(0),
                             latent_image=latent.out(0), seed=int(v["随机种子"]), steps=int(v["采样步数"]), cfg=float(v["CFG"]),
                             sampler_name=v["采样器"], scheduler=v["调度器"], denoise=1.)
            return g.node("VAEDecode", samples=sampled.out(0), vae=vae.out(0)).out(0)

        moss = sample(v["模型"])
        if exact:
            moss = g.node("LiaoMossTypography", image=moss, layout=v["构图数据"],
                          font=v.get("排字字体", AUTO_FONT), clear=bool(v.get("清除框内原字", True))).out(0)
        return {"result": (moss,), "expand": g.finalize()}


def moss_elements(raw, omit_text=False):
    rows = layout_elements(raw)
    boxes = json.loads(raw)
    elements, characters = [], []
    for i, (box, row) in enumerate(zip(boxes, rows), 1):
        item = {"bbox": row["bbox"], "type": "text" if box.get("type") == "text" else "obj",
                "desc": str(box.get("desc", "")), "text": str(box.get("text", ""))}
        if omit_text and item["type"] == "text":
            elements.append({"bbox": row["bbox"], "type": "obj",
                             "desc": "A clean blank space reserved for typography, without any writing or symbols."})
            continue
        if item["type"] == "text" and box.get("direction") == "vertical":
            item["desc"] += " 竖排文字，vertical text, top to bottom."
        if box.get("type") == "character":
            characters.append(str(i))
        palette = box.get("palette", [])
        if isinstance(palette, list) and palette:
            item["color_palette"] = [str(color) for color in palette]
        elements.append(item)
    return elements, ", ".join(characters)


class LiaoMossLayoutElements:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"layout": ("STRING", {"multiline": True})},
                "optional": {"omit_text": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ("ARRAY", "STRING")
    FUNCTION = "build"
    CATEGORY = "Liao2049/Internal"

    def build(self, layout, omit_text=False):
        return moss_elements(layout, omit_text)


class LiaoMossTypography:
    @classmethod
    def INPUT_TYPES(cls):
        fonts = [AUTO_FONT] + list(font_files(folder_paths.models_dir, Path(__file__).parent))
        return {"required": {"image": ("IMAGE",), "layout": ("STRING", {"multiline": True}),
                             "font": (fonts,), "clear": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "Liao2049/Internal"

    def draw(self, image, layout, font=AUTO_FONT, clear=True):
        layout_elements(layout)
        fonts = font_files(folder_paths.models_dir, Path(__file__).parent)
        output = []
        for frame in image:
            source = Image.fromarray((frame.detach().cpu().numpy().clip(0, 1)*255).round().astype(np.uint8))
            result = render_text(source, layout, fonts, font, clear)
            pixels = torch.from_numpy(np.array(result).astype(np.float32)/255).to(device=image.device, dtype=image.dtype)
            composed = frame.clone()
            for box in json.loads(layout):
                if box.get("type") == "text" and str(box.get("text", "")).strip():
                    h, w = frame.shape[:2]
                    x0, y0 = round(box["x"]*w), round(box["y"]*h)
                    x1, y1 = round((box["x"]+box["w"])*w), round((box["y"]+box["h"])*h)
                    composed[y0:y1, x0:x1] = pixels[y0:y1, x0:x1]
            output.append(composed)
        return (torch.stack(output),)



NODE_CLASS_MAPPINGS = {"LiaoMossImageStudio": LiaoMossImageStudio, "LiaoMossLayoutElements": LiaoMossLayoutElements, "LiaoMossTypography": LiaoMossTypography}
NODE_DISPLAY_NAME_MAPPINGS = {"LiaoMossImageStudio": "Liao-MOSS Image 便捷多功能生图"}
