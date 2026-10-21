"""Flux3 Klein generation/editing workbench using the source workflow's graph."""
from collections import OrderedDict
import math
import folder_paths
from comfy_execution.graph_utils import GraphBuilder
from .qwen_image_21_studio import _image_files, _names, _pick, RATIOS
from .flux3_layout import layout_elements, layout_prompt, LAYOUT_EXPANSION_SYSTEM


class LiaoFlux3KleinStudio:
    @classmethod
    def INPUT_TYPES(cls):
        models = _names("diffusion_models", lambda n: "flux3" in n and "klein" in n)
        clips = _names("text_encoders", lambda n: "qwen3vl_4b" in n)
        vaes = _names("vae", lambda n: "flux2" in n)
        images = ["未选择"] + _image_files()
        required = OrderedDict([
            ("模式", (["文生图", "单图编辑", "多图编辑", "构图生图"], {"default": "文生图"})),
            ("提示词", ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True})),
            ("负面提示词", ("STRING", {"default": "", "multiline": True})),
            ("画面比例", (list(RATIOS), {"default": "竖屏 9:16"})),
            ("百万像素", ("FLOAT", {"default": 1.5, "min": .1, "max": 8., "step": .1})),
            ("按首图比例", ("BOOLEAN", {"default": True})),
            ("随机种子", ("INT", {"default": 605452029419059, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})),
            ("采样步数", ("INT", {"default": 16, "min": 1, "max": 100})),
            ("CFG", ("FLOAT", {"default": 1., "min": 0., "max": 20., "step": .1})),
            ("扩散模型", (models, {"default": _pick(models, "flux3_image_klein_sd_int8_convrot")})),
            ("文本编码器", (clips, {"default": _pick(clips, "qwen3vl_4b_int8_convrot")})),
            ("VAE", (vaes, {"default": _pick(vaes, "flux2-vae")})),
        ])
        for i in range(1, 11):
            required[f"参考图{i}"] = (images, {"default": "未选择"})
        loras = ["不使用"] + list(folder_paths.get_filename_list("loras"))
        for i in (1, 2):
            required[f"LoRA{i}"] = (loras, {"default": "不使用"})
            required[f"LoRA{i}强度"] = ("FLOAT", {"default": 1., "min": -4., "max": 4., "step": .05})
        required["构图数据"] = ("STRING", {"default": "[]", "multiline": True})
        required["构图智能扩写"] = ("BOOLEAN", {"default": False})
        upsamplers = ["未选择"] + [n for n in folder_paths.get_filename_list("text_encoders")
                                  if "flux3" in n.lower() and "upsampler" in n.lower()]
        required["构图扩写模型"] = (upsamplers, {"default": upsamplers[1] if len(upsamplers)>1 else "未选择"})
        return {"required": required}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "generate"
    CATEGORY = "Liao2049/Flux3 Klein"

    def generate(self, 模式, 提示词, 负面提示词, 画面比例, 百万像素, 按首图比例,
                 随机种子, 采样步数, CFG, 扩散模型, 文本编码器, VAE, **kwargs):
        if not 提示词.strip():
            raise ValueError("请填写生成或编辑要求。")
        refs = [kwargs.get(f"参考图{i}") for i in range(1, 11)]
        refs = [name for name in refs if name and name != "未选择"]
        if 模式 in {"单图编辑", "多图编辑"} and not refs:
            raise ValueError("图像编辑需要至少一张参考图。")
        if 模式 in {"文生图", "构图生图"}: refs = []
        elif 模式 == "单图编辑": refs = refs[:1]
        g = GraphBuilder()
        model = g.node("UNETLoader", unet_name=扩散模型, weight_dtype="default")
        for i in (1, 2):
            name = kwargs.get(f"LoRA{i}", "不使用")
            if name != "不使用" and float(kwargs.get(f"LoRA{i}强度", 1)) != 0:
                model = g.node("LoraLoaderModelOnly", model=model.out(0), lora_name=name,
                               strength_model=float(kwargs.get(f"LoRA{i}强度", 1)))
        clip = g.node("CLIPLoader", clip_name=文本编码器, type="flux3", device="default")
        vae = g.node("VAELoader", vae_name=VAE)
        positive_text = 提示词
        if 模式 == "构图生图":
            raw_layout = kwargs.get("构图数据", "[]")
            layout_elements(raw_layout)
            layout_inputs = {"prompt": 提示词, "layout": raw_layout}
            if kwargs.get("构图智能扩写", False):
                name = kwargs.get("构图扩写模型", "未选择")
                if name == "未选择" or name not in folder_paths.get_filename_list("text_encoders"):
                    raise ValueError("构图智能扩写需要本地 Flux3 prompt upsampler 模型；请先选择模型，或关闭智能扩写。")
                import nodes
                if "TextGenerate" not in nodes.NODE_CLASS_MAPPINGS:
                    raise ValueError("当前内核缺少 TextGenerate，请更新内核或关闭构图智能扩写。")
                upsampler = g.node("CLIPLoader", clip_name=name, type="flux3", device="default")
                expanded = g.node("TextGenerate", clip=upsampler.out(0),
                                  prompt=layout_prompt(提示词, raw_layout), max_length=3584,
                                  sampling_mode="off", system_prompt=LAYOUT_EXPANSION_SYSTEM,
                                  thinking=False, use_default_template=True, mtp="auto")
                layout_inputs["expanded"] = expanded.out(0)
            positive_text = g.node("LiaoFlux3LayoutPrompt", **layout_inputs).out(0)
        positive = g.node("CLIPTextEncode", clip=clip.out(0), text=positive_text)
        negative = g.node("CLIPTextEncode", clip=clip.out(0), text=负面提示词)
        rw, rh = RATIOS[画面比例]
        scale = math.sqrt(float(百万像素) * 1024 * 1024 / (rw * rh))
        width, height = max(8, round(rw * scale / 8) * 8), max(8, round(rh * scale / 8) * 8)
        latent = g.node("EmptyFlux2LatentImage", width=width, height=height, batch_size=1).out(0)
        if refs:
            images = {f"images.image{i}": g.node("LoadImage", image=name).out(0) for i, name in enumerate(refs)}
            ref = g.node("ReferenceMultiLatent", positive=positive.out(0), negative=negative.out(0),
                         vae=vae.out(0), resolution="resize", **{"resolution.megapixels": float(百万像素)}, **images)
            positive_link, negative_link = ref.out(0), ref.out(1)
            if 按首图比例: latent = ref.out(2)
        else:
            positive_link, negative_link = positive.out(0), negative.out(0)
        sampled = g.node("KSampler", model=model.out(0), positive=positive_link, negative=negative_link,
                         latent_image=latent, seed=int(随机种子), steps=int(采样步数), cfg=float(CFG),
                         sampler_name="euler", scheduler="simple", denoise=1.)
        decoded = g.node("VAEDecode", samples=sampled.out(0), vae=vae.out(0))
        return {"result": (decoded.out(0),), "expand": g.finalize()}


class LiaoFlux3LayoutPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt": ("STRING", {"multiline": True}), "layout": ("STRING", {"multiline": True})},
                "optional": {"expanded": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "build"
    CATEGORY = "Liao2049/Internal"

    def build(self, prompt, layout, expanded=""):
        return (layout_prompt(prompt, layout, expanded),)


NODE_CLASS_MAPPINGS = {"LiaoFlux3KleinStudio": LiaoFlux3KleinStudio, "LiaoFlux3LayoutPrompt": LiaoFlux3LayoutPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"LiaoFlux3KleinStudio": "Liao-Flux3 Klein 便捷多功能生图"}
