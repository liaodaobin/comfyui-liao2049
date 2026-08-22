"""Internal MiniMax H3 learned latent upscaler used by Liao-H3.

This is a small, self-contained inference implementation for the published
24-channel 3D latent-upscaler weights.  It intentionally exposes no user-facing
workflow node and depends only on packages already shipped by ComfyUI.
"""
from __future__ import annotations

import os
import re

import torch
from torch import nn
from torch.nn import functional as F
import folder_paths


MODEL_FOLDER = "latent_upscale_models"
MODEL_CACHE: dict[str, nn.Module] = {}

if MODEL_FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(
        MODEL_FOLDER, os.path.join(folder_paths.models_dir, MODEL_FOLDER)
    )

LATENT_MEAN = (
    0.8580903411, -0.9606591463, 1.0661640167, -0.5090325475,
    -0.2727581855, -1.3675414324, -0.2553254962, -0.2690755427,
    -0.5376840830, -0.0464097299, 0.6657370329, 0.1969012767,
    -0.5460608006, -0.4035342038, -0.2368302494, 0.2592845261,
    -0.3013394475, 0.2113419920, -1.1206848621, 0.3581933379,
    -0.0422514379, 0.2604829967, 0.2286409289, 0.7056031823,
)
LATENT_STD = (
    1.2223774195, 1.2767263651, 1.6831774712, 1.7549455166,
    1.5636216402, 2.1941435337, 0.9653137922, 1.0569885969,
    0.8419489264, 0.7729952931, 1.8955937624, 0.9468418360,
    0.7996809483, 0.4498890042, 0.7197399735, 0.6936293244,
    2.9610950947, 2.7694199085, 3.0496184826, 2.1088054180,
    3.2762262821, 3.1627357006, 2.2816812992, 2.6127843857,
)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


class _Residual3D(nn.Module):
    def __init__(self, channels: int, embedding_channels: int, dropout: float = 0.1):
        super().__init__()
        self.out_channels = channels
        self.in_layers = nn.Sequential(
            _group_norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1)
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embedding_channels, channels * 2))
        self.out_norm = _group_norm(channels)
        final_conv = nn.Conv3d(channels, channels, 3, padding=1)
        final_conv.weight.detach().zero_()
        final_conv.bias.detach().zero_()
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), final_conv)
        self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.in_layers(x)
        scale_shift = self.emb_layers(embedding).to(hidden.dtype)
        while scale_shift.ndim < hidden.ndim:
            scale_shift = scale_shift[..., None]
        scale, shift = scale_shift.chunk(2, dim=1)
        hidden = self.out_norm(hidden) * (1 + scale) + shift
        return self.skip(x) + self.out_layers(hidden)


class _Temporal3D(nn.Module):
    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.norm = _group_norm(channels)
        self.dwconv = nn.Conv3d(
            channels, channels, (kernel, 1, 1), padding=(kernel // 2, 0, 0), groups=channels
        )
        self.pwconv = nn.Conv3d(channels, channels, 1)
        self.pwconv.weight.detach().zero_()
        self.pwconv.bias.detach().zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.dwconv(F.silu(self.norm(x)))
        return x + self.pwconv(hidden)


class _H3LatentResizer3D(nn.Module):
    def __init__(
        self, in_channels: int, channels: int, in_blocks: int,
        out_blocks: int, temporal_every: int, temporal_kernel: int,
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embedding_channels = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embedding_channels), nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.in_blocks = self._make_stack(
            channels, embedding_channels, in_blocks, temporal_every, temporal_kernel
        )
        self.out_blocks = self._make_stack(
            channels, embedding_channels, out_blocks, temporal_every, temporal_kernel
        )
        self.norm_out = _group_norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    @staticmethod
    def _make_stack(channels, embedding_channels, count, temporal_every, temporal_kernel):
        layers = nn.ModuleList()
        for index in range(count):
            layers.append(_Residual3D(channels, embedding_channels))
            if temporal_every > 0 and index % temporal_every == 0:
                layers.append(_Temporal3D(channels, temporal_kernel))
        return layers

    @staticmethod
    def _run_stack(layers, x, embedding):
        for layer in layers:
            x = layer(x, embedding) if isinstance(layer, _Residual3D) else layer(x)
        return x

    def forward(
        self, x: torch.Tensor, scale: float,
        scale_height: float | None = None, scale_width: float | None = None,
    ) -> torch.Tensor:
        scale_height = float(scale_height or scale)
        scale_width = float(scale_width or scale)
        target = (
            x.shape[2], round(x.shape[3] * scale_height), round(x.shape[4] * scale_width)
        )
        embedding = self.embed(x.new_tensor([[scale - 1.0]]))
        x = self._run_stack(self.in_blocks, self.conv_in(x), embedding)
        x = F.interpolate(x, size=target, mode="trilinear", align_corners=False)
        x = self._run_stack(self.out_blocks, x, embedding)
        return self.conv_out(F.silu(self.norm_out(x)))


def _strip_prefix(state):
    if any(key.startswith("upscaler.") for key in state):
        return {
            key.removeprefix("upscaler."): value
            for key, value in state.items() if key.startswith("upscaler.")
        }
    return state


def _architecture(state):
    conv = state["conv_in.weight"]
    residual_in = {
        int(match.group(1)) for key in state
        if (match := re.match(r"in_blocks\.(\d+)\.in_layers\.", key))
    }
    residual_out = {
        int(match.group(1)) for key in state
        if (match := re.match(r"out_blocks\.(\d+)\.in_layers\.", key))
    }
    temporal_keys = [key for key in state if key.endswith("dwconv.weight")]
    return {
        "in_channels": int(conv.shape[1]),
        "channels": int(conv.shape[0]),
        "in_blocks": len(residual_in) or 12,
        "out_blocks": len(residual_out) or 12,
        "temporal_every": 2 if temporal_keys else 0,
        "temporal_kernel": int(state[temporal_keys[0]].shape[2]) if temporal_keys else 5,
    }


def _model_path(name: str) -> str:
    paths = folder_paths.get_folder_paths(MODEL_FOLDER)
    for directory in paths:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"未找到潜空间放大模型 {name}，请放入 ComfyUI/models/{MODEL_FOLDER}"
    )


def _load_cpu_model(name: str) -> nn.Module:
    if name in MODEL_CACHE:
        return MODEL_CACHE[name]
    path = _model_path(name)
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path, device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    state = _strip_prefix(state)
    state = {
        key: value.to(torch.float16) if value.dtype == torch.float8_e4m3fn else value
        for key, value in state.items()
    }
    model = _H3LatentResizer3D(**_architecture(state))
    model.load_state_dict(state, strict=True)
    model.eval()
    MODEL_CACHE[name] = model
    print(f"[Liao-H3] 内置潜空间放大器已加载: {name}")
    return model


class LiaoH3EmbeddedLatentUpscaler3D:
    """Internal graph node; not intended for manual workflow placement."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "model_name": ("STRING", {"default": "minimax_h3_latent_upscaler_3d_fp16.safetensors"}),
            "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
            "scale_height": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
            "scale_width": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0}),
            "device": (["cuda", "cpu"], {"default": "cuda"}),
            "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "Liao2049/Internal"

    def upscale(
        self, latent, model_name, scale=2.0, scale_height=2.0, scale_width=2.0,
        device="cuda", precision="fp16",
    ):
        import comfy.model_management as mm

        if float(scale) <= 1.0:
            return (latent,)
        target_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
        model = _load_cpu_model(str(model_name)).to(device=target_device, dtype=dtype)
        source = latent["samples"]
        was_4d = source.ndim == 4
        samples = source.unsqueeze(2) if was_4d else source
        samples = samples.to(device=target_device, dtype=dtype)
        mean = samples.new_tensor(LATENT_MEAN).view(1, -1, 1, 1, 1)
        std = samples.new_tensor(LATENT_STD).view(1, -1, 1, 1, 1)
        with torch.inference_mode():
            output = model(
                (samples - mean) / std, float(scale),
                float(scale_height), float(scale_width),
            ) * std + mean
        if was_4d:
            output = output.squeeze(2)
        result = latent.copy()
        result["samples"] = output.to(device="cpu", dtype=source.dtype)
        # The following H3 refinement needs all available VRAM.  Keep the
        # reusable model cached on CPU instead of pinning another ~659 MB GPU.
        model.to("cpu")
        del samples, output, mean, std
        mm.soft_empty_cache()
        return (result,)


NODE_CLASS_MAPPINGS = {
    "LiaoH3EmbeddedLatentUpscaler3D": LiaoH3EmbeddedLatentUpscaler3D,
}
