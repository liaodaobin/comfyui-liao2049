"""Optional SelfLift graph route; normal sampling retains its existing node."""


def sample_h3_graph(graph, node_type, selflift, vae, upscaler_model="h3_upscaler_lms_v0.1.safetensors", highres_tiling=True, **inputs):
    if not selflift:
        return graph.node(node_type, **inputs)
    guider = graph.lookup_node(inputs["guider"][0].removeprefix(graph.prefix))
    noise = graph.lookup_node(inputs["noise"][0].removeprefix(graph.prefix))
    schedule = graph.lookup_node(inputs["sigmas"][0].removeprefix(graph.prefix))
    steps = int(schedule.get_input("steps") or 0)
    if steps < 2:
        raise ValueError("SelfLift 至少需要 2 步，并且必须使用完整的采样调度。")
    if steps != 10:
        schedule = graph.node("BasicScheduler", **{**schedule.inputs, "steps": 10})
    transition = 6
    positive = guider.get_input("conditioning")
    negative = graph.node("ConditioningZeroOut", conditioning=positive)
    sampler = graph.node("KSamplerSelect", sampler_name="euler")
    prepared = graph.node("LiaoH3SelfLiftPrepare", model=guider.get_input("model"), samples=inputs["latent_image"])
    sampled = graph.node(
        "SelfLiftH3Sampler", model=prepared.out(0), positive=positive,
        negative=negative.out(0), vae=vae, latent_image=prepared.out(1),
        sampler=sampler.out(0), sigmas=schedule.out(0), seed=int(noise.get_input("noise_seed") or 0),
        cfg=1.0, transition_step=transition, lowres_scale=0.6, rho=0.0,
        w_min=0.5, w_max=1.0, upscaler_model=upscaler_model,
        highres_tiling=highres_tiling, upscaler_unload=True,
    )
    if inputs.get("phase"):
        sampled = graph.node("WenWuH3PhaseMarker", samples=sampled.out(0), phase=inputs["phase"], progress=5, span=80)
    return graph.node("LiaoH3SelfLiftResult", samples=sampled.out(0))


class LiaoH3SelfLiftResult:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    RETURN_TYPES = ("LATENT", "LATENT")
    FUNCTION = "result"
    CATEGORY = "Liao2049/Internal"

    def result(self, samples):
        # SelfLift has one final latent; both custom-sampler output paths use it.
        return samples, samples


class LiaoH3SelfLiftPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "samples": ("LATENT",)}}

    RETURN_TYPES = ("MODEL", "LATENT")
    FUNCTION = "prepare"
    CATEGORY = "Liao2049/Internal"

    def prepare(self, model, samples):
        import torch
        mask = samples.get("noise_mask")
        if mask is None:
            return model, samples
        output = samples.copy()
        nested = getattr(mask, "is_nested", False)
        masks = list(mask.unbind()) if nested else [mask]
        video_mask = masks[0]
        video = samples["samples"]
        streams = list(video.unbind()) if getattr(video, "is_nested", False) else [video]
        if video_mask.ndim == 5 and video_mask.shape[1] != 1:
            single = video_mask[:, :1]
            if not torch.equal(video_mask, single.expand_as(video_mask)):
                raise ValueError("SelfLift 不支持各视频通道使用不同的遮罩，请关闭二采。")
            video_mask = single
        if torch.all(video_mask == 1):
            output.pop("noise_mask", None)
        else:
            output["noise_mask"] = video_mask
        if nested and len(masks) > 1:
            # SelfLift resizes video internally. Anchor audio using its trailing
            # packed region, whose length stays constant at both resolutions.
            audio = torch.cat([s.reshape(s.shape[0], 1, -1) for s in streams[1:]], dim=-1)
            audio_mask = torch.cat([m.expand_as(s).reshape(s.shape[0], 1, -1)
                                    for m, s in zip(masks[1:], streams[1:])], dim=-1)
            if not torch.all(audio_mask == 1):
                model = model.clone()
                def preserve_audio(args):
                    denoised = args["denoised"]
                    count = audio.shape[-1]
                    result = denoised.clone()
                    tail = result[..., -count:]
                    weight = audio_mask.to(device=tail.device, dtype=tail.dtype)
                    anchor = audio.to(device=tail.device, dtype=tail.dtype)
                    result[..., -count:] = tail * weight + anchor * (1 - weight)
                    return result
                model.set_model_sampler_post_cfg_function(preserve_audio)
        return model, output
