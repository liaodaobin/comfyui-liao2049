# ComfyUI-Liao2049

![Liao-H3 智能创作台界面](assets/liao-h3-interface.png)

面向 MiniMax H3 的一体化 ComfyUI 创作节点。核心节点 **Liao-H3 智能创作台**把模型选择、LoRA、VAE、参考素材、提示词增强、生成模式、性能预设、视频编辑和二采放大集中到一个界面中。

> 本仓库不包含任何模型权重。MiniMax H3、文本编码器、VAE、LoRA、GGUF 与潜空间放大模型需由使用者按照各自许可证单独获取。

## 主要功能

### MiniMax H3 六种生成模式

- 图生视频：单张首帧驱动。
- 文生视频：自动忽略参考素材并使用 FL2VA 路线。
- 多参考：图片、视频、音频可同时输入，并通过 `@图片1`、`@视频1`、`@音频1` 明确引用。
- 首尾帧：严格使用两张图片作为起始帧和结束帧。
- 视频编辑：支持通用编辑、去除字幕、动作迁移和角色替换。
- 数字人：支持单人、双人和 MV 三种子模式。
- MV 模式：上传一条完整音乐与 1–20 张图片，节点自动读取音频总时长、按图片数量切分镜头，每段最长 15 秒，逐段生成后自动拼接，并保留完整原始音轨。

### 模型与性能配置

- 自动识别本机 MiniMax H3 相关模型，过滤无关模型。
- 极速 4 步、均衡 8 步、质量 20 步和自定义步数。
- 根据生成模式与性能档位自动匹配 FL2VA/Ref2VA 模型及可用 LoRA。
- 支持手动展开模型配置，自定义 UNet、文本编码器、双 VAE、LoRA、精度、SageAttention 与 SigmaShift。
- 任务开始、条件编码、模型切换和解码前进行显存释放，降低模型重叠驻留。

### 参考素材交互

- 一个入口混合上传图片、视频和音频，自动编号。
- 支持拖拽、文件选择和剪贴板粘贴图片。
- 点击素材缩略图即可把对应的 `@` 标签插入提示词。
- 多参考模式区分人物/动物、商品/物体、场景和局部属性，避免把参考图误当作整张首帧。

### 提示词增强

- 内置 MiniMax H3 官方 Skill 结构化模板。
- 内置 Liao 分镜模板，支持按时长和用户要求组织镜头。
- 本地 Llama.cpp：支持文本 GGUF 与 `mmproj` 视觉识别模型。
- 云端服务：支持 Kimi API、MiniMax API 与 OpenAI 兼容接口。
- 增强功能默认关闭；关闭时不会校验或要求 GGUF/mmproj 模型。
- 可直接采用增强结果，也可保留原始创意继续生成。

### 二采高清放大（更耗显存）

- **潜空间 1.5 倍**：先以 75% 线性尺寸首采，使用 H3 3D 潜空间放大模型进行 2 倍放大，再完成低 Sigma 尾段精修；最终线性尺寸约为用户选择值的 1.5 倍。
- **双模型 2 倍**：首采后解码、像素放大、重新 VAE 编码，再加载同系列 W4A8 Mixed 模型进行 3 步低降噪重绘。
- 两种方式均显著增加显存、内存和耗时；8GB 显存优先使用较低基础分辨率或潜空间方案。

### 兼容节点

仓库还包含从旧 WenWu 插件迁移的兼容实现：

- ACE-Step 1.5 LLM 规划与 Remix 辅助节点。
- Krea2 文生图、风格参考与洗图提示词节点。
- 旧工作流使用的节点类型 ID 尽量保持兼容，无需继续安装完整 WenWu 插件。

## 安装

把整个仓库文件夹复制到：

```text
ComfyUI/custom_nodes/comfyui-liao2049
```

重启 ComfyUI，然后在节点菜单 `Liao2049/MiniMax H3` 中添加 **Liao-H3 智能创作台**。

### 依赖原则

H3 生成主体不通过 `requirements.txt` 强制安装额外 Python 包。所有特殊依赖均应由使用者在节点界面点击“检测依赖”或启动器的“一键补全依赖”后手动确认安装，避免在不同 CUDA、Python 和 PyTorch 环境中破坏现有运行栈。

本地提示词增强是可选功能：

- 需要 `llama-cpp-python>=0.3.35`，或可复用兼容的 `ComfyUI-llama-cpp_vlm` / `ComfyUI-llama-cpp` 运行环境。
- 文本 GGUF 放在 `ComfyUI/models/LLM`。
- 视觉识别所需的 `mmproj` GGUF 同样放在 `ComfyUI/models/LLM`。
- GPU offload 是否可用取决于显卡、CUDA、Python 版本和可用 wheel；通用 PyPI 包不保证带 CUDA 后端。
- OpenCV 只用于可选的视频帧视觉分析，本仓库不会自动安装。

## 模型目录

```text
ComfyUI/models/diffusion_models/       MiniMax H3 FL2VA / Ref2VA 模型
ComfyUI/models/text_encoders/          Qwen MiniMax H3 文本编码器
ComfyUI/models/vae/                    视频 VAE、音频 VAE
ComfyUI/models/loras/                  MiniMax H3 Turbo 4-step / 8-step LoRA
ComfyUI/models/LLM/                    本地提示词 GGUF 与 mmproj
ComfyUI/models/latent_upscale_models/  H3 3D 潜空间放大模型
```

潜空间二采常用模型名：

```text
minimax_h3_latent_upscaler_3d_fp16.safetensors
```

双模型重绘会在本机自动寻找与首采模型同系列、文件名包含 `W4A8` / `Mixed` 的 MiniMax H3 二采模型。没有匹配模型时会明确报错，不会静默换用无关模型。

## 使用建议

- 第一次测试建议使用 0.4MP、5 秒和均衡 8 步。
- 角色替换与动作迁移需要至少一张目标主体图片和一段源视频。
- MV 模式只接收一条主音轨；图片数量至少为 `音频时长 ÷ 15 秒` 向上取整。当前最多 20 张图片，适合约 5 分钟以内的音乐。
- 提示词中使用明确标签，例如：`让 @图片1 的人物执行 @视频1 的动作，保留 @视频1 的镜头和背景。`
- 双模型二采在第二模型加载和高分辨率重绘期间可能长时间停留在某个阶段，这通常是模型换页或高分辨率采样，并不一定是死机。
- 高分辨率、长时长、质量 20 步与二采叠加会显著增加显存压力。

## 可移植性

- 插件不写死盘符，不随仓库分发本地模型路径。
- 提示词增强关闭时，不要求本地 Llama 模型存在。
- 缺少可选模型时提供中文错误提示，不自动联网下载。
- 不自动安装、升级或替换 PyTorch、CUDA、Triton、SageAttention 等底层环境。

## 反馈与教程

- QQ 群：`38251314`
- 更多工具和教程：B站搜索 `liao_2049`

提交问题时建议附上 ComfyUI 错误报告、运行日志、显卡型号、Python/PyTorch/CUDA 版本、所选生成模式以及模型文件名。

## 免责声明

本项目仅提供 ComfyUI 节点代码，不提供或重新分发第三方模型。使用者应遵守 MiniMax H3、相关模型、LoRA、VAE、Llama.cpp 及其他第三方组件各自的许可证和使用条款。
