import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

let lastSageWarningAt = 0;
api.addEventListener("liao_h3_sage_warning", (event) => {
  const now = Date.now();
  if (now - lastSageWarningAt < 2000) return;
  lastSageWarningAt = now;
  const detail = event?.detail || {};
  window.alert(
    `${detail.title || "未检测到 SageAttention 2.2"}\n\n` +
    `${detail.message || "没有安装 SageAttention 仍可继续生成，但速度可能明显变慢，建议安装。"}`
  );
});

const TARGET = "WenWuMiniMaxH3Unified";
const MEDIA = {
  图片: { count: 20, accept: "image/png,image/jpeg,image/webp,image/bmp", icon: "▧" },
  视频: { count: 3, accept: "video/mp4,video/webm,video/quicktime,video/x-matroska", icon: "▶" },
  音频: { count: 3, accept: "audio/mpeg,audio/wav,audio/flac,audio/x-wav", icon: "♫" },
};

const w = (node, name) => node.widgets?.find((item) => item.name === name);
const values = (widget) => widget?.options?.values || widget?.options?.items || [];
// Shared filename normalizer used by the model and LoRA ranking helpers.
// This must live at module scope because the preset buttons call the helpers
// before entering applyPerformancePreset's local scope.
const lower = (value) => String(value || "").replaceAll("\\", "/").toLowerCase();
const PREFERRED_H3_TURBO_LORA = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16";
const PREFERRED_H3_BALANCED_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16";
const PREFERRED_H3_REF_TURBO_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16";
function pickMinimaxH3TurboLora(installedLoras) {
  const ranked = [];
  for (let index = 0; index < installedLoras.length; index++) {
    const value = String(installedLoras[index] || "");
    const normalized = lower(value).replaceAll("\\", "/");
    const compact = normalized.replace(/[^a-z0-9]/g, "");
    if (normalized.includes(PREFERRED_H3_TURBO_LORA)) return value;
    if (!compact.includes("minimaxh3")) continue;
    let score = 0;
    if (compact.includes("fl2v") || compact.includes("fl2va")) score += 50;
    if (compact.includes("turbo")) score += 40;
    if (compact.includes("4step")) score += 30;
    if (compact.includes("comfyui")) score += 10;
    if (compact.includes("bf16")) score += 5;
    if (compact.includes("lightx2v")) score -= 15;
    ranked.push({ value, score, index });
  }
  ranked.sort((a, b) => b.score - a.score || a.index - b.index);
  return ranked[0]?.value || "";
}
function pickMinimaxH3BalancedLora(installedLoras) {
  const ranked = [];
  for (let index = 0; index < installedLoras.length; index++) {
    const value = String(installedLoras[index] || "");
    const normalized = lower(value).replaceAll("\\", "/");
    const compact = normalized.replace(/[^a-z0-9]/g, "");
    if (normalized.includes(PREFERRED_H3_BALANCED_LORA)) return value;
    if (!compact.includes("minimaxh3") || !compact.includes("8step") || compact.includes("4step")) continue;
    let score = 0;
    if (compact.includes("fl2v") || compact.includes("fl2va")) score += 50;
    if (compact.includes("turbo")) score += 40;
    if (compact.includes("comfyui")) score += 10;
    if (compact.includes("bf16")) score += 5;
    ranked.push({ value, score, index });
  }
  ranked.sort((a, b) => b.score - a.score || a.index - b.index);
  return ranked[0]?.value || "";
}
function pickMinimaxH3RefTurboLora(installedLoras, preferredSteps = 4) {
  const ranked = [];
  const stepTag = `${Number(preferredSteps) || 4}step`;
  for (let index = 0; index < installedLoras.length; index++) {
    const value = String(installedLoras[index] || "");
    const normalized = lower(value).replaceAll("\\", "/");
    const compact = normalized.replace(/[^a-z0-9]/g, "");
    if (normalized.includes(PREFERRED_H3_REF_TURBO_LORA)) return value;
    if (!compact.includes("minimaxh3") || (!compact.includes("ref2v") && !compact.includes("ref2va"))) continue;
    let score = 0;
    if (compact.includes("turbo")) score += 80;
    if (compact.includes(stepTag)) score += 40;
    if (compact.includes("comfyui")) score += 10;
    if (compact.includes("bf16")) score += 5;
    ranked.push({ value, score, index });
  }
  ranked.sort((a, b) => b.score - a.score || a.index - b.index);
  return ranked[0]?.value || "";
}
const NUMERIC_WIDGETS = new Set([
  "百万像素", "尺寸倍数", "时长秒", "分镜数", "随机种子", "采样步数", "降噪强度", "Llama上下文",
  "视频SigmaShift", "音频SigmaShift", "LoRA1强度", "LoRA2强度",
]);
const NUMERIC_RULES = {
  百万像素: [0.4, 0.2, 2.0], 尺寸倍数: [32, 32, 32], 时长秒: [5, 1, 15], 分镜数: [2, 1, 12],
  随机种子: [470115107471061, 0, Number.MAX_SAFE_INTEGER], 采样步数: [6, 1, 100],
  降噪强度: [1, 0, 1], Llama上下文: [8192, 1024, 131072],
  视频SigmaShift: [10, 0, 100], 音频SigmaShift: [3, 0, 100],
  LoRA1强度: [0.75, -4, 4], LoRA2强度: [1, -4, 4],
};
function setW(node, name, value) {
  const target = w(node, name);
  if (!target) return;
  if (NUMERIC_WIDGETS.has(name) || typeof target.value === "number") value = Number(value);
  if (typeof target.value === "boolean") value = Boolean(value);
  target.value = value;
  target.callback?.(value);
  node.setDirtyCanvas?.(true, true);
}
function hide(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
  if (widget.element) widget.element.style.display = "none";
}
function repairNumber(node, name, fallback) {
  const widget = w(node, name);
  if (!widget) return;
  const [safeFallback, min, max] = NUMERIC_RULES[name] || [fallback, -Infinity, Infinity];
  const raw = widget.value;
  const numeric = Number(widget.value);
  const valid = raw !== "" && raw !== null && raw !== undefined && Number.isFinite(numeric) && numeric >= min && numeric <= max;
  setW(node, name, valid ? numeric : safeFallback);
}
function repairChoice(node, name, fallback) {
  const widget = w(node, name);
  const choices = values(widget);
  if (!widget || choices.includes(widget.value)) return;
  setW(node, name, choices.includes(fallback) ? fallback : choices[0]);
}
function rememberWidgetChoice(widget, value) {
  if (!widget || !value || value === "未选择") return;
  const choices = values(widget);
  if (Array.isArray(choices) && !choices.includes(value)) choices.push(value);
}
function repairMainState(node) {
  for (const [name, fallback] of [
    ["百万像素", 0.4], ["尺寸倍数", 32], ["时长秒", 5], ["分镜数", 2],
    ["随机种子", 470115107471061], ["采样步数", 6], ["降噪强度", 1],
    ["Llama上下文", 8192],
  ]) repairNumber(node, name, fallback);
  repairChoice(node, "画面比例", "16:9 (Widescreen)");
  repairChoice(node, "采样器", "euler");
  repairChoice(node, "调度器", "simple");
  repairChoice(node, "参考图尺寸", "match");
  repairChoice(node, "Llama运算设备", "自动");
  repairChoice(node, "提示词服务", "本地 Llama");
  const promptTemplate = w(node, "提示词模板");
  repairChoice(node, "提示词模板", "官方 MiniMax H3");
  repairChoice(node, "加速方案", "参考工作流加速");
  repairChoice(node, "文本编码器类型", "minimax");
  repairChoice(node, "文本编码器设备", "default");
  repairChoice(node, "模型权重精度", "default");
  repairChoice(node, "SageAttention", "auto");
  for (const name of ["模型", "文本编码器", "视频VAE", "音频VAE"]) repairChoice(node, name, "");
  // 上传接口返回的新文件不会立即出现在节点创建时缓存的下拉选项里。
  // 不能用 repairChoice 判无效，否则 onSerialize 会把可见素材悄悄重置成“未选择”。
  for (const kind of ["图片", "视频", "音频"]) {
    for (let index = 1; index <= MEDIA[kind].count; index++) {
      const widget = w(node, `${kind}${index}`);
      if (!widget) continue;
      if (!widget.value) setW(node, `${kind}${index}`, "未选择");
      else rememberWidgetChoice(widget, widget.value);
    }
  }
  // 旧版本可能依据错误 MIME 把 .mp4 放进图片槽。按真实扩展名重新分桶并写回工作流。
  const repairedMedia = { 图片: [], 视频: [], 音频: [] };
  for (const declaredKind of Object.keys(MEDIA)) {
    for (const filename of selected(node, declaredKind)) {
      const actualKind = inferKindFromName(filename) || declaredKind;
      if (repairedMedia[actualKind].length < MEDIA[actualKind].count) repairedMedia[actualKind].push(filename);
    }
  }
  for (const kind of Object.keys(MEDIA)) writeSelected(node, kind, repairedMedia[kind]);
  repairChoice(node, "LoRA1", "无");
  repairChoice(node, "LoRA2", "无");
  const editProfile = w(node, "视频编辑模式");
  if (["快速6步", "极速6步", "快速创意编辑6步（弱保留）"].includes(String(editProfile?.value || ""))) {
    setW(node, "视频编辑模式", "极速4步");
  } else if (String(editProfile?.value || "") === "精准人物替换20步") {
    setW(node, "视频编辑模式", "质量20步");
  } else {
    if (["均衡8步", "均衡10步"].includes(String(editProfile?.value || ""))) setW(node, "视频编辑模式", "均衡12步");
    else repairChoice(node, "视频编辑模式", "均衡12步");
  }
  repairChoice(node, "视频编辑功能", "通用编辑");
  for (const name of ["文生视频", "图生视频", "首尾帧", "视频编辑", "数字人", "双人数字人", "MV数字人", "启用提示词增强", "仅增强提示词"]) {
    const widget = w(node, name);
    if (widget && typeof widget.value !== "boolean") setW(node, name, false);
  }
}
function repairConfigState(node) {
  repairNumber(node, "视频SigmaShift", 10.0);
  repairNumber(node, "音频SigmaShift", 3.0);
  repairNumber(node, "LoRA1强度", 0.75);
  repairNumber(node, "LoRA2强度", 1.0);
  repairChoice(node, "加速方案", "参考工作流加速");
  repairChoice(node, "文本编码器类型", "minimax");
  repairChoice(node, "文本编码器设备", "default");
  repairChoice(node, "模型权重精度", "default");
  repairChoice(node, "SageAttention", "auto");
}
function selected(node, kind) {
  const list = [];
  for (let i = 1; i <= MEDIA[kind].count; i++) {
    const value = w(node, `${kind}${i}`)?.value;
    if (value && value !== "未选择") list.push(value);
  }
  return list;
}
function writeSelected(node, kind, list) {
  for (let i = 1; i <= MEDIA[kind].count; i++) {
    const value = list[i - 1] || "未选择";
    const widget = w(node, `${kind}${i}`);
    rememberWidgetChoice(widget, value);
    setW(node, `${kind}${i}`, value);
  }
}
function mediaUrl(filename) {
  return api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input`);
}
function inferKindFromName(filename) {
  const ext = String(filename || "").split("[")[0].split(".").pop()?.toLowerCase();
  if (["png", "jpg", "jpeg", "webp", "bmp", "gif"].includes(ext)) return "图片";
  if (["mp4", "mov", "webm", "mkv", "avi", "m4v"].includes(ext)) return "视频";
  if (["mp3", "wav", "flac", "m4a", "aac", "ogg"].includes(ext)) return "音频";
}
function inferKind(file) {
  const filenameKind = inferKindFromName(file.name);
  if (filenameKind) return filenameKind;
  if (file.type.startsWith("image/")) return "图片";
  if (file.type.startsWith("video/")) return "视频";
  if (file.type.startsWith("audio/")) return "音频";
}
async function upload(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("type", "input");
  body.append("overwrite", "true");
  const response = await api.fetchApi("/upload/image", { method: "POST", body });
  if (!response.ok) throw new Error(`上传失败 HTTP ${response.status}`);
  const result = await response.json();
  return result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
}
async function promptOnly(nodeId) {
  const prompt = structuredClone(await app.graphToPrompt());
  const output = prompt.output || {};
  const keep = new Set();
  const visit = (id) => {
    id = String(id);
    if (!output[id] || keep.has(id)) return;
    keep.add(id);
    for (const value of Object.values(output[id].inputs || {})) {
      if (Array.isArray(value)) visit(value[0]);
    }
  };
  visit(nodeId);
  prompt.output = Object.fromEntries(Object.entries(output).filter(([id]) => keep.has(id)));
  return prompt;
}

function addStyle() {
  if (document.getElementById("liao2049-h3-aurora-v1")) return;
  const style = document.createElement("style");
  style.id = "liao2049-h3-aurora-v1";
  style.textContent = `
.wwh3{width:100%;height:100%;overflow:visible;padding:7px 12px 12px;display:flex;flex-direction:column;gap:10px;box-sizing:border-box;color:#eaffff;font:12px/1.45 'Microsoft YaHei UI',sans-serif;background:radial-gradient(circle at 8% 0,#00dfc044,transparent 30%),radial-gradient(circle at 92% 16%,#6d54ee44,transparent 33%),linear-gradient(155deg,#04151b,#081329 52%,#150b29);border:1px solid #42dfce;border-radius:14px;box-shadow:inset 0 0 35px #00bda826,0 0 20px #5669e644}
  .wwh3 *{box-sizing:border-box}.wwh3-head{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:13px 15px;border:1px solid #62ffe5;border-radius:12px;background:linear-gradient(105deg,#087a82,#2454a1 48%,#873a99);box-shadow:0 0 18px #25d9cc44}.wwh3-title{font-size:17px;font-weight:800;letter-spacing:.5px}.wwh3-chip{padding:4px 9px;border:1px solid #89ffe9;border-radius:99px;background:#071a26aa;color:#bafff3;font-size:10px;white-space:nowrap}
  .wwh3-card{flex:none;border:1px solid #266f83;border-radius:11px;background:linear-gradient(145deg,#071923e8,#0a1024e8 65%,#160d25e8);overflow:hidden}.wwh3-card-title{display:flex;align-items:center;justify-content:space-between;padding:9px 12px;font-weight:700;background:linear-gradient(90deg,#0a4d5c,#17345f 55%,#462556);border-bottom:1px solid #287d8e}.wwh3-card-body{padding:11px;display:flex;flex-direction:column;gap:9px}
  .wwh3 label{display:flex;flex-direction:column;gap:4px;color:#a9deda;font-size:11px;min-width:0}.wwh3 textarea,.wwh3 select,.wwh3 input{width:100%;min-width:0;border:1px solid #288f95;border-radius:7px;background:#041118;color:#eaffff;padding:7px;outline:none}.wwh3 textarea:focus,.wwh3 select:focus,.wwh3 input:focus{border-color:#63f7df;box-shadow:0 0 12px #37dbc955}.wwh3 textarea{min-height:82px;resize:vertical;font:13px/1.55 Consolas,'Microsoft YaHei UI',sans-serif}.wwh3 textarea.wwh3-idea{height:164px;min-height:164px!important}.wwh3-model-file select{font-size:11px;text-overflow:ellipsis;white-space:nowrap;overflow:hidden}
  .wwh3-actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.wwh3 button{border:1px solid #339c9d;border-radius:7px;background:linear-gradient(145deg,#0b343c,#172b58);color:#eaffff;padding:7px 12px;cursor:pointer}.wwh3 button:hover{border-color:#76ffe5;box-shadow:0 0 12px #34d7c766}.wwh3 button:disabled{opacity:.45;cursor:wait}.wwh3-primary{font-weight:800!important;background:linear-gradient(95deg,#009f91,#376cd2 55%,#9c3cae)!important;border-color:#69f8df!important}.wwh3-status{color:#9ceade;font-size:11px}.wwh3-final{border-color:#5c58b8!important;background:#080d20!important}.wwh3-final.editing{border-color:#ff63d1!important}
  .wwh3-progress{display:flex;flex-direction:column;gap:5px;padding:7px 9px;border:1px solid #2b7788;border-radius:8px;background:#040f1c}.wwh3-progress[hidden]{display:none}.wwh3-progress-head{display:flex;justify-content:space-between;gap:12px;color:#a9eee4;font-size:11px}.wwh3-progress-track{height:9px;overflow:hidden;border:1px solid #285d78;border-radius:99px;background:#07101e;box-shadow:inset 0 0 8px #000}.wwh3-progress-bar{height:100%;width:0;border-radius:inherit;background:linear-gradient(90deg,#00b6a8,#387ce4 58%,#b64dc7);box-shadow:0 0 12px #3debd5;transition:width .22s ease}.wwh3-progress.indeterminate .wwh3-progress-bar{width:38%!important;animation:wwh3-progress-scan 1.15s ease-in-out infinite}.wwh3-progress.error{border-color:#d64d73}.wwh3-progress.error .wwh3-progress-bar{background:linear-gradient(90deg,#cb365f,#ff835f)}@keyframes wwh3-progress-scan{0%{transform:translateX(-110%)}100%{transform:translateX(290%)}}
  .wwh3-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.wwh3-llama-grid{grid-template-columns:minmax(0,1.3fr) minmax(0,1.3fr) minmax(105px,.65fr)}.wwh3-core-grid{grid-template-columns:minmax(125px,1fr) minmax(165px,1.22fr) minmax(88px,.68fr) minmax(118px,.86fr) minmax(168px,1.16fr) minmax(112px,.72fr)}.wwh3-field-wide{grid-column:1/-1}.wwh3-mode{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}.wwh3-mode button.on,.wwh3-submode button.on{background:linear-gradient(100deg,#008f87,#4a55c5)!important;border-color:#7fffe6!important;box-shadow:0 0 12px #35dbc866}.wwh3-edit-speed{display:grid;grid-template-columns:minmax(112px,.72fr) repeat(3,minmax(0,1fr));gap:7px;align-items:stretch;padding:8px;border:1px solid #315579;border-radius:12px;background:linear-gradient(100deg,#07111d,#0b1729 60%,#121126);box-shadow:inset 0 0 16px #010712}.wwh3-speed-caption{display:flex;flex-direction:column;justify-content:center;gap:3px;padding:2px 8px;border-right:1px solid #28435e}.wwh3-speed-caption strong{color:#d9ffff;font-size:12px;letter-spacing:.5px}.wwh3-speed-caption small{color:#718fa1;font-size:9px;white-space:nowrap}.wwh3-edit-speed button{display:flex;align-items:center;justify-content:flex-start;gap:8px;min-width:0;padding:8px 10px;border-color:#263f5d;background:#091827!important;text-align:left}.wwh3-speed-icon{display:grid;place-items:center;flex:0 0 25px;height:25px;border-radius:8px;background:#18283b;color:#a8bfd2;font-size:14px}.wwh3-speed-copy{display:flex;flex-direction:column;min-width:0;line-height:1.15}.wwh3-speed-copy b{font-size:11px;color:#e9ffff}.wwh3-speed-copy small{margin-top:3px;color:#7896a8;font-size:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.wwh3-edit-speed button.profile-fast.on{border-color:#ffb05b!important;background:linear-gradient(110deg,#593019,#20203a)!important;box-shadow:0 0 13px #ff8a3444}.wwh3-edit-speed button.profile-fast.on .wwh3-speed-icon{background:#ff9a3d;color:#1b0c00}.wwh3-edit-speed button.profile-balanced.on{border-color:#4de6d0!important;background:linear-gradient(110deg,#06483f,#162552)!important;box-shadow:0 0 13px #29d7c544}.wwh3-edit-speed button.profile-balanced.on .wwh3-speed-icon{background:#32d7c3;color:#021612}.wwh3-edit-speed button.profile-quality.on{border-color:#bd7cff!important;background:linear-gradient(110deg,#3a225b,#1e2859)!important;box-shadow:0 0 13px #a85dff44}.wwh3-edit-speed button.profile-quality.on .wwh3-speed-icon{background:#a968f0;color:#160522}.wwh3-submode{display:flex;gap:7px;padding:7px;border:1px solid #285d78;border-radius:8px;background:#061520}.wwh3-submode:before{content:'数字人类型';align-self:center;color:#9ceade;margin-right:auto}.wwh3-submode button{min-width:110px}
  .wwh3-edit-speed{grid-template-columns:minmax(100px,.62fr) repeat(4,minmax(0,1fr))}.wwh3-custom-speed{display:flex;align-items:center;justify-content:center;gap:6px;min-width:0;padding:7px 8px;border:1px solid #314763;border-radius:7px;background:#091827;cursor:pointer}.wwh3-custom-speed.on{border-color:#ff69cf;box-shadow:0 0 13px #e54ab944;background:linear-gradient(110deg,#41203d,#17264e)}.wwh3-custom-speed span{font-weight:700;white-space:nowrap}.wwh3-custom-speed input{width:54px!important;padding:5px!important;text-align:center}.wwh3-custom-speed small{color:#7896a8;white-space:nowrap}.wwh3-enhance-head{display:grid;grid-template-columns:minmax(180px,220px) minmax(240px,320px);gap:8px;align-items:end;justify-content:start}.wwh3-switch{min-height:57px;display:flex!important;flex-direction:row!important;align-items:center;justify-content:space-between;padding:7px 10px;border:1px solid #288f95;border-radius:7px;background:#061520;color:#bdf4ed!important}.wwh3-switch input{appearance:none;width:44px!important;height:24px!important;border-radius:99px!important;padding:0!important;cursor:pointer;background:radial-gradient(circle at 11px 50%,#c6d2d4 0 7px,transparent 7.5px),#26383e!important;transition:.2s}.wwh3-switch input:checked{border-color:#5effe3!important;background:radial-gradient(circle at 32px 50%,#fff 0 7px,transparent 7.5px),linear-gradient(90deg,#00a995,#5d61e8)!important;box-shadow:0 0 12px #36e5ce88}
  .wwh3-media-toolbar{display:flex;align-items:center;gap:8px;min-height:42px;padding:6px;border:1px dashed #3a9da1;border-radius:9px;background:#050e1a}.wwh3-media-toolbar.drag{border-color:#7effe5;background:#0a333b}.wwh3-media-add{flex:none;font-weight:800!important;background:linear-gradient(95deg,#008f87,#4f55c9)!important}.wwh3-media-hint{min-width:0;flex:1;color:#9bc5ca;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.wwh3-media-count{flex:none;padding:3px 7px;border:1px solid #317b83;border-radius:99px;color:#a8f5e8;background:#061720}.wwh3-media-rail{display:flex;gap:7px;overflow-x:auto;padding:2px 1px 5px;scrollbar-width:thin;scrollbar-color:#3ee8d0 #07131d}.wwh3-media-item{position:relative;flex:0 0 92px;height:72px;border:1px solid #245f76;border-radius:8px;background:#040a13;overflow:hidden;display:flex;align-items:center;justify-content:center;color:#739da2;cursor:pointer}.wwh3-media-item img,.wwh3-media-item video{width:100%;height:100%;object-fit:cover}.wwh3-media-item audio{width:84px;height:32px}.wwh3-media-item.is-audio{flex-basis:132px}.wwh3-media-label{position:absolute;left:3px;bottom:3px;right:3px;padding:2px 4px;border-radius:4px;background:#02070bd9;color:#d8ffff;font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.wwh3-x{position:absolute;right:3px;top:3px;width:20px;height:20px;padding:0!important;border-radius:50%!important;background:#8f284e!important;border-color:#ff75a7!important;z-index:2}.wwh3-note{color:#61e4cf;font-size:10px}.wwh3-config-note{padding:9px 11px;border:1px solid #3a7890;border-radius:8px;background:linear-gradient(90deg,#082a34,#171c46);color:#aeece5}
  .wwh3-mv-timeline{display:flex;flex-direction:column;gap:8px;padding:9px;border:1px solid #286d83;border-radius:10px;background:linear-gradient(135deg,#061521,#10122a);overflow-x:auto;overscroll-behavior:contain}.wwh3-mv-timeline[hidden]{display:none}.wwh3-mv-track{display:grid;grid-template-columns:76px minmax(0,1fr);gap:8px;align-items:stretch}.wwh3-mv-track-label{position:sticky;z-index:8;left:0;display:flex;flex-direction:column;justify-content:center;padding:7px;border-right:1px solid #31516d;background:#081726;color:#bffbf1}.wwh3-mv-track-label b{font-size:11px}.wwh3-mv-track-label small{font-size:8px;color:#769ba9}.wwh3-mv-track-content{display:flex;min-width:0;min-height:68px;border:1px solid #285a72;border-radius:8px;background:#030c17;overflow:hidden}.wwh3-mv-clip{position:relative;min-width:0;overflow:visible;border-right:1px solid #4b7f96;background:linear-gradient(145deg,#093440,#18234b)}.wwh3-mv-clip img{width:100%;height:100%;min-height:68px;object-fit:cover;opacity:.78}.wwh3-mv-clip-info{position:absolute;left:4px;right:4px;bottom:4px;display:flex;align-items:center;justify-content:space-between;gap:4px;padding:2px 4px;border-radius:5px;background:#02080ddb;color:#eaffff;font-size:9px}.wwh3-mv-clip-info input{width:52px!important;padding:2px 3px!important;font-size:9px;text-align:right}.wwh3-mv-boundary{position:absolute;z-index:5;right:-5px;top:0;width:10px;height:100%;padding:0!important;border:0!important;border-radius:0!important;background:linear-gradient(90deg,transparent,#5ff5df 45%,#5ff5df 55%,transparent)!important;cursor:ew-resize;touch-action:none}.wwh3-mv-audio{display:flex;align-items:center;gap:8px;width:100%;padding:8px}.wwh3-mv-audio audio{width:100%;height:38px}.wwh3-mv-empty{display:grid;place-items:center;flex:1;color:#698f9c;font-size:10px}.wwh3-mv-total{color:#73ead8;font-size:9px;white-space:nowrap}
  .wwh3-prompt-editor{position:relative;display:flex;flex-direction:column;gap:4px}.wwh3-prompt-toolbar{display:flex;align-items:center;justify-content:space-between;gap:8px;color:#a9deda;font-size:11px}.wwh3-prompt-tools{display:flex;align-items:center;gap:6px}.wwh3-clear-prompt,.wwh3-latent-refine{padding:3px 10px!important;min-height:24px;font-size:10px}.wwh3-clear-prompt{border-color:#725173!important;background:linear-gradient(145deg,#32172c,#172345)!important;color:#ffd8ef!important}.wwh3-latent-field{min-width:0}.wwh3-latent-box{display:flex;gap:4px;height:34px}.wwh3-latent-field .wwh3-latent-refine{flex:0 0 60px;white-space:nowrap;border-color:#41698e!important;background:linear-gradient(145deg,#09283a,#202454)!important;color:#bfefff!important}.wwh3-latent-refine.on{border-color:#65ffe1!important;background:linear-gradient(100deg,#087d78,#4056bf)!important;color:#fff!important;box-shadow:0 0 10px #31dcc477}.wwh3-refine-method{display:grid;grid-template-columns:1fr 1fr;flex:1;min-width:0;padding:2px;border:1px solid #334c72;border-radius:7px;background:#050f1e}.wwh3-refine-method button{min-width:0;padding:3px 5px!important;border:0!important;background:transparent!important;color:#7597a9;font-size:9px;white-space:nowrap}.wwh3-refine-method button.on{background:linear-gradient(100deg,#087d78,#4056bf)!important;color:#fff;box-shadow:0 0 7px #31dcc455}.wwh3-refine-method.off{opacity:.48}.wwh3-prompt-refs{display:flex;gap:6px;overflow-x:auto;padding:6px 1px 1px}.wwh3-prompt-refs:empty{display:none}.wwh3-prompt-ref{display:flex;align-items:center;gap:6px;flex:0 0 auto;max-width:180px;padding:4px 7px;border:1px solid #35aa9f;border-radius:8px;background:linear-gradient(110deg,#073039,#171d48);color:#dffff9}.wwh3-prompt-ref img,.wwh3-prompt-ref video{width:30px;height:30px;border-radius:5px;object-fit:cover}.wwh3-prompt-ref b{font-size:10px;white-space:nowrap}.wwh3-mention-picker{position:absolute;left:0;right:0;top:100%;z-index:30;display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px;max-height:220px;overflow:auto;padding:8px;border:1px solid #57e4d2;border-radius:10px;background:#061323f5;box-shadow:0 12px 30px #000b,0 0 18px #31d7c955}.wwh3-mention-picker[hidden]{display:none}.wwh3-mention-option{display:flex!important;align-items:center!important;justify-content:flex-start!important;gap:8px;min-width:0;padding:6px!important;text-align:left}.wwh3-mention-option img,.wwh3-mention-option video{flex:0 0 46px;width:46px;height:40px;border-radius:6px;object-fit:cover}.wwh3-mention-audio{display:grid;place-items:center;flex:0 0 46px;height:40px;border-radius:6px;background:#112b46;color:#72ffe5;font-size:18px}.wwh3-mention-copy{display:flex;flex-direction:column;min-width:0}.wwh3-mention-copy b{color:#eaffff}.wwh3-mention-copy small{overflow:hidden;color:#8fbfc4;white-space:nowrap;text-overflow:ellipsis}
  .wwh3 details{flex:none;border:1px solid #285d78;border-radius:9px;overflow:hidden}.wwh3 details[open]{height:auto!important;max-height:none!important}.wwh3 summary{padding:6px 9px;min-height:28px;line-height:15px;cursor:pointer;background:linear-gradient(90deg,#0b3546,#24214e);font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.wwh3 details .wwh3-grid{padding:10px}.wwh3-final{height:360px;min-height:320px!important;resize:vertical!important;overflow:auto!important}
  .wwh3-enhance-head{grid-template-columns:minmax(165px,200px) minmax(190px,240px) minmax(210px,280px) auto}.wwh3-dep-btn{height:34px;white-space:nowrap;align-self:end;background:linear-gradient(145deg,#173451,#3d2867)!important}.wwh3-dep-btn.ok{border-color:#58f3c9!important;color:#9fffe7}.wwh3-dep-btn.bad{border-color:#ff7b9d!important;color:#ffd2df}
  .wwh3-mv-track{grid-template-columns:64px minmax(0,1fr)}.wwh3-mv-track.is-pictures .wwh3-mv-track-content{height:86px;min-height:86px;max-height:86px}.wwh3-mv-track.is-pictures .wwh3-mv-clip{height:84px;min-height:84px;max-height:84px}.wwh3-mv-track.is-pictures .wwh3-mv-clip img{display:block;height:84px!important;min-height:0!important;max-height:84px!important}.wwh3-mv-track.is-music .wwh3-mv-track-content{height:88px;min-height:88px;max-height:88px}.wwh3-mv-track.is-music .wwh3-mv-audio{position:relative;display:grid;grid-template-rows:52px 25px;gap:2px;height:86px;padding:4px 36px 3px 5px}.wwh3-mv-wave{position:relative;overflow:hidden;border:1px solid #24384b;border-radius:5px;background:#02070d;cursor:crosshair}.wwh3-mv-wave canvas{display:block;width:100%;height:50px}.wwh3-mv-audio-controls{display:flex;align-items:center;gap:8px;color:#99afba}.wwh3-mv-play{display:grid!important;place-items:center;width:23px!important;height:23px;padding:0!important;border-radius:50%!important;background:#f6fbff!important;color:#07101a!important;border-color:#fff!important;font-size:10px}.wwh3-mv-time{font:10px/1.1 Consolas,monospace;color:#8fa6b1}.wwh3-mv-audio>.wwh3-x{position:absolute!important;right:5px!important;top:5px!important}
  .wwh3-mv-track.is-music .wwh3-mv-audio{width:100%;padding:0;grid-template-rows:58px 26px}.wwh3-mv-track.is-music .wwh3-mv-wave{width:100%;border-radius:7px 7px 2px 2px}.wwh3-mv-track.is-music .wwh3-mv-wave canvas{height:56px}.wwh3-mv-wave.is-committed{box-shadow:inset 0 0 0 1px #52f4dd}.wwh3-mv-audio-controls{display:flex;align-items:center;gap:7px;min-width:0;padding:0 6px;overflow:hidden}.wwh3-mv-cut{flex:0 0 auto!important;display:grid!important;place-items:center;min-width:29px!important;width:29px!important;height:23px!important;padding:0!important;border-color:#65fff0!important;background:linear-gradient(135deg,#0799a1,#7656e5)!important;color:#fff!important;font-size:15px!important;font-weight:700!important;box-shadow:0 0 9px #23ddcb66!important}.wwh3-mv-extra-audio{position:relative;display:grid;grid-template-columns:minmax(180px,1fr) auto auto;align-items:center;gap:5px 8px;width:100%;padding:5px 34px 5px 8px}.wwh3-mv-extra-audio audio{display:none}.wwh3-mv-extra-wave{position:relative;grid-column:1/-1;height:29px;border:1px solid #28465a;border-radius:6px;background:repeating-linear-gradient(90deg,#2878c9 0 2px,transparent 2px 5px),linear-gradient(#071421,#02070d);overflow:hidden}.wwh3-mv-extra-wave.is-committed{border-color:#52f4dd;box-shadow:inset 0 0 10px #1dd9c833}.wwh3-mv-extra-wave.is-committed input{display:none}.wwh3-mv-extra-wave input[type=range]{position:absolute;inset:0;width:100%!important;height:100%!important;margin:0!important;pointer-events:none;background:transparent!important;appearance:none}.wwh3-mv-extra-wave input[type=range]::-webkit-slider-thumb{width:7px;height:29px;border:1px solid #8ffdf0;border-radius:2px;background:#4be6d3;appearance:none;pointer-events:auto;cursor:ew-resize}.wwh3-mv-extra-range{display:flex;align-items:center;gap:4px;color:#a8dcd8;font-size:9px}.wwh3-mv-extra-range input{width:72px!important;padding:3px 5px!important}.wwh3-mv-extra-audio>.wwh3-x{position:absolute!important;right:5px!important;top:5px!important}.wwh3-mv-range{position:absolute;left:4px;top:4px;padding:2px 4px;border-radius:4px;background:#02080dcc;color:#9ff5e8;font:8px/1.1 Consolas,monospace;white-space:nowrap}.wwh3-mv-clip-info b{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .wwh3-mv-outside{flex:none;background:#01060b;opacity:.8}
  .wwh3-mv-continuation{display:grid;place-items:center;min-width:26px;border-left:1px dashed #56d9c8;background:repeating-linear-gradient(135deg,#092733,#092733 7px,#10213b 7px,#10213b 14px);color:#8eeadd;font-size:9px;text-align:center}.wwh3-mv-track-content.is-overflow{box-shadow:inset 0 0 0 2px #ef586c}.wwh3-mv-trim{position:absolute!important;z-index:8;top:0!important;bottom:0!important;width:10px!important;min-width:10px!important;height:100%!important;padding:0!important;border:0!important;border-radius:0!important;transform:translateX(-50%);background:linear-gradient(90deg,transparent 35%,#52f4dd 35%,#52f4dd 65%,transparent 65%)!important;cursor:ew-resize}.wwh3-mv-trim::after{content:"";position:absolute;left:1px;top:1px;width:8px;height:8px;border-radius:2px;background:#52f4dd}.wwh3-mv-selection{min-width:0;overflow:hidden;text-overflow:ellipsis;color:#72e4d4;font:9px/1.1 Consolas,monospace;white-space:nowrap}
  .wwh3-media-item.is-audio{flex-basis:92px;background:linear-gradient(145deg,#082b39,#18234b)}.wwh3-media-item.is-selected{border-color:#70ffe7;box-shadow:0 0 0 2px #35ead0,0 0 14px #35ead066}.wwh3-media-audio-icon{display:grid;place-items:center;width:44px;height:44px;border:1px solid #42aaad;border-radius:50%;background:linear-gradient(145deg,#0d5961,#4a43a2);color:#eaffff;font-size:23px;box-shadow:0 0 12px #29d8c755}
  `;
  document.head.append(style);
}

function field(node, name, label = name) {
  const widget = w(node, name);
  const wrap = document.createElement("label");
  wrap.append(document.createTextNode(label));
  let input;
  const options = values(widget);
  if (options.length) {
    input = document.createElement("select");
    for (const value of options) {
      const option = document.createElement("option");
      option.value = option.textContent = value;
      input.append(option);
    }
  } else {
    input = document.createElement("input");
    input.type = typeof widget?.value === "boolean" ? "checkbox" : typeof widget?.value === "number" ? "number" : "text";
    if (input.type === "checkbox") input.checked = Boolean(widget?.value);
    if (input.type === "number") {
      const rule = NUMERIC_RULES[name];
      const minimum = Number(widget?.options?.min ?? rule?.[1]);
      const maximum = Number(widget?.options?.max ?? rule?.[2]);
      const step = Number(widget?.options?.step ?? widget?.options?.precision ?? "any");
      if (Number.isFinite(minimum)) input.min = String(minimum);
      if (Number.isFinite(maximum)) input.max = String(maximum);
      input.step = Number.isFinite(step) && step > 0 ? String(step) : "any";
    }
  }
  if (input.type !== "checkbox") input.value = widget?.value ?? "";
  input.dataset.widgetName = name;
  const numberValue = () => {
    const numeric = Number(input.value);
    if (!Number.isFinite(numeric)) return null;
    const minimum = input.min === "" ? -Infinity : Number(input.min);
    const maximum = input.max === "" ? Infinity : Number(input.max);
    return Math.min(maximum, Math.max(minimum, numeric));
  };
  const liveCommit = () => {
    if (input.type !== "number") {
      setW(node, name, input.type === "checkbox" ? input.checked : input.value);
      return;
    }
    const raw = input.value.trim();
    // Keep incomplete numeric text editable (for example `0.`, `-.` or `1e-`).
    // Rewriting it on every keypress previously made decimal LoRA strengths
    // impossible to enter because `0.` was immediately changed back to `0`.
    if (!raw || raw === "-" || raw === "." || raw === "-." || /[eE][+-]?$/.test(raw)) return;
    const numeric = numberValue();
    if (numeric === null) return;
    setW(node, name, numeric);
    // Preserve normal in-range editing, but still expose hard limits immediately.
    if (numeric !== Number(raw)) input.value = String(numeric);
  };
  const finalCommit = () => {
    if (input.type !== "number") {
      liveCommit();
      return;
    }
    const numeric = numberValue();
    if (numeric === null) {
      input.value = String(widget?.value ?? "");
      return;
    }
    input.value = String(numeric);
    setW(node, name, numeric);
  };
  input.onchange = finalCommit;
  // Ctrl+Enter may queue while a numeric field still has focus. `change` only
  // fires after blur, which previously left the hidden ComfyUI widget at its
  // old value (commonly 5 seconds). Keep the real widget live as the user types.
  if (input.type === "number" || input.type === "text") input.oninput = liveCommit;
  wrap.append(input);
  return wrap;
}

const RATIO_LABELS = {
  "1:1 (Square)": "方形 1:1", "2:3 (Portrait Photo)": "竖幅照片 2:3",
  "3:2 (Photo)": "横幅照片 3:2", "3:4 (Portrait Standard)": "标准竖屏 3:4",
  "4:3 (Standard)": "标准横屏 4:3", "9:16 (Portrait Widescreen)": "竖屏 9:16",
  "16:9 (Widescreen)": "横屏 16:9", "21:9 (Ultrawide)": "超宽屏 21:9",
};
const RATIO_NUMBERS = {
  "1:1 (Square)": [1, 1], "2:3 (Portrait Photo)": [2, 3], "3:2 (Photo)": [3, 2],
  "3:4 (Portrait Standard)": [3, 4], "4:3 (Standard)": [4, 3],
  "9:16 (Portrait Widescreen)": [9, 16], "16:9 (Widescreen)": [16, 9], "21:9 (Ultrawide)": [21, 9],
};
function outputDimensions(ratio, megapixels, multiple = 32) {
  const [rw, rh] = RATIO_NUMBERS[ratio] || [16, 9];
  const scale = Math.sqrt(Number(megapixels) * 1024 * 1024 / (rw * rh));
  return [Math.max(multiple, Math.round(rw * scale / multiple) * multiple), Math.max(multiple, Math.round(rh * scale / multiple) * multiple)];
}
function ratioField(node) {
  const wrap = field(node, "画面比例", "画面比例");
  const select = wrap.querySelector("select");
  for (const option of select?.options || []) option.textContent = RATIO_LABELS[option.value] || option.value;
  return wrap;
}
function resolutionField(node) {
  const wrap = field(node, "百万像素", "输出分辨率");
  const select = wrap.querySelector("select");
  const refresh = () => {
    const ratio = String(w(node, "画面比例")?.value || "16:9 (Widescreen)");
    for (const option of select?.options || []) {
      const [width, height] = outputDimensions(ratio, option.value);
      option.textContent = `${width} × ${height}（${option.value}MP）`;
    }
  };
  refresh();
  return { wrap, refresh };
}
function storyboardCountField(node) {
  node.properties ||= {};
  if (typeof node.properties.wwh3StoryboardAuto !== "boolean") node.properties.wwh3StoryboardAuto = true;
  const wrap = document.createElement("label");
  wrap.append(document.createTextNode("分镜数"));
  const select = document.createElement("select");
  select.dataset.widgetName = "分镜数显示";
  const automatic = document.createElement("option");
  automatic.value = "auto";
  automatic.textContent = "自动（按时长）";
  select.append(automatic);
  for (let count = 1; count <= 12; count += 1) {
    const option = document.createElement("option");
    option.value = option.textContent = String(count);
    select.append(option);
  }
  const automaticCount = () => Math.max(1, Math.min(5, Math.ceil(Number(w(node, "时长秒")?.value || 5) / 3)));
  const refresh = () => {
    if (node.properties.wwh3StoryboardAuto) {
      select.value = "auto";
      setW(node, "分镜数", automaticCount());
    } else {
      const count = Math.max(1, Math.min(12, Number(w(node, "分镜数")?.value || 2)));
      select.value = String(count);
      setW(node, "分镜数", count);
    }
  };
  select.onchange = () => {
    node.properties.wwh3StoryboardAuto = select.value === "auto";
    setW(node, "分镜数", node.properties.wwh3StoryboardAuto ? automaticCount() : Number(select.value));
  };
  wrap.append(select);
  refresh();
  return { wrap, refresh };
}

function build(node) {
  const root = document.createElement("div");
  root.className = "wwh3";

  const card = () => {
    const outer = document.createElement("section");
    outer.className = "wwh3-card";
    const body = document.createElement("div");
    body.className = "wwh3-card-body";
    outer.append(body);
    root.append(outer);
    return body;
  };

  const modelDetails = document.createElement("details");
  const modelSummary = document.createElement("summary");
  modelSummary.textContent = "模型、VAE 与 LoRA";
  const modelGrid = document.createElement("div");
  modelGrid.className = "wwh3-grid";
  const modelFileField = (name, label) => {
    const item = field(node, name, label);
    item.classList.add("wwh3-model-file");
    const input = item.querySelector("select,input");
    if (input) {
      input.title = String(w(node, name)?.value || "");
      input.addEventListener("change", () => { input.title = input.value; });
    }
    return item;
  };
  modelGrid.append(
    modelFileField("模型", "MiniMax H3 模型"), modelFileField("文本编码器", "文本编码器"),
    field(node, "视频VAE", "视频 VAE"), field(node, "音频VAE", "音频 VAE"),
    field(node, "LoRA1", "LoRA 1"), field(node, "LoRA1强度", "LoRA 1 强度"),
    field(node, "LoRA2", "LoRA 2"), field(node, "LoRA2强度", "LoRA 2 强度"),
    field(node, "加速方案", "运行方案"), field(node, "SageAttention", "SageAttention"),
    field(node, "文本编码器设备", "文本编码器设备"), field(node, "模型权重精度", "模型权重精度"),
    field(node, "视频SigmaShift", "视频 SigmaShift"), field(node, "音频SigmaShift", "音频 SigmaShift"),
  );
  modelDetails.append(modelSummary, modelGrid);
  const modelSection = document.createElement("div");
  modelSection.className = "wwh3-card wwh3-model-section";
  modelSection.style.padding = "10px";
  modelSection.style.display = "flex";
  modelSection.style.flexDirection = "column";
  modelSection.style.gap = "9px";
  let applyingPerformancePreset = false;
  const manualModelFields = new Set([
    "模型", "文本编码器", "视频VAE", "音频VAE", "LoRA1", "LoRA1强度", "LoRA2", "LoRA2强度",
    "加速方案", "SageAttention", "文本编码器设备", "模型权重精度", "视频SigmaShift", "音频SigmaShift",
  ]);
  modelDetails.addEventListener("toggle", () => node.__wwh3?.resizeForContent?.());
  modelGrid.addEventListener("change", (event) => {
    const name = event.target?.dataset?.widgetName;
    if (applyingPerformancePreset || !manualModelFields.has(name)) return;
    setW(node, "自定义模型配置", true);
    node.__wwh3PresetKey = "custom";
    modelSummary.textContent = "自定义模型配置（优选档暂时失效）";
    speedButtons?.forEach?.((button) => button.classList.remove("on"));
  });
  modelSection.append(modelDetails);
  root.append(modelSection);

  const ideaBody = card();
  const ideaLabel = document.createElement("div");
  ideaLabel.className = "wwh3-prompt-editor";
  const ideaToolbar = document.createElement("div");
  ideaToolbar.className = "wwh3-prompt-toolbar";
  const ideaTitle = document.createElement("span");
  ideaTitle.textContent = "原始创意 / 修改要求";
  const clearIdea = document.createElement("button");
  clearIdea.type = "button";
  clearIdea.className = "wwh3-clear-prompt";
  clearIdea.textContent = "清空";
  const latentRefine = document.createElement("button");
  latentRefine.type = "button";
  latentRefine.className = "wwh3-latent-refine";
  latentRefine.textContent = "关闭";
  latentRefine.title = "性能折中：所选分辨率的75%首采 → 潜空间宽高各2倍 → 放大后二次精修";
  const latentRefineField = document.createElement("label");
  latentRefineField.className = "wwh3-latent-field";
  const latentBox = document.createElement("div");
  latentBox.className = "wwh3-latent-box";
  const refineMethod = document.createElement("div");
  refineMethod.className = "wwh3-refine-method";
  const refineMethodButtons = ["潜空间二采", "双模型重绘"].map((method) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = method === "潜空间二采" ? "潜空间 1.5倍" : "双模型 2倍";
    button.dataset.method = method;
    button.title = method === "潜空间二采"
      ? "原方式：75% 首采、H3 潜空间模型放大、低 Sigma 尾段精修"
      : "对比方式：首采解码、像素放大、重新编码、第二个 H3 模型低降噪重绘";
    button.onclick = () => {
      setW(node, "二采方式", method);
      if (!Boolean(w(node, "二采放大精修")?.value)) setW(node, "二采放大精修", true);
      refreshRefineControl();
      status.textContent = method === "潜空间二采"
        ? "已选择原潜空间二采"
        : "已选择双模型重绘（约 2MP 输出，耗时和显存更高）";
    };
    refineMethod.append(button);
    return button;
  });
  latentBox.append(latentRefine, refineMethod);
  latentRefineField.append(document.createTextNode("二采高清放大（更耗显存）"), latentBox);
  const promptTools = document.createElement("div");
  promptTools.className = "wwh3-prompt-tools";
  const idea = document.createElement("textarea");
  idea.className = "wwh3-idea";
  idea.placeholder = "输入创意或修改要求。可使用 @图片1、@视频1、@音频1 引用素材。";
  idea.value = w(node, "增强源提示词")?.value || w(node, "提示词")?.value || "";
  const promptReferences = document.createElement("div");
  promptReferences.className = "wwh3-prompt-refs";
  const mentionPicker = document.createElement("div");
  mentionPicker.className = "wwh3-mention-picker";
  mentionPicker.hidden = true;
  idea.oninput = () => {
    if (currentMode() === "图生视频" && Number.isInteger(node.__wwh3I2vPromptIndex)) {
      let prompts = [];
      try { prompts = JSON.parse(String(w(node, "图生分段提示词")?.value || "[]")); } catch (_) {}
      const imageCount = selected(node, "图片").slice(0, 20).length;
      while (prompts.length < imageCount) prompts.push("");
      prompts[node.__wwh3I2vPromptIndex] = idea.value;
      setW(node, "图生分段提示词", JSON.stringify(prompts));
      setW(node, "图生当前图片序号", node.__wwh3I2vPromptIndex + 1);
      setW(node, "增强源提示词", idea.value);
      return;
    }
    setW(node, "增强源提示词", idea.value);
    if (!Boolean(w(node, "启用提示词增强")?.value)) setW(node, "提示词", idea.value);
    renderPromptReferences();
    updateMentionPicker();
  };
  idea.onkeydown = (event) => {
    if (event.key === "Escape") mentionPicker.hidden = true;
  };
  idea.onblur = () => setTimeout(() => { mentionPicker.hidden = true; }, 160);
  clearIdea.onclick = () => {
    idea.value = "";
    idea.dispatchEvent(new Event("input", { bubbles: true }));
    idea.focus();
  };
  function refreshRefineControl() {
    const enabled = Boolean(w(node, "二采放大精修")?.value);
    const method = String(w(node, "二采方式")?.value || "潜空间二采");
    latentRefine.classList.toggle("on", enabled);
    latentRefine.textContent = enabled ? "开启" : "关闭";
    refineMethod.classList.toggle("off", !enabled);
    refineMethodButtons.forEach((button) => button.classList.toggle("on", button.dataset.method === method));
  }
  latentRefine.onclick = () => {
    const enabled = !Boolean(w(node, "二采放大精修")?.value);
    setW(node, "二采放大精修", enabled);
    refreshRefineControl();
    const method = String(w(node, "二采方式")?.value || "潜空间二采");
    status.textContent = !enabled ? "二采放大精修已关闭" : method === "潜空间二采"
      ? "潜空间二采已开启：原方式保持不变"
      : "双模型重绘已开启：首采解码 → 像素放大 → 第二模型低降噪重绘";
  };
  promptTools.append(clearIdea);
  ideaToolbar.append(ideaTitle, promptTools);
  ideaLabel.append(ideaToolbar, idea, promptReferences, mentionPicker);
  const actions = document.createElement("div");
  actions.className = "wwh3-actions";
  const enhance = document.createElement("button");
  enhance.className = "wwh3-primary";
  enhance.textContent = "✦ Llama 提示词增强";
  const useEnhanced = document.createElement("button");
  useEnhanced.textContent = "采用增强提示词";
  const useDirect = document.createElement("button");
  useDirect.textContent = "直接采用原始创意";
  const status = document.createElement("span");
  status.className = "wwh3-status";
  status.textContent = "GGUF 由 Liao2049 内置 Llama 直接加载";
  actions.append(enhance, useEnhanced, useDirect, status);
  const progressWrap = document.createElement("div");
  progressWrap.className = "wwh3-progress";
  progressWrap.hidden = true;
  progressWrap.setAttribute("role", "progressbar");
  progressWrap.setAttribute("aria-valuemin", "0");
  progressWrap.setAttribute("aria-valuemax", "100");
  const progressHead = document.createElement("div");
  progressHead.className = "wwh3-progress-head";
  const progressText = document.createElement("span");
  const progressPercent = document.createElement("span");
  const stopGeneration = document.createElement("button");
  stopGeneration.type = "button";
  stopGeneration.textContent = "停止当前任务";
  stopGeneration.title = "中断正在执行的 ComfyUI 任务；清理队列不会停止当前任务";
  stopGeneration.style.cssText = "margin-left:auto;padding:2px 9px;border-radius:8px;border:1px solid #e75f72;background:#381625;color:#ffd9df;cursor:pointer;font-size:11px";
  const progressTrack = document.createElement("div");
  progressTrack.className = "wwh3-progress-track";
  const progressBar = document.createElement("div");
  progressBar.className = "wwh3-progress-bar";
  progressHead.append(progressText, progressPercent, stopGeneration);
  progressTrack.append(progressBar);
  progressWrap.append(progressHead, progressTrack);
  const generationProgress = {
    active: false, promptId: "", startedAt: 0, sawSamplerProgress: false,
    phaseTimers: [], phaseLabel: "", phaseBase: null, phaseSpan: null,
  };
  let autoRunAfterEnhance = false;
  const clearPhaseTimers = () => {
    for (const timer of generationProgress.phaseTimers.splice(0)) clearTimeout(timer);
  };
  const schedulePhaseHints = () => {
    clearPhaseTimers();
    generationProgress.startedAt = Date.now();
    generationProgress.sawSamplerProgress = false;
    generationProgress.phaseLabel = "";
    generationProgress.phaseBase = null;
    generationProgress.phaseSpan = null;
    const hint = (delay, text) => generationProgress.phaseTimers.push(setTimeout(() => {
      if (!generationProgress.active || generationProgress.sawSamplerProgress) return;
      paintProgress(NaN, text);
    }, delay));
    hint(8000, "正在加载模型和应用加速配置…");
    hint(25000, "正在编码参考素材与准备音视频 Latent…");
    hint(60000, "尚未进入采样：高负载电脑可能正在显存换页，可点击停止当前任务");
    hint(180000, "等待已超过3分钟且仍无采样进度，建议停止并降低分辨率或时长");
  };
  const paintProgress = (value, text, state = "running") => {
    const numeric = Number(value);
    const determinate = Number.isFinite(numeric);
    const percent = determinate ? Math.max(0, Math.min(100, Math.round(numeric))) : 0;
    const becameVisible = progressWrap.hidden;
    progressWrap.hidden = false;
    progressWrap.classList.toggle("indeterminate", !determinate && state === "running");
    progressWrap.classList.toggle("error", state === "error");
    progressBar.style.width = `${percent}%`;
    progressText.textContent = text;
    progressPercent.textContent = determinate ? `${percent}%` : "执行中";
    progressWrap.setAttribute("aria-valuenow", String(percent));
    // The progress card is inserted into the normal flow only after execution
    // starts.  Grow the ComfyUI node at that exact transition so the bottom
    // controls can never be clipped by the old, pre-run node height.
    if (becameVisible) requestAnimationFrame(() => node.__wwh3?.resizeForContent?.());
  };
  const eventMatches = (detail) => !generationProgress.promptId
    || !detail?.prompt_id || String(detail.prompt_id) === generationProgress.promptId;
  const onExecutionStart = (event) => {
    // Also observe runs started from ComfyUI's top Run/Ctrl+Enter controls,
    // not only runs submitted by the buttons inside this node.
    if (!generationProgress.active) {
      generationProgress.active = true;
      generationProgress.promptId = String(event.detail?.prompt_id || "");
    }
    if (!eventMatches(event.detail)) return;
    paintProgress(2, "工作流已开始执行");
    schedulePhaseHints();
  };
  const onProgress = (event) => {
    if (!generationProgress.active || !eventMatches(event.detail)) return;
    generationProgress.sawSamplerProgress = true;
    clearPhaseTimers();
    const value = Number(event.detail?.value);
    const max = Number(event.detail?.max);
    const ratio = max > 0 ? value / max : NaN;
    const staged = Number.isFinite(generationProgress.phaseBase)
      && Number.isFinite(generationProgress.phaseSpan);
    const percent = staged && Number.isFinite(ratio)
      ? generationProgress.phaseBase + ratio * generationProgress.phaseSpan
      : (Number.isFinite(ratio) ? ratio * 100 : NaN);
    const phase = generationProgress.phaseLabel || "正在生成";
    paintProgress(percent, `${phase} · ${value || 0}/${max || "?"}`);
  };
  const onH3Phase = (event) => {
    if (!generationProgress.active) return;
    const detail = event.detail || {};
    generationProgress.phaseLabel = String(detail.phase || "处理中");
    generationProgress.phaseBase = Number(detail.progress);
    generationProgress.phaseSpan = Number(detail.span);
    clearPhaseTimers();
    paintProgress(
      Number.isFinite(generationProgress.phaseBase) ? generationProgress.phaseBase : NaN,
      generationProgress.phaseLabel,
    );
  };
  const onExecuting = (event) => {
    if (!generationProgress.active || !eventMatches(event.detail)) return;
    const executingNode = event.detail && typeof event.detail === "object" ? event.detail.node : event.detail;
    if (executingNode == null) {
      generationProgress.active = false;
      clearPhaseTimers();
      paintProgress(100, "生成完成", "done");
      status.textContent = "完整工作流生成完成";
    } else if (!progressWrap.classList.contains("indeterminate") && Number(progressWrap.getAttribute("aria-valuenow")) < 3) {
      paintProgress(NaN, "模型加载与节点执行中");
    }
  };
  const onExecutionError = (event) => {
    const wasEnhancing = Boolean(w(node, "仅增强提示词")?.value);
    if (wasEnhancing) {
      setW(node, "仅增强提示词", false);
      enhance.disabled = false;
      enhance.textContent = "↻ 重新生成增强提示词";
      status.textContent = "提示词增强失败，可重新运行";
    }
    if (autoRunAfterEnhance) {
      autoRunAfterEnhance = false;
      enhance.disabled = false;
      status.textContent = "提示词增强失败，已停止自动生成";
    }
    if (!generationProgress.active || !eventMatches(event.detail)) return;
    generationProgress.active = false;
    clearPhaseTimers();
    paintProgress(Number(progressWrap.getAttribute("aria-valuenow")) || 0, "生成失败，请查看错误信息", "error");
  };
  const onExecutionInterrupted = (event) => {
    const wasEnhancing = Boolean(w(node, "仅增强提示词")?.value);
    setW(node, "仅增强提示词", false);
    autoRunAfterEnhance = false;
    enhance.disabled = false;
    enhance.textContent = "↻ 重新生成增强提示词";
    if (wasEnhancing) status.textContent = "提示词增强已停止，可重新运行";
    if (!generationProgress.active || !eventMatches(event.detail)) return;
    generationProgress.active = false;
    clearPhaseTimers();
    paintProgress(Number(progressWrap.getAttribute("aria-valuenow")) || 0, "任务已停止", "error");
  };
  stopGeneration.onclick = async () => {
    if (!generationProgress.active) return void (status.textContent = "当前没有由本节点启动的生成任务");
    if (!window.confirm("确定停止当前正在运行的任务吗？\n\n清理队列只能清除等待任务；此按钮会请求中断当前执行。若底层正处于CUDA计算或显存换页，中断会在底层返回控制后生效。")) return;
    stopGeneration.disabled = true;
    paintProgress(Number(progressWrap.getAttribute("aria-valuenow")) || 0, "已请求停止，正在等待底层安全中断…", "error");
    status.textContent = "已发送中断请求；请等待当前CUDA调用返回";
    try {
      if (typeof api.interrupt === "function") await api.interrupt();
      else await api.fetchApi("/interrupt", { method: "POST" });
    } catch (error) {
      status.textContent = `中断请求失败：${error?.message || error}`;
    } finally {
      setTimeout(() => { stopGeneration.disabled = false; }, 1500);
    }
  };
  api.addEventListener("execution_start", onExecutionStart);
  api.addEventListener("progress", onProgress);
  api.addEventListener("executing", onExecuting);
  api.addEventListener("execution_error", onExecutionError);
  api.addEventListener("execution_interrupted", onExecutionInterrupted);
  api.addEventListener("liao_h3_phase", onH3Phase);
  const disposeProgressListeners = () => {
    clearPhaseTimers();
    api.removeEventListener("execution_start", onExecutionStart);
    api.removeEventListener("progress", onProgress);
    api.removeEventListener("executing", onExecuting);
    api.removeEventListener("execution_error", onExecutionError);
    api.removeEventListener("execution_interrupted", onExecutionInterrupted);
    api.removeEventListener("liao_h3_phase", onH3Phase);
    window.removeEventListener("keydown", onCtrlEnter, true);
  };
  const finalLabel = document.createElement("label");
  finalLabel.textContent = "最终 H3 提示词";
  const finalPrompt = document.createElement("textarea");
  finalPrompt.className = "wwh3-final";
  finalPrompt.value = w(node, "提示词")?.value || "";
  finalPrompt.placeholder = "增强结果会出现在这里，也可以直接编辑。";
  finalPrompt.oninput = () => setW(node, "提示词", finalPrompt.value);
  finalPrompt.addEventListener("pointerup", () => requestAnimationFrame(() => node.__wwh3?.resizeForContent?.()));
  finalLabel.append(finalPrompt);
  const llamaGrid = document.createElement("div");
  llamaGrid.className = "wwh3-grid wwh3-llama-grid";
  const enhanceHead = document.createElement("div");
  enhanceHead.className = "wwh3-enhance-head";
  const enhanceSwitch = document.createElement("label");
  enhanceSwitch.className = "wwh3-switch";
  enhanceSwitch.append(document.createTextNode("提示词增强"));
  const enhanceToggle = document.createElement("input");
  enhanceToggle.type = "checkbox";
  enhanceToggle.checked = Boolean(w(node, "启用提示词增强")?.value);
  enhanceToggle.dataset.widgetName = "启用提示词增强";
  enhanceSwitch.append(enhanceToggle);
  const serviceField = field(node, "提示词服务", "增强服务");
  const templateField = field(node, "提示词模板", "提示词模板");
  const dependencyButton = document.createElement("button");
  dependencyButton.className = "wwh3-dep-btn";
  dependencyButton.textContent = "检测依赖";
  const localModelField = field(node, "Llama模型", "请下载 Llama 模型至 LLM 文件夹");
  const visionModelField = field(node, "视觉识别模型", "视觉识别模型（mmproj）");
  const attachModelSelector = (fieldWrap, widgetName) => {
    const existing = fieldWrap.querySelector("input,select");
    if (!existing) return null;
    const select = document.createElement("select");
    select.dataset.widgetName = widgetName;
    select.onchange = () => setW(node, widgetName, select.value);
    existing.replaceWith(select);
    return {
      input: select,
      replaceOptions(names) {
        const available = Array.from(new Set(names || []));
        const current = String(w(node, widgetName)?.value || select.value || "");
        select.replaceChildren(...available.map((name) => {
          const option = document.createElement("option");
          option.value = option.textContent = name;
          return option;
        }));
        if (!available.length) {
          const empty = document.createElement("option");
          empty.value = "";
          empty.textContent = "未检测到 GGUF 模型";
          select.append(empty);
        }
        const next = available.includes(current) ? current : (available[0] || "");
        select.value = next;
        select.disabled = !available.length;
        if (next !== current) setW(node, widgetName, next);
      },
    };
  };
  const localModelSuggestions = attachModelSelector(localModelField, "Llama模型");
  const visionModelSuggestions = attachModelSelector(visionModelField, "视觉识别模型");
  let modelRefreshPromise = null;
  const refreshLlamaModels = () => {
    if (modelRefreshPromise) return modelRefreshPromise;
    modelRefreshPromise = api.fetchApi(`/liao_h3/models/llm?fresh=${Date.now()}`, {
      cache: "no-store",
    }).then(async (response) => {
      const data = await response.json();
      if (!response.ok || !data?.ok) return;
    const fill = (target, names) => {
      if (!target) return;
      target.replaceOptions(names);
    };
    fill(localModelSuggestions, data.models);
    fill(visionModelSuggestions, data.vision);
    updateEnhanceUI();
    }).catch(() => {}).finally(() => { modelRefreshPromise = null; });
    return modelRefreshPromise;
  };
  refreshLlamaModels();
  localModelSuggestions?.input.addEventListener("focus", refreshLlamaModels);
  localModelSuggestions?.input.addEventListener("pointerdown", refreshLlamaModels);
  visionModelSuggestions?.input.addEventListener("focus", refreshLlamaModels);
  visionModelSuggestions?.input.addEventListener("pointerdown", refreshLlamaModels);
  const localContextField = field(node, "Llama上下文", "上下文");
  const compatibleBaseField = field(node, "OpenAI兼容地址", "OpenAI 兼容 API 地址");
  compatibleBaseField.classList.add("wwh3-field-wide");
  const compatibleBaseInput = compatibleBaseField.querySelector("input");
  if (compatibleBaseInput) compatibleBaseInput.placeholder = "例如：http://127.0.0.1:11434/v1";
  const cloudKeyField = field(node, "云端APIKey", "云端 API Key（也可使用环境变量）");
  const cloudKeyInput = cloudKeyField.querySelector("input");
  if (cloudKeyInput) {
    cloudKeyInput.type = "password";
    cloudKeyInput.autocomplete = "off";
  }
  const cloudModelField = field(node, "云端模型", "云端模型（留空自动匹配）");
  const cloudModelTextInput = cloudModelField.querySelector("input");
  if (cloudModelTextInput) cloudModelTextInput.placeholder = "填写服务端模型 ID";
  const apiCheckButton = document.createElement("button");
  apiCheckButton.className = "wwh3-dep-btn wwh3-api-check-btn";
  apiCheckButton.textContent = "检测 API 模型";
  enhanceHead.append(enhanceSwitch, serviceField, templateField, dependencyButton);
  llamaGrid.append(
    localModelField, visionModelField, localContextField,
    compatibleBaseField,
    cloudKeyField, cloudModelField, apiCheckButton,
  );
  ideaBody.append(ideaLabel, enhanceHead, llamaGrid, actions, progressWrap, finalLabel);

  const readDependencyStatus = async () => {
    const response = await api.fetchApi("/liao_h3/dependencies");
    const data = await response.json();
    if (!response.ok) throw new Error(data?.error || `HTTP ${response.status}`);
    return data;
  };
  const paintDependencyStatus = (data) => {
    const diagnostics = data?.diagnostics || {};
    dependencyButton.classList.toggle("ok", Boolean(diagnostics.complete));
    dependencyButton.classList.toggle("bad", !diagnostics.complete && !data?.running);
    dependencyButton.textContent = data?.running ? `安装中：${data.stage || "处理中"}`
      : diagnostics.complete ? `依赖完整 · GPU` : "依赖缺失 · 安装";
    dependencyButton.title = diagnostics.complete
      ? `llama_cpp ${diagnostics.version || "unknown"}，Qwen3.5 多模态与 GPU offload 可用`
      : `Python ${diagnostics.python || "unknown"}；需要 llama-cpp-python 0.3.35+、Qwen3.5 多模态处理器与 GPU offload`;
  };
  const pollDependencyInstall = async () => {
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, 1800));
      const data = await readDependencyStatus();
      paintDependencyStatus(data);
      if (!data.running) {
        dependencyButton.disabled = false;
        if (!data.diagnostics?.complete) {
          const tail = (data.log || []).slice(-18).join("\n");
          window.alert(`Liao-H3 依赖安装未完成：\n${tail || "请查看 ComfyUI 控制台。"}`);
        } else {
          window.alert(`依赖安装完成：llama_cpp ${data.diagnostics.version}，Qwen3.5 多模态与 GPU offload 可用。请重启 ComfyUI。`);
        }
        return;
      }
    }
  };
  dependencyButton.onclick = async () => {
    dependencyButton.disabled = true;
    dependencyButton.textContent = "检测中…";
    try {
      let data = await readDependencyStatus();
      paintDependencyStatus(data);
      if (data.running) return void pollDependencyInstall();
      if (data.diagnostics?.complete) {
        window.alert(`依赖完整：llama_cpp ${data.diagnostics.version}，Qwen3.5 多模态与 GPU offload 可用。`);
        dependencyButton.disabled = false;
        return;
      }
      const agreed = window.confirm(
        `检测到本地 Llama CUDA 依赖不完整。\n\n是否自动识别当前 Python、PyTorch CUDA 与 NVIDIA 环境，并安装匹配的 JamePeng llama-cpp-python Qwen3.5 预编译 GPU 版？\n\n安装器会按 Python、Windows x64 与 CUDA 版本从发布页选择 wheel，并依次尝试 GitHub 与国内代理。找不到匹配轮子时会停止，不会安装 CPU 版、不会源码编译，也不会顺带安装无关依赖。\n\n确认后将修改当前 ComfyUI Python 环境。`
      );
      if (!agreed) {
        dependencyButton.disabled = false;
        return;
      }
      const response = await api.fetchApi("/liao_h3/dependencies/install", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm: "INSTALL_LLAMA_GPU" }),
      });
      data = await response.json();
      if (!response.ok) throw new Error(data?.error || `HTTP ${response.status}`);
      paintDependencyStatus(data);
      await pollDependencyInstall();
    } catch (error) {
      dependencyButton.disabled = false;
      dependencyButton.classList.add("bad");
      dependencyButton.textContent = "检测失败";
      window.alert(`依赖检测失败：${error?.message || error}`);
    }
  };
  const cloudErrorText = (data) => {
    const error = data?.error;
    const serialized = typeof error === "string" ? error : JSON.stringify(error || {});
    if (serialized.includes("insufficient_balance") || serialized.includes("1008")) {
      return data?.key_type === "token_plan_or_coding"
        ? "余额检测失败：当前很可能是 Token Plan/编程套餐 Key，不能消费按量充值余额。请换用“账户管理 → 接口密钥”创建的普通按量 API Key。"
        : "MiniMax 返回余额不足（1008）。请确认充值已到账，并确认 Key 属于当前充值账户/Team。";
    }
    if (serialized.includes("2049") || serialized.toLowerCase().includes("invalid") || data?.http_status === 401) {
      return "API Key 无效或与当前接口类型不匹配。";
    }
    return `检测失败（HTTP ${data?.http_status || "未知"}）：${serialized.slice(0, 500)}`;
  };
  apiCheckButton.onclick = async () => {
    const provider = String(w(node, "提示词服务")?.value || "");
    const key = String(cloudKeyInput?.value || w(node, "云端APIKey")?.value || "").trim();
    const compatibleBase = String(compatibleBaseInput?.value || w(node, "OpenAI兼容地址")?.value || "").trim();
    const modelInput = llamaGrid.querySelector('[data-widget-name="云端模型"]');
    const model = String(modelInput?.value || w(node, "云端模型")?.value || "").trim();
    if (!key && provider !== "OpenAI 兼容 API") return void window.alert("请先填写云端 API Key。");
    if (provider === "本地 Llama") return void window.alert("请先选择云端 API 服务。");
    if (provider === "OpenAI 兼容 API" && !compatibleBase) return void window.alert("请先填写 OpenAI 兼容 API 地址。");
    setW(node, "云端APIKey", key);
    apiCheckButton.disabled = true;
    apiCheckButton.textContent = "正在检测…";
    apiCheckButton.classList.remove("ok", "bad");
    try {
      const response = await api.fetchApi("/liao_h3/api/models/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, api_key: key, model, base_url: compatibleBase }),
      });
      const data = await response.json();
      if (!response.ok || !data?.ok) throw Object.assign(new Error(cloudErrorText(data)), { data });
      if (modelInput && data.selected) modelInput.value = data.selected;
      if (data.selected) setW(node, "云端模型", data.selected);
      apiCheckButton.classList.add("ok");
      apiCheckButton.textContent = `可用 · ${data.selected}`;
      apiCheckButton.title = `检测到 ${(data.models || []).length} 个模型：\n${(data.models || []).join("\n")}`;
      status.textContent = `API 可用，已选择 ${data.selected}`;
      window.alert(`API 检测通过。\n\n推荐模型：${data.selected}\n可用模型数：${(data.models || []).length}`);
    } catch (error) {
      apiCheckButton.classList.add("bad");
      apiCheckButton.textContent = "检测失败";
      const message = error?.message || "API 检测失败";
      status.textContent = message;
      window.alert(message);
    } finally {
      apiCheckButton.disabled = false;
    }
  };

  const updateEnhanceUI = () => {
    const enabled = Boolean(w(node, "启用提示词增强")?.value);
    const service = String(w(node, "提示词服务")?.value || "本地 Llama");
    const promptTemplate = String(w(node, "提示词模板")?.value || "自动匹配（按生成模式）");
    const modeName = Boolean(w(node, "MV数字人")?.value) || Boolean(w(node, "双人数字人")?.value) || Boolean(w(node, "数字人")?.value) ? "数字人"
      : ["文生视频", "图生视频", "首尾帧", "视频编辑"].find((name) => Boolean(w(node, name)?.value)) || "多参考";
    const effectiveTemplate = promptTemplate === "自动匹配（按生成模式）"
      ? modeName === "图生视频" ? "Liao 分镜模板"
        : modeName === "视频编辑" ? "官方 H3 · Ref2VA 视频编辑"
          : modeName === "首尾帧" ? "官方 H3 · FL2VA 首尾帧"
            : modeName === "数字人" ? "官方 H3 · 数字人音频锁定"
              : modeName === "多参考" ? "官方 H3 · Ref2VA 多参考"
                : "官方 H3 · T2VA 文生视频"
      : promptTemplate;
    const local = service === "本地 Llama";
    const compatible = service === "OpenAI 兼容 API";
    llamaGrid.style.display = enabled ? "grid" : "none";
    localModelField.style.display = enabled && local ? "flex" : "none";
    visionModelField.style.display = enabled && local ? "flex" : "none";
    localContextField.style.display = enabled && local ? "flex" : "none";
    cloudKeyField.style.display = enabled && !local ? "flex" : "none";
    cloudModelField.style.display = enabled && !local ? "flex" : "none";
    compatibleBaseField.style.display = enabled && compatible ? "flex" : "none";
    apiCheckButton.style.display = enabled && !local ? "inline-flex" : "none";
    const cloudModelInput = llamaGrid.querySelector('[data-widget-name="云端模型"]');
    if (enabled && cloudModelInput) {
      const currentCloudModel = String(cloudModelInput.value || "").trim();
      if (service === "MiniMax API" && (!currentCloudModel || currentCloudModel.includes("自动匹配"))) {
        cloudModelInput.value = "MiniMax-M2.7";
        setW(node, "云端模型", "MiniMax-M2.7");
      } else if (service === "Kimi API" && (!currentCloudModel || currentCloudModel.startsWith("MiniMax-"))) {
        cloudModelInput.value = "自动匹配（优先K3）";
        setW(node, "云端模型", "自动匹配（优先K3）");
      } else if (service === "OpenAI 兼容 API" && (currentCloudModel.startsWith("MiniMax-") || currentCloudModel.includes("自动匹配"))) {
        cloudModelInput.value = "";
        setW(node, "云端模型", "");
      }
    }
    enhance.textContent = local ? "✦ Llama 提示词增强" : `✦ ${service} 提示词增强`;
    actions.style.display = enabled ? "flex" : "none";
    finalLabel.style.display = enabled ? "flex" : "none";
    if (enabled) {
      const model = String(w(node, "Llama模型")?.value || "");
      status.textContent = local
        ? (model && model !== "未找到GGUF模型" ? `模板：${effectiveTemplate}` : "请下载 Llama 模型至 LLM 文件夹")
        : `模板：${effectiveTemplate} · ${service}`;
    }
    if (!enabled) {
      setW(node, "仅增强提示词", false);
      finalPrompt.value = idea.value;
      setW(node, "提示词", idea.value);
    }
    node.__wwh3?.resizeForEnhance?.(enabled);
  };
  enhanceToggle.onchange = () => {
    setW(node, "启用提示词增强", enhanceToggle.checked);
    if (!enhanceToggle.checked) setW(node, "仅增强提示词", false);
    if (enhanceToggle.checked) refreshLlamaModels();
    updateEnhanceUI();
  };
  const llamaSelect = llamaGrid.querySelector('[data-widget-name="Llama模型"]');
  llamaSelect?.addEventListener("change", () => setTimeout(updateEnhanceUI, 0));
  const serviceSelect = enhanceHead.querySelector('[data-widget-name="提示词服务"]');
  serviceSelect?.addEventListener("change", () => {
    const service = String(serviceSelect.value || "本地 Llama");
    setW(node, "提示词服务", service);
    const cloudModel = llamaGrid.querySelector('[data-widget-name="云端模型"]');
    const currentCloudModel = String(cloudModel?.value || "").trim();
    const isLegacyKimi = ["kimi-k2.5", "k2.5"].includes(currentCloudModel.toLowerCase());
    if (service !== "本地 Llama" && service !== "OpenAI 兼容 API" && cloudModel && (!currentCloudModel || (service === "Kimi API" && isLegacyKimi))) {
      cloudModel.value = service === "Kimi API" ? "自动匹配（优先K3）" : "MiniMax-M2.7";
      setW(node, "云端模型", cloudModel.value);
    } else if (service === "OpenAI 兼容 API" && cloudModel && (currentCloudModel.startsWith("MiniMax-") || currentCloudModel.includes("自动匹配"))) {
      cloudModel.value = "";
      setW(node, "云端模型", "");
    }
    updateEnhanceUI();
  });
  const templateSelect = enhanceHead.querySelector('[data-widget-name="提示词模板"]');
  if (templateSelect?.value === "Liao 视频编辑模板") {
    templateSelect.value = "官方 MiniMax H3";
    setW(node, "提示词模板", "官方 MiniMax H3");
  }
  templateSelect?.addEventListener("change", () => {
    setW(node, "提示词模板", templateSelect.value);
    updateEnhanceUI();
  });
  updateEnhanceUI();

  const continueGeneration = async (prompt, label) => {
    prompt = String(prompt || "").trim();
    if (!prompt) return void (status.textContent = label === "增强提示词" ? "请先执行提示词增强" : "请先输入原始创意");
    syncModeWidgets();
    finalPrompt.value = prompt;
    setW(node, "提示词", prompt);
    setW(node, "仅增强提示词", false);
    useEnhanced.disabled = true;
    useDirect.disabled = true;
    enhance.disabled = true;
    status.textContent = `已采用${label}，正在提交完整工作流…`;
    generationProgress.active = true;
    generationProgress.promptId = "";
    schedulePhaseHints();
    paintProgress(NaN, "正在提交生成任务");
    try {
      const queued = await app.queuePrompt(0, 1);
      generationProgress.promptId = String(queued?.prompt_id || queued?.promptId || "");
      status.textContent = `已采用${label}并开始生成`;
    } catch (error) {
      generationProgress.active = false;
      clearPhaseTimers();
      paintProgress(0, "生成任务提交失败", "error");
      status.textContent = `生成提交失败：${error?.message || error}`;
    } finally {
      useEnhanced.disabled = false;
      useDirect.disabled = false;
      enhance.disabled = false;
    }
  };
  useDirect.onclick = () => continueGeneration(idea.value, "原始创意");
  useEnhanced.onclick = () => continueGeneration(finalPrompt.value, "增强提示词");
  enhance.onclick = async () => {
    if (!idea.value.trim()) return void (status.textContent = "请先输入原始创意");
    // Prompt-only queueing bypasses the normal full-workflow queue path, so
    // synchronize the visible mode before serializing the enhancement request.
    syncModeWidgets();
    const service = String(w(node, "提示词服务")?.value || "本地 Llama");
    if (service === "本地 Llama" && (!w(node, "Llama模型")?.value || w(node, "Llama模型")?.value === "未找到GGUF模型")) {
      return void (status.textContent = "请把 GGUF 放入 ComfyUI/models/LLM 后重启");
    }
    setW(node, "增强源提示词", idea.value.trim());
    const requestSerial = (Number(w(node, "增强请求序号")?.value || 0) + 1) % 2147483647;
    setW(node, "增强请求序号", requestSerial);
    setW(node, "仅增强提示词", true);
    enhance.disabled = true;
    status.textContent = `${service} 正在理解创意和素材关系…`;
    try {
      await api.queuePrompt(-1, await promptOnly(node.id));
    } catch (error) {
      autoRunAfterEnhance = false;
      setW(node, "仅增强提示词", false);
      enhance.disabled = false;
      status.textContent = `增强失败：${error?.message || error}`;
    }
  };
  const onCtrlEnter = (event) => {
    if (event.repeat || event.key !== "Enter" || !(event.ctrlKey || event.metaKey)) return;
    if (!Boolean(w(node, "启用提示词增强")?.value)) return;
    if (!idea.value.trim()) return;
    const service = String(w(node, "提示词服务")?.value || "本地 Llama");
    const llamaReady = service !== "本地 Llama"
      || (w(node, "Llama模型")?.value && w(node, "Llama模型")?.value !== "未找到GGUF模型");
    if (!llamaReady || enhance.disabled) return;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    autoRunAfterEnhance = true;
    status.textContent = `Ctrl+Enter：先由${service}增强，再自动生成…`;
    enhance.click();
  };
  window.addEventListener("keydown", onCtrlEnter, true);

  const settingsBody = card();
  const mode = document.createElement("div");
  mode.className = "wwh3-mode";
      const modes = ["图生视频", "文生视频", "多参考", "首尾帧", "视频编辑", "数字人"];
  const modeButtons = [];
  const digitalVariant = () => Boolean(w(node, "MV数字人")?.value) ? "MV数字人" : Boolean(w(node, "双人数字人")?.value) ? "双人数字人" : "单人数字人";
  const modeFromWidgets = () => Boolean(w(node, "MV数字人")?.value) || Boolean(w(node, "双人数字人")?.value) || Boolean(w(node, "数字人")?.value) ? "数字人"
    : ["文生视频", "图生视频", "首尾帧", "视频编辑"].find((name) => Boolean(w(node, name)?.value)) || "多参考";
  node.__wwh3SelectedMode = node.__wwh3SelectedMode || modeFromWidgets();
  const currentMode = () => node.__wwh3SelectedMode || modeFromWidgets();
  const syncModeWidgets = () => {
    const selectedMode = currentMode();
    for (const option of ["文生视频", "图生视频", "首尾帧", "视频编辑"]) setW(node, option, option === selectedMode);
    if (selectedMode !== "数字人") {
      setW(node, "数字人", false);
      setW(node, "双人数字人", false);
      setW(node, "MV数字人", false);
    } else if (!Boolean(w(node, "数字人")?.value) && !Boolean(w(node, "双人数字人")?.value) && !Boolean(w(node, "MV数字人")?.value)) {
      setW(node, "数字人", true);
    }
  };
  const autoCorrectPromptTemplate = (modeName = currentMode(), forceAuto = false) => {
    const current = String(w(node, "提示词模板")?.value || "自动匹配（按生成模式）");
    const desired = forceAuto ? "自动匹配（按生成模式）" : current;
    if (desired !== current) setW(node, "提示词模板", desired);
    const visible = root.querySelector('[data-widget-name="提示词模板"]');
    if (visible) visible.value = desired;
    updateEnhanceUI();
  };
  const autoCorrectModel = (modeName = currentMode()) => {
    const modelWidget = w(node, "模型");
    const installed = values(modelWidget).map(String);
    if (!modelWidget || !installed.length) return "";
    const lower = (value) => String(value || "").replaceAll("\\", "/").toLowerCase();
    const current = String(modelWidget.value || "");
    if (modeName === "视频编辑") return current;
    const needsFl = ["文生视频", "图生视频", "首尾帧"].includes(modeName);
    const correctFamily = needsFl ? "minimax_h3_fl2va" : "minimax_h3_ref2va";
    let chosen = current;
    if (!lower(current).includes(correctFamily)) {
      const family = installed.filter((value) => lower(value).includes(correctFamily));
      chosen = needsFl
        ? family.find((value) => lower(value).includes("int8_convrot")) || family[0] || ""
        : family.find((value) => lower(value).includes("int8_convrot") && !lower(value).includes("pruned"))
          || family.find((value) => lower(value).includes("int8_convrot")) || family[0] || "";
    }
    if (!chosen || chosen === current) return current;
    setW(node, "模型", chosen);
    const visibleModel = root.querySelector('[data-widget-name="模型"]');
    if (visibleModel) visibleModel.value = chosen;
    modelSummary.textContent = `已按${modeName}自动选择：${chosen.split(/[\\/]/).pop()}`;
    return chosen;
  };
    const applyPerformancePreset = (modeName = currentMode(), profile = String(w(node, "视频编辑模式")?.value || "均衡12步")) => {
    applyingPerformancePreset = true;
    setW(node, "自定义模型配置", false);
    try {
    const modelWidget = w(node, "模型");
    const installedModels = values(modelWidget).map(String);
    const lower = (value) => String(value || "").replaceAll("\\", "/").toLowerCase();
    const compact = (value) => lower(value).replace(/[^a-z0-9]/g, "");
    const flFamily = ["文生视频", "图生视频", "首尾帧"].includes(modeName)
      || (modeName === "多参考" && profile === "极速4步");
    const familyTag = flFamily ? "minimaxh3fl2va" : "minimaxh3ref2va";
    const h3Models = installedModels.filter((value) => {
      const name = compact(value);
      return name.includes("minimaxh3") && (name.includes("fl2va") || name.includes("ref2va"));
    });
    const familyModels = h3Models.filter((value) => compact(value).includes(familyTag));
    const pick = (...predicates) => {
      for (const predicate of predicates) {
        const match = familyModels.find((value) => predicate(lower(value)));
        if (match) return match;
      }
      return familyModels[0] || "";
    };
    let chosen = "";
    if (profile === "极速4步") {
      chosen = pick(
        (value) => value.includes("kijai_") && value.includes("w4a8_mixed"),
        (value) => value.includes("w4a8"),
        (value) => value.includes("int8_convrot"),
        (value) => value.includes("bf16"),
      );
    } else if (profile === "质量20步") {
      chosen = pick(
        (value) => value.includes("comfy-org_") && value.includes("bf16"),
        (value) => value.includes("bf16"),
        (value) => value.includes("int8_convrot") && !value.includes("pruned"),
        (value) => value.includes("int8_convrot"),
        (value) => value.includes("w4a8"),
      );
    } else {
      chosen = pick(
        (value) => value.includes("int8_convrot") && !value.includes("pruned"),
        (value) => value.includes("int8_convrot"),
        (value) => value.includes("w4a8"),
        (value) => value.includes("bf16"),
      );
    }
    if (chosen) setW(node, "模型", chosen);
    setW(node, "模型权重精度", "default");
    const loraWidget = w(node, "LoRA1");
    const installedLoras = values(loraWidget).map(String);
    let selectedTurbo = "";
      if (profile === "极速4步" || profile === "均衡12步") {
        const turbo = modeName === "多参考"
          ? pickMinimaxH3RefTurboLora(installedLoras, profile === "均衡12步" ? 8 : 4)
          : (profile === "均衡12步" ? pickMinimaxH3BalancedLora(installedLoras) : pickMinimaxH3TurboLora(installedLoras));
      selectedTurbo = turbo || "";
      setW(node, "LoRA1", turbo || "无");
      setW(node, "LoRA1强度", turbo ? 0.75 : 1.0);
    } else {
      setW(node, "LoRA1", "无");
      setW(node, "LoRA1强度", 1.0);
    }
    setW(node, "LoRA2", "无");
    const visibleModel = root.querySelector('[data-widget-name="模型"]');
    if (visibleModel && chosen) visibleModel.value = chosen;
    for (const name of ["LoRA1", "LoRA1强度", "LoRA2", "模型权重精度"]) {
      const visible = root.querySelector(`[data-widget-name="${name}"]`);
      if (visible) visible.value = w(node, name)?.value ?? "";
    }
    const familyLabel = flFamily ? "FL2VA" : "Ref2VA";
    const chosenLower = lower(chosen);
    const precision = chosenLower.includes("bf16") ? "BF16" : chosenLower.includes("int8") ? "INT8" : chosenLower.includes("w4a8") ? "W4A8" : "已有模型";
    const label = `${familyLabel} ${precision}${selectedTurbo ? " + LoRA 0.75" : ""}`;
    modelSummary.textContent = `${modeName} ${profile}：${label}${chosen ? ` · ${chosen.split(/[\\/]/).pop()}` : "（未找到匹配模型）"}`;
    const profileButton = speedButtons?.find?.((button) => button.dataset.profile === profile);
    const detail = profileButton?.querySelector("small");
    if (detail) detail.textContent = `${profile.match(/\d+/)?.[0] || ""}步 · ${precision}${selectedTurbo ? " + 加速" : ""}`;
    node.__wwh3PresetKey = `${modeName}|${profile}`;
    return chosen;
    } finally {
      applyingPerformancePreset = false;
    }
  };
  const selectMode = (name) => {
    node.__wwh3I2vPromptIndex = null;
    setW(node, "图生当前图片序号", 1);
    node.__wwh3SelectedMode = name;
    for (const option of ["文生视频", "图生视频", "首尾帧", "视频编辑"]) setW(node, option, option === name);
    if (name === "数字人") {
      if (!Boolean(w(node, "数字人")?.value) && !Boolean(w(node, "双人数字人")?.value) && !Boolean(w(node, "MV数字人")?.value)) setW(node, "数字人", true);
    } else {
      setW(node, "数字人", false);
      setW(node, "双人数字人", false);
      setW(node, "MV数字人", false);
    }
    modeButtons.forEach((button) => button.classList.toggle("on", button.dataset.mode === name));
    if (!Boolean(w(node, "自定义模型配置")?.value)) {
      autoCorrectModel(name);
      applyPerformancePreset(name);
    }
    autoCorrectPromptTemplate(name, true);
    renderMedia();
  };
  for (const name of modes) {
    const button = document.createElement("button");
    button.dataset.mode = name;
    button.textContent = name;
    button.onclick = () => selectMode(name);
    mode.append(button);
    modeButtons.push(button);
  }
  const i2vSubmode = document.createElement("div");
  i2vSubmode.className = "wwh3-submode wwh3-i2v-submode";
  const i2vLabelStyle = document.createElement("style");
  i2vLabelStyle.textContent = ".wwh3-i2v-submode:before{content:'图生视频模式'!important}";
  i2vSubmode.append(i2vLabelStyle);
  const i2vButtons = [];
  for (const [label, enabled] of [["单图模式", false], ["连续拼接模式", true]]) {
    const button = document.createElement("button");
    button.textContent = label;
    button.onclick = () => {
      setW(node, "图生连续拼接", enabled);
      node.__wwh3I2vPromptIndex = enabled ? 0 : null;
      setW(node, "图生当前图片序号", 1);
      if (enabled) {
        let prompts = [];
        try { prompts = JSON.parse(String(w(node, "图生分段提示词")?.value || "[]")); } catch (_) {}
        idea.value = String(prompts[0] || "");
        setW(node, "增强源提示词", idea.value);
        idea.placeholder = "图片1 · 输入本段提示词（实时保存）";
      }
      if (!enabled) {
        const images = selected(node, "图片");
        if (images.length > 1) writeSelected(node, "图片", images.slice(0, 1));
      }
      renderMedia();
    };
    i2vSubmode.append(button); i2vButtons.push([button, enabled]);
  }
  const digitalSubmode = document.createElement("div");
  digitalSubmode.className = "wwh3-submode";
  const digitalButtons = [];
  for (const name of ["单人数字人", "双人数字人", "MV数字人"]) {
    const button = document.createElement("button");
    button.textContent = name === "单人数字人" ? "单人" : name === "双人数字人" ? "双人" : "MV模式";
    button.dataset.mode = name;
    button.onclick = () => {
      setW(node, "数字人", name === "单人数字人");
      setW(node, "双人数字人", name === "双人数字人");
      setW(node, "MV数字人", name === "MV数字人");
      if (!Boolean(w(node, "自定义模型配置")?.value)) {
        autoCorrectModel("数字人");
        applyPerformancePreset("数字人");
      }
      autoCorrectPromptTemplate("数字人", true);
      renderMedia();
    };
    digitalSubmode.append(button);
    digitalButtons.push(button);
  }
  const videoEditSubmode = document.createElement("div");
  videoEditSubmode.className = "wwh3-submode wwh3-video-edit-tools";
  const videoEditToolStyle = document.createElement("style");
  videoEditToolStyle.textContent = ".wwh3-video-edit-tools:before{content:'视频编辑功能'}";
  videoEditSubmode.append(videoEditToolStyle);
  const videoEditToolButtons = [];
  const videoEditPromptDefaults = {
    "去除字幕": "处理@视频1：仅去除画面中可见的字幕和说明文字，逐帧重建被文字遮挡的背景。完整保留人物、动作、服装、场景、构图、镜头运动、色彩、光线和原视频音频，不裁切画面，不添加新文字。",
    "动作迁移": "以@图片1中的完整角色为目标主体，让该角色复现@视频1中的全部动作、姿势变化、表情节奏和镜头时序。只参考@视频1的动作与时间结构，不保留源视频人物、服装和身份。需要额外场景时，请上传素材后手动引用对应图片。",
    "角色替换": "用@图片1中的完整角色替换@视频1里的主要人物。完整采用@图片1人物的脸部、发型、体型和服装；保留@视频1的动作、表情时序、镜头、背景、光线和音频，彻底排除源视频人物及其服装。",
  };
  for (const name of ["通用编辑", "去除字幕", "动作迁移", "角色替换"]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = name;
    button.dataset.tool = name;
    button.title = name === "动作迁移"
      ? "图片定义目标主体，视频只提供动作、节奏和镜头时序"
      : name === "角色替换"
        ? "只替换视频主角身份，保留原视频服装、动作、背景和音频"
        : name === "去除字幕"
          ? "只重建字幕遮挡区域，其余画面与音频保持不变"
          : "按输入文字执行一般视频编辑";
    button.onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      setW(node, "视频编辑功能", name);
      const defaultPrompt = videoEditPromptDefaults[name];
      if (defaultPrompt) {
        // Dedicated edit tools own their verified Ref2VA recipe. A stale
        // custom-model flag must not bypass model/LoRA selection.
        setW(node, "自定义模型配置", false);
        setW(node, "视频编辑模式", "均衡12步");
        node.__wwh3PresetKey = "";
        idea.value = defaultPrompt;
        finalPrompt.value = defaultPrompt;
        setW(node, "增强源提示词", defaultPrompt);
        setW(node, "提示词", defaultPrompt);
        setW(node, "仅增强提示词", false);
        status.textContent = `${name}提示词已写入，可直接修改后运行`;
      }
      renderMedia();
    };
    videoEditSubmode.append(button);
    videoEditToolButtons.push(button);
  }
  const videoEditProfile = document.createElement("div");
  videoEditProfile.className = "wwh3-edit-speed";
  const speedCaption = document.createElement("div");
  speedCaption.className = "wwh3-speed-caption";
  speedCaption.innerHTML = "<strong>模型配置</strong><small>按本机模型自动分档</small>";
  speedCaption.style.cursor = "pointer";
  speedCaption.title = "点击展开模型、VAE 与 LoRA 选项";
  speedCaption.onclick = () => {
    modelDetails.open = true;
    node.__wwh3?.resizeForContent?.();
  };
  videoEditProfile.append(speedCaption);
  const speedButtons = [];
  const speedMeta = {
    "极速4步": ["⚡", "极速", "4步 · Turbo 0.75", "profile-fast"],
      "均衡12步": ["◐", "均衡", "12步 · Turbo 8step 0.75", "profile-balanced"],
    "质量20步": ["◆", "质量", "20步 · BF16", "profile-quality"],
  };
    for (const name of ["极速4步", "均衡12步", "质量20步"]) {
    const button = document.createElement("button");
    button.type = "button";
    const [icon, title, detail, className] = speedMeta[name];
    button.className = className;
    button.innerHTML = `<span class="wwh3-speed-icon">${icon}</span><span class="wwh3-speed-copy"><b>${title}</b><small>${detail}</small></span>`;
    button.dataset.profile = name;
    button.onclick = (event) => {
      event.preventDefault();
      event.stopPropagation();
      // Switch the visible state first. Model/LoRA matching must never make
      // a performance button appear locked when an optional file is missing.
      setW(node, "自定义模型配置", false);
      speedButtons.forEach((item) => item.classList.toggle("on", item.dataset.profile === name));
      customSpeed.classList.remove("on");
      setW(node, "视频编辑模式", name);
      try {
        applyPerformancePreset(currentMode(), name);
      } catch (error) {
        node.__wwh3PresetKey = `${currentMode()}|${name}`;
        modelSummary.textContent = `${name} 已选择；模型自动匹配失败：${error?.message || error}`;
        console.error("[Liao2049 H3] performance preset failed", error);
      }
      renderMedia();
    };
    videoEditProfile.append(button);
    speedButtons.push(button);
  }
  const customSpeed = document.createElement("div");
  customSpeed.className = "wwh3-custom-speed";
  const customSpeedLabel = document.createElement("span");
  customSpeedLabel.textContent = "自定义";
  const customSpeedInput = document.createElement("input");
  customSpeedInput.type = "number";
  customSpeedInput.min = "1";
  customSpeedInput.max = "100";
  customSpeedInput.step = "1";
  customSpeedInput.value = String(w(node, "采样步数")?.value || 10);
  const customSpeedUnit = document.createElement("small");
  customSpeedUnit.textContent = "步";
  const activateCustomSpeed = () => {
    const steps = Math.max(1, Math.min(100, Math.round(Number(customSpeedInput.value) || 10)));
    customSpeedInput.value = String(steps);
    setW(node, "采样步数", steps);
    setW(node, "自定义模型配置", true);
    node.__wwh3PresetKey = "custom";
    speedButtons.forEach((button) => button.classList.remove("on"));
    customSpeed.classList.add("on");
    modelDetails.open = true;
    modelSummary.textContent = `自定义模型配置 · ${steps}步（优选档暂时失效）`;
    node.__wwh3?.resizeForContent?.();
  };
  customSpeed.onclick = (event) => { if (event.target !== customSpeedInput) activateCustomSpeed(); };
  customSpeedInput.onchange = activateCustomSpeed;
  customSpeedInput.oninput = activateCustomSpeed;
  customSpeed.append(customSpeedLabel, customSpeedInput, customSpeedUnit);
  videoEditProfile.append(customSpeed);
  modelSection.append(videoEditProfile);
  const coreGrid = document.createElement("div");
  coreGrid.className = "wwh3-grid wwh3-core-grid";
  const ratioControl = ratioField(node);
  const resolutionControl = resolutionField(node);
  const durationControl = field(node, "时长秒", "时长（秒）");
  const storyboardControl = storyboardCountField(node);
  ratioControl.querySelector("select")?.addEventListener("change", resolutionControl.refresh);
  const durationInput = durationControl.querySelector("input,select");
  durationInput?.addEventListener("change", storyboardControl.refresh);
  const applyDurationToTimeline = () => {
    const isMv = currentMode() === "数字人" && digitalVariant() === "MV数字人";
    const isI2vSequence = currentMode() === "图生视频" && Boolean(w(node, "图生连续拼接")?.value);
    if (!isMv && !isI2vSequence) return;
    const minimum = isI2vSequence ? 3 : 2;
    const clipDuration = Math.max(minimum, Math.min(15, Number(w(node, "时长秒")?.value || 5)));
    const imageCount = selected(node, "图片").slice(0, 20).length;
    if (imageCount) writeMvDurations(Array(imageCount).fill(clipDuration));
    renderMvTimeline();
  };
  durationInput?.addEventListener("change", applyDurationToTimeline);
  durationInput?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    event.currentTarget.blur();
    applyDurationToTimeline();
  });
  coreGrid.append(ratioControl, resolutionControl.wrap, durationControl, storyboardControl.wrap, latentRefineField, field(node, "随机种子", "随机种子"));
  settingsBody.append(mode, i2vSubmode, digitalSubmode, videoEditSubmode, coreGrid);

  const mediaBody = card();
  const note = document.createElement("div");
  note.className = "wwh3-note";
  note.textContent = "一个入口混合上传图片、视频和音频；素材按类型自动编号，可同时使用。";
  const limits = () => {
    const m = currentMode();
    if (m === "文生视频") return { 图片: 0, 视频: 0, 音频: 0 };
    if (m === "图生视频") return { 图片: Boolean(w(node, "图生连续拼接")?.value) ? 20 : 1, 视频: 0, 音频: 0 };
    if (m === "首尾帧") return { 图片: 2, 视频: 0, 音频: 0 };
    if (m === "视频编辑") return { 图片: 9, 视频: 3, 音频: 3 };
    if (m === "数字人" && digitalVariant() === "单人数字人") return { 图片: 1, 视频: 0, 音频: 1 };
    if (m === "数字人" && digitalVariant() === "双人数字人") return { 图片: 2, 视频: 0, 音频: 2 };
    if (m === "数字人" && digitalVariant() === "MV数字人") return { 图片: 20, 视频: 0, 音频: 3 };
    return { 图片: 9, 视频: 3, 音频: 3 };
  };
  const toolbar = document.createElement("div");
  toolbar.className = "wwh3-media-toolbar";
  const addButton = document.createElement("button");
  addButton.className = "wwh3-media-add";
  addButton.textContent = "＋ 上传参考内容";
  const hint = document.createElement("span");
  hint.className = "wwh3-media-hint";
  hint.textContent = "点击、拖入或 Ctrl+V 粘贴图片 / 视频 / 音频";
  const count = document.createElement("span");
  count.className = "wwh3-media-count";
  const rail = document.createElement("div");
  rail.className = "wwh3-media-rail";
  const mvTimeline = document.createElement("div");
  mvTimeline.className = "wwh3-mv-timeline";
  mvTimeline.hidden = true;
  node.__wwh3MvZoom = Math.max(3, Math.min(48, Number(node.__wwh3MvZoom) || 8));
  mvTimeline.addEventListener("wheel", (event) => {
    const blank = event.target === mvTimeline || event.target.classList?.contains("wwh3-mv-track-content") || event.target.classList?.contains("wwh3-mv-outside") || event.target.classList?.contains("wwh3-mv-continuation");
    if (!event.ctrlKey && !blank) return;
    event.preventDefault();
    const oldZoom = node.__wwh3MvZoom;
    const oldScroll = mvTimeline.scrollLeft;
    const pointer = Math.max(0, event.clientX - mvTimeline.getBoundingClientRect().left);
    const timeAtPointer = (oldScroll + pointer) / oldZoom;
    node.__wwh3MvZoom = Math.max(3, Math.min(48, oldZoom * (event.deltaY < 0 ? 1.14 : 0.88)));
    renderMvTimeline();
    requestAnimationFrame(() => {
      mvTimeline.scrollLeft = Math.max(0, timeAtPointer * node.__wwh3MvZoom - pointer);
    });
  }, { passive: false });
  const picker = document.createElement("input");
  picker.type = "file";
  picker.multiple = true;
  picker.accept = Object.values(MEDIA).map((item) => item.accept).join(",");
  picker.hidden = true;
  toolbar.append(addButton, hint, count, picker);
  mediaBody.append(note, toolbar, rail, mvTimeline);

  const readMvDurations = () => {
    try {
      const parsed = JSON.parse(String(w(node, "MV图片时长")?.value || "[]"));
      return Array.isArray(parsed) ? parsed.map(Number).filter(Number.isFinite) : [];
    } catch (_) { return []; }
  };
  const writeMvDurations = (durations) => {
    const minimum = currentMode() === "图生视频" && Boolean(w(node, "图生连续拼接")?.value) ? 3 : 2;
    const clean = durations.map((value) => Math.round(Math.max(minimum, Math.min(15, Number(value) || minimum)) * 10) / 10);
    setW(node, "MV图片时长", JSON.stringify(clean));
    return clean;
  };
  const balancedMvDurations = (total, count) => {
    if (!count) return [];
    const minimum = currentMode() === "图生视频" && Boolean(w(node, "图生连续拼接")?.value) ? 3 : 2;
    const defaultClip = Math.max(minimum, Math.min(15, Number(w(node, "时长秒")?.value || 5)));
    return Array(count).fill(Math.round(defaultClip * 10) / 10);
  };
  const readAudioTrimConfig = () => {
    try {
      const parsed = JSON.parse(String(w(node, "音频剪切配置")?.value || "{}"));
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
    } catch (_) { return {}; }
  };
  const writeAudioTrimConfig = (config) => {
    setW(node, "音频剪切配置", JSON.stringify(config || {}));
  };
  const mvAudioRange = () => {
    const source = Math.max(0, Number(node.__wwh3MvAudioDuration || 0));
    const start = Math.max(0, Math.min(Math.max(0, source - 2), Number(w(node, "MV音乐开始秒")?.value || 0)));
    const requestedEnd = Number(w(node, "MV音乐结束秒")?.value || 0);
    const end = source > 0 ? Math.max(start + 2, Math.min(source, requestedEnd > start ? requestedEnd : source)) : 0;
    return { source, start, end, duration: Math.max(0, end - start) };
  };
  const resolveMvDurations = (count) => {
    let durations = readMvDurations();
    const total = mvAudioRange().duration;
    durations = durations.slice(0, count).filter((value) => Number.isFinite(value) && value >= 2 && value <= 15);
    const defaultClip = Math.max(2, Math.min(15, Number(w(node, "时长秒")?.value || 5)));
    while (durations.length < count) durations.push(defaultClip);
    durations = writeMvDurations(durations);
    return durations;
  };
  function renderMvTimeline() {
    const activeMode = currentMode();
    const isMv = activeMode === "数字人" && digitalVariant() === "MV数字人";
    const isI2vSequence = activeMode === "图生视频" && Boolean(w(node, "图生连续拼接")?.value);
    const images = selected(node, "图片").slice(0, 20);
    const audios = selected(node, "音频").slice(0, 3);
    // T2V / I2V / FL2V never consume an audio reference.  Old workflows may
    // still retain a filename in a hidden audio widget after switching modes;
    // do not let that stale value bring the music editor back into view.
    // The large waveform editor belongs only to MV production. Multi-reference
    // audio is a normal selectable reference card beside images/videos.
    mvTimeline.hidden = !isMv && !isI2vSequence;
    if (mvTimeline.hidden) return;
    mvTimeline.replaceChildren();
    let durations = resolveMvDurations(images.length);
    const rangeState = mvAudioRange();
    const trimConfigForTotal = readAudioTrimConfig();
    node.__wwh3AudioReedit ||= {};
    const firstAudioCollapsed = Boolean(trimConfigForTotal["1"]?.committed) && !node.__wwh3AudioReedit["1"];
    const combinedAudioDuration = audios.reduce((sum, _filename, index) => {
      const slot = String(index + 1);
      const source = index === 0
        ? Number(node.__wwh3MvAudioDuration || 0)
        : Number(node.__wwh3AudioDurations?.[slot] || 0);
      const saved = trimConfigForTotal[slot];
      const effective = saved?.committed
        ? Math.max(0, Math.min(source || Number(saved.end), Number(saved.end)) - Math.max(0, Number(saved.start)))
        : source;
      return sum + (Number.isFinite(effective) ? effective : 0);
    }, 0);
    const timelineDuration = Math.max(2, combinedAudioDuration || rangeState.duration || durations.reduce((sum, value) => sum + value, 0) || 5);
    // A committed trim becomes a new zero-based master timeline. While the
    // user is re-editing we temporarily restore source-file timestamps so both
    // handles can be positioned against the original waveform.
    const scaleDuration = firstAudioCollapsed ? timelineDuration : Math.max(timelineDuration, rangeState.source || 0);
    const timelineOrigin = firstAudioCollapsed ? 0 : rangeState.start;
    const pixelsPerSecond = Math.max(3, Math.min(48, Number(node.__wwh3MvZoom) || 8));
    const laneWidth = Math.max(320, Math.ceil(scaleDuration * pixelsPerSecond));
    // Both lanes must use the exact same time ruler.  Leaving the content
    // column as `1fr` made it keep the old viewport width after an audio trim,
    // even though the calculated duration had already changed.
    const syncTrackWidth = (track, content) => {
      track.style.gridTemplateColumns = `64px ${laneWidth}px`;
      track.style.width = `${laneWidth + 72}px`;
      track.style.minWidth = `${laneWidth + 72}px`;
      content.style.width = `${laneWidth}px`;
      content.style.minWidth = `${laneWidth}px`;
      content.style.maxWidth = `${laneWidth}px`;
    };
    const formatMvStamp = (seconds) => {
      const safe = Math.max(0, Number(seconds) || 0);
      const minutes = Math.floor(safe / 60);
      return `${String(minutes).padStart(2, "0")}:${(safe - minutes * 60).toFixed(1).padStart(4, "0")}`;
    };
    const pictureTrack = document.createElement("div");
    pictureTrack.className = "wwh3-mv-track is-pictures";
    const pictureLabel = document.createElement("div");
    pictureLabel.className = "wwh3-mv-track-label";
    pictureLabel.innerHTML = `<b>图片轨道</b><small>拖边界 · 空白滚轮缩放<br>${pixelsPerSecond.toFixed(1)} 像素/秒</small>`;
    const pictureContent = document.createElement("div");
    pictureContent.className = "wwh3-mv-track-content";
    if (!firstAudioCollapsed && rangeState.start > 0 && rangeState.source > 0) {
      const beforeSelection = document.createElement("div");
      beforeSelection.className = "wwh3-mv-outside";
      beforeSelection.style.flex = `0 0 ${rangeState.start * pixelsPerSecond}px`;
      pictureContent.append(beforeSelection);
    }
    if (!images.length) {
      const empty = document.createElement("div");
      empty.className = "wwh3-mv-empty";
      empty.textContent = "请上传图片，可连续添加最多20张";
      pictureContent.append(empty);
    }
    const updateClipDuration = (index, nextDuration) => {
      if (index < 0 || index >= durations.length) return;
      durations[index] = Math.max(isI2vSequence ? 3 : 2, Math.min(15, Number(nextDuration) || durations[index]));
      durations = writeMvDurations(durations);
      renderMvTimeline();
    };
    images.forEach((filename, index) => {
      const clip = document.createElement("div");
      clip.className = "wwh3-mv-clip";
      if (isI2vSequence && node.__wwh3I2vPromptIndex === index) {
        clip.style.boxShadow = "0 0 0 2px #35ead0,0 0 14px #35ead066";
      }
      clip.style.flex = `0 0 ${Math.max(2, durations[index] || 2) * pixelsPerSecond}px`;
      const image = document.createElement("img");
      image.src = mediaUrl(filename);
      const clipStart = timelineOrigin + durations.slice(0, index).reduce((sum, value) => sum + value, 0);
      const range = document.createElement("span");
      range.className = "wwh3-mv-range";
      range.textContent = `${formatMvStamp(clipStart)}–${formatMvStamp(clipStart + durations[index])}`;
      const removeImage = document.createElement("button");
      removeImage.type = "button";
      removeImage.className = "wwh3-x";
      removeImage.textContent = "×";
      removeImage.onclick = (event) => {
        event.stopPropagation();
        const list = selected(node, "图片");
        list.splice(index, 1);
        writeSelected(node, "图片", list);
        const nextDurations = durations.slice();
        nextDurations.splice(index, 1);
        writeMvDurations(nextDurations);
        renderMedia();
      };
      const info = document.createElement("div");
      info.className = "wwh3-mv-clip-info";
      const alias = document.createElement("b");
      alias.textContent = `图片${index + 1}`;
      const seconds = document.createElement("input");
      seconds.type = "number";
      seconds.min = isI2vSequence ? "3" : "2";
      seconds.max = "15";
      seconds.step = ".1";
      seconds.value = String(durations[index] || 0);
      seconds.title = "该图片在MV中持续的秒数";
      seconds.onchange = () => {
        updateClipDuration(index, Number(seconds.value));
      };
      const unit = document.createElement("span");
      unit.textContent = "秒";
      info.append(alias, seconds, unit);
      clip.append(image, removeImage, range, info);
      if (isI2vSequence) clip.onclick = (event) => {
        if (event.target.closest("button,input")) return;
        let prompts = [];
        try { prompts = JSON.parse(String(w(node, "图生分段提示词")?.value || "[]")); } catch (_) {}
        node.__wwh3I2vPromptIndex = index;
        // A blank segment must stay blank. Pulling the global/final prompt here
        // silently made several pictures share one prompt.
        idea.value = String(prompts[index] || "");
        setW(node, "图生当前图片序号", index + 1);
        setW(node, "增强源提示词", idea.value);
        idea.placeholder = `图片${index + 1} · 输入本段提示词（实时保存）`;
        renderMvTimeline();
        idea.focus();
      };
      {
        const boundary = document.createElement("button");
        boundary.type = "button";
        boundary.className = "wwh3-mv-boundary";
        boundary.title = "左右拖动，只调整当前图片的持续时间（2至15秒）";
        boundary.onpointerdown = (event) => {
          event.preventDefault();
          event.stopPropagation();
          const startX = event.clientX;
          const startLeft = durations[index];
          document.body.style.userSelect = "none";
          document.body.style.cursor = "ew-resize";
          const move = (moveEvent) => {
            const delta = (moveEvent.clientX - startX) / pixelsPerSecond;
            const value = Math.max(isI2vSequence ? 3 : 2, Math.min(15, startLeft + delta));
            seconds.value = value.toFixed(1);
            range.textContent = `${formatMvStamp(clipStart)}–${formatMvStamp(clipStart + value)}`;
            clip.style.flex = `0 0 ${value * pixelsPerSecond}px`;
          };
          const up = (upEvent) => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            document.body.style.userSelect = "";
            document.body.style.cursor = "";
            const delta = (upEvent.clientX - startX) / pixelsPerSecond;
            updateClipDuration(index, startLeft + delta);
          };
          window.addEventListener("pointermove", move);
          window.addEventListener("pointerup", up, { once: true });
        };
        clip.append(boundary);
      }
      pictureContent.append(clip);
    });
    const assignedDuration = durations.reduce((sum, value) => sum + value, 0);
    if (images.length && assignedDuration < timelineDuration - .05) {
      const continuation = document.createElement("div");
      continuation.className = "wwh3-mv-continuation";
      continuation.style.flex = `0 0 ${(timelineDuration - assignedDuration) * pixelsPerSecond}px`;
      continuation.textContent = `尾帧自动续接 ${(timelineDuration - assignedDuration).toFixed(1)}秒`;
      pictureContent.append(continuation);
    }
    if (!firstAudioCollapsed && rangeState.source > 0 && rangeState.end < rangeState.source) {
      const afterSelection = document.createElement("div");
      afterSelection.className = "wwh3-mv-outside";
      afterSelection.style.flex = `0 0 ${(rangeState.source - rangeState.end) * pixelsPerSecond}px`;
      pictureContent.append(afterSelection);
    }
    if (assignedDuration > timelineDuration + .05) pictureContent.classList.add("is-overflow");
    pictureTrack.append(pictureLabel, pictureContent);
    syncTrackWidth(pictureTrack, pictureContent);

    const musicTrack = document.createElement("div");
    musicTrack.className = "wwh3-mv-track is-music";
    const musicLabel = document.createElement("div");
    musicLabel.className = "wwh3-mv-track-label";
    musicLabel.innerHTML = "<b>音乐轨道</b><small>决定MV总时间线</small>";
    const musicContent = document.createElement("div");
    musicContent.className = "wwh3-mv-track-content";
    if (audios.length) {
      const audioWrap = document.createElement("div");
      audioWrap.className = "wwh3-mv-audio";
      const audio = document.createElement("audio");
      audio.src = mediaUrl(audios[0]);
      audio.preload = "metadata";
      audio.hidden = true;
      const wave = document.createElement("div");
      wave.className = "wwh3-mv-wave";
      const canvas = document.createElement("canvas");
      const trimStart = document.createElement("button");
      const trimEnd = document.createElement("button");
      trimStart.type = trimEnd.type = "button";
      trimStart.className = "wwh3-mv-trim is-start";
      trimEnd.className = "wwh3-mv-trim is-end";
      trimStart.title = "拖动选择音乐开始时间";
      trimEnd.title = "拖动选择音乐结束时间";
      wave.append(canvas, trimStart, trimEnd);
      const controls = document.createElement("div");
      controls.className = "wwh3-mv-audio-controls";
      const timeText = document.createElement("span");
      timeText.className = "wwh3-mv-time";
      const selectionText = document.createElement("span");
      selectionText.className = "wwh3-mv-selection";
      const play = document.createElement("button");
      play.type = "button";
      play.className = "wwh3-mv-play";
      play.textContent = "▶";
      const formatTime = (seconds) => {
        const safe = Math.max(0, Number(seconds) || 0);
        const minutes = Math.floor(safe / 60);
        const rest = (safe - minutes * 60).toFixed(3).padStart(6, "0");
        return `${String(minutes).padStart(2, "0")}:${rest}`;
      };
      let peaks = [];
      let animationFrame = 0;
      node.__wwh3AudioReedit ||= {};
      const drawWave = () => {
        const rect = canvas.getBoundingClientRect();
        const ratio = Math.max(1, window.devicePixelRatio || 1);
        const width = Math.max(1, Math.round(rect.width * ratio));
        const height = Math.max(1, Math.round(rect.height * ratio));
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
        }
        const context = canvas.getContext("2d");
        context.clearRect(0, 0, width, height);
        const selected = mvAudioRange();
        const committed = Boolean(readAudioTrimConfig()["1"]?.committed) && !node.__wwh3AudioReedit["1"];
        const viewStart = committed ? selected.start : 0;
        const viewEnd = committed ? selected.end : (audio.duration || selected.end);
        const viewDuration = Math.max(.001, viewEnd - viewStart);
        const progress = Math.max(0, Math.min(1, (audio.currentTime - viewStart) / viewDuration));
        const startFraction = committed ? 0 : (audio.duration > 0 ? selected.start / audio.duration : 0);
        const endFraction = committed ? 1 : (audio.duration > 0 ? selected.end / audio.duration : 1);
        const waveformHeight = Math.max(1, height - 12 * ratio);
        const center = waveformHeight / 2;
        const bars = Math.max(1, Math.floor(width / (2 * ratio)));
        context.lineWidth = Math.max(1, ratio);
        for (let index = 0; index < bars; index++) {
          const sourceFraction = audio.duration > 0 ? (viewStart + index / bars * viewDuration) / audio.duration : index / bars;
          const value = peaks.length ? peaks[Math.min(peaks.length - 1, Math.floor(sourceFraction * peaks.length))] : .04;
          const amplitude = Math.max(1.5 * ratio, value * waveformHeight * .46);
          context.strokeStyle = index / bars <= progress ? "#318bff" : "#3b3f44";
          context.beginPath();
          const x = (index + .5) / bars * width;
          context.moveTo(x, center - amplitude);
          context.lineTo(x, center + amplitude);
          context.stroke();
        }
        context.fillStyle = "#02070dbb";
        context.fillRect(0, 0, startFraction * width, waveformHeight);
        context.fillRect(endFraction * width, 0, (1 - endFraction) * width, waveformHeight);
        context.font = `${8 * ratio}px Consolas,monospace`;
        context.fillStyle = "#6f8792";
        context.strokeStyle = "#263640";
        context.lineWidth = Math.max(.5, ratio * .5);
        for (let tick = 0; tick <= 4; tick++) {
          const fraction = tick / 4;
          const x = fraction * width;
          context.beginPath();
          context.moveTo(x, 0);
          context.lineTo(x, waveformHeight);
          context.stroke();
          const label = formatTime((committed ? 0 : viewStart) + viewDuration * fraction).slice(0, -2);
          context.textAlign = tick === 0 ? "left" : tick === 4 ? "right" : "center";
          context.fillText(label, x, height - 2 * ratio);
        }
        const cursorX = progress * width;
        context.strokeStyle = "#ff263a";
        context.lineWidth = Math.max(1, ratio);
        context.beginPath();
        context.moveTo(cursorX, 0);
        context.lineTo(cursorX, height);
        context.stroke();
        timeText.textContent = committed
          ? `${formatTime(Math.max(0, audio.currentTime - viewStart))} / ${formatTime(viewDuration)}`
          : `${formatTime(audio.currentTime)} / ${formatTime(audio.duration)}`;
        selectionText.textContent = committed
          ? `仅保留 ${formatTime(selected.start)}–${formatTime(selected.end)}（${selected.duration.toFixed(1)}秒）`
          : `选择 ${formatTime(selected.start)}–${formatTime(selected.end)}（${selected.duration.toFixed(1)}秒）`;
        trimStart.style.left = `${startFraction * 100}%`;
        trimEnd.style.left = `${endFraction * 100}%`;
        trimStart.hidden = trimEnd.hidden = committed;
        wave.classList.toggle("is-committed", committed);
      };
      const animate = () => {
        const selected = mvAudioRange();
        if (!audio.paused && selected.end > selected.start && audio.currentTime >= selected.end) {
          audio.pause();
          audio.currentTime = selected.end;
        }
        drawWave();
        if (!audio.paused && !audio.ended) animationFrame = requestAnimationFrame(animate);
      };
      play.onclick = async () => {
        if (audio.paused) {
          const selected = mvAudioRange();
          if (audio.currentTime < selected.start || audio.currentTime >= selected.end) audio.currentTime = selected.start;
          try { await audio.play(); } catch (_) { return; }
        } else audio.pause();
      };
      audio.onplay = () => { play.textContent = "❚❚"; cancelAnimationFrame(animationFrame); animate(); };
      audio.onpause = () => { play.textContent = "▶"; cancelAnimationFrame(animationFrame); drawWave(); };
      audio.onended = () => { play.textContent = "▶"; drawWave(); };
      const cutAudio = document.createElement("button");
      cutAudio.type = "button";
      cutAudio.className = "wwh3-mv-cut";
      cutAudio.textContent = "✂";
      cutAudio.title = readAudioTrimConfig()["1"]?.committed && !node.__wwh3AudioReedit["1"]
        ? "重新选择剪切区域" : "保留当前选中区域";
      cutAudio.onclick = () => {
        const existing = readAudioTrimConfig()["1"];
        if (existing?.committed && !node.__wwh3AudioReedit["1"]) {
          node.__wwh3AudioReedit["1"] = true;
          setW(node, "MV音乐开始秒", Number(existing.start) || 0);
          setW(node, "MV音乐结束秒", Number(existing.end) || 0);
          renderMvTimeline();
          return;
        }
        const selectedRange = mvAudioRange();
        const next = readAudioTrimConfig();
        next["1"] = { start: selectedRange.start, end: selectedRange.end, committed: true };
        writeAudioTrimConfig(next);
        node.__wwh3AudioReedit["1"] = false;
        renderMvTimeline();
        // The retained range is a new zero-based timeline.  Do not leave the
        // horizontal viewport parked at the old source-file timestamp.
        requestAnimationFrame(() => { mvTimeline.scrollLeft = 0; });
      };
      const seek = (event) => {
        if (!Number.isFinite(audio.duration) || audio.duration <= 0) return;
        const rect = wave.getBoundingClientRect();
        const selected = mvAudioRange();
        const committed = Boolean(readAudioTrimConfig()["1"]?.committed);
        const viewStart = committed ? selected.start : 0;
        const viewEnd = committed ? selected.end : audio.duration;
        audio.currentTime = Math.max(viewStart, Math.min(viewEnd, viewStart + (event.clientX - rect.left) / rect.width * (viewEnd - viewStart)));
        drawWave();
      };
      wave.onpointerdown = (event) => {
        seek(event);
        const move = (moveEvent) => seek(moveEvent);
        const up = () => {
          window.removeEventListener("pointermove", move);
          window.removeEventListener("pointerup", up);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up);
      };
      const bindTrim = (handle, isStart) => {
        handle.onpointerdown = (event) => {
          event.preventDefault();
          event.stopPropagation();
          const move = (moveEvent) => {
            if (!Number.isFinite(audio.duration) || audio.duration < 2) return;
            const rect = wave.getBoundingClientRect();
            const value = Math.max(0, Math.min(audio.duration, (moveEvent.clientX - rect.left) / rect.width * audio.duration));
            const current = mvAudioRange();
            if (isStart) setW(node, "MV音乐开始秒", Math.min(current.end - 2, value));
            else setW(node, "MV音乐结束秒", Math.max(current.start + 2, value));
            drawWave();
          };
          const up = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            renderMvTimeline();
          };
          window.addEventListener("pointermove", move);
          window.addEventListener("pointerup", up, { once: true });
        };
      };
      bindTrim(trimStart, true);
      bindTrim(trimEnd, false);
      controls.append(timeText, play, cutAudio, selectionText);
      audio.onloadedmetadata = () => {
        if (!Number.isFinite(audio.duration) || audio.duration <= 0) return;
        const changed = Math.abs(Number(node.__wwh3MvAudioDuration || 0) - audio.duration) > .1;
        node.__wwh3MvAudioDuration = audio.duration;
        node.__wwh3AudioDurations ||= {};
        node.__wwh3AudioDurations["1"] = audio.duration;
        const existingEnd = Number(w(node, "MV音乐结束秒")?.value || 0);
        if (existingEnd <= Number(w(node, "MV音乐开始秒")?.value || 0) || existingEnd > audio.duration) {
          setW(node, "MV音乐结束秒", audio.duration);
        }
        drawWave();
        if (changed) {
          resolveMvDurations(images.length);
          requestAnimationFrame(renderMvTimeline);
        }
      };
      fetch(audio.src).then((response) => response.arrayBuffer()).then(async (buffer) => {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;
        const audioContext = new AudioCtx();
        try {
          const decoded = await audioContext.decodeAudioData(buffer.slice(0));
          const data = decoded.getChannelData(0);
          const bucketCount = 640;
          const bucketSize = Math.max(1, Math.floor(data.length / bucketCount));
          peaks = Array.from({ length: bucketCount }, (_, index) => {
            const start = index * bucketSize;
            const end = Math.min(data.length, start + bucketSize);
            let peak = 0;
            for (let offset = start; offset < end; offset += Math.max(1, Math.floor(bucketSize / 64))) {
              peak = Math.max(peak, Math.abs(data[offset] || 0));
            }
            return peak;
          });
          drawWave();
        } finally { audioContext.close?.(); }
      }).catch(() => drawWave());
      requestAnimationFrame(drawWave);
      const removeAudio = document.createElement("button");
      removeAudio.type = "button";
      removeAudio.className = "wwh3-x";
      removeAudio.textContent = "×";
      removeAudio.onclick = () => {
        audio.pause();
        cancelAnimationFrame(animationFrame);
        const list = selected(node, "音频");
        list.splice(0, 1);
        writeSelected(node, "音频", list);
        node.__wwh3MvAudioDuration = 0;
        setW(node, "MV音乐开始秒", 0);
        setW(node, "MV音乐结束秒", 0);
        writeAudioTrimConfig({});
        renderMedia();
      };
      audioWrap.append(audio, wave, controls, removeAudio);
      musicContent.append(audioWrap);
    } else {
      const empty = document.createElement("div");
      empty.className = "wwh3-mv-empty";
      empty.textContent = "请上传一段完整音乐";
      musicContent.append(empty);
    }
    musicTrack.append(musicLabel, musicContent);
    syncTrackWidth(musicTrack, musicContent);
    if (isMv || isI2vSequence) mvTimeline.append(pictureTrack);
    if (!isI2vSequence) mvTimeline.append(musicTrack);

    // Additional clips share the same independent music-track contract. They
    // are concatenated by the backend in this visible order.
    audios.slice(1).forEach((filename, extraIndex) => {
      const slot = extraIndex + 2;
      const extraTrack = document.createElement("div");
      extraTrack.className = "wwh3-mv-track is-music";
      const extraLabel = document.createElement("div");
      extraLabel.className = "wwh3-mv-track-label";
      extraLabel.innerHTML = `<b>音乐轨道 ${slot}</b><small>剪切后顺序拼接</small>`;
      const extraContent = document.createElement("div");
      extraContent.className = "wwh3-mv-track-content";
      const panel = document.createElement("div");
      panel.className = "wwh3-mv-extra-audio";
      const player = document.createElement("audio");
      player.src = mediaUrl(filename);
      player.controls = true;
      player.preload = "metadata";
      const startInput = document.createElement("input");
      const endInput = document.createElement("input");
      const extraWave = document.createElement("div");
      extraWave.className = "wwh3-mv-extra-wave";
      const startSlider = document.createElement("input");
      const endSlider = document.createElement("input");
      startSlider.type = endSlider.type = "range";
      startSlider.min = endSlider.min = "0";
      startSlider.step = endSlider.step = ".1";
      extraWave.append(startSlider, endSlider);
      startInput.type = endInput.type = "number";
      startInput.min = endInput.min = "0";
      startInput.step = endInput.step = ".1";
      const config = readAudioTrimConfig();
      const saved = config[String(slot)] || {};
      node.__wwh3AudioReedit ||= {};
      const extraCommitted = Boolean(saved.committed) && !node.__wwh3AudioReedit[String(slot)];
      startInput.value = String(Number(saved.start || 0));
      endInput.value = String(Number(saved.end || 0));
      startSlider.value = startInput.value;
      endSlider.value = endInput.value;
      const cut = document.createElement("button");
      cut.type = "button";
      cut.className = "wwh3-mv-cut";
      cut.textContent = "✂";
      cut.title = extraCommitted ? "重新选择剪切区域" : "保留当前选中区域";
      extraWave.classList.toggle("is-committed", extraCommitted);
      player.onloadedmetadata = () => {
        node.__wwh3AudioDurations ||= {};
        node.__wwh3AudioDurations[String(slot)] = player.duration;
        endInput.max = startInput.max = String(player.duration);
        startSlider.max = endSlider.max = String(player.duration);
        if (!(Number(endInput.value) > Number(startInput.value))) endInput.value = player.duration.toFixed(1);
        startSlider.value = startInput.value;
        endSlider.value = endInput.value;
      };
      const syncExtraRange = (fromSlider) => {
        const duration = Number.isFinite(player.duration) ? player.duration : Number(endInput.max || endInput.value || 0);
        let start = Number(fromSlider ? startSlider.value : startInput.value) || 0;
        let end = Number(fromSlider ? endSlider.value : endInput.value) || duration;
        start = Math.max(0, Math.min(Math.max(0, end - .05), start));
        end = Math.max(start + .05, Math.min(duration, end));
        startInput.value = start.toFixed(1);
        endInput.value = end.toFixed(1);
        startSlider.value = String(start);
        endSlider.value = String(end);
        cut.textContent = "✂";
        cut.title = "保留当前选中区域";
      };
      startSlider.oninput = endSlider.oninput = () => syncExtraRange(true);
      startInput.oninput = endInput.oninput = () => syncExtraRange(false);
      cut.onclick = () => {
        if (saved.committed && !node.__wwh3AudioReedit[String(slot)]) {
          node.__wwh3AudioReedit[String(slot)] = true;
          renderMvTimeline();
          return;
        }
        const duration = Number.isFinite(player.duration) ? player.duration : Number(endInput.value || 0);
        const start = Math.max(0, Math.min(duration, Number(startInput.value) || 0));
        const end = Math.max(start + .05, Math.min(duration, Number(endInput.value) || duration));
        const next = readAudioTrimConfig();
        next[String(slot)] = { start, end, committed: true };
        writeAudioTrimConfig(next);
        node.__wwh3AudioReedit[String(slot)] = false;
        renderMvTimeline();
      };
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "wwh3-x";
      remove.textContent = "×";
      remove.onclick = () => {
        const list = selected(node, "音频");
        list.splice(slot - 1, 1);
        writeSelected(node, "音频", list);
        writeAudioTrimConfig({});
        renderMedia();
      };
      const range = document.createElement("span");
      range.className = "wwh3-mv-extra-range";
      if (extraCommitted) {
        range.textContent = `仅保留 ${Number(saved.start).toFixed(1)}–${Number(saved.end).toFixed(1)} 秒（${Math.max(0, Number(saved.end) - Number(saved.start)).toFixed(1)}秒）`;
      } else {
        range.append("开始 ", startInput, " 秒　结束 ", endInput, " 秒");
      }
      panel.append(player, extraWave, range, cut, remove);
      extraContent.append(panel);
      extraTrack.append(extraLabel, extraContent);
      syncTrackWidth(extraTrack, extraContent);
      mvTimeline.append(extraTrack);
    });
  }
  function activeMediaEntries() {
    const caps = limits();
    const entries = [];
    for (const kind of Object.keys(MEDIA)) {
      if (kind === "音频") continue;
      selected(node, kind).slice(0, caps[kind]).forEach((filename, index) => {
        entries.push({ kind, index: index + 1, filename, alias: `@${kind}${index + 1}` });
      });
    }
    return entries;
  }
  function visualPreview(entry) {
    if (entry.kind === "图片") {
      const preview = document.createElement("img");
      preview.src = mediaUrl(entry.filename);
      return preview;
    }
    if (entry.kind === "视频") {
      const preview = document.createElement("video");
      preview.src = mediaUrl(entry.filename);
      preview.muted = true;
      preview.preload = "metadata";
      return preview;
    }
    const preview = document.createElement("span");
    preview.className = "wwh3-mention-audio";
    preview.textContent = "♫";
    return preview;
  }
  function mentionContext() {
    const caret = Number.isFinite(idea.selectionStart) ? idea.selectionStart : idea.value.length;
    const before = idea.value.slice(0, caret);
    const match = before.match(/@([^\s@]*)$/);
    return match ? { start: caret - match[0].length, end: caret, query: match[1].toLowerCase() } : null;
  }
  function insertReferenceAlias(alias) {
    const context = mentionContext();
    const start = context ? context.start : (Number.isFinite(idea.selectionStart) ? idea.selectionStart : idea.value.length);
    const end = context ? context.end : (Number.isFinite(idea.selectionEnd) ? idea.selectionEnd : start);
    const before = idea.value.slice(0, start);
    const spacerBefore = before && !/[\s，。；：、(（]$/.test(before) ? " " : "";
    idea.setRangeText(`${spacerBefore}${alias} `, start, end, "end");
    idea.dispatchEvent(new Event("input", { bubbles: true }));
    mentionPicker.hidden = true;
    idea.focus();
  }
  function renderPromptReferences() {
    const previousCount = promptReferences.childElementCount;
    const aliases = new Set(idea.value.match(/@(图片|视频|音频)\d+/g) || []);
    promptReferences.replaceChildren();
    for (const entry of activeMediaEntries()) {
      if (!aliases.has(entry.alias)) continue;
      const chip = document.createElement("div");
      chip.className = "wwh3-prompt-ref";
      chip.title = `${entry.alias} · ${entry.filename.split("/").pop()}`;
      chip.append(visualPreview(entry));
      const text = document.createElement("b");
      text.textContent = entry.alias;
      chip.append(text);
      promptReferences.append(chip);
    }
    if (previousCount !== promptReferences.childElementCount) {
      requestAnimationFrame(() => node.__wwh3?.resizeForContent?.());
    }
  }
  function updateMentionPicker() {
    const context = mentionContext();
    if (!context) {
      mentionPicker.hidden = true;
      return;
    }
    const entries = activeMediaEntries().filter((entry) => {
      const query = context.query;
      return !query || entry.alias.slice(1).toLowerCase().includes(query) || entry.filename.toLowerCase().includes(query);
    });
    mentionPicker.replaceChildren();
    for (const entry of entries) {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "wwh3-mention-option";
      option.onmousedown = (event) => event.preventDefault();
      option.onclick = () => insertReferenceAlias(entry.alias);
      const copy = document.createElement("span");
      copy.className = "wwh3-mention-copy";
      const title = document.createElement("b");
      title.textContent = entry.alias;
      const filename = document.createElement("small");
      filename.textContent = entry.filename.split("/").pop();
      copy.append(title, filename);
      option.append(visualPreview(entry), copy);
      mentionPicker.append(option);
    }
    mentionPicker.hidden = !entries.length;
  }
  const addFiles = async (files) => {
    const caps = limits();
    const lists = Object.fromEntries(Object.keys(MEDIA).map((kind) => [kind, selected(node, kind)]));
    for (const file of files) {
      const kind = inferKind(file);
      if (!kind || !caps[kind] || lists[kind].length >= caps[kind]) continue;
      try { lists[kind].push(await upload(file)); } catch (error) { alert(error.message); }
    }
    for (const kind of Object.keys(MEDIA)) writeSelected(node, kind, lists[kind]);
    renderMedia();
  };
  picker.onchange = () => { addFiles([...picker.files]); picker.value = ""; };
  addButton.onclick = () => picker.click();
  toolbar.onclick = (event) => { if (event.target === toolbar || event.target === hint) picker.click(); };
  toolbar.ondragover = (event) => { event.preventDefault(); toolbar.classList.add("drag"); };
  toolbar.ondragleave = () => toolbar.classList.remove("drag");
  toolbar.ondrop = (event) => { event.preventDefault(); toolbar.classList.remove("drag"); addFiles([...event.dataTransfer.files]); };
  const pasteMedia = (event) => {
    if (!root.isConnected) return;
    const active = document.activeElement;
    if (active && (active.tagName === "TEXTAREA" || active.tagName === "INPUT" || active.isContentEditable)) return;
    const files = [...(event.clipboardData?.items || [])]
      .filter((item) => item.kind === "file")
      .map((item) => item.getAsFile())
      .filter(Boolean);
    if (!files.length) return;
    event.preventDefault();
    event.stopPropagation();
    addFiles(files);
  };
  root.tabIndex = 0;
  root.addEventListener("paste", pasteMedia);
  toolbar.title = "点击上传；也可先点击节点空白处，再按 Ctrl+V 粘贴剪贴板图片";
  function renderMedia() {
    const caps = limits();
    refreshRefineControl();
    modeButtons.forEach((button) => button.classList.toggle("on", button.dataset.mode === currentMode()));
    digitalSubmode.style.display = currentMode() === "数字人" ? "flex" : "none";
    i2vSubmode.style.display = currentMode() === "图生视频" ? "flex" : "none";
    const i2vContinuous = Boolean(w(node, "图生连续拼接")?.value);
    i2vButtons.forEach(([button, enabled]) => button.classList.toggle("on", enabled === i2vContinuous));
    digitalButtons.forEach((button) => button.classList.toggle("on", button.dataset.mode === digitalVariant()));
    videoEditSubmode.style.display = currentMode() === "视频编辑" ? "flex" : "none";
    videoEditToolButtons.forEach((button) => button.classList.toggle("on", button.dataset.tool === String(w(node, "视频编辑功能")?.value || "通用编辑")));
    videoEditProfile.style.display = "grid";
    const customModelConfig = Boolean(w(node, "自定义模型配置")?.value);
      speedButtons.forEach((button) => button.classList.toggle("on", !customModelConfig && button.dataset.profile === String(w(node, "视频编辑模式")?.value || "均衡12步")));
    customSpeed.classList.toggle("on", customModelConfig);
    customSpeedInput.value = String(w(node, "采样步数")?.value || 10);
    rail.replaceChildren();
    let total = 0;
    let capacity = 0;
    for (const kind of Object.keys(MEDIA)) {
      const cap = caps[kind];
      const list = selected(node, kind);
      const activeList = list.slice(0, cap);
      total += activeList.length;
      capacity += cap;
      for (let i = 0; i < activeList.length; i++) {
        // Only digital-human MV uses the large waveform/music track. In
        // multi-reference and video-edit modes audio stays a compact card.
        const usesMvMusicTrack = currentMode() === "数字人" && digitalVariant() === "MV数字人";
        if (kind === "音频" && usesMvMusicTrack) continue;
        const item = document.createElement("div");
        item.className = `wwh3-media-item${kind === "音频" ? " is-audio" : ""}`;
        let preview;
        if (kind === "图片") { preview = document.createElement("img"); preview.src = mediaUrl(activeList[i]); }
        else if (kind === "视频") { preview = document.createElement("video"); preview.src = mediaUrl(activeList[i]); preview.muted = true; }
        else {
          preview = document.createElement("span");
          preview.className = "wwh3-media-audio-icon";
          preview.textContent = "♫";
        }
        const remove = document.createElement("button");
        remove.className = "wwh3-x";
        remove.textContent = "×";
        remove.onclick = (event) => { event.stopPropagation(); list.splice(i, 1); writeSelected(node, kind, list); renderMedia(); };
        const label = document.createElement("span");
        label.className = "wwh3-media-label";
        label.textContent = currentMode() === "首尾帧" ? (i ? "尾帧" : "首帧") : `@${kind}${i + 1}`;
        const alias = `@${kind}${i + 1}`;
        item.classList.toggle("is-selected", String(idea.value || "").includes(alias));
        item.title = `点击插入 ${alias} · ${activeList[i].split("/").pop()}`;
        item.onclick = (event) => {
          if (event.target.closest(".wwh3-x")) return;
          insertReferenceAlias(alias);
          item.classList.add("is-selected");
        };
        item.append(preview, remove, label);
        rail.append(item);
      }
    }
    count.textContent = `${total}/${capacity}`;
    const isMvTimeline = (currentMode() === "数字人" && digitalVariant() === "MV数字人") || (currentMode() === "图生视频" && Boolean(w(node, "图生连续拼接")?.value));
    const cardTotal = selected(node, "图片").slice(0, caps.图片).length
      + selected(node, "视频").slice(0, caps.视频).length
      + (currentMode() === "数字人" && digitalVariant() === "MV数字人" ? 0 : selected(node, "音频").slice(0, caps.音频).length);
    rail.style.display = cardTotal && !isMvTimeline ? "flex" : "none";
    renderMvTimeline();
    toolbar.style.opacity = capacity ? "1" : ".55";
    addButton.disabled = !capacity || total >= capacity;
    hint.textContent = currentMode() === "文生视频"
      ? "文生视频会自动忽略槽位中已有的全部参考素材"
      : currentMode() === "首尾帧"
        ? "只使用两张图片：图片1为首帧，图片2为尾帧"
        : currentMode() === "图生视频"
          ? (Boolean(w(node, "图生连续拼接")?.value)
            ? "连续添加图片；拖动每段右边界设置3–15秒，点击图片编辑该段提示词"
            : "单图模式只使用1张首帧图片")
        : currentMode() === "数字人" && digitalVariant() === "MV数字人"
          ? "图片按轨道分段；可加入最多3段音乐，每段剪切后按顺序拼接"
        : currentMode() === "数字人" && digitalVariant() === "单人数字人"
          ? "需要1张人物图和1段驱动音频"
          : currentMode() === "数字人" && digitalVariant() === "双人数字人"
            ? "需要2张人物图和2段音频，音频中间自动加入1秒静音"
          : currentMode() === "数字人" && digitalVariant() === "MV数字人"
            ? "上轨排列图片并拖动边界设置每张时长；下轨音乐决定MV总时长"
        : capacity ? "点击或拖入图片 / 视频 / 音频；点击素材可插入提示词" : "当前生成方式不需要参考素材";
    renderPromptReferences();
    if (!mentionPicker.hidden) updateMentionPicker();
    // Mode switches and media changes can add/remove whole rows.  Always let
    // the outer node re-wrap the complete visible panel afterwards.
    requestAnimationFrame(() => node.__wwh3?.resizeForContent?.());
  }
  renderMedia();
  // 提示词紧跟参考素材，符合“先放素材、再描述如何使用”的创作顺序。
  mediaBody.parentElement.after(ideaBody.parentElement);

  const sync = () => {
    storyboardControl.refresh();
    idea.value = w(node, "增强源提示词")?.value || w(node, "提示词")?.value || "";
    finalPrompt.value = w(node, "提示词")?.value || "";
    for (const input of root.querySelectorAll("[data-widget-name]")) {
      const widget = w(node, input.dataset.widgetName);
      if (!widget) continue;
      if (input.type === "checkbox") input.checked = Boolean(widget.value);
      else input.value = widget.value ?? "";
    }
    autoCorrectModel(currentMode());
      const presetKey = `${currentMode()}|${String(w(node, "视频编辑模式")?.value || "均衡12步")}`;
    if (node.__wwh3PresetKey !== presetKey) applyPerformancePreset();
    autoCorrectPromptTemplate(currentMode());
    updateEnhanceUI();
    renderMedia();
  };
  const consumeAutoRun = () => {
    const pending = autoRunAfterEnhance;
    autoRunAfterEnhance = false;
    return pending;
  };
  node.__wwh3 = {
    root, idea, finalPrompt, enhance, status, renderMedia, sync, modelDetails,
    disposeProgressListeners, consumeAutoRun, continueGeneration, syncModeWidgets,
    syncStoryboardCount: storyboardControl.refresh,
  };
  return root;
}

app.registerExtension({
  name: "liao2049.h3.aurora.v1",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET) return;
    addStyle();
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      repairMainState(this);
      for (const widget of this.widgets || []) hide(widget);
      this.color = "#08606b";
      this.bgcolor = "#071326";
      this.boxcolor = "#53f2dd";
      const panel = build(this);
      const shell = document.createElement("div");
      let panelHeight = 790;
      const nodeChromeHeight = 60;
      shell.style.cssText = `width:720px;min-width:720px;height:${panelHeight}px`;
      shell.append(panel);
      const dom = this.addDOMWidget("liao2049_h3_panel", "div", shell, { serialize: false, hideOnZoom: false, getHeight: () => panelHeight });
      const syncPanelSize = () => {
        const width = Math.max(640, Number(this.size?.[0] || 740) - 20);
        // Keep widget height independent from the outer LiteGraph node height.
        // Feeding this.size[1] back into computeSize makes each layout pass grow.
        const height = panelHeight;
        shell.style.width = `${width}px`;
        shell.style.minWidth = `${width}px`;
        shell.style.height = `${height}px`;
        panel.style.width = `${width}px`;
        panel.style.height = `${height}px`;
        if (dom?.element) {
          dom.element.style.width = `${width}px`;
          dom.element.style.minWidth = `${width}px`;
          dom.element.style.height = `${height}px`;
        }
        if (dom) dom.computedHeight = height;
      };
      if (dom) dom.computeSize = () => [
        Math.max(640, Number(this.size?.[0] || 740) - 20),
        panelHeight,
      ];
      this.__wwh3.syncPanelSize = syncPanelSize;
      let resizingForContent = false;
      const resizeForContent = () => {
        if (resizingForContent) return;
        resizingForContent = true;
        try {
        const modelsOpen = Boolean(this.__wwh3?.modelDetails?.open);
        const minimumNodeWidth = modelsOpen ? 980 : 740;
        const nodeWidth = Math.max(Number(this.size?.[0] || 0), minimumNodeWidth);
        // Keep a near-square, fully usable default node while the DOM is still mounting.
        // A freshly-added DOM widget can report zero child heights for its first frame.
        const minimumHeight = 730;
        shell.style.height = "auto";
        panel.style.height = "auto";
        const panelStyle = getComputedStyle(panel);
        const visibleChildren = Array.from(panel.children).filter((child) => getComputedStyle(child).display !== "none");
        const verticalPadding = (parseFloat(panelStyle.paddingTop) || 0) + (parseFloat(panelStyle.paddingBottom) || 0);
        const verticalBorder = (parseFloat(panelStyle.borderTopWidth) || 0) + (parseFloat(panelStyle.borderBottomWidth) || 0);
        const gap = parseFloat(panelStyle.rowGap || panelStyle.gap) || 0;
        const childrenHeight = visibleChildren.reduce(
          (total, child) => total + Math.max(child.offsetHeight, child.scrollHeight),
          0,
        );
        const naturalHeight = Math.ceil(verticalPadding + verticalBorder + childrenHeight + gap * Math.max(0, visibleChildren.length - 1));
        panelHeight = Math.min(4096, Math.max(minimumHeight, naturalHeight));
        this.setSize([nodeWidth, panelHeight + nodeChromeHeight]);
        syncPanelSize();
        this.graph?.setDirtyCanvas?.(true, true);
        } finally {
          resizingForContent = false;
        }
      };
      this.__wwh3.resizeForContent = resizeForContent;
      this.__wwh3.resizeForEnhance = () => requestAnimationFrame(resizeForContent);
      this.__wwh3.ensureOuterWrap = () => {
        const requiredHeight = panelHeight + nodeChromeHeight;
        if (Number(this.size?.[1] || 0) + 1 < requiredHeight) {
          this.setSize([Math.max(740, Number(this.size?.[0] || 740)), requiredHeight]);
        }
      };
      resizeForContent();
      syncPanelSize();
      requestAnimationFrame(resizeForContent);
      setTimeout(resizeForContent, 80);
      setTimeout(resizeForContent, 240);
      return result;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = configured?.call(this, info);
      setTimeout(() => {
        repairMainState(this);
        this.__wwh3?.sync();
        this.__wwh3?.resizeForContent?.();
      }, 30);
      return result;
    };
    const resized = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      const result = resized?.call(this, size);
      this.__wwh3?.syncPanelSize?.();
      // Users may widen the node freely, but its height must never become
      // smaller than the complete inner panel.  One deferred correction keeps
      // the outer LiteGraph frame coordinated with all visible DOM controls.
      requestAnimationFrame(() => this.__wwh3?.ensureOuterWrap?.());
      return result;
    };
    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this.__wwh3?.disposeProgressListeners?.();
      return removed?.apply(this, arguments);
    };
    const serialized = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (info) {
      this.__wwh3?.syncModeWidgets?.();
      this.__wwh3?.syncStoryboardCount?.();
      repairMainState(this);
      return serialized?.call(this, info);
    };
    const executed = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = executed?.apply(this, arguments);
      const prompt = String(message?.aurora_enhanced_prompt?.[0] || "").trim();
      if (prompt && this.__wwh3) {
        const shouldAutoRun = this.__wwh3.consumeAutoRun?.() || false;
        this.__wwh3.finalPrompt.value = prompt;
        if (Boolean(w(this, "图生视频")?.value) && Boolean(w(this, "图生连续拼接")?.value)
            && Number.isInteger(this.__wwh3I2vPromptIndex)) {
          let prompts = [];
          try { prompts = JSON.parse(String(w(this, "图生分段提示词")?.value || "[]")); } catch (_) {}
          const imageCount = selected(this, "图片").slice(0, 20).length;
          while (prompts.length < imageCount) prompts.push("");
          prompts[this.__wwh3I2vPromptIndex] = prompt;
          setW(this, "图生分段提示词", JSON.stringify(prompts));
          setW(this, "增强源提示词", prompt);
          this.__wwh3.idea.value = prompt;
        } else {
          setW(this, "提示词", prompt);
        }
        setW(this, "仅增强提示词", false);
        this.__wwh3.enhance.disabled = false;
        this.__wwh3.enhance.textContent = "↻ 重新生成增强提示词";
        this.__wwh3.status.textContent = shouldAutoRun
          ? "增强完成，正在自动提交视频生成…"
          : "增强完成并已自动采用，可直接运行生成";
        if (shouldAutoRun) {
          setTimeout(() => this.__wwh3?.continueGeneration?.(prompt, "增强提示词"), 0);
        }
      }
      return result;
    };
  },
});

app.registerExtension({
  name: "liao2049.h3.model.lora.config.repair.v1",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "WenWuH3ModelLoraConfig") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      repairConfigState(this);
      return result;
    };
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = configured?.call(this, info);
      setTimeout(() => repairConfigState(this), 20);
      return result;
    };
  },
});
