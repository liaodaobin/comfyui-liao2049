import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const css = document.createElement("style");
css.textContent = `
.liao-krea2{box-sizing:border-box;width:100%;padding:10px;color:#dffcff;font:12px/1.4 Arial;background:linear-gradient(160deg,#041724,#11102b);border:1px solid #28d6c6;border-radius:14px}.liao-krea2 *{box-sizing:border-box}.lk-head{display:flex;align-items:center;justify-content:space-between;padding:8px 11px;margin-bottom:9px;border-radius:10px;background:linear-gradient(90deg,#078d92,#5367da);font-weight:700;font-size:14px}.lk-modes,.lk-grid{display:grid;gap:7px}.lk-modes{grid-template-columns:repeat(3,1fr);margin-bottom:9px}.lk-modes button,.lk-upload{padding:8px;border:1px solid #258b9a;border-radius:8px;background:#0b2940;color:#e8ffff;font-weight:700}.lk-modes button.on{background:linear-gradient(90deg,#08a69f,#596cdb);border-color:#58f2dd}.lk-card{padding:9px;margin-top:8px;border:1px solid #245f78;border-radius:10px;background:#071421}.lk-label{display:block;margin-bottom:4px;color:#9feee5;font-size:10px}.lk-prompt{width:100%;height:150px;resize:vertical;padding:9px;border:1px solid #218b91;border-radius:8px;background:#020d16;color:#efffff;outline:none}.lk-ref{display:flex;align-items:center;gap:9px}.lk-ref img{width:72px;height:72px;object-fit:cover;border:1px solid #44dccb;border-radius:8px}.lk-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.lk-grid select,.lk-grid input{width:100%;padding:7px;border:1px solid #237b88;border-radius:7px;background:#03131e;color:#edffff}.lk-models summary{cursor:pointer;font-weight:700;color:#bbfff7}.lk-model-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:8px}.lk-toggle{display:flex;align-items:center;gap:7px}.lk-note{color:#75bfc1;font-size:10px}.lk-ref[hidden]{display:none}`;
document.head.append(css);

const widget = (node, name) => (node.widgets || []).find(w => w.name === name);
const setValue = (node, name, value) => { const w = widget(node, name); if (!w) return; w.value = value; w.callback?.(value); };
const hideWidget = w => { if (!w) return; w.computeSize = () => [0, -4]; w.hidden = true; if (w.element) w.element.style.display = "none"; };
const options = w => Array.isArray(w?.options?.values) ? w.options.values : [];
const mediaUrl = filename => api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=`);
async function uploadImage(file){ const body=new FormData(); body.append("image",file); body.append("type","input"); body.append("overwrite","true"); const response=await api.fetchApi("/upload/image",{method:"POST",body}); if(!response.ok) throw new Error(`上传失败 HTTP ${response.status}`); const result=await response.json(); return result.subfolder?`${result.subfolder}/${result.name}`:result.name; }

function selectControl(node, name, label){ const wrap=document.createElement("label"); const title=document.createElement("span"); title.className="lk-label"; title.textContent=label; const select=document.createElement("select"); for(const value of options(widget(node,name))){const option=document.createElement("option"); option.value=option.textContent=String(value); select.append(option)} select.value=String(widget(node,name)?.value??""); select.onchange=()=>setValue(node,name,select.value); wrap.append(title,select); return wrap; }
function inputControl(node,name,label,type="text"){const wrap=document.createElement("label");const title=document.createElement("span");title.className="lk-label";title.textContent=label;const input=document.createElement("input");input.type=type;input.value=String(widget(node,name)?.value??"");input.onchange=()=>setValue(node,name,type==="number"?Number(input.value):input.value);wrap.append(title,input);return wrap;}

app.registerExtension({
  name:"Liao2049.Krea2Studio",
  async beforeRegisterNodeDef(nodeType,nodeData){
    if(nodeData.name!=="LiaoKrea2Studio") return;
    const original=nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated=function(){
      original?.apply(this,arguments);
      const node=this; (node.widgets||[]).forEach(hideWidget);
      const root=document.createElement("div"); root.className="liao-krea2";
      const head=document.createElement("div");head.className="lk-head";head.innerHTML="<span>Liao-Krea2 智能生图台</span><small>Raw + Turbo 双阶段</small>";
      const modes=document.createElement("div");modes.className="lk-modes";
      const ref=document.createElement("div");ref.className="lk-card lk-ref";const preview=document.createElement("img");const pick=document.createElement("button");pick.className="lk-upload";pick.textContent="＋ 上传参考图片";const refText=document.createElement("span");refText.className="lk-note";const file=document.createElement("input");file.type="file";file.accept="image/*";file.hidden=true;pick.onclick=()=>file.click();
      const promptCard=document.createElement("div");promptCard.className="lk-card";const promptLabel=document.createElement("span");promptLabel.className="lk-label";promptLabel.textContent="创意描述 / 修改要求";const prompt=document.createElement("textarea");prompt.className="lk-prompt";prompt.value=String(widget(node,"提示词")?.value||"");prompt.oninput=()=>setValue(node,"提示词",prompt.value);promptCard.append(promptLabel,prompt);
      const refresh=()=>{const mode=String(widget(node,"模式")?.value||"文生图");[...modes.children].forEach(b=>b.classList.toggle("on",b.dataset.mode===mode));ref.hidden=mode==="文生图";refText.textContent=mode==="洗图"?"重建图片内容并自动清除文字、标识和水印":"只提取配色、光线、材质与镜头气质，不复制原图主体";const name=String(widget(node,"参考图片")?.value||"");preview.hidden=!name||name==="未选择";if(!preview.hidden)preview.src=mediaUrl(name);node.setSize([Math.max(node.size[0],620),mode==="文生图"?570:660]);};
      for(const mode of ["文生图","洗图","风格参考"]){const button=document.createElement("button");button.textContent=mode;button.dataset.mode=mode;button.onclick=()=>{setValue(node,"模式",mode);refresh()};modes.append(button)}
      file.onchange=async()=>{if(!file.files?.[0])return;try{const name=await uploadImage(file.files[0]);setValue(node,"参考图片",name);refresh()}catch(error){alert(error.message||error)}};
      ref.append(preview,pick,refText,file);
      const basics=document.createElement("div");basics.className="lk-card lk-grid";basics.append(selectControl(node,"画面比例","画面比例"),selectControl(node,"输出分辨率","输出分辨率"),inputControl(node,"随机种子","随机种子","number"));
      const smart=document.createElement("label");smart.className="lk-card lk-toggle";const check=document.createElement("input");check.type="checkbox";check.checked=!!widget(node,"智能改写")?.value;check.onchange=()=>setValue(node,"智能改写",check.checked);smart.append(check,document.createTextNode("智能提示词改写（洗图和风格参考会自动启用视觉识别）"));
      const models=document.createElement("details");models.className="lk-card lk-models";const summary=document.createElement("summary");summary.textContent="模型配置";const modelGrid=document.createElement("div");modelGrid.className="lk-model-grid";modelGrid.append(selectControl(node,"Raw模型","Krea2 Raw 首采模型"),selectControl(node,"Turbo模型","Krea2 Turbo 二采模型"),selectControl(node,"文本编码器","Krea2 文本编码器"),selectControl(node,"VAE","Qwen Image VAE"),inputControl(node,"Llama模型","本地 Llama GGUF"),inputControl(node,"视觉模型","视觉 mmproj"));models.append(summary,modelGrid);
      root.append(head,modes,basics,ref,promptCard,smart,models);node.addDOMWidget("krea2_studio_ui","div",root,{serialize:false});node.title="Liao-Krea2 智能生图台";refresh();
    };
  }
});

