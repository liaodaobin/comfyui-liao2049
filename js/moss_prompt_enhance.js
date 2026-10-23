import { api } from "../../../scripts/api.js";

export function createMossEnhancer({node,get,set,field,changed}) {
  const make=(tag,text,css)=>{const e=document.createElement(tag);if(text)e.textContent=text;if(css)e.style.cssText=css;return e;};
  const asBool=value=>value===true||value===1||["true","1","on"].includes(String(value).toLowerCase());
  const card=make("div");card.className="lq21-card lq21-llm";
  const toggle=make("label"),toggleText=make("span","提示词增强（开启后运行时自动推理）"),toggleInput=make("input");
  toggleText.className="lq21-label";toggleInput.type="checkbox";toggleInput.setAttribute("aria-label","提示词增强");toggle.append(toggleText,toggleInput);
  toggle.style.cssText="display:flex;align-items:center;justify-content:space-between;padding:7px;border:1px solid #28546b;border-radius:7px;cursor:pointer";
  toggle.querySelector("span").style.margin="0";toggle.querySelector("input").style.cssText="width:16px;height:16px;margin:0;accent-color:#16a6a8";
  const body=make("div"),top=make("div",null,"display:grid;grid-template-columns:1fr 1.5fr 1fr;gap:7px;margin-top:9px;align-items:end");
  const fixed=(title,value)=>{const label=make("label"),caption=make("span",title),input=make("input");caption.className="lq21-label";input.value=value;input.readOnly=true;label.append(caption,input);return label;};
  const service=fixed("增强服务","本地 Llama"),template=fixed("约束模板","MOSS 官方规则 · 文生图"),check=make("button","检测依赖");check.type="button";check.className="lq21-upload";
  top.append(service,template,check);
  const modelRow=make("div",null,"display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) 85px;gap:7px;margin-top:8px");
  modelRow.append(field("增强模型","增强 LLM 模型（GGUF）"),field("增强视觉模型","视觉模型（mmproj）"),field("增强上下文","上下文"));
  const note=make("div",null,"font-size:10px;color:#91cdd2;margin:7px 0");
  const actions=make("div",null,"display:flex;align-items:center;gap:7px;flex-wrap:wrap;margin:9px 0");
  const button=text=>{const b=make("button",text);b.type="button";b.className="lq21-upload";return b;};
  const enhance=button("✦ Llama 提示词增强"),adopt=button("采用增强提示词"),original=button("直接采用原始创意"),rerun=make("label"),rerunInput=make("input");
  rerunInput.type="checkbox";rerunInput.setAttribute("aria-label","每次推理");rerun.append(make("span","每次推理"),rerunInput);
  rerun.style.cssText="display:flex;align-items:center;gap:6px;margin-left:auto";rerun.querySelector("span").style.margin="0";
  rerun.querySelector("input").style.cssText="width:16px;height:16px;accent-color:#16a6a8";
  actions.append(enhance,adopt,original,rerun);
  const status=make("div","等待增强，或直接填写提示词后生成。","font-size:10px;color:#91cdd2;margin:6px 0");
  const label=make("span","最终提示词预览（可编辑）");label.className="lq21-label";
  const preview=make("textarea");preview.className="lq21-preview";preview.style.height="220px";preview.placeholder="点击增强后预览，也可直接编辑并采用。";
  const request=()=>({mode:String(get("模式")?.value||"文生图"),prompt:String(get("提示词")?.value||"").trim(),layout:String(get("构图数据")?.value||"[]"),negative:String(get("负面提示词")?.value||"").trim(),model:String(get("增强模型")?.value||""),vision_model:String(get("增强视觉模型")?.value||"无"),context:Number(get("增强上下文")?.value||8192)});
  const key=async q=>{const raw=JSON.stringify([q.mode,q.prompt,q.mode==="构图生图"?q.layout:"[]",q.negative]);const digest=await crypto.subtle.digest("SHA-256",new TextEncoder().encode(raw));return [...new Uint8Array(digest)].map(v=>v.toString(16).padStart(2,"0")).join("");};
  let signature="",disposed=false;
  const showError=error=>{status.textContent=`增强失败：${error.message||error}；原始描述保持不变。`;};
  preview.oninput=()=>{set("增强提示词",preview.value);set("增强来源","");status.textContent="预览已编辑，点击“采用增强提示词”后用于生成。";};
  enhance.onclick=async()=>{
    const q=request();if(!q.prompt){status.textContent="请先填写画面描述。";return;}
    enhance.disabled=true;status.textContent="正在本地增强，请稍候…";
    try{
      const response=await api.fetchApi("/liao_moss/enhance",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(q)});
      if(response.status===404||response.status===405)throw new Error("请重启 ComfyUI 加载 MOSS 增强接口");
      const data=await response.json();if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);
      if(disposed)return;
      if(await key(request())!==data.key){status.textContent="原始描述或构图已变化，请重新增强。";return;}
      set("增强提示词",data.prompt);set("增强来源","");preview.value=data.prompt;
      status.textContent="增强完成，可编辑；点击“采用增强提示词”后，再运行生成。";changed();
    }catch(error){showError(error);}finally{enhance.disabled=false;}
  };
  adopt.onclick=async()=>{
    if(!preview.value.trim()){status.textContent="请先增强或填写预览提示词。";return;}
    const q=request();set("增强提示词",preview.value.trim());set("增强来源",await key(q));set("提示词增强",true);set("每次推理",false);
    refresh();status.textContent="已采用增强提示词，点击运行生成；原始创意未被覆盖。";
  };
  original.onclick=()=>{set("提示词增强",false);set("每次推理",false);set("增强来源","");refresh();status.textContent="已切回原始创意，点击运行生成。";};
  check.onclick=async()=>{try{const r=await api.fetchApi("/liao_moss/enhance/status");if(!r.ok)throw new Error("请重启 ComfyUI 加载依赖检测接口");const data=await r.json();status.textContent=data.message+(data.models?.length?` 检测到 ${data.models.length} 个 GGUF 模型。`:" 未检测到 GGUF 模型。");}catch(error){showError(error);}};
  const changeEnabled=enabled=>{
    set("提示词增强",enabled);
    if(!enabled)set("每次推理",false);
    refresh();status.textContent=enabled?"运行时自动增强已开启，也可先点击增强预览。":"增强已关闭，运行时使用原始创意。";
  };
  toggleInput.onchange=()=>changeEnabled(toggleInput.checked);
  toggle.onpointerdown=event=>event.stopPropagation();
  toggle.onclick=event=>{event.stopPropagation();if(event.target===toggleInput)return;event.preventDefault();changeEnabled(!asBool(get("提示词增强")?.value));};
  rerunInput.onchange=()=>{const enabled=rerunInput.checked;set("每次推理",enabled);if(enabled)set("提示词增强",true);refresh();status.textContent=enabled?"每次运行都会重新增强。":"可采用预览结果重复生成。";};
  rerun.onpointerdown=event=>event.stopPropagation();
  const onDone=event=>{const d=event.detail||{};if(String(d.node)!==String(node.id))return;set("增强提示词",d.prompt);set("增强来源",d.key);preview.value=d.prompt;status.textContent="运行时增强完成，已进入图像生成。";};
  api.addEventListener("liao_moss_prompt_enhanced",onDone);
  function refresh(){
    const q=request(),current=JSON.stringify([q.mode,q.prompt,q.mode==="构图生图"?q.layout:"[]",q.negative,q.model,q.vision_model,q.context]);
    if(signature&&signature!==current){set("增强来源","");status.textContent="描述、构图或模型已变化；运行时将重新增强，也可重新预览。";}
    signature=current;body.hidden=!asBool(get("提示词增强")?.value);
      card.classList.toggle("expanded",!body.hidden);
    toggleInput.checked=asBool(get("提示词增强")?.value);rerunInput.checked=asBool(get("每次推理")?.value);
    toggle.setAttribute("aria-expanded",String(!body.hidden));
    template.querySelector("input").value=q.mode==="构图生图"?"MOSS 官方规则 · 构图锁定":"MOSS 官方规则 · 文生图";
    note.textContent=q.mode==="构图生图"?"增强整体画面描述；程序保留框位置、人物/物体类别、配色和文字原文。纯文字增强可选“无”；mmproj 需与主模型配套。":"补充可见画面细节，保留主体、数量、风格、排除要求及指定文字；默认英文描述。纯文字增强可选“无”；mmproj 需与主模型配套。";
    if(document.activeElement!==preview)preview.value=String(get("增强提示词")?.value||"");
    changed();
  }
  body.append(top,modelRow,note,actions,status,label,preview);card.append(toggle,body);
  return {element:card,refresh,dispose(){disposed=true;api.removeEventListener("liao_moss_prompt_enhanced",onDone);}};
}
