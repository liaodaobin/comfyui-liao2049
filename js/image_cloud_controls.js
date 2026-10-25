import { api } from "../../../scripts/api.js";
export function imageCloudControls(node,{onChange=()=>{},onStatus=()=>{}}={}){
  const get=name=>node.widgets?.find(w=>w.name===name);
  for(const [name,type,value,options] of [["增强服务","combo","本地 Llama",{values:["本地 Llama","DeepSeek API"]}],["云端增强模型","text","deepseek-flash",{}],["云端凭据会话","text","",{}]]){
    if(!get(name))node.addWidget(type,name,value,undefined,options);
  }
  const wrap=document.createElement("div");wrap.style.cssText="display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:8px";
  const field=(title,input)=>{const box=document.createElement("label"),caption=document.createElement("span");caption.className="lq21-label";caption.textContent=title;box.append(caption,input);wrap.append(box);return box;};
  const key=document.createElement("input");key.type="password";key.autocomplete="off";key.placeholder="本机浏览器保存，不写入工作流";
  const model=document.createElement("input");model.value=get("云端增强模型")?.value||"deepseek-flash";
  field("DeepSeek API Key（本机记忆）",key);field("云端模型",model);
  const storage="liao2049.h3.cloudApiKey";let ready=Promise.resolve(),version=0;
  try{key.value=localStorage.getItem(storage)||"";}catch{}
  const prepare=()=>{
    const current=++version,value=key.value.trim();get("云端凭据会话").value="";
    if(!value){ready=Promise.resolve();return ready;}
    ready=api.fetchApi("/liao/image/cloud-session",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({api_key:value})}).then(async response=>{
      if(!response.ok)throw new Error(response.status===404?"请重启 ComfyUI 加载云端增强接口":"本机凭据会话创建失败");
      const data=await response.json();if(current===version)get("云端凭据会话").value=data.token;
    }).catch(error=>{onStatus(error.message);});return ready;
  };
  key.oninput=()=>{const value=key.value.trim();try{if(value)localStorage.setItem(storage,value);else localStorage.removeItem(storage);}catch{};get("云端凭据会话").value="";version++;};
  key.onchange=()=>prepare();key.onblur=()=>prepare();
  model.onchange=()=>{get("云端增强模型").value=model.value.trim()||"deepseek-flash";onChange();};
  const provider=()=>get("增强服务")?.value||"本地 Llama";
  const refresh=()=>{wrap.hidden=provider()!=="DeepSeek API";};
  const configure=node.onConfigure;node.onConfigure=function(){const result=configure?.apply(this,arguments);get("云端凭据会话").value="";refresh();prepare();return result;};
  prepare();refresh();
  return {element:wrap,provider,refresh,setProvider(value){get("增强服务").value=value;refresh();onChange();},settings(){return {provider:provider(),cloud_model:model.value.trim()||"deepseek-flash",api_key:key.value.trim()};}};
}
