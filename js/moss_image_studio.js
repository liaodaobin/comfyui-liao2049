import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { seedControl } from "./seed_control.js";
import { createLayoutEditor } from "./flux3_layout_editor.js";
import { createMossEnhancer } from "./moss_prompt_enhance.js";
import "./qwen_image_21_studio.js";

const PROFILES=["SFT 标准","Base 标准","SFT 极速8步"];
app.registerExtension({
  name:"liao2049.MossImageStudio",
  async beforeRegisterNodeDef(type,data){
    if(data.name!=="LiaoMossImageStudio")return;
    // Older running backends do not advertise this newly added optional widget yet.
    if(!data.input.optional?.["增强视觉模型"]){
      let visual=[];
      try{const response=await api.fetchApi("/models/LLM");if(response.ok)visual=(await response.json()).filter(n=>/mmproj/i.test(n)&&/\.gguf$/i.test(n));}catch(error){console.warn("[Liao MOSS] 视觉模型列表读取失败",error);}
      data.input.optional??={};
      data.input.optional["增强视觉模型"]=[["无",...visual],{default:"无"}];
    }
    for(const i of [3,4,5]){
      data.input.optional??={};
      data.input.optional[`LoRA${i}`]??=[data.input.required.LoRA1[0],{default:"不使用"}];
      data.input.optional[`LoRA${i}强度`]??=["FLOAT",{default:1,min:-4,max:4,step:.05}];
    }
    const created=type.prototype.onNodeCreated;
    type.prototype.onNodeCreated=function(){
      created?.apply(this,arguments);
      const node=this, get=name=>node.widgets.find(w=>w.name===name);
      for(const [name,spec] of Object.entries(data.input?.optional||{})){
        if(get(name))continue;
        const [kind,options={}]=spec,values=Array.isArray(kind)?kind:null;
        const widgetType=values?"combo":kind==="BOOLEAN"?"toggle":kind==="INT"||kind==="FLOAT"?"number":"text";
        node.addWidget(widgetType,name,options.default??(values?.[0]??""),undefined,{...options,...(values?{values}:{})});
      }
      const set=(name,value)=>{const w=get(name);if(w){w.value=value;w.callback?.(value);}node.graph?.setDirtyCanvas(true,true);};
      const make=(tag,text,css)=>{const e=document.createElement(tag);if(text)e.textContent=text;if(css)e.style.cssText=css;return e;};
      const root=make("div");root.className="lq21 liao-moss-workbench";
      node.drawBadges=()=>{};
      if(!document.getElementById("liao-moss-badge-style")){
        const style=make("style");style.id="liao-moss-badge-style";
        style.textContent="[data-node-id]:has(.liao-moss-workbench) .h-5.w-full.gap-2.px-2.text-muted-foreground{display:none!important}";
        document.head.append(style);
      }
      const refreshers=[];
      const field=(name,title=name)=>{
        const w=get(name),spec=data.input?.required?.[name]||data.input?.optional?.[name],label=make("label"),caption=make("span",title);caption.className="lq21-label";
        let input;
        if(Array.isArray(w?.options?.values)){
          input=make("select");for(const v of w.options.values){const o=make("option",v);o.value=v;input.append(o);}
        }else{
          input=make("input");input.type=spec?.[0]==="BOOLEAN"||typeof w?.value==="boolean"?"checkbox":["INT","FLOAT"].includes(spec?.[0])||typeof w?.value==="number"?"number":"text";
          if(input.type==="number"){input.min=w.options?.min??0;input.max=w.options?.max??1000;input.step=w.options?.step??1;}
        }
        const sync=()=>{if(input.type==="checkbox")input.checked=w?.value===true||w?.value===1||["true","1","on"].includes(String(w?.value).toLowerCase());else input.value=w?.value??"";input.title=String(w?.value??"");};
        input.onchange=()=>{set(name,input.type==="checkbox"?input.checked:input.type==="number"?Number(input.value):input.value);refresh();};
        refreshers.push(sync);sync();label.append(caption,input);return label;
      };
      const grid=(names,columns=2)=>{const e=make("div",null,`display:grid;grid-template-columns:repeat(${columns},minmax(0,1fr));gap:8px`);for(const n of names)e.append(field(n));return e;};
      const card=()=>{const e=make("div");e.className="lq21-card";return e;};
      const models=make("details");models.className="lq21-models";
      const head=make("summary");head.className="lq21-head";head.append(make("span","MOSS Image · 模型配置"),make("small","liao2049"));
      const modelGrid=grid(["Base模型","SFT模型","文本编码器","VAE"]);
      modelGrid.style.padding="10px";
      const loras=make("div",null,"grid-column:1/-1;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;border-top:1px solid #28546b;padding-top:9px");
      for(const i of [1,2,3,4,5]){const row=make("div",null,"display:grid;grid-template-columns:minmax(0,1fr) 65px;gap:6px;align-items:end");row.append(field(`LoRA${i}`),field(`LoRA${i}强度`,"强度"));loras.append(row);}
      const sampling=make("div",null,"grid-column:1/-1;display:grid;gap:8px;border-top:1px solid #28546b;padding-top:9px");
      sampling.setAttribute("aria-label","采样与 SLG 设置");
      const samplerRow=make("div",null,"display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr) 80px;gap:8px;align-items:end");
      const slgSwitch=field("SLG");
      slgSwitch.style.cssText="display:flex;align-items:center;justify-content:center;gap:7px;height:34px;border:1px solid #326479;border-radius:7px;cursor:pointer";
      slgSwitch.querySelector("span").style.margin="0";
      slgSwitch.querySelector("input").style.cssText="width:16px;height:16px;margin:0;accent-color:#16a6a8";
      samplerRow.append(field("采样器"),field("调度器"),slgSwitch);
      sampling.append(samplerRow,grid(["SLG双流层","SLG单流层","SLG开始","SLG结束"],4));
      const fastRow=make("div",null,"display:grid;grid-template-columns:minmax(0,1fr) 65px;gap:6px;align-items:end");
      fastRow.append(field("极速LoRA","极速 LoRA（仅极速8步生效）"),field("极速LoRA强度","强度"));
      loras.prepend(fastRow);
      modelGrid.append(loras,sampling);models.append(head,modelGrid);
      const tabs=make("div");tabs.className="lq21-tabs";tabs.style.gridTemplateColumns="repeat(2,minmax(0,1fr))";
      for(const mode of ["文生图","构图生图"]){const b=make("button",mode);b.onclick=()=>{set("模式",mode);refresh();};tabs.append(b);}
      const plans=card(),buttons=make("div",null,"display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px");
      for(const profile of PROFILES){const b=make("button",profile);b.className="lq21-upload";b.onclick=()=>{
        set("生成方案",profile);const fast=profile==="SFT 极速8步";
        set("采样步数",fast?8:50);set("CFG",fast?1:3.5);set("采样器",fast?"er_sde":get("模式")?.value==="构图生图"?"er_sde":"euler");set("调度器","simple");refresh();
      };buttons.append(b);}
      const planNote=make("div",null,"font-size:11px;color:#91cdd2;margin-top:8px");plans.append(buttons,planNote);
      const basic=card();const basicGrid=make("div",null,"display:grid;grid-template-columns:1fr .75fr .7fr .55fr 1.2fr;gap:8px");
      basicGrid.append(field("画面比例"),field("百万像素"),field("采样步数"),field("CFG"),seedControl(node,field("随机种子")));basic.append(basicGrid);
      const layout=createLayoutEditor({get,set,changed:()=>{resize();enhancer.refresh();},moss:true});
      const typography=card(),fontOptions=grid(["文字绘制","排字字体"]),clear=field("清除框内原字");
      clear.style.cssText="display:flex;align-items:center;gap:8px;margin-top:8px";
      const fontNote=make("div","精确排字使用真实字体，只绘制“文字内容”。按描述自动匹配支持楷体、宋体、黑体、微软雅黑；文字框配色决定字色。清除原字会用纯色底板覆盖框内画面。","font-size:11px;color:#91cdd2;margin-top:7px");
      typography.append(fontOptions,clear,fontNote);
      const promptCard=card(),prompt=make("textarea"),negative=make("textarea");
      prompt.className=negative.className="lq21-prompt";negative.style.height="68px";prompt.placeholder="描述主体、场景、风格和需要呈现的文字";negative.placeholder="负面提示词（可留空）";
      prompt.oninput=()=>{set("提示词",prompt.value);enhancer.refresh();};negative.oninput=()=>{set("负面提示词",negative.value);enhancer.refresh();};
      for(const [title,input] of [["画面描述",prompt],["负面提示词",negative]]){const caption=make("span",title);caption.className="lq21-label";promptCard.append(caption,input);}
      const enhancer=createMossEnhancer({node,get,set,field,changed:()=>resize()});
      root.append(models,tabs,plans,basic,typography,layout.element,promptCard,enhancer.element);
      let panelHeight=700,dom,pending=false;
      const shell=make("div",null,"width:700px;min-width:650px;overflow:visible");shell.append(root);
      const resize=()=>{if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;if(!root.isConnected)return;const width=Math.max(720,Number(node.size?.[0])||720);shell.style.width=root.style.width=`${width-20}px`;shell.style.height=root.style.height="auto";panelHeight=Math.ceil(root.scrollHeight);shell.style.height=`${panelHeight}px`;if(dom)dom.computedHeight=panelHeight;if(Math.abs(node.size[1]-panelHeight-75)>3)node.setSize([width,panelHeight+75]);node.graph?.setDirtyCanvas(true,true);});};
      const refresh=()=>{
        refreshers.forEach(f=>f());prompt.value=get("提示词")?.value||"";negative.value=get("负面提示词")?.value||"";
        if(["洗图","Fibo2文生图","单图编辑（Fibo2）","多图编辑（Fibo2）"].includes(get("模式")?.value))set("模式","文生图");
        const mode=get("模式")?.value,profile=get("生成方案")?.value;
        for(const b of tabs.children)b.classList.toggle("on",b.textContent===mode);
        for(const b of buttons.children){const active=b.textContent===profile;b.setAttribute("aria-pressed",String(active));b.style.background=active?"linear-gradient(90deg,#078f94,#4058bb)":"#102434";}
        fastRow.style.opacity=profile==="SFT 极速8步"?"1":"0.55";layout.element.hidden=mode!=="构图生图";
        typography.hidden=mode!=="构图生图";
        const exact=get("文字绘制")?.value==="字体精确排字";
        fontOptions.children[1].hidden=clear.hidden=!exact;
        fontNote.textContent=exact?"精确排字使用真实字体，只绘制“文字内容”。按描述自动匹配支持楷体、宋体、黑体、微软雅黑；文字框配色决定字色。清除原字会用纯色底板覆盖框内画面。":"模型生成文字沿用原版流程，字体样式由模型理解，可能出现多字或字体不准确。";
        if(mode==="构图生图")layout.refresh();
        enhancer.refresh();
        planNote.textContent=profile==="SFT 极速8步"?"原工作流：SFT + 极速LoRA，8步 / CFG 1 / er_sde；参数仍可手动调整。":"原工作流标准参数：50步 / CFG 3.5；可在下方自定义。";
        resize();
      };
      for(const w of node.widgets){w.hidden=true;w.computeSize=()=>[0,-4];w.draw=()=>{};}
      dom=node.addDOMWidget("moss_image_ui","div",shell,{serialize:false,hideOnZoom:false,getHeight:()=>panelHeight});dom.computeSize=()=>[Math.max(650,(node.size?.[0]||720)-20),panelHeight];
      node.title="Liao-MOSS Image 便捷多功能生图";node.__mossRefresh=refresh;node.__mossResize=resize;
      models.addEventListener("toggle",resize);const observer=new ResizeObserver(resize);observer.observe(root);
      const removed=node.onRemoved;node.onRemoved=function(){observer.disconnect();enhancer.dispose();removed?.apply(this,arguments);};
      refresh();setTimeout(resize,100);setTimeout(resize,350);
    };
    for(const method of ["onConfigure","onGraphConfigured"]){const previous=type.prototype[method];type.prototype[method]=function(){if(method==="onConfigure"&&arguments[0]?.widgets_values?.length===36)arguments[0].widgets_values.splice(23,8);const r=previous?.apply(this,arguments);if(this.outputs?.length>1){for(let i=this.outputs.length-1;i>0;i--)this.removeOutput(i);}this.__mossRefresh?.();setTimeout(()=>this.__mossResize?.(),100);return r;};}
    const resized=type.prototype.onResize;type.prototype.onResize=function(){const r=resized?.apply(this,arguments);this.__mossResize?.();return r;};
  }
});
