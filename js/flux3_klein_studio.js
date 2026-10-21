import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { seedControl } from "./seed_control.js";
import { createLayoutEditor } from "./flux3_layout_editor.js";
import "./qwen_image_21_studio.js";

app.registerExtension({
  name: "liao2049.Flux3KleinStudio",
  async beforeRegisterNodeDef(type, data) {
    if (data.name !== "LiaoFlux3KleinStudio") return;
    const created = type.prototype.onNodeCreated;
    type.prototype.onNodeCreated = function () {
      created?.apply(this, arguments);
      const node = this;
      const widget = name => node.widgets.find(w => w.name === name);
      const set = (name, value) => { const w=widget(name); if(w) {w.value=value;w.callback?.(value);} node.graph?.setDirtyCanvas(true,true); };
      const root = document.createElement("div"); root.className="lq21";
      const fields = [];
      const field = (name, title=name) => {
        const w=widget(name), label=document.createElement("label"), caption=document.createElement("span");
        caption.className="lq21-label";caption.textContent=title;label.append(caption);
        let input;
        if(Array.isArray(w?.options?.values)) {
          input=document.createElement("select");
          for(const value of w.options.values) {const option=document.createElement("option");option.value=value;option.textContent=value;input.append(option);}
        } else { input=document.createElement("input");input.type=typeof w?.value==="boolean"?"checkbox":typeof w?.value==="number"?"number":"text";
          if(input.type==="number") {input.min=w.options?.min;input.max=w.options?.max;input.step=name.includes("强度")?.05:name==="百万像素"||name==="CFG"?.1:1;}
        }
        const refresh=()=>{if(input.type==="checkbox")input.checked=Boolean(w?.value);else input.value=w?.value??"";input.title=String(w?.value??"");};
        input.onchange=()=>{set(name,input.type==="checkbox"?input.checked:input.type==="number"?Number(input.value):input.value);refresh();};
        refresh();fields.push(refresh);label.append(input);return label;
      };
      const models=document.createElement("details");models.className="lq21-models";
      const head=document.createElement("summary");head.className="lq21-head";head.innerHTML="<span>Flux3 Klein · 模型配置</span><small>liao2049</small>";
      const modelGrid=document.createElement("div");modelGrid.className="lq21-advanced-grid";
      for(const name of ["扩散模型","文本编码器","VAE"])modelGrid.append(field(name));
      const loraGrid=document.createElement("div");
      loraGrid.style.cssText="grid-column:1/-1;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding-top:10px;margin-top:3px;border-top:1px solid #28546b";
      for(const index of [1,2]) {
        const row=document.createElement("div");
        row.style.cssText="display:grid;grid-template-columns:minmax(0,1fr) 70px;gap:6px;align-items:end;min-width:0";
        const strength=field(`LoRA${index}强度`,"强度");
        row.append(field(`LoRA${index}`,`LoRA ${index}`),strength);
        loraGrid.append(row);
      }
      modelGrid.append(loraGrid);
      models.append(head,modelGrid);
      const tabs=document.createElement("div");tabs.className="lq21-tabs";
      tabs.style.gridTemplateColumns="repeat(4,minmax(0,1fr))";
      for(const mode of ["文生图","单图编辑","多图编辑","构图生图"]) {const b=document.createElement("button");b.textContent=mode;b.onclick=()=>{set("模式",mode);refresh();};tabs.append(b);}
      const basic=document.createElement("div");basic.className="lq21-card lq21-grid";basic.style.gridTemplateColumns="1fr .8fr .65fr .55fr 1.3fr";
      const sizeMode=document.createElement("div");
      sizeMode.style.cssText="grid-column:1/-1;display:flex;align-items:center;gap:7px;padding:8px;border:1px solid #28546b;border-radius:8px;margin-bottom:3px";
      const sizeCaption=document.createElement("span");sizeCaption.textContent="编辑输出尺寸";sizeCaption.style.cssText="color:#91cdd2;white-space:nowrap";
      const followImage=document.createElement("button"), customRatio=document.createElement("button");
      followImage.textContent="按原图比例";customRatio.textContent="按设定比例";
      for(const button of [followImage,customRatio]) {button.type="button";button.className="lq21-upload";button.style.cssText="padding:6px 10px;cursor:pointer";}
      followImage.onclick=()=>{set("按首图比例",true);refresh();};
      customRatio.onclick=()=>{set("按首图比例",false);refresh();};
      sizeMode.append(sizeCaption,followImage,customRatio);
      const ratioField=field("画面比例");
      basic.append(sizeMode,ratioField,field("百万像素"),field("采样步数"),field("CFG"),seedControl(node,field("随机种子")));
      const refs=document.createElement("div");refs.className="lq21-card";
      const tools=document.createElement("div");tools.className="lq21-upload-row";
      const upload=document.createElement("button");upload.className="lq21-upload";upload.textContent="＋ 上传参考图";
      const paste=document.createElement("button");paste.className="lq21-upload";paste.textContent="粘贴";
      const note=document.createElement("span");note.className="lq21-note";note.textContent="可拖入或 Ctrl+V；多图最多10张";
      const picker=document.createElement("input");picker.type="file";picker.accept="image/*";picker.multiple=true;picker.hidden=true;
      const thumbs=document.createElement("div");thumbs.className="lq21-thumbs";
      const imageNames=()=>Array.from({length:10},(_,i)=>widget(`参考图${i+1}`)?.value).filter(v=>v&&v!=="未选择");
      const writeImages=names=>{for(let i=1;i<=10;i++)set(`参考图${i}`,names[i-1]||"未选择");refresh();};
      const addFiles=async files=>{
        const names=imageNames();
        try {for(const file of files) {
          if(!file.type.startsWith("image/"))continue;
          if(names.length>=10)throw new Error("参考图最多10张，请先删除已有图片。");
          const form=new FormData();form.append("image",file,file.name);form.append("type","input");
          const response=await api.fetchApi("/upload/image",{method:"POST",body:form});
          if(!response.ok)throw new Error(`上传失败：${response.status}`);
          const info=await response.json(), name=(info.subfolder?info.subfolder+"/":"")+info.name;
          names.push(name);for(let i=1;i<=10;i++) {const w=widget(`参考图${i}`);if(w&&!w.options.values.includes(name))w.options.values.push(name);}
          writeImages(names);
        }}catch(error){alert(error.message);}
      };
      upload.onclick=()=>picker.click();picker.onchange=()=>{addFiles([...picker.files]);picker.value="";};
      paste.onclick=async()=>{try {const files=[];for(const item of await navigator.clipboard.read()) {const mime=item.types.find(t=>t.startsWith("image/"));if(mime)files.push(new File([await item.getType(mime)],`clipboard-${Date.now()}.png`,{type:mime}));}if(!files.length)throw new Error("请先复制图片或截图。");await addFiles(files);}catch(error){alert(`无法粘贴：${error.message}，也可在参考区按 Ctrl+V。`);}};
      root.tabIndex=0;root.addEventListener("paste",e=>{if(["INPUT","TEXTAREA"].includes(e.target.tagName))return;const files=[...(e.clipboardData?.files||[])];if(files.length){e.preventDefault();e.stopPropagation();addFiles(files);}});
      refs.ondragover=e=>e.preventDefault();refs.ondrop=e=>{e.preventDefault();e.stopPropagation();addFiles([...(e.dataTransfer?.files||[])]);};
      tools.append(upload,paste,note,picker);refs.append(tools,thumbs);
      const promptCard=document.createElement("div");promptCard.className="lq21-card";
      const title=document.createElement("span");title.className="lq21-label";title.textContent="生成 / 编辑要求";
      const prompt=document.createElement("textarea");prompt.className="lq21-prompt";prompt.oninput=()=>set("提示词",prompt.value);
      const advanced=document.createElement("div"),summary=document.createElement("span");summary.className="lq21-label";summary.textContent="负面提示词";advanced.style.marginTop="9px";
      const negative=document.createElement("textarea");negative.className="lq21-prompt";negative.style.height="70px";negative.placeholder="填写不希望出现的内容（可留空）";negative.oninput=()=>set("负面提示词",negative.value);advanced.append(summary,negative);
      const layoutEditor=createLayoutEditor({get:widget,set,changed:()=>requestAnimationFrame(resize)});
      basic.addEventListener("change",()=>layoutEditor.refresh());
      promptCard.append(title,prompt,advanced);root.append(models,tabs,basic,refs,layoutEditor.element,promptCard);
      let panelHeight=600, dom, resizePending=false;
      const shell=document.createElement("div");
      shell.style.cssText="width:650px;min-width:650px;height:auto;overflow:visible";
      shell.append(root);
      const resize=()=>{
        if(resizePending)return;
        resizePending=true;
        requestAnimationFrame(()=>{
          resizePending=false;
          if(!root.isConnected)return;
          const width=Math.max(670,Number(node.size?.[0])||670), contentWidth=width-20;
          shell.style.width=root.style.width=`${contentWidth}px`;
          shell.style.height=root.style.height="auto";
          panelHeight=Math.max(1,Math.ceil(Math.max(root.scrollHeight,root.offsetHeight)));
          shell.style.height=`${panelHeight}px`;
          if(dom)dom.computedHeight=panelHeight;
          const height=panelHeight+75;
          if(Math.abs(node.size[0]-width)>1||Math.abs(node.size[1]-height)>3)node.setSize([width,height]);
          node.graph?.setDirtyCanvas(true,true);
        });
      };
      const refresh=()=>{
        fields.forEach(fn=>fn());prompt.value=widget("提示词")?.value||"";negative.value=widget("负面提示词")?.value||"";
        const mode=widget("模式")?.value;[...tabs.children].forEach(b=>b.classList.toggle("on",b.textContent===mode));refs.hidden=mode==="文生图"||mode==="构图生图";thumbs.replaceChildren();
        layoutEditor.element.hidden=mode!=="构图生图";
        if(mode==="构图生图")layoutEditor.refresh();
        const editing=["单图编辑","多图编辑"].includes(mode), follow=editing&&Boolean(widget("按首图比例")?.value);
        sizeMode.hidden=!editing;
        for(const [button,active] of [[followImage,follow],[customRatio,!follow]]) {
          button.setAttribute("aria-pressed",String(active));
          button.style.background=active?"linear-gradient(90deg,#078f94,#4058bb)":"#102434";
          button.style.borderColor=active?"#66e3d4":"#276c85";
          button.style.color=active?"#ffffff":"#91aab4";
        }
        const ratioSelect=ratioField.querySelector("select");if(ratioSelect)ratioSelect.disabled=follow;
        ratioField.style.opacity=follow?"0.4":"1";
        ratioField.title=follow?"当前按原图比例输出；点击‘按设定比例’可启用此处设置":"按选择的画面比例输出";
        imageNames().forEach((name,index)=>{const card=document.createElement("div");card.className="lq21-ref";
          const img=document.createElement("img"),parts=name.replaceAll("\\","/").split("/"),filename=parts.pop();img.src=api.apiURL(`/view?type=input&filename=${encodeURIComponent(filename)}&subfolder=${encodeURIComponent(parts.join("/"))}`);
          const remove=document.createElement("button");remove.className="lq21-remove";remove.textContent="×";remove.onclick=()=>{const names=imageNames();names.splice(index,1);writeImages(names);};
          const text=document.createElement("span");text.className="lq21-ref-label";text.textContent=`参考图${index+1}`;card.title=name;card.append(img,remove,text);thumbs.append(card);
        });requestAnimationFrame(resize);
      };
      for(const w of node.widgets) {w.hidden=true;w.computeSize=()=>[0,-4];w.draw=()=>{};}
      dom=node.addDOMWidget("flux3_klein_ui","div",shell,{serialize:false,hideOnZoom:false,getHeight:()=>panelHeight});dom.computeSize=()=>[Math.max(650,Number(node.size?.[0]||670)-20),panelHeight];
      node.__fluxRefresh=refresh;node.__fluxResize=resize;node.title="Liao-Flux3 Klein 便捷多功能生图";
      models.addEventListener("toggle",resize);
      const observer=new ResizeObserver(()=>requestAnimationFrame(resize));observer.observe(root);
      const removed=node.onRemoved;node.onRemoved=function(){observer.disconnect();removed?.apply(this,arguments);};
      refresh();setTimeout(resize,100);setTimeout(resize,350);
    };
    const configured=type.prototype.onConfigure;
    type.prototype.onConfigure=function(){const result=configured?.apply(this,arguments);this.__fluxRefresh?.();setTimeout(()=>this.__fluxResize?.(),0);setTimeout(()=>this.__fluxResize?.(),250);return result;};
    const resized=type.prototype.onResize;
    type.prototype.onResize=function(){const result=resized?.apply(this,arguments);this.__fluxResize?.();return result;};
    const graphConfigured=type.prototype.onGraphConfigured;
    type.prototype.onGraphConfigured=function(){const result=graphConfigured?.apply(this,arguments);setTimeout(()=>this.__fluxResize?.(),0);setTimeout(()=>this.__fluxResize?.(),250);return result;};
  }
});
