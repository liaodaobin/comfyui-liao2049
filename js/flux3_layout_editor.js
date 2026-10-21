// Normalized boxes stay independent of display zoom and output resolution.
export function createLayoutEditor({get, set, changed, moss=false}) {
  const panel=document.createElement("div");panel.className="lq21-card";
  const make=(tag,text,css)=>{const el=document.createElement(tag);if(text)el.textContent=text;if(css)el.style.cssText=css;return el;};
  const toolbar=make("div",null,"display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-bottom:8px");
  let boxes=[], selected=-1, tool="obj", gesture=null, loaded="";
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const button=(text,action)=>{const b=make("button",text);b.type="button";b.className="lq21-upload";b.onclick=action;return b;};
  const drawObject=button("＋ 画物体框",()=>{tool="obj";paint();});
  const drawText=button("＋ 画文字框",()=>{tool="text";paint();});
  const move=button("选择 / 移动",()=>{tool="select";paint();});
  const add=button("添加居中框",()=>{if(boxes.length>=40)return;boxes.push({x:.25,y:.25,w:.35,h:.35,type:tool==="text"?"text":"obj",desc:"",text:"",direction:"horizontal"});selected=boxes.length-1;save();paint();});
  const remove=button("删除选中",()=>{if(selected<0)return;boxes.splice(selected,1);selected=Math.min(selected,boxes.length-1);save();paint();});
  toolbar.append(drawObject,drawText,move,add,remove);
  const note=make("div","拖出方框；拖动框可移动，右下角手柄可缩放。坐标引导构图，生成位置可能有偏差。","font-size:10px;color:#91cdd2;margin-bottom:8px");
  const canvasWrap=make("div",null,"display:flex;justify-content:center;background:#050d15;padding:10px;border:1px solid #28546b;border-radius:8px");
  const board=make("div",null,"position:relative;touch-action:none;user-select:none;overflow:hidden;flex:none;border:1px solid #48869b;background-color:#102534;background-image:linear-gradient(#31536666 1px,transparent 1px),linear-gradient(90deg,#31536666 1px,transparent 1px);background-size:10% 10%");
  board.setAttribute("aria-label","构图画布");canvasWrap.append(board);
  const size=make("div",null,"color:#91cdd2;font-size:10px;margin:5px 0 9px");
  const empty=make("div","在上方画布拖动鼠标，添加第一个物体或文字框。","font-size:11px;color:#91cdd2;margin:8px 0");
  const fields=make("div",null,"display:grid;grid-template-columns:1fr 1fr;gap:7px");
  const field=(title,input)=>{const label=make("label"),caption=make("span",title);caption.className="lq21-label";label.append(caption,input);fields.append(label);return label;};
  const list=make("select"),kind=make("select"),direction=make("select"),text=make("input"),desc=make("textarea");
  for(const [value,label] of [["obj","物体"],["text","文字"]]){const o=make("option",label);o.value=value;kind.append(o);}
  if(moss){const o=make("option","人物");o.value="character";kind.append(o);}
  for(const [value,label] of [["horizontal","横排"],["vertical","竖排"]]){const o=make("option",label);o.value=value;direction.append(o);}
  field("当前元素",list);field("元素类型",kind);
  const textField=field("文字内容（原样保留）",text),directionField=field("文字方向",direction);
  desc.className="lq21-prompt";desc.style.height="75px";desc.placeholder="例如：一只白色小熊玩具；文字框可填写字体、颜色等";
  const descField=field("物体描述 / 文字样式",desc);descField.style.gridColumn="1/-1";
  const palette=make("input");
  if(moss){palette.placeholder="#ffffff, #68a879";const p=field("元素配色（逗号分隔，可留空）",palette);p.style.gridColumn="1/-1";}
  const coordinates=make("div",null,"display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;grid-column:1/-1");
  const numbers={};
  for(const [key,label] of [["x","左 %"],["y","上 %"],["w","宽 %"],["h","高 %"]]){
    const wrap=make("label"),caption=make("span",label),input=make("input");caption.className="lq21-label";
    input.type="number";input.min="0";input.max="100";input.step="0.1";
    input.oninput=input.onchange=()=>{const b=boxes[selected];if(!b||input.value==="")return;const v=Number(input.value)/100;if(!Number.isFinite(v))return;
      if(key==="x")b.x=clamp(v,0,1-b.w);if(key==="y")b.y=clamp(v,0,1-b.h);
      if(key==="w")b.w=clamp(v,.005,1-b.x);if(key==="h")b.h=clamp(v,.005,1-b.y);
      save();paint();};numbers[key]=input;wrap.append(caption,input);coordinates.append(wrap);
  }
  fields.append(coordinates);
  const save=()=>{loaded=JSON.stringify(boxes);set("构图数据",loaded);changed();};
  const edit=()=>{const b=boxes[selected];if(!b)return;b.type=kind.value;b.text=text.value;b.desc=desc.value;b.direction=direction.value;if(moss)b.palette=palette.value.split(/[,，]/).map(v=>v.trim()).filter(Boolean);save();paintBoxes();textField.hidden=directionField.hidden=b.type!=="text";updateOptions();};
  kind.onchange=text.oninput=direction.onchange=desc.oninput=palette.oninput=edit;
  list.onchange=()=>{selected=Number(list.value);paint();};
  const updateOptions=()=>{list.replaceChildren();boxes.forEach((b,i)=>{const o=make("option",`${i+1}. ${b.type==="text"?"文字":b.type==="character"?"人物":"物体"} · ${(b.type==="text"?b.text:b.desc)||"待填写"}`);o.value=String(i);list.append(o);});list.value=String(selected);};
  const paintBoxes=()=>{
    board.replaceChildren();
    boxes.forEach((b,i)=>{
      const color=b.type==="text"?"#e9b55b":"#51d5d1";
      const box=make("div",null,`position:absolute;left:${b.x*100}%;top:${b.y*100}%;width:${b.w*100}%;height:${b.h*100}%;border:${selected===i?2:1}px solid ${color};background:${selected===i?"#49c7cf30":"#49c7cf12"};cursor:move;overflow:hidden`);
      box.dataset.index=String(i);
      const label=make("span",`${i+1} · ${(b.type==="text"?b.text:b.desc)|| (b.type==="text"?"文字":"物体")}`,`pointer-events:none;display:block;padding:3px;font-size:11px;line-height:1.3;color:${color};overflow:hidden;max-height:100%;white-space:pre-wrap;word-break:break-word`);
      if(b.type==="text"&&b.direction==="vertical"){
        label.textContent=b.text||"文字";label.style.writingMode="vertical-rl";label.style.textOrientation="upright";label.style.height="100%";
      }
      box.append(label);
      if(selected===i){const handle=make("span",null,`position:absolute;right:0;bottom:0;width:13px;height:13px;background:${color};cursor:nwse-resize`);handle.dataset.resize="true";box.append(handle);}
      board.append(box);
    });
    for(const key of Object.keys(numbers))numbers[key].value=boxes[selected]?String(Math.round(boxes[selected][key]*1000)/10):"";
  };
  const paint=()=>{
    size.textContent=size.textContent.replace(/·.*$/,`· ${boxes.length}/40 个元素`);
    updateOptions();paintBoxes();const b=boxes[selected];empty.hidden=boxes.length>0;fields.hidden=!b;remove.disabled=!b;add.disabled=boxes.length>=40;
    if(b){kind.value=b.type;desc.value=b.desc||"";text.value=b.text||"";direction.value=b.direction||"horizontal";if(moss)palette.value=(b.palette||[]).join(", ");textField.hidden=directionField.hidden=b.type!=="text";}
    for(const [b,name] of [[drawObject,"obj"],[drawText,"text"],[move,"select"]]){b.setAttribute("aria-pressed",String(tool===name));b.style.opacity=tool===name?"1":".55";}
    board.style.cursor=tool==="select"?"default":"crosshair";
  };
  const point=e=>{const r=board.getBoundingClientRect();return {x:clamp((e.clientX-r.left)/r.width,0,1),y:clamp((e.clientY-r.top)/r.height,0,1)};};
  board.onpointerdown=e=>{
    if(e.button!==0)return;e.preventDefault();e.stopPropagation();const p=point(e),item=e.target.closest("[data-index]");
    if(item){selected=Number(item.dataset.index);gesture={type:e.target.dataset.resize?"resize":"move",start:p,box:{...boxes[selected]}};}
    else if(tool!=="select"){
      if(boxes.length>=40){alert("最多添加40个构图元素。");return;}
      boxes.push({x:p.x,y:p.y,w:.001,h:.001,type:tool,desc:"",text:"",direction:"horizontal"});selected=boxes.length-1;gesture={type:"draw",start:p};
    }else{selected=-1;paint();return;}
    board.setPointerCapture(e.pointerId);paint();
  };
  board.onpointermove=e=>{
    if(!gesture)return;e.preventDefault();e.stopPropagation();const p=point(e),b=boxes[selected],g=gesture;
    if(g.type==="draw"){b.x=Math.min(g.start.x,p.x);b.y=Math.min(g.start.y,p.y);b.w=Math.abs(p.x-g.start.x);b.h=Math.abs(p.y-g.start.y);}
    if(g.type==="move"){b.x=clamp(g.box.x+p.x-g.start.x,0,1-b.w);b.y=clamp(g.box.y+p.y-g.start.y,0,1-b.h);}
    if(g.type==="resize"){b.w=clamp(g.box.w+p.x-g.start.x,.005,1-b.x);b.h=clamp(g.box.h+p.y-g.start.y,.005,1-b.y);}
    paintBoxes();
  };
  const finish=e=>{if(!gesture)return;e.preventDefault();e.stopPropagation();const b=boxes[selected];if(gesture.type==="draw"&&(b.w<.005||b.h<.005)){boxes.splice(selected,1);selected=boxes.length-1;}gesture=null;save();paint();if(board.hasPointerCapture(e.pointerId))board.releasePointerCapture(e.pointerId);};
  board.onpointerup=finish;board.onpointercancel=finish;
  const expansion=make("label",null,"display:flex;align-items:center;gap:7px;margin-top:10px");
  const enabled=make("input");enabled.type="checkbox";enabled.onchange=()=>{set("构图智能扩写",enabled.checked);refresh();};expansion.append(enabled,make("span","智能扩写物体描述（保留框位置和文字）"));
  const modelLabel=make("label"),modelCaption=make("span","扩写模型"),model=make("select");modelCaption.className="lq21-label";
  model.onchange=()=>set("构图扩写模型",model.value);modelLabel.style.marginTop="7px";modelLabel.append(modelCaption,model);
  const modelNote=make("div","可选模型：qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors；关闭扩写可直接按框生图。","font-size:10px;color:#91cdd2;margin-top:6px");
  panel.append(toolbar,note,canvasWrap,size,empty,fields);
  if(!moss)panel.append(expansion,modelLabel,modelNote);
  function refresh(){
    const raw=String(get("构图数据")?.value||"[]");
    if(raw!==loaded){try{const parsed=JSON.parse(raw);boxes=Array.isArray(parsed)?parsed:[];}catch{boxes=[];}loaded=raw;selected=clamp(selected,boxes.length?0:-1,boxes.length-1);}
    const ratio=String(get("画面比例")?.value||"1:1").match(/(\d+)\s*:\s*(\d+)/),rw=Number(ratio?.[1]||1),rh=Number(ratio?.[2]||1);
    const multiple=moss?16:8;
    const k=Math.sqrt(Number(get("百万像素")?.value||1.5)*1024*1024/(rw*rh));size.textContent=`输出画布 ${Math.max(multiple,Math.round(rw*k/multiple)*multiple)} × ${Math.max(multiple,Math.round(rh*k/multiple)*multiple)} · ${boxes.length}/40 个元素`;
    const maxWidth=460*rw/rh;board.style.width=`min(100%, ${maxWidth}px)`;board.style.aspectRatio=`${rw}/${rh}`;
    enabled.checked=Boolean(get("构图智能扩写")?.value);modelLabel.hidden=!enabled.checked;
    model.replaceChildren();for(const name of get("构图扩写模型")?.options?.values||["未选择"]){const o=make("option",name);o.value=name;model.append(o);}model.value=get("构图扩写模型")?.value||"未选择";
    paint();changed();
  }
  return {element:panel,refresh};
}
