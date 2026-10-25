import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
export async function assetDropFiles(dt){
  const files=Array.from(dt?.files||[]);if(files.length)return files;
  let info={};try{info=JSON.parse(dt?.getData("application/x-comfy-asset-info")||"{}");}catch{}
  const uri=(dt?.getData("text/uri-list")||dt?.getData("text/x-moz-url")||"").split(/\r?\n/).find(x=>x&&!x.startsWith("#"));
  let src=info.media_url||uri||info.preview_url;
  if(!src&&info.filename)src=api.apiURL(`/view?${new URLSearchParams({filename:info.filename,subfolder:info.subfolder||"",type:info.type||"output"})}`);
  if(!src)return [];
  const url=new URL(src,location.href);if(url.origin!==location.origin)throw new Error("???????????");
  const response=await fetch(url);if(!response.ok)throw new Error(`???????${response.status}`);
  const blob=await response.blob();return [new File([blob],info.filename||info.display_name||url.searchParams.get("filename")||"asset.png",{type:blob.type})];
}
// Standalone Fibo2 workflows use ComfyUI's native LoadImage node.
app.registerExtension({name:"liao2049.AssetImageDrop",beforeRegisterNodeDef(type,data){
  if(data.name!=="LoadImage")return;
  const original=type.prototype.onDragDrop;
  type.prototype.onDragDrop=async function(event){
    if(!event.dataTransfer?.types?.includes("application/x-comfy-asset-info"))return original?.apply(this,arguments);
    try{
      const files=await assetDropFiles(event.dataTransfer);const file=files.find(f=>f.type.startsWith("image/"));if(!file)return false;
      const body=new FormData();body.append("image",file);body.append("type","input");
      const response=await api.fetchApi("/upload/image",{method:"POST",body});if(!response.ok)throw new Error(`?????${response.status}`);
      const result=await response.json();const value=result.subfolder?`${result.subfolder}/${result.name}`:result.name;
      const widget=this.widgets?.find(w=>w.name==="image");if(!widget)return false;
      if(Array.isArray(widget.options?.values)&&!widget.options.values.includes(value))widget.options.values.push(value);
      widget.value=value;widget.callback?.(value);this.graph?.setDirtyCanvas(true,true);return true;
    }catch(error){alert(error.message);return true;}
  };
}});
