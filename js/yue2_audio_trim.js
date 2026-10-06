export function attachAudioTrimmer({parent,player,get,set,status}) {
 const panel=document.createElement('div');panel.hidden=true;panel.style.cssText='padding:10px;margin-top:8px;background:#080d15;border:1px solid #3c566c;border-radius:10px';parent.append(panel);
 const canvas=document.createElement('canvas');canvas.width=1000;canvas.height=120;canvas.style.cssText='display:block;width:100%;height:120px;background:#080d15;border:1px solid #3c566c;border-radius:8px;cursor:crosshair;touch-action:none';canvas.setAttribute('aria-label','拖动波形选择原曲片段');panel.append(canvas);
 const playback=document.createElement('div');playback.style.cssText='display:flex;align-items:center;gap:5px;margin:8px 0;flex-wrap:nowrap;overflow-x:auto';panel.append(playback);
 const play=document.createElement('button');play.type='button';play.textContent='▶ 播放';play.style.cssText='margin:0;white-space:nowrap';playback.append(play);
 const clock=document.createElement('span');clock.style.cssText='font-size:12px;white-space:nowrap;color:#a7c6dd';clock.textContent='00:00.00 / 00:00.00';playback.append(clock);
 const markStart=document.createElement('button'),markEnd=document.createElement('button');
 for(const [b,title] of [[markStart,'设为起点'],[markEnd,'设为终点']]){b.type='button';b.textContent=title;b.style.cssText='margin:0;white-space:nowrap';b.style.padding='7px 8px';playback.append(b);}
 const parseTime=value=>{
  const text=String(value).trim();
  if(/^\d+(?:\.\d+)?$/.test(text))return Number(text);
  if(/^\d+:\d{1,2}(?:\.\d+)?$/.test(text)){const[m,sec]=text.split(':').map(Number);if(sec<60)return m*60+sec;}
  throw Error('请输入秒数（如90.5）或分:秒（如01:30.50）。');
 };
 const editBoundary=startBoundary=>{
  if(!buffer)return;
  const[start,end]=range();
  const input=globalThis.prompt(startBoundary?'输入起点时间：秒数或分:秒':'输入终点时间：秒数或分:秒',time(startBoundary?start:end));
  if(input===null)return;
  try{
   const t=parseTime(input),nextStart=startBoundary?t:start,nextEnd=startBoundary?end:t;
   if(!Number.isFinite(t)||t<0||t>buffer.duration)throw Error('时间需在原曲时长范围内。');
   if(nextEnd-nextStart<1)throw Error('终点需比起点至少晚1秒。');
   if(nextEnd-nextStart>900)throw Error('所选片段最长900秒。');
   segmentPlaying=false;select(nextStart,nextEnd);status.textContent='选段时间已更新。';
  }catch(error){status.textContent=error.message;}
 };
 markStart.textContent='起点选择';markEnd.textContent='终点选择';
 markStart.onclick=()=>editBoundary(true);markEnd.onclick=()=>editBoundary(false);
 const label=document.createElement('div');label.className='note';panel.append(label);
 const buttons=document.createElement('div');buttons.style.display='contents';playback.append(buttons);
 let buffer=null,peaks=[],source='',revision=0,abort=null,drag=null,segmentPlaying=false,dead=false;
 const range=()=>{const duration=buffer?.duration||0,start=Math.max(0,Math.min(Number(get('参考起点秒'))||0,Math.max(0,duration-1)));return [start,Math.min(duration,start+Math.max(1,Math.min(900,Number(get('参考长度秒'))||1)))];};
 const time=s=>`${Math.floor(s/60).toString().padStart(2,'0')}:${(s%60).toFixed(2).padStart(5,'0')}`;
 function draw(){
  if(!buffer)return;const [start,end]=range(),ctx=canvas.getContext('2d');
  label.textContent=`选段 ${time(start)} → ${time(end)} · ${(end-start).toFixed(2)}秒 / 全曲 ${time(buffer.duration)}　拖动选段，拖动两端调整`;
  if(!ctx)return;ctx.clearRect(0,0,1000,120);ctx.fillStyle='#142e46';ctx.fillRect(start/buffer.duration*1000,0,(end-start)/buffer.duration*1000,120);
  for(let i=0;i<peaks.length;i++){const t=i/peaks.length*buffer.duration,h=Math.max(2,peaks[i]*100);ctx.fillStyle=t>=start&&t<=end?'#398bf5':'#555b65';ctx.fillRect(i/peaks.length*1000,60-h/2,1.5,h);}
  ctx.fillStyle='#ec846d';for(const t of [start,end])ctx.fillRect(t/buffer.duration*1000-1,0,2,120);
  ctx.fillStyle='#eff7ff';ctx.fillRect((player.currentTime||0)/buffer.duration*1000,0,2,120);
  clock.textContent=`${time(player.currentTime||0)} / ${time(buffer.duration)}`;
 }
 function select(start,end){if(!buffer)return;start=Math.max(0,Math.min(start,buffer.duration-1));end=Math.min(buffer.duration,Math.max(start+1,Math.min(start+900,end)));set('参考起点秒',Number(start.toFixed(2)));set('参考长度秒',Number((end-start).toFixed(2)));draw();}
 const point=event=>{const rect=canvas.getBoundingClientRect();return Math.max(0,Math.min(buffer.duration,(event.clientX-rect.left)/rect.width*buffer.duration));};
 canvas.addEventListener('pointerdown',event=>{if(!buffer||buffer.duration<1)return;event.preventDefault();const t=point(event),[s,e]=range(),tolerance=buffer.duration*12/canvas.getBoundingClientRect().width;drag={mode:Math.abs(t-s)<tolerance?'start':Math.abs(t-e)<tolerance?'end':'new',anchor:t,start:s,end:e};canvas.setPointerCapture(event.pointerId);});
 canvas.addEventListener('pointermove',event=>{if(!drag||!buffer)return;const t=point(event);if(drag.mode==='start')select(Math.min(t,drag.end-1),drag.end);else if(drag.mode==='end')select(drag.start,Math.max(t,drag.start+1));else select(Math.min(t,drag.anchor),Math.max(t,drag.anchor));});
 const finish=event=>{drag=null;if(canvas.hasPointerCapture?.(event.pointerId))canvas.releasePointerCapture(event.pointerId);};canvas.addEventListener('pointerup',finish);canvas.addEventListener('pointercancel',finish);
 const button=(title,fn)=>{const b=document.createElement('button');b.type='button';b.textContent=title;b.onclick=fn;b.style.cssText='margin:0;padding:7px 8px;white-space:nowrap;flex-shrink:0';buttons.append(b);return b;};
 button('随机选段',()=>{if(!buffer)return;const length=Math.min(buffer.duration,Math.max(1,Math.min(900,Number(get('参考长度秒'))||30))),start=Math.random()*Math.max(0,buffer.duration-length);select(start,start+length);});
 button('试听选段',async()=>{if(!buffer)return;const[s]=range();player.pause();player.currentTime=s;segmentPlaying=true;try{await player.play();}catch(error){segmentPlaying=false;status.textContent='试听失败：'+error.message;}});
 play.onclick=async()=>{if(!source)return;try{if(!player.paused)player.pause();else{segmentPlaying=false;await player.play();}}catch(error){status.textContent='播放失败：'+error.message;}};

 function onTime(){if(segmentPlaying&&player.currentTime>=range()[1]){player.pause();segmentPlaying=false;}draw();}function onPause(){segmentPlaying=false;play.textContent='▶ 播放';}function onPlay(){play.textContent='❚❚ 暂停';}
 player.addEventListener('timeupdate',onTime);player.addEventListener('pause',onPause);player.addEventListener('play',onPlay);player.addEventListener('ended',onPause);
 button('下载选段 WAV',()=>{
  if(!buffer)return;const[s,e]=range(),rate=buffer.sampleRate,first=Math.round(s*rate),last=Math.min(buffer.length,Math.round(e*rate)),frames=last-first,channels=buffer.numberOfChannels;
  const bytes=new ArrayBuffer(44+frames*channels*2),view=new DataView(bytes);const str=(offset,value)=>{for(let i=0;i<value.length;i++)view.setUint8(offset+i,value.charCodeAt(i));};
  str(0,'RIFF');view.setUint32(4,bytes.byteLength-8,true);str(8,'WAVE');str(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,channels,true);view.setUint32(24,rate,true);view.setUint32(28,rate*channels*2,true);view.setUint16(32,channels*2,true);view.setUint16(34,16,true);str(36,'data');view.setUint32(40,frames*channels*2,true);
  const samples=Array.from({length:channels},(_,c)=>buffer.getChannelData(c));let offset=44;for(let i=first;i<last;i++)for(let c=0;c<channels;c++){const v=Math.max(-1,Math.min(1,samples[c][i]));view.setInt16(offset,Math.round(v*(v<0?32768:32767)),true);offset+=2;}
  const a=document.createElement('a'),url=URL.createObjectURL(new Blob([bytes],{type:'audio/wav'}));a.href=url;a.download=`原曲选段_${s.toFixed(2)}-${e.toFixed(2)}.wav`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 });
 async function load(url){
  if(source===url)return;source=url;const current=++revision;abort?.abort();abort=new AbortController();buffer=null;panel.hidden=!url;
  if(!url)return;label.textContent='正在读取波形…';let context;
  try{const response=await fetch(url,{signal:abort.signal});if(!response.ok)throw Error('HTTP '+response.status);const data=await response.arrayBuffer();const AudioCtx=globalThis.AudioContext||globalThis.webkitAudioContext;if(!AudioCtx)throw Error('浏览器不支持音频波形解码');context=new AudioCtx();const decoded=await context.decodeAudioData(data);if(current!==revision||dead)return;buffer=decoded;peaks=[];const tracks=Array.from({length:buffer.numberOfChannels},(_,c)=>buffer.getChannelData(c)),bins=1000;
   for(let i=0;i<bins;i++){let peak=0;const start=Math.floor(i/bins*buffer.length),end=Math.floor((i+1)/bins*buffer.length),step=Math.max(1,Math.floor((end-start)/512));for(let j=start;j<end;j+=step)for(const track of tracks)peak=Math.max(peak,Math.abs(track[j]));peaks.push(peak);}
   draw();
  }catch(error){if(current===revision&&!dead&&error.name!=='AbortError')label.textContent='波形读取失败：'+error.message+'；仍可填写起点和长度选择片段。';}finally{await context?.close();}
 }
 return {load,draw,select,range,buttons,destroy(){dead=true;++revision;abort?.abort();buffer=null;peaks=[];player.removeEventListener('timeupdate',onTime);player.removeEventListener('pause',onPause);player.removeEventListener('play',onPlay);player.removeEventListener('ended',onPause);}};
}
