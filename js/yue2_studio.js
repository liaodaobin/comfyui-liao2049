import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { attachAudioTrimmer } from './yue2_audio_trim.js';
const presets = {"官方默认·夏日独立流行": "Upbeat indie pop with warm female vocals, bright electric guitars, punchy drums, melodic bass, and subtle synth layers. Catchy and energetic, with an uplifting summer atmosphere, a memorable chorus, and polished modern production.", "流行情歌": "Contemporary pop ballad, intimate expressive lead vocals, piano-led verses with rounded bass and restrained drums. Straight ballad pulse, sustained melodic phrases and space for breath. Strings enter in the second verse; the chorus opens into a fuller band, followed by a quiet final resolution. Clear close vocals and controlled reverb.", "木吉他民谣": "Acoustic storytelling folk, natural conversational lead vocals, fingerpicked steel-string guitar and upright bass. Gentle alternating-bass pulse, subtle hand percussion and short harmonica answers. Keep verses spare, add a second guitar in the refrain, and finish with an exposed acoustic ending. Organic room sound and modest dynamics.", "史诗悲壮": "Tragic cinematic ballad, restrained low-register verses rising into an anguished sustained chorus. Somber piano, low strings, French horns and distant orchestral drums; broad half-time pulse with long pauses. Build tension through descending melodic gestures and layered strings, then end on a subdued unresolved feeling. Dark spacious orchestral sound.", "钢琴抒情": "Intimate piano ballad, softly articulated lead vocals with unhurried phrasing. Delicate arpeggiated piano, cello counterlines and occasional brushed cymbals; minimal bass movement. Leave silence between phrases, develop the melody through small variations, and keep the refrain lyrical rather than explosive. Close piano detail and a small warm room.", "热血摇滚": "Arena rock, forceful sustained lead vocals, distorted rhythm guitars, driving eighth-note bass and strong snare backbeats. Palm-muted verses expand into open power chords and a chant-like chorus; add a melodic guitar solo before the final refrain. Tight live-band energy, big drums and clearly separated guitars.", "慢摇电子": "Mid-tempo melodic electronic pop, smooth restrained lead vocals, deep synth bass, soft pads and a steady four-on-the-floor kick. Gentle sidechain motion, sparse syncopated plucks and a repeated melodic hook. Filtered verses lead into a restrained instrumental drop; avoid an aggressive festival climax. Deep clean low end and spacious stereo textures.", "国风流行": "Contemporary Chinese-inspired pop, graceful lyrical lead vocals, guzheng figures, erhu responses, piano and modern drums. Flowing pentatonic phrases over a steady pop backbeat; leave space for traditional instruments between sung lines. A restrained verse grows into a sweeping melodic chorus. Balance acoustic ornamentation with a clear modern rhythm section.", "复古港风": "Classic Hong Kong pop ballad, expressive sustained lead vocals with measured vibrato. Electric piano, clean guitar fills, melodic electric bass, live drums and lush strings. A broad straight ballad groove, long vocal arcs and a brief instrumental interlude. Gradually widen the final chorus. Warm tape-like texture and rich studio reverb.", "R&B灵魂": "Contemporary slow R&B, intimate lead vocals with restrained melisma and breathy phrase endings. Rhodes chords, deep sub bass, sparse syncopated kick and snare, and silky pads. Laid-back pocket, offbeat vocal entries and short call-and-response harmonies. Keep the hook compact and the arrangement uncluttered. Dry close vocals with spacious background textures.", "爵士酒馆": "Small-club jazz ballad, smoky conversational lead vocals, acoustic piano trio, upright bass and brushed drums. Gentle swing and relaxed behind-the-beat phrasing; saxophone answers only between vocal lines. Spare verses, tasteful piano voicings and a short improvised instrumental passage. Warm room ambience, natural dynamics and no oversized pop climax.", "城市放克": "Urban funk-pop, rhythmic articulated lead vocals, muted sixteenth-note guitar, slap bass, dry drums and punchy brass stabs. Tight syncopation, stop-time breaks and a bass-led groove. Use short repeated vocal hooks and instrumental responses; keep the arrangement crisp rather than lush. Dry upfront rhythm section and sharply defined attacks.", "梦幻流行": "Accessible dream pop, soft floating lead vocals, chorus-treated guitars, gentle synth pads and a steady understated drum groove. Clear melodic hooks with long shimmering tails. Keep verses intimate, widen the chorus with stacked textures, and finish with a drifting fade. Smooth luminous production while retaining a clear vocal melody.", "伤感慢歌": "Sparse melancholic pop ballad, fragile close lead vocals, low piano chords and occasional muted guitar. Slow straight pulse with minimal percussion; descending phrases, hesitant entrances and long held endings. Keep the first chorus almost bare, add restrained strings late, and end with voice and piano. Intimate dry detail rather than cinematic grandeur.", "治愈轻音乐歌": "Gentle acoustic vocal pop, soothing softly articulated lead vocals, fingerpicked guitar, light piano and rounded bass. An easy even pulse with quiet shaker and brushed snare. Short lyrical phrases, a modest comforting refrain and small instrumental answers. Keep the energy calm throughout, with clear natural acoustics and no dramatic build.", "流行朋克": "Fast pop punk, youthful direct lead vocals, crunchy power chords, pick-played bass and rapid live drums. Driving eighth-note guitars, energetic snare backbeats and quick fills. Tight verse-to-chorus transitions, a shout-along refrain and a half-time bridge before the final chorus. Bright aggressive band sound with little ambient padding.", "重金属": "Melodic heavy metal, powerful gritty lead vocals, tightly doubled distorted guitars, double-kick drums and heavy bass. Precise riff-driven rhythm, alternating chugging passages and sustained melodic phrases. Add harmonized guitar leads and a brief solo; use a dramatic half-time breakdown before the final refrain. Dense but articulate guitars and controlled low end.", "嘻哈说唱": "Melodic hip-hop, clearly articulated rhythmic verses and a compact sung hook. Dry kick and snare, deep 808 bass, a sparse piano sample and subtle atmospheric textures. Syncopated flows with deliberate rests and small hi-hat variations. Drop the drums for a short transition, then return to the main groove. Upfront vocal intelligibility and uncluttered production.", "拉丁夏日": "Acoustic Latin pop, light playful lead vocals, nylon-string guitar, shakers, hand percussion and melodic bass. Breezy syncopated groove with gentle bossa-influenced guitar patterns. Easy melodic verses, a relaxed singalong refrain and a short guitar interlude. Warm open-air atmosphere and light acoustic production.", "乡村公路": "Modern country road song, warm storytelling lead vocals, strummed acoustic guitar, pedal steel, electric bass and steady live drums. A rolling train-like pulse, clear syllabic phrasing and twangy instrumental fills. Add fiddle or steel responses between lines, with a broad open chorus and a brief guitar solo. Natural band sound and warm vocal presence.", "电影主题曲": "Cinematic orchestral pop theme, lyrical lead vocals and a clear recurring melody. Piano introduces the theme; strings, woodwinds and orchestral percussion expand it gradually. Flowing measured pulse, long melodic arcs and a quieter bridge before the thematic return. Emphasize emotional continuity and orchestral color over a heavy drum groove.", "LoFi卧室流行": "Lo-fi bedroom pop, intimate lightly sung lead vocals, muted guitar, soft electric piano and dusty sampled drums. Slightly loose relaxed groove, small repetitive melodic cells and restrained bass. Keep the chorus understated, with a brief instrumental loop and a gentle ending. Narrow cozy textures, subtle tape wear and close informal vocals.", "迪斯科舞曲": "Contemporary disco-pop, confident rhythmic lead vocals, four-on-the-floor kick, clean funk guitar, active bass and bright synth hooks. Crisp dance-floor pulse with handclaps and short rhythmic vocal phrases. Build into a hook-led chorus and a bass-focused dance break. Tight modern low end and sparkling clean percussion.", "华语青春校园": "Youthful campus acoustic pop, clear unforced lead vocals, bright strummed guitar, piano and a light band groove. Straight buoyant pulse and easy syllabic melodies. Storytelling verses lead to a simple communal singalong chorus; add light backing harmonies near the end. Fresh acoustic detail, natural room sound and hopeful restraint.", "都市夜色City Pop": "Late-night Japanese city-pop-inspired arrangement, smooth melodic lead vocals, electric piano, clean chorus guitar, rounded fingerstyle bass and relaxed live drums. Subtle syncopation with jazz-colored chords and saxophone fills. Silky verses, a breezy melodic chorus and a polished instrumental interlude. Warm analog sheen and spacious night-drive ambience.", "梦幻独立Dream Pop": "Atmospheric indie dream pop, hushed distant lead vocals, delay-soaked guitars, slow-moving synth clouds and minimal drums. Hypnotic repeating figures, blurred phrase boundaries and a patient gradual build instead of a conventional large chorus. Let guitar textures carry the transitions. Hazy wide ambience and subdued vocal presence.", "英伦另类摇滚": "British alternative guitar rock, earnest slightly gritty lead vocals, jangly rhythm guitar, melodic bass and lively live drums. Propulsive straight groove with a few offbeat guitar accents. Narrow tense verses open into a wide melodic refrain; add a contrasting guitar countermelody and a brief stripped bridge. Raw-edged band dynamics with clear vocal focus.", "后摇氛围": "Post-rock vocal song, restrained sparse lead vocals, interlocking delay guitars, deep bass and slowly developing live drums. Repeating melodic motifs evolve through texture and dynamics rather than frequent hooks. Begin nearly bare, build in long waves to a guitar-led climax, then decay into ambience. Wide spatial sound and strong quiet-to-loud contrast.", "灵魂福音": "Soul-gospel song, expressive lead vocals with emphatic phrase endings and responsive backing harmonies. Piano, Hammond organ, bass and live drums. A rolling soulful groove, call-and-response refrains and gradual harmonic intensity. Start with solo voice and piano, then add choir responses and organ swells. Organic ensemble energy and rich vocal blend.", "雷鬼阳光": "Roots-reggae-inspired pop, relaxed rhythmic lead vocals, offbeat guitar chops, bubbling organ, deep bass and spacious drums. A laid-back one-drop feel with clear gaps in the rhythm. Short melodic hooks, bass-led instrumental passages and occasional percussion accents. Warm low end, dry guitar skank and generous dub-like space.", "拉丁热情": "Energetic salsa-inspired Latin pop, assertive rhythmic lead vocals, piano montuno patterns, tumbao bass, congas, timbales and brass. Interlocking syncopation and bright percussion breaks. Use a responsive chorus, short brass answers and an extended rhythmic dance passage. Live ensemble energy and clear separation of percussion layers.", "探戈戏剧": "Dramatic tango song, precise emotionally charged lead vocals, bandoneon, violin, piano and double bass. Sharp accented pulses, sudden pauses and alternating clipped and sustained phrases. Tense verses move into an expansive melodic refrain, followed by a bandoneon-led interlude. Intimate theatrical acoustics with strong dynamic contrasts.", "凯尔特幻想": "Celtic folk song, clear flowing lead vocals, fiddle, tin whistle, harp, acoustic guitar and bodhran. Lilting compound-feel rhythm and ornamented instrumental responses. Narrative verses lead into a communal refrain; alternate sung passages with a lively fiddle-and-whistle interlude. Natural acoustic blend and an adventurous open landscape feel.", "东方禅意": "Contemplative East Asian acoustic song, quiet unhurried lead vocals, guqin, bamboo flute and sparse low strings. Spacious phrasing with soft isolated percussion and long silences. Keep melodies modest and ornamentation delicate; let flute answer the voice without a large pop chorus. Dry plucked-string detail and gentle natural reverberation.", "武侠江湖": "Wuxia cinematic song, bold expressive lead vocals, erhu, bamboo flute, pipa and orchestral strings. Strong martial drum accents alternate with lyrical spacious passages. Pentatonic melodic gestures, a resolute refrain and a contrasting flute-led bridge. Heroic momentum with bittersweet melodic turns and dramatic acoustic depth.", "童谣轻快": "Playful nursery-rhyme-style song, bright gentle lead vocals with clear simple articulation. Ukulele, toy piano, glockenspiel and soft hand percussion. A bouncing even pulse, short stepwise melodies and repeated easy phrases. Use small call-and-response moments and playful instrument answers. Friendly light production without dense bass or dramatic effects.", "音乐剧叙事": "Narrative musical-theater song, articulate expressive lead vocals with speech-like verses and sustained emotional peaks. Piano-led accompaniment expands into strings and woodwinds. Follow the vocal phrasing with dynamic accents, pauses and changing accompaniment density. Build toward a dramatic melodic reprise rather than a looped pop hook. Clear theatrical vocal presence.", "复古迪斯科": "Late-1970s disco arrangement, buoyant soulful lead vocals, octave-driven electric bass, clipped funk guitar, strings, brass and four-on-the-floor drums. Open hi-hats and a continuous dance groove. Repeated ensemble refrains lead to an extended strings-and-bass break. Warm live-orchestra texture and vintage studio ambience.", "暗黑电子": "Minimal dark electronic pop, low intimate lead vocals, ominous synth bass, sparse broken beats and metallic percussion. Tense repeating motifs, whispered-to-firm vocal dynamics and deliberate empty space. Gradually introduce distorted textures, then strip the final section back. Cold close detail, deep low end and narrow unsettling ambience.", "未来感Synthwave": "Retro-futuristic synthwave song, smooth sustained lead vocals, analog polysynth chords, pulsing octave bass, sequenced arpeggios and gated snare. Steady driving electronic pulse and broad melodic lines. A restrained verse opens into a wide synth-led chorus, with an instrumental arpeggio break. Saturated analog color and cinematic night-drive depth.", "摇篮曲温柔": "Tender lullaby song, very soft close lead vocals, gentle piano, celesta and occasional sustained strings. Slow rocking pulse, narrow-range stepwise melody and repeated soothing phrases. Maintain quiet dynamics with no percussion climax; finish with a short fading piano figure. Warm intimate sound and soft attacks throughout.", "史诗凯旋": "Triumphant orchestral anthem, resolute projected lead vocals, heroic brass, rising strings, timpani and large ensemble drums. Firm marching pulse and ascending melodic gestures. Build from a focused statement to a broad choral refrain, with a brass-led return of the theme. Bright grand orchestration and decisive cadences.", "自定义": ""};
const w=(node,name)=>node.widgets?.find(x=>x.name===name);
const misplacedLyrics=value=>/^\s*\[(verse|chorus|bridge|主歌|副歌)/im.test(String(value||''))&&! /^(X:|K:|V:)/m.test(String(value||''));
let requiredNames=[];
const css=document.createElement('style');
css.textContent=`.lyue{box-sizing:border-box;padding:14px;background:linear-gradient(145deg,#101c2c,#20203b);border:1px solid #477b9e;border-radius:12px;color:#e4eff7;font:13px/1.5 sans-serif;width:100%}.lyue *{box-sizing:border-box}.lyue h3{margin:0 0 10px;color:#a9dbef}.lyue .grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px}.lyue label{display:block;margin:7px 0;color:#b4cfdf}.lyue input,.lyue select,.lyue textarea{display:block;width:100%;padding:8px;background:#0b1423;color:#edf6fa;border:1px solid #3c566c;border-radius:6px;font:13px/1.5 sans-serif}.lyue textarea{resize:vertical;min-height:80px}.lyue input[type=checkbox]{width:auto;display:inline;margin:8px}.lyue button{padding:9px 12px;margin:6px 5px 0 0;border:1px solid #538eac;border-radius:7px;background:#244b67;color:white;cursor:pointer}.lyue button.primary{background:#256c84}.lyue details{margin-top:10px;padding:8px;background:#101929;border-radius:8px}.lyue summary{cursor:pointer;color:#a4d4e7}.lyue .note{font-size:12px;color:#8facbf;margin:8px 0}.lyue .report{max-height:180px;overflow:auto;white-space:pre-wrap;color:#b2d5bd;font-size:12px}.lyue [hidden]{display:none!important}`;
document.head.append(css);
app.registerExtension({name:'liao2049.YuE2Studio',
 beforeConfigureGraph(graph){
  const migrated=new Set();
  for(const node of graph.nodes||[]){
   if(node.type!=='LiaoYuE2Studio')continue;
   const removedInputs=new Set((node.inputs||[]).filter(x=>['原曲','原生管线'].includes(x.name)).map(x=>x.link).filter(x=>x!=null));
   node.inputs=(node.inputs||[]).filter(x=>!['原曲','原生管线'].includes(x.name));
   graph.links=(graph.links||[]).filter(link=>!removedInputs.has(link[0]));
   for(const other of graph.nodes||[])for(const output of other.outputs||[])if(output.links)output.links=output.links.filter(id=>!removedInputs.has(id));
   const named=node.widgets_values_named;
   if(named&&typeof named==='object'){
    if(named['风格模板']==='夏日独立流行')named['风格模板']='官方默认·夏日独立流行';
    named['目标BPM']??=0;named['拍号']??='自动/沿用';named['调性根音']??='自动/沿用';
    const abc=String(named['ABC乐谱']||'');
    if(misplacedLyrics(abc)){if(!String(named['歌词']||'').trim())named['歌词']=abc;named['ABC乐谱']='';}
    if(['本地Whisper','跳过使用已填歌词'].includes(named['Whisper模型路径']))named['Whisper模型路径']='';
    if(named['额外乐器与要求'])named['风格提示词']=String(named['风格提示词']||'')+'\n('+named['额外乐器与要求']+')';
    delete named['额外乐器与要求'];
    if(requiredNames.length&&requiredNames.every(name=>Object.hasOwn(named,name))){
     node.widgets_values=requiredNames.flatMap(name=>name==='种子'?[named[name],named.control_after_generate||'fixed']:[named[name]]);
     node.widgets_values.push('');
    }
   }
   if(node.outputs?.length!==5)continue;
   const vals=node.widgets_values;
   if(!named&&Array.isArray(vals)&&vals.length>=28){
    const extra=String(vals[6]||'').trim();
    if(extra)vals[2]=String(vals[2]||'')+'\n('+extra+')';
    vals.splice(6,1);
   }
   node.outputs=node.outputs.slice(0,1);migrated.add(node.id);
  }
  const removed=new Set();
  graph.links=(graph.links||[]).filter(link=>{if(migrated.has(link[1])&&link[2]>0){removed.add(link[0]);return false;}return true;});
  for(const node of graph.nodes||[])for(const input of node.inputs||[])if(removed.has(input.link))input.link=null;
 },
 async beforeRegisterNodeDef(type,data){
 if(data.name!=='LiaoYuE2Studio')return;
 requiredNames=Object.keys(data.input.required);
 const created=type.prototype.onNodeCreated;
 type.prototype.onNodeCreated=function(){
  created?.apply(this,arguments);const node=this,fields=new Map(),modeButtons=new Map();let trimmer,insertLyrics;
  node.properties||={};if(typeof node.properties.liao_yue2_random_seed!=='boolean')node.properties.liao_yue2_random_seed=true;
  for(const x of node.widgets||[]){x.computeSize=()=>[0,-4];x.hidden=true;if(x.element)x.element.style.display='none';}
  const root=document.createElement('div');root.className='lyue';const head=document.createElement('h3');head.textContent='♫ Yue2 写歌 · 改歌工作室';root.append(head);
  const scoreSlot=()=>w(node,'操作')?.value==='按编辑后的ABC改歌'?'cover':'write';let activeScoreSlot=null;
  const saveScore=()=>{node.properties.liao_yue2_scores||={};node.properties.liao_yue2_scores[activeScoreSlot||scoreSlot()]=String(w(node,'ABC乐谱')?.value||'');};
  const set=(name,value)=>{const x=w(node,name);if(x)x.value=value;const f=fields.get(name);if(f){if(f.type==='checkbox')f.checked=Boolean(value);else f.value=String(value);}if(name==='ABC乐谱')saveScore();if(name==='原曲文件')syncPreview();node.graph?.setDirtyCanvas(true,true);};
  const applyOperationMode=()=>{
   const slot=scoreSlot();node.properties.liao_yue2_scores||={};
   if(activeScoreSlot!==slot){
    if(activeScoreSlot)saveScore();
    const existing=String(w(node,'ABC乐谱')?.value||'');
    const value=node.properties.liao_yue2_scores[slot]??(activeScoreSlot?'':existing);
    activeScoreSlot=slot;set('ABC乐谱',value);
   }
   const action=w(node,'操作')?.value;
   if(action==='写歌并生成'){set('乐谱模式','full');set('自由重编伴奏',false);}
   else if(action==='按编辑后的ABC改歌'){set('乐谱模式','melody');set('自由重编伴奏',true);}
  };
  const clearMisplacedScore=()=>{
   const score=String(w(node,'ABC乐谱')?.value||'');
   if(misplacedLyrics(score)){if(!String(w(node,'歌词')?.value||'').trim())set('歌词',score);set('ABC乐谱','');}
   const cached=node.properties?.liao_yue2_result;if(cached&&misplacedLyrics(cached.abc))cached.abc='';
  };
  const field=(name,parent=root)=>{
   const x=w(node,name);if(!x)return;const label=document.createElement('label');label.textContent=name==='风格提示词'?'风格提示词（可输入其他任意要求）':name;const values=name==='操作'?['写歌并生成','按编辑后的ABC改歌','只规划ABC']:data.input.required[name]?.[0];let el;
   if(name==='歌词'){
    label.textContent='';const header=document.createElement('div');header.style.cssText='display:flex;justify-content:space-between;align-items:center;gap:8px';
    const title=document.createElement('span');title.textContent='歌词';
    const insert=document.createElement('button');insert.type='button';insert.textContent='插入歌词结构';
    insert.onclick=event=>{event.preventDefault();event.stopPropagation();if(w(node,'歌词').value.trim()&&!confirm('替换现有歌词为结构模板？'))return;set('歌词','[verse]\n在这里填写主歌歌词\n\n[chorus]\n在这里填写副歌歌词\n\n[verse]\n第二段主歌\n\n[bridge]\n桥段\n\n[chorus]\n最终副歌');};
    insertLyrics=insert;header.append(title);label.append(header);
   }
   if(Array.isArray(values)){el=document.createElement('select');for(const value of values){if(name==='操作'&&['原曲提取ABC与歌词','直接生成不规划'].includes(value))continue;const o=document.createElement('option');o.value=value;o.textContent=name==='操作'&&value==='按编辑后的ABC改歌'?'改歌翻唱':value;el.append(o);}}
   else if(['风格提示词','歌词','ABC乐谱'].includes(name)){el=document.createElement('textarea');el.rows=name==='歌词'?8:name==='ABC乐谱'?7:4;}
   else{el=document.createElement('input');el.type=typeof x.value==='boolean'?'checkbox':typeof x.value==='number'?'number':'text';if(el.type==='number'){el.min=x.options?.min??0;el.max=x.options?.max??1e18;el.step='any';}}
   if(name==='操作'){
    label.textContent='';label.style.gridColumn='1 / -1';el.hidden=true;
    const choices=document.createElement('div');choices.style.cssText='display:flex;gap:10px';
    for(const [value,title] of [['写歌并生成','AI写歌'],['按编辑后的ABC改歌','改歌翻唱']]){
     const b=document.createElement('button');b.type='button';b.textContent=title;b.style.cssText='flex:1;margin:0;padding:12px;font-size:16px;font-weight:bold';
     b.onclick=()=>{set('操作',value);applyOperationMode();refresh();};modeButtons.set(value,b);choices.append(b);
    }
    label.append(choices);
   }
   el.setAttribute('aria-label',name);fields.set(name,el);if(el.type==='checkbox')el.checked=Boolean(x.value);else el.value=String(x.value??'');
   el.addEventListener('input',()=>{x.value=el.type==='checkbox'?el.checked:el.type==='number'?Number(el.value):el.value;if(name==='ABC乐谱')saveScore();});
   el.addEventListener('change',()=>{if(name==='风格模板'&&el.value!=='自定义'&&presets[el.value])set('风格提示词',presets[el.value]);if(name==='操作'){applyOperationMode();refresh();}if(name==='原曲文件')syncPreview();if(['参考起点秒','参考长度秒'].includes(name))trimmer?.draw();});
   if(name==='种子'){
    const row=document.createElement('div');row.style.cssText='display:flex;align-items:center;gap:6px';el.style.minWidth='0';el.style.flex='1';
    const randomLabel=document.createElement('label');randomLabel.style.cssText='display:flex;align-items:center;margin:0;white-space:nowrap;flex-shrink:0';
    const random=document.createElement('input');random.type='checkbox';random.setAttribute('aria-label','每次随机种子');random.checked=Boolean(node.properties?.liao_yue2_random_seed);
    const text=document.createElement('span');text.textContent='随机种子';randomLabel.append(random,text);
    const fixedControl=()=>{const control=w(node,'control_after_generate');if(control)control.value='fixed';};fixedControl();
    random.addEventListener('change',()=>{node.properties||={};node.properties.liao_yue2_random_seed=random.checked;fixedControl();node.graph?.setDirtyCanvas(true,true);});
    node.__yueSeedSwitch=random;
    const beforeQueued=x.beforeQueued;x.beforeQueued=function(...args){fixedControl();if(node.properties?.liao_yue2_random_seed){const values=new Uint32Array(2);globalThis.crypto.getRandomValues(values);set('种子',(values[0]&0x1fffff)*4294967296+values[1]);}return beforeQueued?.apply(this,args);};
    row.append(el,randomLabel);label.append(row);
   }else label.append(el);
   if(name==='Whisper模型路径'){
    const guide=document.createElement('details');guide.style.cssText='padding:6px;margin-top:6px';
    const title=document.createElement('summary');title.textContent='Whisper 缺失？安装指引';guide.append(title);
    const content=document.createElement('div');content.className='note';content.style.whiteSpace='pre-wrap';
    content.textContent='歌词识别是可选功能，缺少Whisper仍可手动填歌词并写歌、翻唱。\n\n1. 使用运行ComfyUI的Python安装依赖（不是系统中其他Python）：\npython -m pip install "faster-whisper>=1.2.1"\n便携版示例：\n.\\python_embeded\\python.exe -m pip install "faster-whisper>=1.2.1"\n你的整合包可能叫python或python_embeded，请以实际目录为准。\n\n2. 准备faster-whisper格式的本地模型目录，需包含：model.bin、config.json、tokenizer.json、vocabulary.txt。普通Whisper的.pt文件不能直接使用。\n\n3. 默认目录：ComfyUI/models/whisper/faster-whisper-base/\n也可将完整模型目录填到上面的Whisper模型路径。\n\n4. 重启ComfyUI后点击“识别歌词”。\n本节点不会自动下载或安装。';guide.append(content);label.append(guide);
   }
   parent.append(label);return el;
  };
  const grid=document.createElement('div');grid.className='grid';grid.style.gridTemplateColumns='repeat(4,minmax(0,1fr))';root.append(grid);for(const name of ['操作','风格模板','人声','语言'])field(name,grid);
  const instrumentLabel=document.createElement('label');instrumentLabel.textContent='乐器';
  const instruments=document.createElement('select');instruments.setAttribute('aria-label','添加乐器');
  const instrumentNames=[['选择添加乐器',''],['钢琴','piano'],['电钢琴','electric piano'],['木吉他','acoustic guitar'],['电吉他','electric guitar'],['尼龙吉他','nylon-string guitar'],['贝斯','bass guitar'],['低音提琴','double bass'],['架子鼓','drums'],['手鼓','hand percussion'],['小提琴','violin'],['大提琴','cello'],['弦乐组','orchestral strings'],['长笛','flute'],['萨克斯','saxophone'],['小号','trumpet'],['圆号','French horn'],['竖琴','harp'],['合成器','synthesizer'],['古筝','guzheng'],['古琴','guqin'],['二胡','erhu'],['琵琶','pipa'],['竹笛','bamboo flute'],['扬琴','yangqin'],['唢呐','suona'],['口琴','harmonica'],['手风琴','accordion'],['尤克里里','ukulele'],['钟琴','glockenspiel']];
  for(const[cn,en]of instrumentNames){const o=document.createElement('option');o.value=en;o.textContent=cn;instruments.append(o);}
  instruments.addEventListener('change',()=>{
   const value=instruments.value;if(!value)return;
   const current=String(w(node,'风格提示词')?.value||'').trim();
   const added=[];
   const base=current.replace(/(?:^|\n)\s*(?:Additional instrument(?:s)?|Additional instrumentation):\s*([^\n]+)/gi,(_,list)=>{
    added.push(...list.replace(/\.$/,'').split(/,\s*|\s+and\s+/i).map(x=>x.trim()).filter(Boolean));return '';
   }).trim();
   added.push(value);
   const unique=[...new Set(added.map(x=>x.toLowerCase()))].filter(name=>!base.toLowerCase().includes(name));
   const list=unique.length<2?unique[0]:unique.slice(0,-1).join(', ')+' and '+unique.at(-1);
   set('风格提示词',base+(list?(base?'\n':'')+'Additional instrumentation: '+list+'.':''));
   instruments.value='';
  });instrumentLabel.append(instruments);grid.append(instrumentLabel);field('风格提示词');const editors=document.createElement('div');editors.className='grid';root.append(editors);field('歌词',editors);
  const button=(title,fn,parent=root)=>{const b=document.createElement('button');b.type='button';b.textContent=title;b.onclick=fn;parent.append(b);return b;};
  const ref=document.createElement('div');root.append(ref);const upload=document.createElement('input');upload.type='file';upload.accept='audio/*,.mp3,.wav,.flac,.m4a,.ogg';upload.hidden=true;ref.append(upload);
  const uploadRow=document.createElement('div');ref.append(uploadRow);
  const uploadButton=button('上传原歌曲',()=>upload.click(),uploadRow);
  const recognizeButton=button('识别歌词',async()=>{
   const filename=String(w(node,'原曲文件')?.value||'').trim();
   if(!filename){status.textContent='请先上传原歌曲。';return;}
   const previousLyrics=w(node,'歌词').value;
   recognizeButton.disabled=uploadButton.disabled=true;recognizeButton.textContent='识别中…';
   status.textContent=`正在识别所选片段：起点${w(node,'参考起点秒').value}秒，长度${w(node,'参考长度秒').value}秒…`;
   try{
    const payload={filename,start_seconds:w(node,'参考起点秒').value,length_seconds:w(node,'参考长度秒').value,language:w(node,'语言').value,model_path:w(node,'Whisper模型路径').value};
    const request=()=>api.fetchApi('/liao_yue2/recognize_lyrics',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    let response=await request();
    if(response.status===404||response.status===405)throw Error('请重启ComfyUI，加载新版歌词识别接口。');
    let result=await response.json();
    if(!response.ok&&payload.model_path&&/没有本地faster-whisper模型|未找到歌词识别模型/.test(result.error||'')){
     payload.model_path='';set('Whisper模型路径','');status.textContent='指定模型路径无效，正在使用已安装的默认模型重试…';response=await request();result=await response.json();
    }
    if(!response.ok)throw Error(result.error||`HTTP ${response.status}`);
    if(String(w(node,'原曲文件').value).trim()!==filename){status.textContent='原曲已更换，请重新识别。';return;}
    if(w(node,'歌词').value!==previousLyrics){node.properties||={};node.properties.liao_yue2_pending_lyrics=result.lyrics;status.textContent='识别完成；识别期间歌词有修改，结果已保留，请点击“填入识别结果”。';adoptLyrics.hidden=false;return;}
    set('歌词',result.lyrics);clearMisplacedScore();status.textContent=`已识别${result.seconds}秒音频并填入歌词框。${result.note||''} 仅识别歌词；旋律ABC需运行“原曲提取ABC与歌词”。`;
   }catch(error){status.textContent=error.message||String(error);}
   finally{recognizeButton.disabled=uploadButton.disabled=false;recognizeButton.textContent='识别歌词';resize();}
  },uploadRow);recognizeButton.className='primary';
  const adoptLyrics=button('填入识别结果',()=>{const value=node.properties?.liao_yue2_pending_lyrics;if(value){set('歌词',value);delete node.properties.liao_yue2_pending_lyrics;adoptLyrics.hidden=true;status.textContent='已填入识别歌词，请校对。';}},uploadRow);adoptLyrics.hidden=true;
  field('原曲文件',ref);
  const preview=document.createElement('audio');preview.controls=false;preview.preload='metadata';preview.style.display='none';preview.setAttribute('aria-label','原歌曲试听');preview.hidden=true;ref.append(preview);
  let previewFilename='';
  const syncPreview=()=>{
   const filename=String(w(node,'原曲文件')?.value||'').trim().replaceAll('\\','/');
   if(filename===previewFilename)return;
   previewFilename=filename;preview.pause?.();preview.hidden=true;
   if(!filename){preview.removeAttribute('src');preview.load?.();trimmer?.load('');return;}
   const split=filename.lastIndexOf('/');const params=new URLSearchParams({filename:filename.slice(split+1),subfolder:split>=0?filename.slice(0,split):'',type:'input'});
   preview.src=api.apiURL('/view?'+params.toString());preview.load?.();trimmer?.load(preview.src);
  };
  preview.addEventListener('error',()=>{status.textContent='原曲无法播放，请检查文件是否存在或重新上传。';});
  const status=document.createElement('div');status.className='note';ref.append(status);
  trimmer=attachAudioTrimmer({parent:ref,player:preview,get:name=>w(node,name)?.value,set,status});
  upload.onchange=async()=>{const file=upload.files?.[0];if(!file)return;uploadButton.disabled=recognizeButton.disabled=true;status.textContent='正在上传…';try{const body=new FormData();body.append('image',file);body.append('type','input');const response=await api.fetchApi('/upload/image',{method:'POST',body});if(!response.ok)throw Error('HTTP '+response.status);const result=await response.json();set('原曲文件',result.subfolder?result.subfolder+'/'+result.name:result.name);status.textContent='上传完成，点击旁边“识别歌词”即可填入歌词框。';}catch(e){status.textContent='上传失败：'+e.message;}finally{uploadButton.disabled=recognizeButton.disabled=false;upload.value='';}};
  const score=document.createElement('div');editors.append(score);field('ABC乐谱',score);
  for(const name of ['歌词','ABC乐谱'])fields.get(name).style.cssText='height:260px;min-height:160px;resize:vertical;overflow-y:auto';
  const lyricEditor=fields.get('歌词'),abcEditor=fields.get('ABC乐谱');
  let editorHeight=260;
  const editorObserver=new ResizeObserver(entries=>{
   for(const entry of entries){
    const box=entry.borderBoxSize;const next=Math.round((Array.isArray(box)?box[0]?.blockSize:box?.blockSize)||entry.target.offsetHeight);
    if(!next||Math.abs(next-editorHeight)<2)continue;
    editorHeight=Math.max(160,next);
    lyricEditor.style.height=abcEditor.style.height=editorHeight+'px';
    requestAnimationFrame(resize);break;
   }
  });
  editorObserver.observe(lyricEditor);editorObserver.observe(abcEditor);

  const recognizeABC=button('识别旋律',async()=>{
   const filename=String(w(node,'原曲文件')?.value||'').trim();if(!filename){status.textContent='请先上传原歌曲。';return;}
   const previous=w(node,'ABC乐谱').value;recognizeABC.disabled=true;recognizeABC.textContent='旋律识别中…';status.textContent='正在提取波形选中片段的ABC旋律…';
   try{
    const response=await api.fetchApi('/liao_yue2/recognize_abc',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename,start_seconds:w(node,'参考起点秒').value,length_seconds:w(node,'参考长度秒').value,encoder:w(node,'旋律识别模型').value,melody_only:fields.get('自由重编伴奏').checked})});
    if(response.status===404||response.status===405)throw Error('请重启ComfyUI，加载新版ABC识别接口。');
    const result=await response.json();if(!response.ok)throw Error(result.error||`HTTP ${response.status}`);
    if(String(w(node,'原曲文件').value).trim()!==filename){status.textContent='原曲已更换，请重新识别ABC。';return;}
    if(w(node,'ABC乐谱').value!==previous){node.properties||={};node.properties.liao_yue2_pending_abc=result.abc;adoptABC.hidden=false;status.textContent='ABC识别完成；编辑内容已保留，可点击“填入ABC识别结果”。';return;}
    set('ABC乐谱',result.abc);status.textContent='已将所选片段的旋律填入ABC框，歌词保留。编辑后选择“改歌翻唱”即可生成。';
   }catch(error){status.textContent=error.message||String(error);}
   finally{recognizeABC.disabled=false;recognizeABC.textContent='识别旋律';resize();}
  },uploadRow);recognizeABC.className='primary';
  uploadRow.insertBefore(recognizeABC,adoptLyrics);
  const recognitionNote=document.createElement('span');recognitionNote.textContent='识别歌词仅作参考';recognitionNote.style.cssText='display:inline-block;margin-left:6px;font-size:12px;color:#8facbf';uploadRow.insertBefore(recognitionNote,adoptLyrics);

  const adoptABC=button('填入ABC识别结果',()=>{const value=node.properties?.liao_yue2_pending_abc;if(value){set('ABC乐谱',value);delete node.properties.liao_yue2_pending_abc;adoptABC.hidden=true;status.textContent='已填入ABC识别结果。';}},score);adoptABC.hidden=true;
  const basic=document.createElement('div');basic.className='grid';root.append(basic);field('最长秒数',basic);field('种子',basic);basic.style.gridTemplateColumns='minmax(0,1fr) minmax(0,1.3fr) auto';basic.style.alignItems='end';insertLyrics.style.cssText='margin:0 0 7px;white-space:nowrap';const basicActions=document.createElement('div');basicActions.style.cssText='display:flex;align-items:center;gap:4px';basic.append(basicActions);basicActions.append(insertLyrics);
  const advanced=document.createElement('details'),title=document.createElement('summary');title.textContent='♫ Yue2 写歌 · 改歌工作室';title.style.cssText='font-size:16px;font-weight:bold;color:#a9dbef';title.title='点击展开或收起专业设置';advanced.append(title);advanced.style.cssText='margin:0 0 10px;padding:0;background:transparent';head.hidden=true;root.insertBefore(advanced,grid);const ag=document.createElement('div');ag.className='grid';advanced.append(ag);
  for(const name of ['模型','旋律识别模型','乐谱模式','目标BPM','拍号','调性根音','采样步数','速度倍率','升降调半音','自由重编伴奏','自动时长预算','歌词识别','Whisper模型路径','规划温度','音乐温度'])field(name,ag);
  const download=(value,filename)=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([value],{type:'text/plain;charset=utf-8'}));a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
  const downloadABC=button('下载ABC',()=>download(w(node,'ABC乐谱').value,'yue2_score.abc'),basicActions);
  const downloadLyrics=button('下载歌词',()=>download(w(node,'歌词').value,'yue2_lyrics.txt'),basicActions);
  for(const b of [downloadABC,downloadLyrics,insertLyrics])b.style.cssText='margin:0 0 7px;padding:8px;white-space:nowrap';basicActions.insertBefore(downloadABC,insertLyrics);basicActions.insertBefore(downloadLyrics,insertLyrics);
  const note=document.createElement('div');note.className='note';root.append(note);const reportDetails=document.createElement('details');const reportTitle=document.createElement('summary');reportTitle.textContent='运行详情';reportDetails.append(reportTitle);const report=document.createElement('div');report.className='report';reportDetails.append(report);root.append(reportDetails);
  const shell=document.createElement('div');shell.style.cssText='width:100%;min-width:0;height:auto;overflow:visible';shell.append(root);
  let height=1000;const dom=node.addDOMWidget('liao_yue2_ui','div',shell,{serialize:false,hideOnZoom:false,getHeight:()=>height});dom.computeSize=()=>[480,height];
  let sizing=false;
  const resize=()=>{if(sizing)return;sizing=true;try{shell.style.width=Math.max(460,node.size[0]-20)+'px';height=Math.ceil(root.scrollHeight)+8;const target=[Math.max(480,node.size[0]),height+40];if(Math.abs(node.size[0]-target[0])>1||Math.abs(node.size[1]-target[1])>1)node.setSize(target);node.graph?.setDirtyCanvas(true,true);}finally{sizing=false;}};
  const previousResize=node.onResize;
  node.onResize=function(size){previousResize?.apply(this,arguments);if(sizing)return;shell.style.width=Math.max(460,size[0]-20)+'px';requestAnimationFrame(resize);};
  const refresh=()=>{if(w(node,'风格模板')?.value==='夏日独立流行')set('风格模板','官方默认·夏日独立流行');if(node.__yueSeedSwitch)node.__yueSeedSwitch.checked=Boolean(node.properties?.liao_yue2_random_seed);const action=w(node,'操作')?.value;if(action==='原曲提取ABC与歌词')set('操作','按编辑后的ABC改歌');else if(['直接生成不规划','只规划ABC'].includes(action))set('操作','写歌并生成');const scoreField=fields.get('ABC乐谱');if(scoreField){scoreField.readOnly=w(node,'操作').value==='写歌并生成';scoreField.setAttribute('aria-readonly',String(scoreField.readOnly));scoreField.title=scoreField.readOnly?'AI写歌生成的乐谱仅供查看；切换改歌翻唱后可编辑。':'可编辑旋律乐谱';}for(const[value,b]of modeButtons){const selected=w(node,'操作').value===value;b.className=selected?'primary':'';b.style.opacity=selected?'1':'0.65';b.setAttribute('aria-pressed',String(selected));}clearMisplacedScore();for(const[name,el]of fields){const value=w(node,name)?.value;if(el.type==='checkbox')el.checked=Boolean(value);else el.value=String(value??'');}note.textContent=w(node,'操作').value==='写歌并生成'?'AI写歌：填写风格提示词和歌词，运行即可生成；ABC乐谱自动生成，仅供查看。':'改歌翻唱：上传原曲并选择片段，识别歌词与旋律，校对或修改后运行。也可以直接填写歌词、粘贴ABC乐谱。';reportDetails.hidden=!node.properties?.liao_yue2_result?.report;ref.hidden=!['原曲提取ABC与歌词','按编辑后的ABC改歌'].includes(w(node,'操作').value);if(ref.hidden)preview.pause?.();syncPreview();report.textContent=node.properties?.liao_yue2_result?.report||'';resize();};
  node.__yueResetScore=()=>{activeScoreSlot=null;};node.__yueRefresh=refresh;node.__yueSet=set;node.__yueApplyMode=applyOperationMode;const observer=new ResizeObserver(resize);observer.observe(root);node.__yueDispose=()=>{observer.disconnect();editorObserver.disconnect();trimmer?.destroy();preview.pause?.();preview.removeAttribute('src');preview.load?.();};advanced.addEventListener('toggle',resize);reportDetails.addEventListener('toggle',resize);requestAnimationFrame(()=>{applyOperationMode();refresh();});
 };
 const configured=type.prototype.onConfigure;type.prototype.onConfigure=function(){configured?.apply(this,arguments);requestAnimationFrame(()=>{this.__yueResetScore?.();this.__yueApplyMode?.();this.__yueRefresh?.();});};
 const removed=type.prototype.onRemoved;type.prototype.onRemoved=function(){this.__yueDispose?.();removed?.apply(this,arguments);};
 const executed=type.prototype.onExecuted;type.prototype.onExecuted=function(message){executed?.apply(this,arguments);const result=message.liao_yue2?.[0];if(!result)return;this.properties||={};this.properties.liao_yue2_result=result;if(['原曲提取ABC与歌词','只规划ABC'].includes(result.action)){this.__yueSet?.('ABC乐谱',result.abc);this.__yueSet?.('歌词',result.lyrics);}else if(result.action==='写歌并生成'&&result.abc){this.properties.liao_yue2_scores||={};this.properties.liao_yue2_scores.write=result.abc;if(w(this,'操作')?.value!=='按编辑后的ABC改歌')this.__yueSet?.('ABC乐谱',result.abc);}this.__yueRefresh?.();};
}});
