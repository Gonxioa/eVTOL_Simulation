import {lowerIndex,clampTime,rotation,transform,tiltDegrees,phase,className,predictionAt} from './core.js';

const $=id=>document.getElementById(id);
const state={catalog:null,record:null,normal:null,time:0,playing:false,lastFrame:0,loading:0,abort:null,
  camera:{azimuth:-.95,elevation:.45,zoom:1},drag:null};
const colors={actual:'#57c7ff',reference:'#ffc66b',normal:'#aab9ca',fault:'#ff8286',grid:'#304155',text:'#afc0d2'};
const fmt=(v,n=2)=>Number.isFinite(v)?v.toFixed(n):'—';
const vec=(a,n=3)=>a.map(v=>fmt(v,n)).join(' / ');
function error(message){$('error').textContent=message;$('error').hidden=!message;}
function option(value,text){const o=document.createElement('option');o.value=value;o.textContent=text;return o;}
function pause(){state.playing=false;$('play').textContent='播放';}
function setTime(value){if(!state.record)return;state.time=clampTime(value,state.record.meta.actual_end_s);$('time').value=state.time;render();}
function index(){return Math.max(0,lowerIndex(state.record.series.time,state.time));}
async function getJson(url,signal){const r=await fetch(url,{signal});const data=await r.json();if(!r.ok)throw new Error(data.error||'无法加载记录');return data;}

async function loadRecord(){
  pause();error('');state.abort?.abort();state.abort=new AbortController();const token=++state.loading;
  const group=Number($('group').value),label=Number($('label').value);
  state.record=null;state.normal=null;state.time=0;$('record-name').textContent='正在加载记录…';
  for(const id of ['position','velocity','speed-norm','reference-position','attitude','tilt','body-rate','prediction','fault-score','alarm-summary','phase','fault-clock','termination'])$(id).textContent='—';
  $('motors').replaceChildren();$('injection-status').textContent='正在加载…';
  for(const id of ['play','reset','jump-fault','time'])$(id).disabled=true;
  for(const id of ['scene','height-chart','tilt-chart','score-chart']){const c=$(id);c.getContext('2d').clearRect(0,0,c.width,c.height);}
  try{
    const r=await getJson(`/api/record?group=${group}&label=${label}`,state.abort.signal);
    if(token!==state.loading)return;
    state.record=r;state.camera={azimuth:-.95,elevation:.45,zoom:1};
    $('record-name').textContent=`${r.meta.record_id} · ${className(label)}`;
    $('time-end').textContent=fmt(r.meta.actual_end_s);$('planned-end').textContent=`计划 ${fmt(r.meta.requested_end_s)} s`;
    $('time').max=r.meta.actual_end_s;$('time').value=0;
    for(const id of ['play','reset','time'])$(id).disabled=false;
    $('jump-fault').disabled=r.meta.first_fault_sample_s===null;
    $('fault-clock').textContent=r.meta.fault_time_s===null?'本条无故障注入':`注入 ${fmt(r.meta.fault_time_s,6)} s · 首次故障采样 ${fmt(r.meta.first_fault_sample_s)} s`;
    $('diagnostic-note').textContent=r.meta.diagnostics_error?`预测记录不可用：${r.meta.diagnostics_error}`:
      r.meta.diagnostics_available?'30轮开发模型 · 故障得分未经概率校准':'本组没有保存诊断预测；不以注入真值替代预测。';
    $('motors').replaceChildren();
    for(let m=0;m<6;m++){const tr=document.createElement('tr');tr.id='motor-'+m;for(let j=0;j<5;j++){const td=document.createElement('td');td.textContent=j===0?'M'+m:'—';tr.append(td);}$('motors').append(tr);}
    metadata(r.meta);render();
    if($('normal-overlay').checked)await loadNormal();
  }catch(e){if(e.name!=='AbortError')error(e.message);}
}
async function loadNormal(){
  const r=state.record;if(!r)return;const token=state.loading;
  try{const normal=r.meta.label===0?r:await getJson(`/api/record?group=${r.meta.group}&label=0`,state.abort.signal);
    if(token===state.loading){state.normal=normal;render();}
  }catch(e){if(e.name!=='AbortError')error('正常对照不可用：'+e.message);}
}
function metadata(meta){
  const s=meta.task;const description=s.family==='hover'?'位置固定，偏航平滑调整':s.family==='line'?`位移 ${fmt(s.displacement_m,6)} m，方位角 ${fmt(s.heading_rad*180/Math.PI,3)}°`:
    s.family==='vertical'?`竖直位移 ${fmt(s.displacement_m,6)} m`:`半径 ${fmt(s.radius_m,6)} m，${s.turn_rad>0?'逆时针':'顺时针'}一圈`;
  const entries=[['任务',description],['参考起点，m',vec(s.origin_m,6)],['运动 / 偏航时段',`${s.motion_start_s}–${s.motion_end_s} s`],
    ['偏航参考',`${fmt(s.yaw_start_rad*180/Math.PI,3)}° → ${fmt((s.yaw_start_rad+s.yaw_change_rad)*180/Math.PI,3)}°`],
    ['数据身份','ENU / FLU，轨迹和姿态为仿真真值，参数为研究设计'],['结束状态',meta.termination==='completed'?'达到计划时长':meta.termination_reasons.join('；')],
    ['源记录',meta.csv],['CSV SHA256',meta.csv_sha256]];
  $('metadata').replaceChildren();for(const [label,value] of entries){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;$('metadata').append(dt,dd);}
}

function context(id){const canvas=$(id),rect=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
  const w=rect.width,h=rect.height;if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}
  const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);return {canvas,ctx,w,h};}
function scene(){
  const r=state.record;if(!r)return;const {ctx,w,h}=context('scene'),s=r.series,i=index(),cam=state.camera;
  const all=[...s.position,...r.planned.position,[0,0,0]];
  const follow=$('follow').checked;
  const min=follow?s.position[i].map(v=>v-.48):[0,1,2].map(k=>Math.min(...all.map(p=>p[k])));
  const max=follow?s.position[i].map(v=>v+.48):[0,1,2].map(k=>Math.max(...all.map(p=>p[k])));
  const center=min.map((v,k)=>(v+max[k])/2),ca=Math.cos(cam.azimuth),sa=Math.sin(cam.azimuth),ce=Math.cos(cam.elevation),se=Math.sin(cam.elevation);
  const raw=p=>{const d=p.map((x,k)=>x-center[k]);return [-sa*d[0]+ca*d[1],-se*ca*d[0]-se*sa*d[1]+ce*d[2],ce*ca*d[0]+ce*sa*d[1]+se*d[2]];};
  const margin=follow?.12:.65,zmargin=follow?.12:.4;
  const corners=[];for(const x of [min[0]-margin,max[0]+margin])for(const y of [min[1]-margin,max[1]+margin])for(const z of [min[2]-zmargin,max[2]+zmargin])corners.push(raw([x,y,z]));
  const xmin=Math.min(...corners.map(p=>p[0])),xmax=Math.max(...corners.map(p=>p[0])),ymin=Math.min(...corners.map(p=>p[1])),ymax=Math.max(...corners.map(p=>p[1]));
  const scale=Math.min((w-60)/(xmax-xmin),(h-55)/(ymax-ymin))*cam.zoom;
  const project=p=>{const v=raw(p);return [w/2+(v[0]-(xmin+xmax)/2)*scale,h/2-(v[1]-(ymin+ymax)/2)*scale,v[2]];};
  function path(points,color,width=1,dash=[]){if(!points.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();points.forEach((p,j)=>{const q=project(p);j?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});ctx.stroke();ctx.setLineDash([]);}
  const gminx=Math.floor(min[0]-.5),gmaxx=Math.ceil(max[0]+.5),gminy=Math.floor(min[1]-.5),gmaxy=Math.ceil(max[1]+.5);
  for(let x=gminx;x<=gmaxx;x++)path([[x,gminy,0],[x,gmaxy,0]],colors.grid,.6);
  for(let y=gminy;y<=gmaxy;y++)path([[gminx,y,0],[gmaxx,y,0]],colors.grid,.6);
  ctx.font='11px Microsoft YaHei, sans-serif';
  for(const [v,name,col] of [[[.8,0,0],'x / 东',colors.actual],[[0,.8,0],'y / 北',colors.normal],[[0,0,.8],'z / 上',colors.reference]]){
    path([[0,0,0],v],col,1.5);const q=project(v);ctx.fillStyle=col;ctx.fillText(name,q[0]+5,q[1]-4);}
  path(r.planned.position,colors.reference,1.6,[5,4]);
  if($('normal-overlay').checked&&state.normal){const normal=state.normal.series;const ni=Math.max(0,lowerIndex(normal.time,state.time));path(normal.position.slice(0,ni+1),colors.normal,1.6,[3,3]);}
  path(s.position.slice(0,i+1),colors.actual,2.5);
  const ref=project(s.reference[i]);ctx.strokeStyle=colors.reference;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(ref[0],ref[1],5,0,Math.PI*2);ctx.stroke();
  const yaw=s.reference_yaw[i];path([s.reference[i],s.reference[i].map((v,k)=>v+(k===0?.18*Math.cos(yaw):k===1?.18*Math.sin(yaw):0))],colors.reference,1.5);
  const R=rotation(s.attitude[i]),p=s.position[i],geometry=r.meta.geometry,rotors=[];
  for(let m=0;m<6;m++){const angle=m*Math.PI/3;const local=[geometry.arm_radius_m*Math.cos(angle),geometry.arm_radius_m*Math.sin(angle),0];const wp=transform(R,local,p);rotors.push({m,local,wp,depth:project(wp)[2]});path([p,wp],colors.normal,3);}
  const body=[];for(let n=0;n<=20;n++)body.push(transform(R,[geometry.body_radius_m*Math.cos(n*Math.PI/10),geometry.body_radius_m*Math.sin(n*Math.PI/10),0],p));
  path(body,colors.actual,2);path([p,transform(R,[geometry.arm_radius_m*.8,0,.03],p)],'#edf3fa',3);
  for(const rotor of rotors.sort((a,b)=>a.depth-b.depth)){
    const ring=[];for(let n=0;n<=24;n++){const a=n*Math.PI/12;ring.push(transform(R,[rotor.local[0]+geometry.rotor_radius_m*Math.cos(a),rotor.local[1]+geometry.rotor_radius_m*Math.sin(a),0],p));}
    const failed=s.effectiveness[i][rotor.m]<1,col=failed?colors.fault:colors.actual;
    ctx.beginPath();ring.forEach((v,j)=>{const q=project(v);j?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});ctx.closePath();
    const relative=s.rotor_speed[i][rotor.m]/geometry.omega_max_rad_s;ctx.fillStyle=failed?`rgba(255,130,134,${.08+.3*relative})`:`rgba(87,199,255,${.08+.3*relative})`;ctx.fill();ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.stroke();
    const label=project(transform(R,[rotor.local[0]*1.48,rotor.local[1]*1.48,0],p));
    const center2d=project(p),dx=label[0]-center2d[0],dy=label[1]-center2d[1],length=Math.hypot(dx,dy);
    if(length>0&&length<37){label[0]=center2d[0]+dx*37/length;label[1]=center2d[1]+dy*37/length;}
    ctx.fillStyle=col;ctx.textAlign='center';ctx.fillText('M'+rotor.m,label[0],label[1]+4);
  }
  ctx.textAlign='left';ctx.fillStyle=colors.text;ctx.fillText('位置单位 m · 圆盘半径与机臂尺寸来自保存配置',12,h-12);
}

function chart(id,lines,options={}){
  const r=state.record;if(!r)return;const {ctx,w,h}=context(id);const pad={l:43,r:13,t:18,b:27};
  const xmin=0,xmax=r.meta.requested_end_s,all=lines.flatMap(line=>line.y).filter(Number.isFinite);
  let ymin=options.min??Math.min(...all,0),ymax=options.max??Math.max(...all,1);if(ymax-ymin<.2){ymax+=.2;ymin-=.2;}
  if(options.min===undefined){const margin=(ymax-ymin)*.12;ymin-=margin;ymax+=margin;}
  const x=t=>pad.l+(t-xmin)/(xmax-xmin)*(w-pad.l-pad.r),y=v=>h-pad.b-(v-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
  if(r.meta.actual_end_s<xmax){ctx.fillStyle='rgba(170,185,202,.07)';ctx.fillRect(x(r.meta.actual_end_s),pad.t,x(xmax)-x(r.meta.actual_end_s),h-pad.b-pad.t);ctx.fillStyle=colors.text;ctx.font='10px sans-serif';ctx.fillText('无记录',x(r.meta.actual_end_s)+7,pad.t+11);}
  ctx.font='10px Microsoft YaHei, sans-serif';ctx.lineWidth=.6;ctx.strokeStyle=colors.grid;ctx.fillStyle=colors.text;
  for(let k=0;k<=3;k++){const v=ymin+(ymax-ymin)*k/3;ctx.beginPath();ctx.moveTo(pad.l,y(v));ctx.lineTo(w-pad.r,y(v));ctx.stroke();ctx.textAlign='right';ctx.fillText(fmt(v,options.decimals??1),pad.l-5,y(v)+3);}
  for(let t=0;t<=xmax;t+=2){ctx.textAlign='center';ctx.fillText(String(t),x(t),h-9);}ctx.textAlign='right';ctx.fillText('s',w-2,h-9);ctx.textAlign='left';
  ctx.save();ctx.beginPath();ctx.rect(pad.l,pad.t,w-pad.l-pad.r,h-pad.t-pad.b);ctx.clip();
  for(const line of lines){ctx.strokeStyle=line.color;ctx.lineWidth=line.width??1.6;ctx.setLineDash(line.dash??[]);ctx.beginPath();let started=false;
    line.t.forEach((t,i)=>{if(!Number.isFinite(line.y[i])){started=false;return;}if(started)ctx.lineTo(x(t),y(line.y[i]));else{ctx.moveTo(x(t),y(line.y[i]));started=true;}});ctx.stroke();}
  ctx.setLineDash([]);
  if(r.meta.fault_time_s!==null){ctx.strokeStyle=colors.fault;ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(x(r.meta.fault_time_s),pad.t);ctx.lineTo(x(r.meta.fault_time_s),h-pad.b);ctx.stroke();ctx.setLineDash([]);}
  if(options.threshold!==undefined){ctx.strokeStyle=colors.reference;ctx.setLineDash([2,4]);ctx.beginPath();ctx.moveTo(pad.l,y(options.threshold));ctx.lineTo(w-pad.r,y(options.threshold));ctx.stroke();ctx.setLineDash([]);}
  ctx.strokeStyle=colors.text;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(state.time),pad.t);ctx.lineTo(x(state.time),h-pad.b);ctx.stroke();
  if(options.alarms)for(const a of options.alarms){ctx.fillStyle=colors.fault;ctx.beginPath();ctx.arc(x(a.time),y(a.fault_probability),3,0,Math.PI*2);ctx.fill();}
  ctx.restore();
  $(id).onclick=e=>{pause();const rect=e.currentTarget.getBoundingClientRect();setTime((e.clientX-rect.left-pad.l)/(w-pad.l-pad.r)*xmax);};
}
function render(){
  const r=state.record;if(!r)return;const s=r.series,i=index(),t=s.time[i],atEnd=state.time>=r.meta.actual_end_s-1e-9;
  $('time-now').textContent=fmt(t);$('time').value=state.time;$('phase').textContent=(atEnd?'回放结束 · ':'')+phase(r.meta,t);
  const early=r.meta.termination!=='completed';$('termination').textContent=early?`提前结束 ${fmt(r.meta.actual_end_s)} s · ${r.meta.termination_reasons.join('；')}`:`完整记录至${fmt(r.meta.actual_end_s)} s`;$('termination').classList.toggle('ended',early&&atEnd);
  const fault=s.effectiveness[i].some(v=>v<1);$('injection-status').textContent=fault?'注入真值：'+className(r.meta.label):'当前采样：未注入故障';$('injection-status').classList.toggle('fault',fault);
  $('position').textContent=vec(s.position[i]);$('reference-position').textContent=vec(s.reference[i]);$('attitude').textContent=vec(s.attitude[i].map(v=>v*180/Math.PI),2);$('body-rate').textContent=vec(s.body_rate[i]);$('tilt').textContent=fmt(tiltDegrees(s.attitude[i]),2);
  $('velocity').textContent=vec(s.velocity[i]);$('speed-norm').textContent=fmt(Math.hypot(...s.velocity[i]),3);
  for(let m=0;m<6;m++){const tr=$('motor-'+m);tr.classList.toggle('fault',s.effectiveness[i][m]<1);const values=[fmt(s.command[i][m],1),fmt(s.rotor_speed[i][m],1),fmt(s.thrust[i][m],3),fmt(s.effectiveness[i][m],2)];values.forEach((v,j)=>tr.cells[j+1].textContent=v);}
  const prediction=predictionAt(r.diagnostics,t);
  $('prediction').textContent=!r.meta.diagnostics_available?'没有保存预测':!prediction?'尚未到首个诊断窗口':!prediction.valid?'当前诊断观测无效':'预测：'+className(prediction.predicted_label);
  $('fault-score').textContent=prediction?.valid?'故障得分 '+fmt(prediction.fault_probability,3):'—';
  const alarmHistory=(r.diagnostics??[]).filter(p=>p.alarm&&p.time<=t+1e-10);
  $('alarm-summary').textContent=r.meta.diagnostics_available?`截至当前采样报警 ${alarmHistory.length} 次`+(alarmHistory.length?` · 最近 ${fmt(alarmHistory.at(-1).time)} s，${className(alarmHistory.at(-1).alarm_motor_severity_label)}`:''):'诊断预测仅保存于 g002 / g004 / g006 / g008；不在回放时重新训练或推理。';
  scene();chart('height-chart',[{t:r.planned.time,y:r.planned.position.map(p=>p[2]),color:colors.reference,dash:[4,3]},{t:s.time,y:s.position.map(p=>p[2]),color:colors.actual}]);
  chart('tilt-chart',[{t:s.time,y:s.attitude.map(tiltDegrees),color:colors.actual}],{min:0,max:Math.max(35,...s.attitude.map(tiltDegrees))*1.05,decimals:0});
  const predictions=(r.diagnostics??[]).filter(p=>p.valid);
  chart('score-chart',predictions.length?[{t:predictions.map(p=>p.time),y:predictions.map(p=>p.fault_probability),color:colors.reference}]:[],{min:0,max:1,threshold:predictions.length ? .8 : undefined,alarms:predictions.filter(p=>p.alarm)});
}

$('play').onclick=()=>{if(state.playing){pause();return;}if(state.time>=state.record.meta.actual_end_s)setTime(0);state.playing=true;state.lastFrame=performance.now();$('play').textContent='暂停';};
$('reset').onclick=()=>{pause();setTime(0);};$('jump-fault').onclick=()=>{pause();setTime(state.record.meta.first_fault_sample_s);};
$('time').oninput=e=>{pause();setTime(Number(e.target.value));};
$('label').onchange=loadRecord;
$('group').onchange=()=>{const group=state.catalog.groups.find(g=>g.id===Number($('group').value));for(const o of $('label').options)o.disabled=!group.labels.includes(Number(o.value));if(!group.labels.includes(Number($('label').value)))$('label').value=group.labels[0];loadRecord();};
$('normal-overlay').onchange=()=>{if($('normal-overlay').checked)loadNormal();else render();};
$('follow').onchange=render;
$('camera-reset').onclick=()=>{state.camera={azimuth:-.95,elevation:.45,zoom:1};render();};
$('scene').onpointerdown=e=>{state.drag={x:e.clientX,y:e.clientY};e.currentTarget.setPointerCapture(e.pointerId);};
$('scene').onpointermove=e=>{if(!state.drag)return;state.camera.azimuth+=(e.clientX-state.drag.x)*.008;state.camera.elevation=Math.max(-1.2,Math.min(1.3,state.camera.elevation+(e.clientY-state.drag.y)*.006));state.drag={x:e.clientX,y:e.clientY};render();};
$('scene').onpointerup=()=>state.drag=null;$('scene').onpointercancel=()=>state.drag=null;
$('scene').addEventListener('wheel',e=>{e.preventDefault();state.camera.zoom=Math.max(.45,Math.min(3.5,state.camera.zoom*Math.exp(-e.deltaY*.001)));render();},{passive:false});
$('scene').onkeydown=e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();state.camera.azimuth+=e.key==='ArrowLeft'?-.12:e.key==='ArrowRight'?.12:0;state.camera.elevation=Math.max(-1.2,Math.min(1.3,state.camera.elevation+(e.key==='ArrowUp'?.1:e.key==='ArrowDown'?-.1:0)));render();};
window.addEventListener('resize',render);document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
function tick(now){if(state.playing&&state.record){const elapsed=(now-state.lastFrame)/1000;state.lastFrame=now;state.time=clampTime(state.time+elapsed*Number($('speed').value),state.record.meta.actual_end_s);if(state.time>=state.record.meta.actual_end_s)pause();render();}requestAnimationFrame(tick);}requestAnimationFrame(tick);

try{state.catalog=await getJson('/api/catalog');$('catalog-count').textContent=`${state.catalog.groups.length} 组 · ${state.catalog.count} 条记录`;
  for(const g of state.catalog.groups)$('group').append(option(g.id,g.name));for(const c of state.catalog.classes)$('label').append(option(c.label,c.name));
  const query=new URLSearchParams(location.search),requestedGroup=Number(query.get('group')??6),requestedLabel=Number(query.get('label')??0);
  $('group').value=state.catalog.groups.some(g=>g.id===requestedGroup)?requestedGroup:state.catalog.groups[0].id;
  const group=state.catalog.groups.find(g=>g.id===Number($('group').value));for(const o of $('label').options)o.disabled=!group.labels.includes(Number(o.value));$('label').value=group.labels.includes(requestedLabel)?requestedLabel:group.labels[0];await loadRecord();
}catch(e){error(e.message);}
