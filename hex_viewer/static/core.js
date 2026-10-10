// Pure replay geometry and time rules, independently usable in checks.
export function lowerIndex(times, time) {
  let lo = 0, hi = times.length;
  while (lo < hi) {const mid = (lo + hi) >>> 1; if (times[mid] <= time + 1e-10) lo = mid + 1; else hi = mid;}
  return lo - 1;
}
export function clampTime(time, end) {
  if (!Number.isFinite(time) || !Number.isFinite(end) || end < 0) throw new Error('Invalid replay clock');
  return Math.max(0, Math.min(end, time));
}
export function rotation([roll, pitch, yaw]) {
  const cr=Math.cos(roll),sr=Math.sin(roll),cp=Math.cos(pitch),sp=Math.sin(pitch),cy=Math.cos(yaw),sy=Math.sin(yaw);
  return [[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]];
}
export function transform(R, point, offset=[0,0,0]) {
  return R.map((row,i)=>row.reduce((sum,x,j)=>sum+x*point[j],offset[i]));
}
export function tiltDegrees(attitude) {
  return Math.acos(Math.max(-1,Math.min(1,Math.cos(attitude[0])*Math.cos(attitude[1]))))*180/Math.PI;
}
export function phase(meta, time) {
  const s=meta.task;
  if(time<s.motion_start_s)return '起点保持';
  if(time>=s.motion_end_s)return '终点保持';
  if(s.family==='hover')return '定点悬停 · 偏航调整';
  const mid=(s.motion_start_s+s.motion_end_s)/2;
  const name=s.family==='circle'?'绕圈':s.family==='line'?'直线运动':s.displacement_m>0?'上升':'下降';
  return (time<mid?'平滑加速 · ':'平滑减速 · ')+name;
}
export function className(label) {
  if(label===0)return 'Normal / 健康';
  if(!Number.isInteger(label)||label<1||label>18)return '未知';
  return ['Full','Severe','Partial'][Math.floor((label-1)/6)]+' / M'+((label-1)%6);
}
export function predictionAt(predictions,time) {
  if(!predictions)return null;
  const i=lowerIndex(predictions.map(p=>p.time),time);
  return i<0?null:predictions[i];
}
