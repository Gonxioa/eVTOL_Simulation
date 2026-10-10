// Run: node tests/check_hex_replay.mjs (Node is a test tool, not a viewer dependency).
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
const source=await readFile(new URL('../hex_viewer/static/core.js',import.meta.url),'utf8');
const {lowerIndex,clampTime,rotation,transform,tiltDegrees,phase,predictionAt}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
assert.equal(lowerIndex([0,.01,.02],.019),1);
assert.equal(lowerIndex([0,.01],-.1),-1);
assert.equal(clampTime(8,3.8),3.8);
assert.throws(()=>clampTime(NaN,8));
assert.equal(predictionAt([{time:.49,valid:true},{time:.59,valid:true}],.48),null);
assert.equal(predictionAt([{time:.49,valid:true},{time:.59,valid:true}],.58).time,.49);
// Positive yaw maps body +x to world +y; positive roll maps +y to +z.
const yaw=transform(rotation([0,0,Math.PI/2]),[1,0,0]);
near(yaw[0],0);near(yaw[1],1);near(yaw[2],0);
const roll=transform(rotation([Math.PI/2,0,0]),[0,1,0]);
near(roll[0],0);near(roll[1],0);near(roll[2],1);
near(tiltDegrees([0,0,3]),0);near(tiltDegrees([Math.PI/2,0,1]),90);
const moving={task:{family:'circle',motion_start_s:1,motion_end_s:7}};
assert.equal(phase(moving,0),'起点保持');
assert.equal(phase(moving,2.6),'平滑加速 · 绕圈');
assert.equal(phase(moving,4),'平滑减速 · 绕圈');
assert.equal(phase(moving,7),'终点保持');
console.log('PASS: saved clock/early end, causal prediction lookup, ENU/FLU pose, task phases');
