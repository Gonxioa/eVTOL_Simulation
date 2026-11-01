function hex_validate_parameters(P)
assert(isfinite(P.duration) && P.duration>0,'Duration must be finite and positive.');
assert(P.fault_motor==fix(P.fault_motor) && P.fault_motor>=0 && P.fault_motor<=6,'FaultMotor is 0 (healthy) or 1..6.');
assert(isfinite(P.fault_factor) && P.fault_factor>=0 && P.fault_factor<=1,'FaultFactor must be in [0,1].');
assert(isfinite(P.fault_start) && P.fault_start>=0 && P.fault_end>P.fault_start,'Invalid fault/control interval.');
assert(P.task_family>=0 && P.task_family<=6 && P.task_family==fix(P.task_family),'Invalid task selection.');
assert(isfield(P.control,'yaw_hold_rad') && isscalar(P.control.yaw_hold_rad) && isfinite(P.control.yaw_hold_rad),'Internal fixed heading control.yaw_hold_rad must be a finite scalar in radians.');
assert(P.motion_end>P.motion_start && P.motion_start>=0,'Invalid motion interval.');
assert(P.sensor_dt==P.control_dt,'This delivery requires matching sensor and controller rates.');
assert(P.integration_dt<=.2*min(P.tau),'RK4 step too large for motor dynamics.');
assert(P.feedback_delay>=0 && P.feedback_delay<=1000*P.sensor_dt,'Feedback delay exceeds the 1001-sample buffer.');
assert(abs(P.duration/P.export_dt-round(P.duration/P.export_dt))<1e-8,'Duration must align with the export clock.');
assert(size(P.exogenous,1)>=ceil(P.duration/P.sensor_dt)+1 && size(P.exogenous,2)==24,'Noise input table has the wrong size.');
assert(all(isfinite(P.exogenous),'all') && all(isfinite(P.initial_state)) && norm(P.initial_state(7:10))>1e-12,'Nonfinite input or zero quaternion.');
rates=[P.control_dt P.sensor_dt P.export_dt]/P.base_dt;
assert(all(abs(rates-round(rates))<1e-8),'Sample periods must be integer multiples of the base period.');
end
