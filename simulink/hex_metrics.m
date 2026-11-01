function item=hex_metrics(result)
%HEX_METRICS Control/rejoin acceptance is independent of mission end.
P=result.parameters; t=result.time; s=result.state; ref=result.reference;
error=vecnorm(s(:,1:3)-ref(:,1:3),2,2);
tilt=acosd(min(max(1-2*(s(:,7).^2+s(:,8).^2),-1),1));
rate=vecnorm(s(:,11:12),2,2); origin=P.task_origin';
relative=s(:,1:3)-origin;
if P.task_family==2
    center=-P.radius*[cos(P.heading),sin(P.heading)];
    horizontal=vecnorm(relative(:,1:2)-center,2,2)-P.radius;
    path_error=hypot(horizontal,relative(:,3));
elseif P.task_family==1 || P.task_family==3
    direction=[cos(P.heading);sin(P.heading);0];
    if P.task_family==3, direction=[0;0;1]; end
    distance=min(max(relative*direction,0),P.displacement);
    path_error=vecnorm(relative-distance*direction',2,2);
else
    path_error=error;
end
post=t>=P.fault_start; late=t>=P.duration-3;
violates=error>.15 | tilt>15 | rate>1 | s(:,3)<P.task_origin(3)-.5;
last=find(post & violates,1,'last'); return_time=0;
if ~isempty(last), return_time=t(last)-P.fault_start; end
if ~any(late) || isempty(t) || t(end)<P.duration-P.export_dt/2
    return_time=Inf; late_error=Inf; late_path=Inf;
else
    late_error=sqrt(mean(error(late).^2)); late_path=sqrt(mean(path_error(late).^2));
end
bounded=all(result.motor_command>=-1e-8 & result.motor_command<=P.omega_max+1e-8,'all');
finite=all(isfinite(s),'all') && all(isfinite(result.control_audit),'all');
integrals=all(abs(result.control_audit(:,13:15))<=P.control.velocity_integral_limit_m_s2'+1e-8,'all') && ...
    all(abs(result.control_audit(:,16:18))<=P.control.rate_integral_limit_rad_s2'+1e-8,'all');
completed=abs(t(end)-P.duration)<1e-8;
stable=completed && finite && bounded && integrals && max(tilt)<=35 && ...
    min(s(:,3))>=P.task_origin(3)-1 && max(rate)<=5 && max(abs(s(:,13)))<=10;
returned=late_error<=.10 && late_path<=.10 && return_time<=15;
phase=result.reference_audit(end,1);
item=struct('task',P.task_family,'motor',P.fault_motor,'factor',P.fault_factor,...
    'seed',P.seed,'observed_feedback',P.observed_feedback,'feedback_delay_s',P.feedback_delay,...
    'duration_s',t(end),'completed',completed,'finite',finite,'commands_bounded',bounded,...
    'integrals_bounded',integrals,'peak_tracking_error_m',max(error),...
    'peak_path_error_m',max(path_error),'late_tracking_rmse_m',late_error,...
    'late_path_rmse_m',late_path,'final_tracking_error_m',error(end),...
    'peak_tilt_deg',max(tilt),'peak_rp_rate_rad_s',max(rate),...
    'peak_yaw_rate_rad_s',max(abs(s(:,13))),'minimum_altitude_m',min(s(:,3)),...
    'return_time_s',return_time,'phase_s',phase,'mission_phase_complete',phase>=P.motion_end,...
    'estimated_effectiveness',result.control_audit(end,44:49),...
    'stable',stable,'returned',returned,'passed',stable && returned);
end
