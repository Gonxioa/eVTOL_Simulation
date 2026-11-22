function item=hex_recovery_metrics(result)
%HEX_RECOVERY_METRICS Existing flight limits plus explicit recovery status.
P=result.parameters;t=result.time;s=result.state;ref=result.reference;
assert(~isempty(t),'hex:EmptyResult','No state samples were produced.');
position=vecnorm(s(:,1:3)-ref(:,1:3),2,2);
tilt=acosd(min(max(1-2*(s(:,7).^2+s(:,8).^2),-1),1));
rate=vecnorm(s(:,11:12),2,2);loss=P.task_origin(3)-s(:,3);
relative=s(:,1:3)-P.task_origin';
if P.task_family==2
    center=-P.radius*[cos(P.heading),sin(P.heading)];
    path_error=hypot(vecnorm(relative(:,1:2)-center,2,2)-P.radius,relative(:,3));
elseif P.task_family==1 || P.task_family==3
    direction=[cos(P.heading);sin(P.heading);0];
    if P.task_family==3,direction=[0;0;1];end
    along=min(max(relative*direction,0),P.displacement);
    path_error=vecnorm(relative-along*direction',2,2);
else
    path_error=position;
end
completed=abs(t(end)-P.duration)<1e-8 && abs(t(1))<1e-8 && ...
    numel(t)==round(P.duration/P.export_dt)+1;
finite=all(isfinite(s),'all') && all(isfinite(result.control_audit),'all');
if isfield(result,'controller_diagnostic'),finite=finite && all(isfinite(result.controller_diagnostic),'all');end
bounded=all(result.motor_command>=P.omega_min-1e-8 & result.motor_command<=P.omega_max+1e-8,'all');
integrals=all(abs(result.control_audit(:,13:15))<=P.control.velocity_integral_limit_m_s2'+1e-8,'all') && ...
    all(abs(result.control_audit(:,16:18))<=P.control.rate_integral_limit_rad_s2'+1e-8,'all');
post=t>=P.fault_start;late=t>=P.duration-3;
violates=[position>.15,tilt>15,rate>1,loss>.5];
last=find(post & any(violates,2),1,'last');last_time=NaN;offset=0;entry=P.fault_start;
status='recovered';criteria={};labels={'position','tilt','roll_pitch_rate','height_loss'};
if ~any(post)
    status='no_postfault_samples';entry=NaN;offset=Inf;
elseif ~isempty(last)
    last_time=t(last);offset=last_time-P.fault_start;criteria=labels(violates(last,:));
    if last==numel(t),status='not_recovered';entry=NaN;
    else,entry=t(last+1);end
end
return_time=offset;
if ~completed,status='incomplete';return_time=Inf;entry=NaN;
elseif ~strcmp(status,'recovered'),return_time=Inf;end
if completed && any(late)
    late_error=sqrt(mean(position(late).^2));late_path=sqrt(mean(path_error(late).^2));
else
    late_error=Inf;late_path=Inf;
end
stable=completed && finite && bounded && integrals && max(tilt)<=35 && ...
    min(s(:,3))>=P.task_origin(3)-1 && max(rate)<=5 && max(abs(s(:,13)))<=10;
returned=strcmp(status,'recovered') && late_error<=.10 && late_path<=.10 && return_time<=15;
phase=result.reference_audit(end,1);
item=struct('task',P.task_family,'motor',P.fault_motor,'factor',P.fault_factor,...
    'seed',P.seed,'observed_feedback',P.observed_feedback,'feedback_delay_s',P.feedback_delay,...
    'duration_s',t(end),'completed',completed,'finite',finite,'commands_bounded',bounded,...
    'integrals_bounded',integrals,'peak_tracking_error_m',max(position),...
    'peak_path_error_m',max(path_error),'late_tracking_rmse_m',late_error,...
    'late_path_rmse_m',late_path,'final_tracking_error_m',position(end),...
    'peak_tilt_deg',max(tilt),'peak_rp_rate_rad_s',max(rate),...
    'peak_yaw_rate_rad_s',max(abs(s(:,13))),'minimum_altitude_m',min(s(:,3)),...
    'return_time_s',return_time,'phase_s',phase,'mission_phase_complete',phase>=P.motion_end,...
    'estimated_effectiveness',result.control_audit(end,44:49),...
    'stable',stable,'returned',returned,'passed',stable && returned,...
    'recovery_status',status,'last_violation_time_s',last_time,...
    'last_violation_offset_s',offset,'sustained_entry_time_s',entry,'last_violation_criteria',{criteria});
if size(result.reference_audit,2)>=8
    nominal=result.reference_audit(:,3:5);
    nominal_error=vecnorm(s(:,1:3)-nominal,2,2);
    item.nominal_tracking_rmse_m=sqrt(mean(nominal_error.^2));
    item.late_nominal_rmse_m=sqrt(mean(nominal_error(late).^2));
end
item.position_rmse_m=sqrt(mean(position.^2));
item.position_iae_m_s=trapz(t,position);
item.actuator_saturated_time_s=sum(diff(t).*double(any(result.motor_command(1:end-1,:)>=P.omega_max-1e-6,2)));
if isfield(result,'controller_diagnostic')
    d=result.controller_diagnostic;gate=d(:,34:36)>.5;
    item.max_velocity_integral_m_s2=max(abs(result.control_audit(:,13:15)),[],1);
    item.max_rate_integral_rad_s2=max(abs(result.control_audit(:,16:18)),[],1);
    item.rate_gate_longest_s=zeros(1,3);
    for axis=1:3,item.rate_gate_longest_s(axis)=longest(gate(:,axis),t);end
    item.late_rate_error_rms_rad_s=sqrt(mean(result.control_audit(late,25:27).^2,1));
    item.velocity_backcalc_components_m_s2=sum(d(:,49:63),1);
    detailed=hex_execution_diagnostics(result);
    angle=detailed.data(:,size(d,2)+19);
    item.late_thrust_direction_error_deg=max(angle(late));
    item.peak_thrust_direction_error_deg=max(angle);
end
end

function duration=longest(flags,time)
duration=0;current=0;
for i=1:numel(time)-1
    if flags(i),current=current+time(i+1)-time(i);else,current=0;end
    duration=max(duration,current);
end
end
