function [reference,audit]=recovery_reference_step(t,observation,P)
%#codegen
% Retimes the original geometric path using measured tracking/stability.
% No fault identity, effectiveness, schedule or plant audit is available.
persistent phase progress initialized
if isempty(initialized)
    phase=0; progress=1; initialized=false;
end
r=P.control.recovery;
previous=task_reference(phase,P);
error=norm(previous(1:3)-observation(1:3));
q=observation(7:10)/norm(observation(7:10));
tilt=acosd(min(max(1-2*(q(1)^2+q(2)^2),-1),1));
rate=norm(observation(11:12));
request=min([ramp(error,r.position_soft_m,r.position_hard_m),...
    ramp(tilt,r.tilt_soft_deg,r.tilt_hard_deg),...
    ramp(rate,r.rate_soft_rad_s,r.rate_hard_rad_s)]);
tau=r.progress_fast_tau_s;
if request<progress, tau=r.progress_slow_tau_s; end
if initialized
    progress=progress+(1-exp(-P.control_dt/tau))*(request-progress);
    phase=min(t,phase+P.control_dt*progress);
end
reference=task_reference(phase,P);
reference(4:6)=progress*reference(4:6);
nominal=task_reference(t,P);
audit=[phase;progress;nominal;error;tilt];
initialized=true;
end

function value=ramp(x,soft,hard)
value=min(max((hard-x)/(hard-soft),0),1);
end
