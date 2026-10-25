function [feedback,observation,imu]=sensor_step(t,state,wind,loads,P)
%#codegen
% Direct noisy observations, not an estimator. IMU is sensor-frame force/rate.
persistent history samples
if isempty(samples)
    history=zeros(19,1001); samples=0;
end
k=min(size(P.exogenous,1),round(t/P.sensor_dt)+1);
errors=P.exogenous(k,:)';
R=hex_rotation(state(7:10)); w=state(11:13);
ds=hex_derivative(state,zeros(6,1),ones(6,1),wind,loads,P);
r=P.sensor.lever_m; install=P.sensor.installation_matrix;
body=R'*(ds(4:6)+[0;0;P.gravity])+cross(ds(11:13),r)+cross(w,cross(w,r));
accel=install'*body; gyro=install'*w;
observation=state;
if P.noise_enabled
    accel=accel+errors(4:6); gyro=gyro+errors(7:9);
    observation(1:3)=state(1:3)+errors(10:12);
    observation(4:6)=state(4:6)+errors(13:15);
    theta=errors(16:18); angle=norm(theta); dq=[0;0;0;1];
    if angle>1e-14
        dq=[theta*(sin(angle/2)/angle);cos(angle/2)];
    end
    observation(7:10)=hex_multiply(state(7:10),dq);
    observation(7:10)=observation(7:10)/norm(observation(7:10));
    observation(14:19)=state(14:19)+errors(19:24);
end
observation(11:13)=install*gyro;
imu=[accel;gyro];
slot=mod(samples,1001)+1; history(:,slot)=observation; samples=samples+1;
% Latest sensor timestamp <= control timestamp - delay; hold the first sample
% during startup, as the Python history selector does.
delay_ticks=ceil(P.feedback_delay/P.sensor_dt-1e-10);
selected=max(0,samples-1-delay_ticks);
feedback=state;
if P.observed_feedback
    feedback=history(:,mod(selected,1001)+1);
end
end
