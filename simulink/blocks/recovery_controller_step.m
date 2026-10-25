function [command,audit]=recovery_controller_step(observation,reference,P)
%#codegen
% Motor response is estimated from RPM feedback and delayed command history.
% Fault truth is not an input. eta_hat describes command-to-thrust loss only
% for this project's first-order motor-target failure model.
persistent previous_w previous_v filtered_alpha filtered_accel vi ri last_qd
persistent previous_rpm speed_ratio command_history ticks initialized
if isempty(initialized)
    previous_w=zeros(3,1); previous_v=zeros(3,1);
    filtered_alpha=zeros(3,1); filtered_accel=zeros(3,1);
    vi=zeros(3,1); ri=zeros(3,1); last_qd=[0;0;0;1];
    previous_rpm=zeros(6,1); speed_ratio=ones(6,1);
    command_history=zeros(6,1001); ticks=0; initialized=false;
end
x=observation(1:3); v=observation(4:6);
q=observation(7:10); q=q/norm(q); w=observation(11:13);
rpm=observation(14:19); R=hex_rotation(q);
c=P.control; r=c.recovery; dt=P.control_dt;
delay_ticks=0;
if P.observed_feedback, delay_ticks=ceil(P.feedback_delay/P.sensor_dt-1e-10); end
if initialized && ticks>delay_ticks
    previous_command=command_history(:,mod(ticks-delay_ticks-1,1001)+1);
    decay=exp(-dt./P.tau);
    for i=1:6
        if previous_command(i)>=r.estimator_min_command_rad_s
            observed=(rpm(i)-decay(i)*previous_rpm(i))/...
                ((1-decay(i))*previous_command(i));
            observed=min(max(observed,0),1.1);
            speed_ratio(i)=speed_ratio(i)+(1-exp(-dt/r.estimator_tau_s))*(observed-speed_ratio(i));
        end
    end
end
eta=min(speed_ratio.^2,1);
eta(speed_ratio>=r.healthy_speed_ratio)=1;
eta(speed_ratio<r.failed_speed_ratio)=0;
damage=min(max((1-min(eta))/.8,0),1);
position_p=(1-damage)*c.position_p+damage*r.degraded_position_p;
velocity_p=(1-damage)*c.velocity_p+damage*r.degraded_velocity_p;
acc_limit=(1-damage)*c.acceleration_limit_m_s2+damage*r.degraded_acceleration_limit_m_s2;
tilt_limit=(1-damage)*c.tilt_limit_deg+damage*r.degraded_tilt_limit_deg;
vi_applied=vi; ri_applied=ri;
ev=min(max(reference(4:6)+position_p.*(reference(1:3)-x),...
    -c.velocity_limit_m_s),c.velocity_limit_m_s)-v;
filt=1-exp(-2*pi*c.derivative_filter_hz*dt);
if initialized
    filtered_alpha=filtered_alpha+filt*((w-previous_w)/dt-filtered_alpha);
    filtered_accel=filtered_accel+filt*((v-previous_v)/dt-filtered_accel);
end
a_raw=velocity_p.*ev+vi-c.velocity_d.*filtered_accel;
acc=min(max(a_raw,-acc_limit),acc_limit);
F=P.mass*(acc+[0;0;P.gravity]); F(3)=max(F(3),0);
max_horizontal=F(3)*tan(tilt_limit*pi/180);
if norm(F(1:2))>max_horizontal
    F(1:2)=F(1:2)*max_horizontal/max(norm(F(1:2)),1e-12);
end
maximum=sum(eta.*P.thrust_max);
if norm(F)>maximum, F=F*maximum/max(norm(F),1e-12); end
qd=hex_desired_attitude(F,c.yaw_hold_rad,last_qd);
e=hex_multiply([-q(1:3);q(4)],qd);
sgn=1; if e(4)<0, sgn=-1; end
attitude_error=2*sgn*e(1:3);
if damage>.1
    % Thrust direction has priority over heading during degraded flight.
    attitude_error=R'*cross(R(:,3),F/max(norm(F),1e-12));
end
rates=min(max(c.attitude_p.*attitude_error,-c.rate_limit_rad_s),c.rate_limit_rad_s);
error=rates-w;
alpha=c.rate_p.*error+ri-c.rate_d.*filtered_alpha;
if damage>.1
    rates(3)=0; error(3)=-w(3);
    alpha(3)=-r.yaw_damping_s_inv*w(3); ri(3)=0;
end
moment=P.J*alpha+cross(w,P.J*w);
desired=[max(F'*R(:,3),0);moment];
Q=P; Q.thrust_max=max(eta.*P.thrust_max,1e-9);
Q.thrust_min=zeros(6,1);
if damage>.1, Q.row_weights=r.degraded_wrench_priority./P.wrench_scale; end
[thrust,residual,flags]=hex_allocate(desired,Q);
thrust=min(thrust,eta.*P.thrust_max);
di=c.rate_i.*error*dt; di((residual(2:4).*error)>1e-8)=0;
if damage>.1, di(3)=0; end
ri=min(max(ri+di,-c.rate_integral_limit_rad_s2),c.rate_integral_limit_rad_s2);
limited_acc=F/P.mass-[0;0;P.gravity]; direction=F/max(norm(F),1e-12);
vi=min(max(vi+dt*(c.velocity_i.*ev+c.antiwindup_gain_s_inv*...
    (limited_acc-a_raw-direction*residual(1)/P.mass)),...
    -c.velocity_integral_limit_m_s2),c.velocity_integral_limit_m_s2);
command=min(sqrt(max(thrust,0)./P.kT)./max(sqrt(eta),1e-8),P.omega_max);
command(eta==0)=0;
audit=[desired;P.A*thrust;residual;vi;ri;reference(1:3)-x;ev;error;qd;rates;flags;...
    vi_applied;ri_applied;eta;damage];
command_history(:,mod(ticks,1001)+1)=command;
previous_rpm=rpm; previous_w=w; previous_v=v; last_qd=qd;
ticks=ticks+1; initialized=true;
end
