function [command,audit]=controller_step(observation,reference,P)
%#codegen
% No effectiveness, fault label or plant audit enters this block.
persistent previous_w previous_v filtered_alpha filtered_accel vi ri last_qd initialized
if isempty(initialized)
    previous_w=zeros(3,1); previous_v=zeros(3,1);
    filtered_alpha=zeros(3,1); filtered_accel=zeros(3,1);
    vi=zeros(3,1); ri=zeros(3,1); last_qd=[0;0;0;1]; initialized=false;
end
x=observation(1:3); v=observation(4:6); q=observation(7:10); q=q/norm(q); w=observation(11:13);
R=hex_rotation(q); c=P.control; dt=P.control_dt;
vi_applied=vi; ri_applied=ri;
ev=min(max(reference(4:6)+c.position_p.*(reference(1:3)-x),-c.velocity_limit_m_s),c.velocity_limit_m_s)-v;
filt=1-exp(-2*pi*c.derivative_filter_hz*dt);
if initialized
    filtered_alpha=filtered_alpha+filt*((w-previous_w)/dt-filtered_alpha);
    filtered_accel=filtered_accel+filt*((v-previous_v)/dt-filtered_accel);
end
% Task input is position/velocity only; acceleration comes from feedback.
a_raw=c.velocity_p.*ev+vi-c.velocity_d.*filtered_accel;
acc=min(max(a_raw,-c.acceleration_limit_m_s2),c.acceleration_limit_m_s2);
F=P.mass*(acc+[0;0;P.gravity]); F(3)=max(F(3),0);
max_horizontal=F(3)*tan(c.tilt_limit_deg*pi/180);
if norm(F(1:2))>max_horizontal
    F(1:2)=F(1:2)*max_horizontal/max(norm(F(1:2)),1e-12);
end
maximum=sum(P.thrust_max);
if norm(F)>maximum
    F=F*maximum/norm(F);
end
qd=hex_desired_attitude(F,c.yaw_hold_rad,last_qd);
e=hex_multiply([-q(1:3);q(4)],qd);
sgn=1;
if e(4)<-1e-12
    sgn=-1;
elseif abs(e(4))<=1e-12
    for j=1:3
        if abs(e(j))>1e-12
            if e(j)<0, sgn=-1; end
            break
        end
    end
end
attitude_error=2*sgn*e(1:3);
rates=min(max(c.attitude_p.*attitude_error,-c.rate_limit_rad_s),c.rate_limit_rad_s);
error=rates-w;
alpha=c.rate_p.*error+ri-c.rate_d.*filtered_alpha;
moment=P.J*alpha+cross(w,P.J*w);
desired=[max(F'*R(:,3),0);moment];
[thrust,residual,flags]=hex_allocate(desired,P);
di=c.rate_i.*error*dt; di((residual(2:4).*error)>1e-8)=0;
ri=min(max(ri+di,-c.rate_integral_limit_rad_s2),c.rate_integral_limit_rad_s2);
limited_acc=F/P.mass-[0;0;P.gravity]; direction=F/max(norm(F),1e-12);
vi=min(max(vi+dt*(c.velocity_i.*ev+c.antiwindup_gain_s_inv*(limited_acc-a_raw-direction*residual(1)/P.mass)),...
    -c.velocity_integral_limit_m_s2),c.velocity_integral_limit_m_s2);
command=sqrt(thrust./P.kT);
audit=[desired;P.A*thrust;residual;vi;ri;reference(1:3)-x;ev;error;qd;rates;flags;vi_applied;ri_applied];
previous_w=w; previous_v=v; last_qd=qd; initialized=true;
end
