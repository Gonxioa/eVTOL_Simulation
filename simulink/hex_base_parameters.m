function P = hex_base_parameters()
%HEX_BASE_PARAMETERS Configuration reader and derived constants; use hex_parameters.
root = fileparts(mfilename('fullpath'));
c = jsondecode(fileread(fullfile(root, 'configs', 'research.json')));
a = c.airframe; p = c.propulsion;
P.mass = a.center_mass_kg + 6*a.hub_mass_kg;
P.gravity = a.gravity_m_s2;
phi = (0:5)'*pi/3;
P.rotor_positions = a.arm_radius_m*[cos(phi),sin(phi),zeros(6,1)];
J = diag([repmat(a.center_mass_kg*(3*a.center_radius_m^2+a.center_height_m^2)/12,1,2),a.center_mass_kg*a.center_radius_m^2/2]);
for i=1:6
    r=P.rotor_positions(i,:)';
    J=J+a.hub_mass_kg*((r'*r)*eye(3)-r*r');
end
P.J=J; P.invJ=inv(J);
P.kT=p.kT*ones(6,1); P.kQ=p.kT*p.Q_over_T_m*ones(6,1);
P.yaw_sign=[1;-1;1;-1;1;-1];
P.tau=p.tau_s*ones(6,1);
P.omega_min=p.omega_min_rad_s; P.omega_max=p.omega_max_rad_s;
P.body_drag=c.aerodynamics.body_drag_kg_m(:);
P.rotor_drag=c.aerodynamics.rotor_drag_kg_per_rad(:);
P.A=[ones(1,6);P.rotor_positions(:,2)';-P.rotor_positions(:,1)';(P.yaw_sign.*P.kQ./P.kT)'];
P.pinvA=pinv(P.A);
P.thrust_min=P.kT*P.omega_min^2; P.thrust_max=P.kT*P.omega_max^2;
P.wrench_scale=sum(abs(P.A).*P.thrust_max',2);
P.row_weights=c.allocator.wrench_priority(:)./P.wrench_scale;
P.regularization=c.allocator.regularization;
P.control=c.control;
P.sensor=c.sensor;
P.sensor.installation_matrix=local_rotation(c.sensor.installation_rotation_xyzw(:));
P.wind=c.wind; P.termination=c.termination;
P.integration_dt=1/c.timing.integration_hz;
P.control_dt=1/c.timing.control_hz;
P.sensor_dt=1/c.sensor.rate_hz;
P.export_dt=1/c.timing.export_hz;
P.base_dt=min([P.control_dt,P.sensor_dt,P.export_dt]);
P.duration=8;
P.reference_mode='position_velocity_fixed_heading';
P.control_tuning_version=c.control_tuning_version;
P.initial_state=[0;0;2.5;zeros(3,1);0;0;0;1;zeros(3,1);sqrt(P.mass*P.gravity/sum(P.kT))*ones(6,1)];
% 0 hover; 1 line; 2 circle; 3 vertical; 4 legacy tracking; 5 step; 6 saturation.
P.task_family=0; P.task_origin=[0;0;2.5];
P.motion_start=1; P.motion_end=7;
P.heading=0; P.displacement=1.2; P.radius=.3; P.turn=2*pi;
% Public rotor numbering: 0 means healthy; 1..6 map to Python motors 0..5.
P.fault_motor=0; P.fault_factor=.8; P.fault_start=3; P.fault_end=Inf;
P.observed_feedback=true; P.noise_enabled=true; P.wind_enabled=true;
P.feedback_delay=c.sensor.feedback_delay_s;
P.seed=c.seed;
P.external_force=zeros(3,1); P.external_torque=zeros(3,1);
% [wind3 accel_error3 gyro_error3 position_error3 velocity_error3 rotvec3 rpm_error6]
P.exogenous=zeros(ceil(P.duration/P.sensor_dt)+2,24);
end

function R=local_rotation(q)
q=q/norm(q); v=q(1:3); w=q(4);
S=[0 -v(3) v(2);v(3) 0 -v(1);-v(2) v(1) 0];
R=eye(3)+2*w*S+2*S*S;
end
