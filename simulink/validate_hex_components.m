function report=validate_hex_components()
%VALIDATE_HEX_COMPONENTS Independent physical identities beyond trajectories.
root=fileparts(mfilename('fullpath'));
addpath(fullfile(root,'blocks'),fullfile(root,'functions'));
P=hex_parameters(); s=P.initial_state; command=s(14:19); eta=ones(6,1);
ds=hex_derivative(s,command,eta,zeros(3,1),zeros(6,1),P);
assert(norm(ds)<1e-11,'Analytical hover balance failed.');
identities={'hover_balance'};
% Per-rotor lever-arm and yaw signs, independent of controller and allocator.
for rotor=1:6
    z=s; z(14:19)=0; z(13+rotor)=100;
    dz=hex_derivative(z,zeros(6,1),eta,zeros(3,1),zeros(6,1),P);
    thrust=P.kT(rotor)*100^2; r=P.rotor_positions(rotor,:)';
    expected=[r(2)*thrust;-r(1)*thrust;P.yaw_sign(rotor)*P.kQ(rotor)*100^2];
    assert(norm(P.J*dz(11:13)-expected)<1e-12,'Rotor moment sign failed.');
end
identities{end+1}='six_individual_rotor_moments';
z=s; z(14:19)=0; clear sensor_step
[~,~,imu]=sensor_step(0,z,zeros(3,1),zeros(6,1),P);
assert(norm(imu)<1e-12,'Free fall must measure zero specific force.');
identities{end+1}='freefall_imu';
% At t=0 the sensor lever term includes alpha x r and w x (w x r).
custom=P; custom.sensor.lever_m=[.1;0;0];
custom.sensor.installation_matrix=[0 -1 0;1 0 0;0 0 1];
z=s; z(11:13)=[0;0;2]; clear sensor_step
[~,~,imu]=sensor_step(0,z,zeros(3,1),zeros(6,1),custom);
dz=hex_derivative(z,zeros(6,1),eta,zeros(3,1),zeros(6,1),custom);
force_body=dz(4:6)+[0;0;P.gravity];
expected=custom.sensor.installation_matrix'*(force_body+cross(dz(11:13),custom.sensor.lever_m)+[-.4;0;0]);
assert(norm(imu(1:3)-expected)<1e-12 && norm(imu(4:6)-[0;0;2])<1e-12,'Sensor installation/lever arm failed.');
identities{end+1}='sensor_installation_and_lever_arm';
stopP=P; stopP.fault_motor=1; stopP.fault_factor=0; stopP.fault_start=0;
[segments,~]=fault_schedule(0,stopP);
next=plant_step(s,command,segments,zeros(3,1),zeros(6,1),stopP);
expected_speed=command(1)*exp(-P.base_dt/P.tau(1));
assert(abs(next(14)-expected_speed)<5e-6,'Mechanical speed must follow the analytical motor decay.');
identities{end+1}='motor_exponential_decay';
% Control results must reset exactly for independent flights.
clear controller_step
[a,~]=controller_step(s,task_reference(0,P),P);
clear controller_step
[b,~]=controller_step(s,task_reference(0,P),P);
assert(isequal(a,b),'Controller state was not reset.');
identities{end+1}='controller_reset';
ref=task_reference(0,P);
assert(isequal(size(ref),[6 1]),'Task input must contain only three positions and three velocities.');
heading=P; heading.control.yaw_hold_rad=pi/3; clear controller_step
[~,fixed_heading]=controller_step(s,ref,heading);
expected_q=[0;0;sin(pi/6);cos(pi/6)];
assert(norm(fixed_heading(28:31)-expected_q)<1e-12,'Controller must use its internal fixed heading.');
identities{end+1}='six_element_reference_and_internal_heading';
clear controller_step
[level,~]=controller_step(s,ref,P);
tilted=s; tilted(7:10)=[sin(.015);0;0;cos(.015)]; clear controller_step
[attitude_response,~]=controller_step(tilted,ref,P);
rotating=s; rotating(11)=.02; clear controller_step
[rate_response,~]=controller_step(rotating,ref,P);
assert(norm(attitude_response-level)>1e-5 && norm(rate_response-level)>1e-5,...
    'Attitude and angular-rate feedback must remain active.');
identities{end+1}='attitude_and_rate_feedback_retained';
ref=task_reference(0,P); clear controller_step
[positive,~]=controller_step(tilted,ref,P);
tilted(7:10)=-tilted(7:10); clear controller_step
[negative,~]=controller_step(tilted,ref,P);
assert(norm(positive-negative)<1e-10,'Antipodal quaternions must produce identical commands.');
identities{end+1}='antipodal_quaternion_controller';
P.noise_enabled=true; P.wind_enabled=true;
a=hex_prepare_inputs(P); b=hex_prepare_inputs(P);
assert(isequal(a.exogenous,b.exogenous),'Random inputs must be reproducible.');
identities{end+1}='independent_input_stream_reproducibility';
report=struct('passed',true,'analytical_checks',{identities});
fprintf('ANALYTICAL_CHECKS_PASSED %d\n',numel(identities));
end
