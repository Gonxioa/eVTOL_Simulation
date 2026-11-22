function P=hex_parameters()
%HEX_PARAMETERS Current six-rotor simulation and control parameters.
% Fault truth remains confined to the plant.
P=hex_base_parameters();
P.duration=24; P.motion_start=1; P.motion_end=7; P.fault_start=3;
P.reference_mode='path_progress';
% Physical, controller and environment values share configs/research.json.
% Degraded Kp/Ki = 4/1.6 gives a 2.5-s integral time scale.
P.control_tuning_version='control_recovery_v4_20261122';
P=hex_prepare_inputs(P);
end
