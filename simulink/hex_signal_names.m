function names=hex_signal_names()
%HEX_SIGNAL_NAMES Header order for the exported signals.csv.
names={'time_s'};
names=[names,{'east_m','north_m','up_m','ve_m_s','vn_m_s','vu_m_s','qx','qy','qz','qw','wx_rad_s','wy_rad_s','wz_rad_s'},series('rotor_actual_rad_s',6)];
names=[names,series('reference_position_m',3),series('reference_velocity_m_s',3)];
names=[names,series('rotor_command_rad_s',6),series('accelerometer_sensor_m_s2',3),series('gyro_sensor_rad_s',3),series('observation_state',19),series('wind_enu_m_s',3),series('plant_effectiveness',6)];
names=[names,series('wrench_desired',4),series('wrench_nominal_allocated',4),series('wrench_nominal_residual',4),...
    series('velocity_integral_m_s2',3),series('rate_integral_rad_s2',3),series('position_error_m',3),series('velocity_error_m_s',3),...
    series('rate_error_rad_s',3),series('quaternion_desired_xyzw',4),series('rate_reference_rad_s',3),...
    {'allocator_success','allocator_active_bound_count','allocator_wrench_feasible'},...
    series('velocity_integral_applied_m_s2',3),series('rate_integral_applied_rad_s2',3),series('feedback_state',19)];
end

function values=series(prefix,n)
values=cell(1,n);
for k=1:n, values{k}=sprintf('%s_%d',prefix,k); end
end
