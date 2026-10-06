# 参数实际读取路径

| 键 | 单位 | 身份 | 读取/适用模块 |
| --- | --- | --- | --- |
| model_version | identifier | research_design_assumption | configuration metadata/signature |
| parameter_status | evidence status | research_design_assumption | configuration metadata/signature |
| frame | coordinate convention | research_design_assumption | configuration metadata/signature |
| airframe.center_mass_kg | kg | research_design_assumption | configuration.physical |
| airframe.center_radius_m | m | research_design_assumption | configuration.physical |
| airframe.center_height_m | m | research_design_assumption | configuration.physical |
| airframe.hub_mass_kg | kg | research_design_assumption | configuration.physical |
| airframe.arm_radius_m | m | research_design_assumption | configuration.physical |
| airframe.rotor_radius_m | m | research_design_assumption | configuration.physical |
| airframe.gravity_m_s2 | m/s2 | research_design_assumption | configuration.physical |
| propulsion.kT | N/(rad/s)^2 | research_design_assumption | configuration.physical; dynamics; control; simulation.export |
| propulsion.Q_over_T_m | m | research_design_assumption | configuration.physical; dynamics; control; simulation.export |
| propulsion.tau_s | s | research_design_assumption | configuration.physical; dynamics; control; simulation.export |
| propulsion.omega_min_rad_s | rad/s | research_design_assumption | configuration.physical; dynamics; control; simulation.export |
| propulsion.omega_max_rad_s | rad/s | research_design_assumption | configuration.physical; dynamics; control; simulation.export |
| aerodynamics.model | model identity | research_design_assumption | dynamics.HexPlant.wrench |
| aerodynamics.body_drag_kg_m | kg/m | research_design_assumption | dynamics.HexPlant.wrench |
| aerodynamics.rotor_drag_kg_per_rad | kg/rad | research_design_assumption | dynamics.HexPlant.wrench |
| control.position_p | 1/s | research_design_assumption | control.CascadeController.update |
| control.velocity_p | 1/s | research_design_assumption | control.CascadeController.update |
| control.velocity_i | 1/s2 | research_design_assumption | control.CascadeController.update |
| control.velocity_d | 1 | research_design_assumption | control.CascadeController.update |
| control.attitude_p | 1/s | research_design_assumption | control.CascadeController.update |
| control.rate_p | 1/s | research_design_assumption | control.CascadeController.update |
| control.rate_i | 1/s2 | research_design_assumption | control.CascadeController.update |
| control.rate_d | 1 | research_design_assumption | control.CascadeController.update |
| control.velocity_limit_m_s | m/s | research_design_assumption | control.CascadeController.update |
| control.rate_limit_rad_s | rad/s | research_design_assumption | control.CascadeController.update |
| control.velocity_integral_limit_m_s2 | m/s2 | research_design_assumption | control.CascadeController.update |
| control.rate_integral_limit_rad_s2 | rad/s2 | research_design_assumption | control.CascadeController.update |
| control.acceleration_limit_m_s2 | m/s2 | research_design_assumption | control.CascadeController.update |
| control.tilt_limit_deg | deg | research_design_assumption | control.CascadeController.update |
| control.derivative_filter_hz | Hz | research_design_assumption | control.CascadeController.update |
| control.antiwindup_gain_s_inv | 1/s | research_design_assumption | control.CascadeController.update |
| allocator.wrench_priority | relative normalized weight | research_design_assumption | control.CascadeController -> control_allocation.BoundedControlAllocator |
| allocator.regularization | dimensionless objective weight | research_design_assumption | control.CascadeController -> control_allocation.BoundedControlAllocator |
| allocator.solver_tolerance | dimensionless numerical tolerance | research_design_assumption | control.CascadeController -> control_allocation.BoundedControlAllocator |
| allocator.feasibility_tolerance | normalized wrench tolerance | research_design_assumption | control.CascadeController -> control_allocation.BoundedControlAllocator |
| allocator.bound_tolerance | N | research_design_assumption | control.CascadeController -> control_allocation.BoundedControlAllocator |
| allocator.max_iterations | iteration count | research_design_assumption | control.CascadeController -> control_allocation.BoundedControlAllocator |
| sensor.rate_hz | Hz | research_design_assumption | observation.Sensors; simulation feedback queue |
| sensor.accel_noise_density | (m/s2)*sqrt(s) | research_design_assumption | observation.Sensors; simulation feedback queue |
| sensor.gyro_noise_density | (rad/s)*sqrt(s) | research_design_assumption | observation.Sensors; simulation feedback queue |
| sensor.accel_bias_diffusion | (m/s2)/sqrt(s) | research_design_assumption | observation.Sensors; simulation feedback queue |
| sensor.gyro_bias_diffusion | (rad/s)/sqrt(s) | research_design_assumption | observation.Sensors; simulation feedback queue |
| sensor.additional_accel_sample_std | m/s2 | research_design_assumption | observation.Sensors; simulation feedback queue |
| sensor.position_sample_std_m | m | research_design_assumption | observation.Sensors; simulation feedback queue |
| sensor.velocity_sample_std_m_s | m/s | research_design_assumption | observation.Sensors; simulation feedback queue |
| sensor.attitude_sample_std_rad | rad | research_design_assumption | observation.Sensors; simulation feedback queue |
| sensor.rpm_sample_std_rad_s | rad/s | research_design_assumption | observation.Sensors; simulation feedback queue |
| sensor.lever_m | m | research_design_assumption | observation.Sensors; simulation feedback queue |
| sensor.installation_rotation_xyzw | unit quaternion xyzw S->B | research_design_assumption | observation.Sensors; simulation feedback queue |
| sensor.feedback_delay_s | s | research_design_assumption | observation.Sensors; simulation feedback queue |
| wind.mean_m_s | m/s | research_design_assumption | observation.OUWind |
| wind.ou_stationary_std_m_s | m/s | research_design_assumption | observation.OUWind |
| wind.ou_correlation_time_s | s | research_design_assumption | observation.OUWind |
| wind.update_rate_hz | Hz | research_design_assumption | observation.OUWind |
| timing.control_hz | Hz | research_design_assumption | simulation event clock; dynamics.integrate |
| timing.integration_hz | Hz (maximum step inverse) | research_design_assumption | simulation event clock; dynamics.integrate |
| timing.export_hz | Hz | research_design_assumption | simulation event clock; dynamics.integrate |
| termination.position_norm_m | m | research_design_assumption | simulation.simulate |
| termination.speed_m_s | m/s | research_design_assumption | simulation.simulate |
| termination.body_rate_rad_s | rad/s | research_design_assumption | simulation.simulate |
| termination.minimum_altitude_m | m | research_design_assumption | simulation.simulate |
| acceptance.status | evidence status | research_design_assumption | validation.normal |
| acceptance.hover_final_position_error_m | m | research_design_assumption | validation.normal |
| acceptance.tracking_rmse_m | m | research_design_assumption | validation.normal |
| acceptance.maximum_normal_tilt_deg | deg | research_design_assumption | validation.normal |
| acceptance.integration_half_step_position_difference_m | m | research_design_assumption | validation.normal |
| acceptance.control_half_step_position_difference_m | m | research_design_assumption | validation.normal |
| acceptance.post_saturation_recovery_error_m | m | research_design_assumption | validation.normal |
| seed | integer | research_design_assumption | simulation.simulate |
| numerical.maximum_control_interval_s | s | explicit_numerical_method_assumption | control.update |
| numerical.actuator_step_to_tau_guard | 1 | explicit_numerical_method_assumption | dynamics.integrate |
| numerical.zero_force_tolerance | N | explicit_numerical_method_assumption | math3d.desired_attitude |
| numerical.quaternion_zero_norm_tolerance | 1 | explicit_numerical_method_assumption | math3d.quaternion |
| numerical.supported_num_rotors | count | explicit_numerical_method_assumption | configuration.physical; shape contracts |
| real_aircraft.mass | kg | unknown | not instantiated as real-aircraft model |
| real_aircraft.COM | m | unknown | not instantiated as real-aircraft model |
| real_aircraft.inertia | kg m2 | unknown | not instantiated as real-aircraft model |
| real_aircraft.kT | N/(rad/s)^2 | unknown | not instantiated as real-aircraft model |
| real_aircraft.kQ | N m/(rad/s)^2 | unknown | not instantiated as real-aircraft model |
| real_aircraft.motor_tau | s | unknown | not instantiated as real-aircraft model |
| real_aircraft.motor_ESC_prop_voltage | hardware identities | unknown | not instantiated as real-aircraft model |
| real_aircraft.sensor_pose | m / rad | unknown | not instantiated as real-aircraft model |
| real_aircraft.sensor_noise_and_bias_spectra | per channel | unknown | not instantiated as real-aircraft model |
| real_aircraft.aero_coefficients | per model | unknown | not instantiated as real-aircraft model |
| derived.mass | kg | derived_from_explicit_research_design | configuration.physical; dynamics; control |
| derived.inertia | kg m2 | derived_from_explicit_research_design | configuration.physical; dynamics; control |
| derived.rotor_positions | m | derived_from_explicit_research_design | configuration.physical; dynamics; control |
| derived.yaw_sign | 1 | derived_from_explicit_research_design | configuration.physical; dynamics; control |
| derived.kQ | N m/(rad/s)^2 | derived_from_explicit_research_design | configuration.physical; dynamics; control |
| legacy_candidate_inertia | per legacy definition | legacy_candidate_not_active | legacy_reference only; excluded from current core |
| legacy_candidate_thrust_coefficient | per legacy definition | legacy_candidate_not_active | legacy_reference only; excluded from current core |
| legacy_SE3_gains | per legacy definition | legacy_candidate_not_active | legacy_reference only; excluded from current core |
| legacy_kh | per legacy definition | legacy_candidate_not_active | legacy_reference only; excluded from current core |
| legacy_full_overrides | per legacy definition | legacy_candidate_not_active | legacy_reference only; excluded from current core |
