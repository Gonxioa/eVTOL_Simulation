"""Evidence descriptors for the implemented equations and explicit method choices."""

ANALYSIS_FIELDS = {
    "low_rate_hz": ("Hz", "Low-frequency analysis observation rate"),
    "lowpass_hz": ("Hz", "Offline low-frequency Butterworth cutoff"),
    "filter_order": ("1", "Butterworth order; forward-backward filtering is acausal"),
    "rpm_interpolation": ("method", "ESC telemetry interpolation; not an increase in measurement rate"),
    "edge_trim_s": ("s", "Trim at both segment edges after filtering"),
    "max_rpm_gap_s": ("s", "Maximum supported consecutive ESC telemetry gap"),
    "segment_min_s": ("s", "Minimum usable segment duration after trimming"),
    "integral_window_s": ("s", "Nonoverlapping angular identification interval"),
    "replay_horizon_s": ("s", "Prediction interval with measured/estimated initial conditions"),
    "vibration_rate_hz": ("Hz", "Vibration analysis rate after anti-alias filtering"),
    "vibration_highpass_hz": ("Hz", "Offline vibration lower cutoff"),
    "vibration_lowpass_hz": ("Hz", "Highest retained vibration band cutoff"),
    "welch_segment_s": ("s", "Disjoint spectrum window length; windows remain correlated within flights"),
    "order_band_halfwidth_hz": ("Hz", "Rotational fundamental centered observation band halfwidth"),
    "seed": ("integer", "Declared local random seed; regression and PSD are deterministic"),
}

DEMO_UNITS = {
    "gravity_m_s2": "m/s2", "mass_kg": "kg", "inertia_kg_m2": "kg m2",
    "arm_radius_m": "m", "kT_N_per_rad_s_squared": "N/(rad/s)^2",
    "kQ_Nm_per_rad_s_squared": "N m/(rad/s)^2", "modal_frequency_hz": "Hz",
    "modal_damping_ratio": "1", "imbalance_kg_m": "kg m", "rotor_frequency_hz": "Hz",
    "actuator_tau_s": "s", "sensor_lever_m": "m", "seed": "integer",
}


def equations():
    # Source identifiers resolve to sources.json; derivations and signs to MODEL_SPEC.
    rows = [
        ("E1", "pdot=v; vdot=g_world+R F_body/m", "m/s; m/s2", ["S03", "S05"], "dynamics.Body.derivative", "freefall_acceleration; known_external_force; hover_collective", "Constant rigid-body mass; no drag or wind"),
        ("E2", "J omega_dot=tau-omega cross(J omega)", "N m", ["S03", "S05"], "dynamics.Body.derivative", "constant_principal_torque_rate; RK4_final_error", "COM rigid-body inertia; rotor angular momentum omitted"),
        ("E3", "qdot=0.5 q HamiltonProduct [0,omega_body]", "1/s", ["S05"], "dynamics.qmul; dynamics.rk4", "constant_principal_torque_attitude; rotation_orthogonality", "Hamilton wxyz, body to world; step normalization"),
        ("E4", "T=kT Omega^2; Q=kQ Omega^2; tau=r cross F-s Q e3", "N; N m", ["S03", "S05"], "dynamics.Body.wrench", "hover_torque; per_motor_wrench_sign; RPM_conversion_60_equals_2pi", "Coplanar fixed pitch; scales do not map to 5/10 labels"),
        ("E5", "f=R.T(a_COM-g)+alpha cross r+omega cross(omega cross r)+a_elastic", "m/s2", [], "dynamics.specific_force; dynamics.imu_observe", "static_IMU_sign; freefall_specific_force; IMU_centripetal_lever", "Own rigid-point acceleration derivation; sensor fixed in body"),
        ("E6", "RK4 with time-varying stage inputs; R_new=S_world R_old S_body", "state units; 1", [], "dynamics.rk4; dynamics.old_enu_flu_to_ned_frd", "RK4_convergence_ratio; frame_adapter", "Own numerical implementation; coordinate adapter is not aircraft equivalence"),
        ("E7", "theta_dot=s Omega; F_imb=mu(Omega^2 e_r-s Omega_dot e_t)", "rad/s; N", ["S05"], "vibration.imbalance_force; vibration.integrate_phase", "imbalance_Omega_squared; imbalance_tangential_reaction; phase_spin_direction", "Own support-reaction derivation; body-motion/Coriolis interactions omitted"),
        ("E8", "qdd=-wn^2 q-2 zeta wn qd+input_gain F; a_elastic=output_shape qdd", "m/s2", [], "vibration.LocalModes", "modal_ringdown_exact_ZOH; demo internal-rate comparison", "Assumed linear mass-normalized local elastic modes; not recovered aircraft modes"),
        ("E9", "Lowpass before sampling; retained cutoff<0.45 min(fs_internal,fs_observation)", "Hz", [], "vibration.checked_sample; signals.segments", "anti_alias_230_to_70_energy_ratio; Nyquist_violation_rejected", "Nyquist requirement plus explicitly selected guard margin; acausal offline observation"),
        ("E10", "-fz=a_z sum(Omega^2)+b_z", "m/s2", ["S01", "S03", "S04"], "analysis.calibrate; analysis.replay", "NO_FAULT4 specific_force_m_s2; per_flight_estimates", "E1/E4 reduced fit; nuisance intercept; no physical kT/m identification claim"),
        ("E11", "Delta omega_j=gamma_j integral(U_j dt)+b_j h", "rad/s", ["S01", "S03", "S04"], "analysis.integral_examples; analysis.calibrate", "NO_FAULT4 angular_rate_rad_s versus persistence", "Conditional base mixer; coupling, aerodynamic damping and delays omitted"),
        ("E12", "One-second RPM-driven rate/attitude/velocity prediction initialized per interval", "rad/s; rad; m/s", ["S01"], "analysis.replay", "normal_replay.json; attitude and horizontal velocity worse than persistence", "Effective normalized dynamics; estimated ATT/XKF reference is not external truth"),
        ("E13", "Rotational band energy E_j approximately G_j Omega^4", "(m/s2)^2", ["S01", "S05", "S06"], "analysis.spectrum_rows; analysis.vibration", "NO_FAULT4/FAULT_M3_5 vibration_validation.json", "E7 amplitude-square derivation; overlapping bands and unknown transfer/filter; unvalidated cross-motor damage gain"),
        ("E14", "Omega(t+dt)=steady+(Omega(t)-steady) exp(-dt/tau)", "rad/s", ["S03"], "dynamics.first_order_step", "actuator_exact_step", "Assumed first-order actuator; true PWM/voltage map, delay and tau unknown"),
    ]
    return [{"id": r[0], "relation": r[1], "units": r[2], "source_ids": r[3],
             "source_location": "sources.json/SOURCE_REGISTER; MODEL_SPEC section " + r[0],
             "evidence_status": "own_derivation_or_explicit_reduced_assumption; source context only",
             "coordinate": "NED world / FRD body; Hamilton wxyz body-to-world",
             "implementation": r[4], "verification": r[5], "limits": r[6]} for r in rows]
