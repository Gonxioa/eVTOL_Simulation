"""Predeclared analytical verification; all test parameters are explicitly demo."""
import numpy as np
from .common import CONFIG, read_json, write_json, digest
from .dynamics import Body, from_euler, rotation, rk4, specific_force, first_order_step, rpm_to_rad_s, old_enu_flu_to_ned_frd
from .vibration import LocalModes, imbalance_force, integrate_phase, checked_sample
from .analysis import ANGLES, SPIN


def demo_body():
    c = read_json(CONFIG / "demo.json")
    positions = c["arm_radius_m"]*np.c_[np.cos(ANGLES), np.sin(ANGLES), np.zeros(6)]
    return Body(c["mass_kg"], np.diag(c["inertia_kg_m2"]), positions, SPIN,
                np.full(6, c["kT_N_per_rad_s_squared"]), np.full(6, c["kQ_Nm_per_rad_s_squared"]), c["gravity_m_s2"], "demo")


def state0():
    return np.r_[np.zeros(6), [1., 0., 0., 0.], np.zeros(3)]


def analytical(out):
    b, zero = demo_body(), np.zeros(6)
    x = state0()
    tests = []
    def check(name, error, tolerance, units):
        entry = {"name": name, "error": float(error), "tolerance_predeclared": tolerance, "units": units,
                 "passed": bool(error <= tolerance), "parameter_status": "demo"}
        tests.append(entry)
    dx = b.derivative(x, zero)
    check("freefall_acceleration", np.linalg.norm(dx[3:6]-[0, 0, b.gravity]), 1e-12, "m/s2")
    check("freefall_specific_force", np.linalg.norm(specific_force(x, dx, b.gravity, [0, 0, 0])), 1e-12, "m/s2")
    dx = b.derivative(x, zero, [0, 0, -b.mass*b.gravity])
    check("static_support", np.linalg.norm(dx), 1e-12, "state_derivative")
    check("static_IMU_sign", abs(specific_force(x, dx, b.gravity, [0, 0, 0])[2]+b.gravity), 1e-12, "m/s2")
    dx = b.derivative(x, zero, [b.mass, 2*b.mass, -b.mass*b.gravity])
    check("known_external_force", np.linalg.norm(dx[3:6]-[1, 2, 0]), 1e-12, "m/s2")
    hover = np.full(6, np.sqrt(b.mass*b.gravity/(6*b.kT[0])))
    force, torque = b.wrench(hover)
    check("hover_collective", abs(force[2]+b.mass*b.gravity), 1e-12, "N")
    check("hover_torque", np.linalg.norm(torque), 1e-12, "Nm")
    # Independent per-motor cross-product sign; yaw is negative of rotor spin.
    error = []
    for i in range(6):
        O = np.zeros(6); O[i] = 100
        f, tau = b.wrench(O)
        expected = np.cross(b.rotor_positions[i], np.array([0., 0., -b.kT[i]*100**2]))
        expected[2] -= b.spin_sign[i]*b.kQ[i]*100**2
        error.append(np.linalg.norm(tau-expected))
    check("per_motor_wrench_sign", max(error), 1e-12, "Nm")
    x = state0(); O = lambda t: zero
    extra = lambda t: (np.array([0, 0, -b.mass*b.gravity]), np.array([.004, 0, 0]))
    for k in range(100):
        x = rk4(b, x, O, k*.01, .01, extra)
    check("constant_principal_torque_rate", abs(x[10]-.004/b.inertia[0, 0]), 1e-12, "rad/s")
    expected_R = rotation(from_euler(.5*.004/b.inertia[0, 0], 0, 0))
    check("constant_principal_torque_attitude", np.linalg.norm(rotation(x[6:10])-expected_R), 1e-8, "rotation_matrix")
    # Nontrivial gyroscopic coupling converges against an independent solver.
    from scipy.integrate import solve_ivp
    initial = state0(); initial[10:] = [.5, .4, .3]
    ref = solve_ivp(lambda t, x: b.derivative(x, hover), [0, 1], initial, rtol=1e-11, atol=1e-13).y[:, -1]
    errors = []
    for dt in (.1, .05, .025):
        x = initial.copy()
        for k in range(round(1/dt)):
            x = rk4(b, x, lambda t: hover, k*dt, dt)
        errors.append(np.linalg.norm(x[10:]-ref[10:]))
    check("RK4_final_error", errors[-1], 1e-8, "rad/s")
    check("RK4_convergence_ratio", 0. if errors[0] > 8*errors[1] > 64*errors[2] else 1., 0., "boolean_error")
    check("RPM_conversion_60_equals_2pi", abs(rpm_to_rad_s(60)-2*np.pi), 1e-12, "rad/s")
    R = rotation(from_euler(.2, .3, -.7))
    check("rotation_orthogonality", np.linalg.norm(R.T@R-np.eye(3)), 1e-12, "dimensionless")
    x = state0(); x[10:] = [0, 0, 2]; dx = np.zeros(13)
    lever_acc = specific_force(x, dx, b.gravity, [.1, 0, 0])
    check("IMU_centripetal_lever", np.linalg.norm(lever_acc-[-.4, 0, -b.gravity]), 1e-12, "m/s2")
    check("actuator_exact_step", abs(first_order_step(0., 100., .1, .1)-100*(1-np.exp(-1))), 1e-12, "rad/s")
    check("imbalance_Omega_squared", np.linalg.norm(imbalance_force(1e-5, 200, 0, 1)-[.4, 0, 0]), 1e-12, "N")
    check("imbalance_tangential_reaction", np.linalg.norm(imbalance_force(1e-5, 200, 0, 1, 10)-[.4, -.0001, 0]), 1e-12, "N")
    t = np.arange(0, .5, .001)
    phase = integrate_phase(t, np.full(len(t), 100.), -1, .2)
    check("phase_spin_direction", np.max(np.abs(phase-(.2-100*t))), 1e-10, "rad")
    modes = LocalModes([20], [.05], [[1]], [[1], [0], [0]])
    t = np.arange(0, 1, .001)
    states, _ = modes.simulate(t, np.zeros((len(t), 1)), initial=[1., 0.])
    wn, zeta = 2*np.pi*20, .05
    wd = wn*np.sqrt(1-zeta*zeta)
    exact = np.exp(-zeta*wn*t)*(np.cos(wd*t)+zeta*wn/wd*np.sin(wd*t))
    check("modal_ringdown_exact_ZOH", np.max(np.abs(states[:, 0]-exact)), 1e-11, "m")
    t = np.arange(0, 3, 1/2000)
    signal = np.column_stack([np.sin(2*np.pi*40*t)+np.sin(2*np.pi*230*t)]*3)
    tobs = np.arange(.1, 2.9, 1/300)
    sample = checked_sample(t, signal, tobs, 120)
    from scipy.signal import welch
    f, p = welch(sample[:, 0], fs=300, nperseg=600)
    check("anti_alias_230_to_70_energy_ratio", p[np.argmin(abs(f-70))]/p[np.argmin(abs(f-40))], .01, "power_ratio")
    try:
        checked_sample(t, signal, tobs, 160)
    except ValueError:
        rejected = True
    else:
        rejected = False
    check("Nyquist_violation_rejected", 0 if rejected else 1, 0, "boolean_error")
    p, v, Rt, w = old_enu_flu_to_ned_frd(np.array([1, 2, 3]), np.zeros(3), np.eye(3), np.array([1, 2, 3]))
    check("frame_adapter", np.linalg.norm(p-[2, 1, -3])+np.linalg.norm(w-[1, -2, -3])+abs(np.linalg.det(Rt)-1), 1e-12, "coordinate_identity")
    report = {"status": "analytical_demo_only", "checks": tests, "all_passed": all(t["passed"] for t in tests),
              "integration_errors": errors, "configuration_sha256": digest(CONFIG / "demo.json")}
    write_json(out / "analytical_validation.json", report)
    print(f"analytical: {sum(t['passed'] for t in tests)}/{len(tests)} passed", flush=True)
    if not report["all_passed"]:
        raise AssertionError([t for t in tests if not t["passed"]])
    return report


def demo(out):
    c = read_json(CONFIG / "demo.json")
    t = np.arange(0, 3, 1/2000)
    Omega = np.full(len(t), 2*np.pi*c["rotor_frequency_hz"])
    phase = integrate_phase(t, Omega, 1, 0.)
    force = np.array([imbalance_force(c["imbalance_kg_m"], O, ph, 1) for O, ph in zip(Omega, phase)])
    modes = LocalModes(c["modal_frequency_hz"], c["modal_damping_ratio"], [[1, 0, 0], [0, 1, 0]], [[1, 0], [0, 1], [.2, .3]])
    _, elastic = modes.simulate(t, force)
    observation_t = np.arange(.1, 2.9, 1/300)
    sampled = checked_sample(t, elastic, observation_t, 120)
    np.savez(out / "demo_vibration.npz", t=observation_t, accel_elastic=sampled, status="demo_not_UAVFD_calibrated")
    # Modal simulation step convergence with same continuous rotating force.
    errors = []
    for rate in (1000, 2000, 4000):
        ti = np.arange(0, 3, 1/rate)
        ph = 2*np.pi*c["rotor_frequency_hz"]*ti
        fi = np.c_[np.cos(ph), np.sin(ph), np.zeros(len(ph))]*c["imbalance_kg_m"]*(2*np.pi*c["rotor_frequency_hz"])**2
        _, ai = modes.simulate(ti, fi)
        samples = checked_sample(ti, ai, observation_t, 120)
        errors.append(samples)
    diffs = [float(np.sqrt(np.mean((errors[i]-errors[i+1])**2))) for i in range(2)]
    write_json(out / "demo_vibration.json", {"config": c, "status": "demo_not_UAVFD_calibrated",
               "modal_internal_rate_step_comparison_rms_m_s2": diffs, "rates_hz": [1000, 2000, 4000],
               "comparison": "1000-vs-2000,2000-vs-4000; exact modal ZOH, varying-force discretization remains"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(observation_t, sampled)
    ax.set(xlim=(2, 2.1), xlabel="Time [s]", ylabel="Local elastic acceleration [m/s^2]", title="DEMO parameters only: rotating imbalance + local modes")
    fig.tight_layout(); fig.savefig(out / "demo_vibration.png", dpi=160); plt.close(fig)
    print("demo vibration convergence RMS:", diffs, flush=True)
