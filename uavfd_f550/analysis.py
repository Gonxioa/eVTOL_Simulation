"""Combination identification, short-horizon replay and observable vibration.

This stage does not assign mass/inertia/modal parameters to the real aircraft.
"""
from pathlib import Path
import re
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import welch, coherence
from .common import split, write_json, read_json, manifest, CONFIG
from .signals import segments
from .dynamics import from_euler, rotation, qmul


# Logged base commit AP_MotorsMatrix.cpp:793--799, Motor1--6 (not test order).
ANGLES = np.deg2rad([90, -90, -30, 150, 30, -150])
SPIN = np.array([1, -1, 1, -1, -1, 1])


def inputs(Omega):
    squared = Omega**2
    return np.column_stack([squared.sum(axis=1), squared@(-np.sin(ANGLES)),
                            squared@np.cos(ANGLES), squared@(-SPIN)])


def fit_regression(X, y, weights=None):
    weights = np.ones(len(y)) if weights is None else np.asarray(weights)
    if weights.shape != (len(y),) or np.any(weights <= 0):
        raise ValueError("Positive observation weights required")
    Xw, yw = X*np.sqrt(weights[:, None]), y*np.sqrt(weights)
    scale = np.linalg.norm(Xw, axis=0)
    if np.any(scale <= 0):
        raise ValueError("Unexcited regression column")
    normalized = Xw/scale
    beta, _, rank, singular = np.linalg.lstsq(normalized, yw, rcond=None)
    beta = beta/scale
    residual = y-X@beta
    return {"coefficients": beta.tolist(), "normalized_rank": int(rank),
            "normalized_condition": float(singular[0]/singular[-1]) if singular[-1] > 0 else None,
            "normalized_singular_values": singular.tolist(), "rmse": float(np.sqrt(np.average(residual**2, weights=weights))),
            "residual_mean": float(np.average(residual, weights=weights)),
            "status": "flight_identified_effective_combination_not_physical_parameter"}


def integral_examples(seg, config):
    t, y, u = seg["t"], seg["imu"][:, 3:], inputs(seg["Omega"])[:, 1:]
    n = int(round(config["integral_window_s"]*config["low_rate_hz"]))
    start = np.arange(0, len(t)-n, n)  # Nonoverlapping equations, still one flight.
    integral = cumulative_trapezoid(u, t, axis=0, initial=0)
    h = t[start+n]-t[start]
    return integral[start+n]-integral[start], y[start+n]-y[start], h


def calibrate(data_dir, out, config):
    names = [n for n in split()["calibration"] if n.startswith("NO_FAULT")]
    rows_z, rows_w = [], [[], [], []]
    provenance = {}
    independent_rotor_rows = []
    for name in names:
        segs = segments(data_dir, name, config, role="fit")
        provenance[name] = [s["selection"] for s in segs]
        Xz, yz = [], []
        axis_x, axis_y = [[], [], []], [[], [], []]
        for s in segs:
            u = inputs(s["Omega"])
            Xz.append(np.c_[u[:, 0], np.ones(len(u))])
            yz.append(-s["imu"][:, 2])
            iw, dw, h = integral_examples(s, config)
            for ax in range(3):
                axis_x[ax].append(np.c_[iw[:, ax], h])
                axis_y[ax].append(dw[:, ax])
            independent_rotor_rows.append(s["Omega"]**2)
        # Equal total weight per independent flight, regardless of duration.
        Xz, yz = np.concatenate(Xz), np.concatenate(yz)
        rows_z.append((Xz, yz, np.full(len(yz), 1/len(yz))))
        for ax in range(3):
            xx, yy = np.concatenate(axis_x[ax]), np.concatenate(axis_y[ax])
            rows_w[ax].append((xx, yy, np.full(len(yy), 1/len(yy))))
    z = fit_regression(*(np.concatenate([row[i] for row in rows_z]) for i in range(3)))
    w = [fit_regression(*(np.concatenate([row[i] for row in rows]) for i in range(3))) for rows in rows_w]
    design = np.concatenate(independent_rotor_rows)
    normalized = design/np.linalg.norm(design, axis=0)
    singular = np.linalg.svd(normalized, compute_uv=False)
    model = {"kind": "reduced_combination_model", "calibration_flights": names, "segments": provenance,
             "vertical": z, "angular": w, "motor_geometry_evidence": "logged_base_mixer_X_not_custom_wiring_verification",
             "gravity": {"value": 9.80665, "status": "standard_gravity_convention_not_site_measurement"},
             "six_independent_coefficients_diagnostic": {"normalized_singular_values": singular.tolist(),
                 "condition": float(singular[0]/singular[-1]), "relative_smallest": float(singular[-1]/singular[0]),
                 "interpretation": "numerical_rank_does_not_prove_physical_identifiability"},
             "limitations": ["unknown COM/sensor lever arm", "no aerodynamic drag/cross-axis rigid-body coupling",
                 "uniform rotor coefficients and regular equal-radius X geometry conditional",
                 "ESC sampling about 9Hz during logging; no precise sub-sample dynamics or phase",
                 "unknown delays; no offline time-shift fitted", "full closed-loop stability not established"],
             "physical_gain_signs_ok": bool(z["coefficients"][0] > 0 and all(a["coefficients"][0] > 0 for a in w)),
             "manifest": manifest(data_dir, names, config)}
    # Each independent fit is reported; their spread is not a population CI (n=3).
    model["per_flight_estimates"] = {name: {"vertical": fit_regression(*rows_z[i]),
                                   "angular": [fit_regression(*rows_w[ax][i]) for ax in range(3)]}
                                   for i, name in enumerate(names)}
    write_json(out / "combination_model.json", model)
    print("combination gains:", z["coefficients"], [a["coefficients"] for a in w], flush=True)
    return model


def metrics(error):
    x = np.asarray(error)
    return {"rmse": np.sqrt(np.mean(x*x, axis=0)).tolist(), "mean": np.mean(x, axis=0).tolist(),
            "p95_absolute": np.quantile(np.abs(x), .95, axis=0).tolist()}


def replay(data_dir, out, config):
    model = read_json(out / "combination_model.json")
    names = [n for n in split()["calibration"]+split()["development"] if n.startswith("NO_FAULT")]
    gains = np.array([a["coefficients"] for a in model["angular"]])
    rows, plotting = {}, None
    n = int(round(config["replay_horizon_s"]*config["low_rate_hz"]))
    for name in names:
        err_w, base_w, err_att, base_att, err_vel, base_vel = [], [], [], [], [], []
        err_acc, selections = [], []
        for s in segments(data_dir, name, config):
            t, imu, u = s["t"], s["imu"], inputs(s["Omega"])
            derivative_w = u[:, 1:]*gains[:, 0]+gains[:, 1]
            vertical = model["vertical"]["coefficients"]
            force_z = -(vertical[0]*u[:, 0]+vertical[1])
            accel_predict = np.column_stack([np.zeros(len(t)), np.zeros(len(t)), force_z])
            err_acc.append(accel_predict-imu[:, :3])
            selections.append(s["selection"])
            for start in range(0, len(t)-n, n):
                w = imu[start, 3:].copy()
                q = from_euler(*s["angles"][start])
                v = s["velocity"][start].copy()
                initial_R = rotation(q)
                for k in range(start, start+n):
                    dt = t[k+1]-t[k]
                    alpha = .5*(derivative_w[k]+derivative_w[k+1])
                    w_mid = w+alpha*dt/2
                    q_mid = q+dt/4*qmul(q, np.r_[0., w_mid])
                    force_mid = .5*(force_z[k]+force_z[k+1])
                    v += dt*(np.array([0, 0, 9.80665])+rotation(q_mid)@np.array([0, 0, force_mid]))
                    q += dt/2*qmul(q_mid, np.r_[0., w_mid])
                    q /= np.linalg.norm(q)
                    w += dt*alpha
                stop = start+n
                err_w.append(w-imu[stop, 3:])
                base_w.append(imu[start, 3:]-imu[stop, 3:])
                observed_R = rotation(from_euler(*s["angles"][stop]))
                from scipy.spatial.transform import Rotation
                err_att.append(Rotation.from_matrix(observed_R.T@rotation(q)).as_rotvec())
                base_att.append(Rotation.from_matrix(observed_R.T@initial_R).as_rotvec())
                err_vel.append(v-s["velocity"][stop])
                base_vel.append(s["velocity"][start]-s["velocity"][stop])
            if name == "NO_FAULT4.mat" and plotting is None:
                plotting = (t-t[0], imu[:, 2], force_z)
        if not err_w:
            raise ValueError("No short prediction examples")
        rows[name] = {"role": "development" if name in split()["development"] else "calibration",
                      "n_nonoverlap_predictions": len(err_w), "independent_flights": 1,
                      "specific_force_m_s2": metrics(np.concatenate(err_acc)),
                      "angular_rate_rad_s": metrics(err_w), "persistence_angular_rate_rad_s": metrics(base_w),
                      "attitude_rotation_error_rad": metrics(err_att), "persistence_attitude_error_rad": metrics(base_att),
                      "estimated_velocity_error_m_s": metrics(err_vel), "persistence_velocity_error_m_s": metrics(base_vel),
                      "segments": selections, "rotation_metric": "rotation_vector_log(R_observed.T @ R_predicted)",
                      "velocity_reference": "XKF1 estimated_not_ground_truth"}
    write_json(out / "normal_replay.json", {"results": rows, "horizon_s": config["replay_horizon_s"],
               "status": "conditional_combination_short_horizon_evaluation_not_closed_loop_validation",
               "manifest": manifest(data_dir, names, config)})
    if plotting:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(plotting[0], plotting[1], label="logged 3-IMU mean / analysis LP")
        ax.plot(plotting[0], plotting[2], label="RPM-driven combination prediction")
        ax.set(xlabel="Time in first valid segment [s]", ylabel="FRD specific force z [m/s^2]", title="NO_FAULT4: development, not final test")
        ax.legend(); fig.tight_layout(); fig.savefig(out / "normal_replay.png", dpi=160); plt.close(fig)
    print("normal development:", rows["NO_FAULT4.mat"]["angular_rate_rad_s"], flush=True)
    return rows


def spectrum_rows(data_dir, name, config, role):
    fs, window = config["vibration_rate_hz"], config["welch_segment_s"]
    n = int(round(fs*window))
    result, spectra = [], []
    for seg in segments(data_dir, name, config, vibration=True, role=role):
        # No overlapping windows used as independent repeated flights.
        for i in range(0, len(seg["t"])-n+1, n):
            sl = slice(i, i+n)
            acc = seg["imu"][sl, :3]
            Omega = np.mean(seg["Omega"][sl], axis=0)
            f, P = welch(acc, fs=fs, nperseg=n, noverlap=0, axis=0)
            energy = []
            for center in Omega/(2*np.pi):
                band = (f >= center-config["order_band_halfwidth_hz"]) & (f <= center+config["order_band_halfwidth_hz"])
                energy.append(np.trapezoid(P[band], f[band], axis=0))
            broad = (f >= config["vibration_highpass_hz"]) & (f <= config["vibration_lowpass_hz"])
            totals = np.trapezoid(P[broad], f[broad], axis=0)
            result.append({"name": name, "t_s": float(seg["t"][i]), "Omega": Omega.tolist(),
                           "voltage": float(np.mean(seg["voltage"][sl])), "rotor_1x_band_energy": np.array(energy).tolist(),
                           "axis_band_energy": totals.tolist(), "axis_ratio": (totals/totals.sum()).tolist(),
                           "peak_frequency_hz": f[np.argmax(P[broad].sum(axis=1))+np.flatnonzero(broad)[0]].item(),
                           "native_log_rate_regularized_hz": seg["selection"]["native_log_rate_regularized_hz"]})
            spectra.append((f, P))
    if not result:
        raise ValueError(f"No eligible vibration windows: {name}")
    return result, spectra


def vibration(data_dir, out, config):
    groups = split()
    rows, psd = {}, {}
    for name in groups["calibration"]+groups["development"]:
        role = "fit" if name in groups["calibration"] else "development"
        rows[name], spectra = spectrum_rows(data_dir, name, config, role)
        psd[name] = (spectra[0][0], np.mean([p for _, p in spectra], axis=0))
        print(f"vibration {name}: {len(rows[name])} disjoint windows (one flight)", flush=True)
    healthy = [n for n in groups["calibration"] if n.startswith("NO_FAULT")]
    # Median across windows within a flight, then median across independent flights.
    # Effective 1x energy / Omega^4, includes mass imbalance, transfer and firmware.
    healthy_gains = []
    for name in healthy:
        e = np.array([r["rotor_1x_band_energy"] for r in rows[name]])
        O = np.array([r["Omega"] for r in rows[name]])
        healthy_gains.append(np.median(e/O[:, :, None]**4, axis=0))
    baseline = np.median(healthy_gains, axis=0)
    fault_gains = {}
    for name in groups["calibration"]:
        if name.startswith("FAULT"):
            motor = int(re.search(r"M(\d)", name).group(1))-1
            gains = [np.array(r["rotor_1x_band_energy"])[motor]/r["Omega"][motor]**4-baseline[motor] for r in rows[name]]
            raw = np.median(gains, axis=0)
            fault_gains[name] = {"raw_increment": raw.tolist(), "nonnegative_increment": np.maximum(raw, 0).tolist(),
                                  "motor": motor+1, "status": "effective_energy_combination_not_mu_or_modal_gain"}
    shared_increment = np.mean([v["nonnegative_increment"] for v in fault_gains.values()], axis=0)
    healthy_rows = sum((rows[name] for name in healthy), [])
    speed_support = [min(min(r["Omega"]) for r in healthy_rows), max(max(r["Omega"]) for r in healthy_rows)]
    volt_support = [min(r["voltage"] for r in healthy_rows), max(r["voltage"] for r in healthy_rows)]
    reports = {}
    for name, rr in rows.items():
        # Normal: report the Motor1-centered reference band. Fault: the labeled motor band.
        # A centered band is an observation interval, not isolated rotor energy.
        motor = int(re.search(r"M(\d)", name).group(1))-1 if name.startswith("FAULT") else 0
        gain = baseline[motor] + (shared_increment if name.startswith("FAULT") else 0)
        observed = np.array([r["rotor_1x_band_energy"][motor] for r in rr])
        predicted = np.array([gain*r["Omega"][motor]**4 for r in rr])
        mask = np.array([speed_support[0] <= r["Omega"][motor] <= speed_support[1] and
                         volt_support[0] <= r["voltage"] <= volt_support[1] for r in rr])
        reports[name] = {"role": "development" if name in groups["development"] else "calibration",
                         "windows_n": len(rr), "independent_flights": 1, "within_speed_voltage_support_n": int(mask.sum()),
                         "rotor_1x_band_energy_observed_mean": observed.mean(axis=0).tolist(),
                         "rotor_1x_band_energy_predicted_mean": predicted.mean(axis=0).tolist(),
                         "prediction_error": metrics(predicted-observed),
                         "common_support_error": metrics((predicted-observed)[mask]) if mask.any() else None,
                         "median_peak_hz": float(np.median([r["peak_frequency_hz"] for r in rr])),
                         "rotor_frequency_hz_range": [float(min(min(r["Omega"]) for r in rr)/(2*np.pi)),
                                                       float(max(max(r["Omega"]) for r in rr)/(2*np.pi))],
                         "mean_axis_energy_ratio": np.mean([r["axis_ratio"] for r in rr], axis=0).tolist()}
    # Axis coherence diagnostic from a valid segment; no cross-axis phases fitted.
    s = segments(data_dir, "FAULT_M3_5.mat", config, vibration=True)[0]
    fc, coh = coherence(s["imu"][:, 0], s["imu"][:, 1], fs=config["vibration_rate_hz"], nperseg=int(config["vibration_rate_hz"]*2))
    f, mean_psd = psd["FAULT_M3_5.mat"]
    write_json(out / "vibration_model.json", {"healthy_per_motor_band_gain": baseline.tolist(),
               "healthy_flight_gain_spread": healthy_gains, "fault_flight_gains": fault_gains,
               "shared_damage_increment": shared_increment.tolist(), "speed_support_rad_s": speed_support,
               "voltage_support_V": volt_support, "calibration_flights": groups["calibration"],
               "status": "effective_observation_band_model_not_structural_modal_calibration",
               "limitations": ["1x bands from multiple rotors can overlap; cannot attribute all band energy to damaged rotor",
                   "unknown log anti-alias/filter path; not raw sensor PSD", "n=2 fault calibration flights insufficient for motor-position transfer",
                   "mass imbalance and structural/log gain not separable", "speed-voltage overlap is insufficient to ensure comparable maneuvers",
                   "2s windows correlated within flights; no classifier accuracy or independent window CI"],
               "manifest": manifest(data_dir, groups["calibration"]+groups["development"], config)})
    write_json(out / "vibration_validation.json", {"results": reports, "windows": rows,
               "development_xy_coherence": {"frequency_hz": fc.tolist(), "coherence": coh.tolist()},
               "development_PSD": {"frequency_hz": f.tolist(), "PSD_m2_s4_per_hz": mean_psd.tolist()},
               "status": "development_metrics_no_fidelity_pass_claim"})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ["NO_FAULT1.mat", "NO_FAULT4.mat", "FAULT_M1_5.mat", "FAULT_M3_5.mat"]:
        f, p = psd[name]
        axes[0].semilogy(f, p.sum(axis=1), label=name.removesuffix(".mat"))
    axes[0].set(xlim=(10, 120), xlabel="Frequency [Hz]", ylabel="Sum acceleration PSD [(m/s^2)^2/Hz]")
    axes[0].legend(fontsize=7)
    axes[1].plot(fc, coh)
    axes[1].set(xlim=(10, 120), ylim=(0, 1), xlabel="Frequency [Hz]", ylabel="x-y coherence", title="FAULT_M3_5 development")
    fig.tight_layout(); fig.savefig(out / "vibration_validation.png", dpi=160); plt.close(fig)
    return reports


def run(command, data_dir, out, config):
    if command == "calibrate":
        calibrate(data_dir, out, config)
    elif command == "replay":
        replay(data_dir, out, config)
    elif command == "vibration":
        vibration(data_dir, out, config)
    elif command == "demo":
        from .verification import demo
        demo(out)
    elif command == "validate":
        from .verification import analytical
        analytical(out)
    else:
        raise ValueError(command)
