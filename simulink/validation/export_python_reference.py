"""Generate PV fixtures using frozen Python dynamics/controller with an explicit input adapter.

Only validation uses Python. The delivered SLX does not call Python.
The adapter sets the legacy acceleration/yaw-rate inputs to zero and holds yaw
at zero. Frozen Python source files are not modified. Legacy fixtures remain.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np
from scipy.io import savemat

BASELINE = "a6b9f1dd3d839416e469755490b8ff7733acd4ee"
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
OUT = ROOT / "output" / "validation"
SNAPSHOT = OUT / "python_baseline"
REFERENCE_MODE = "position_velocity_fixed_heading"


class PositionVelocityTask:
    """Adapt only the task input; preserve the frozen baseline implementation."""
    def __init__(self, provider):
        self.provider = provider
        self.name = provider.name
        self.spec = dict(provider.spec, reference_mode=REFERENCE_MODE)

    def reset(self):
        self.provider.reset()

    def at(self, t):
        values = self.provider.at(t)
        return dict(values, x_ddot=np.zeros(3), yaw=0., yaw_dot=0.)


def snapshot():
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE], cwd=REPO, text=True
    ).splitlines()
    selected = [p for p in paths if (
        p.startswith("hex_research/") and (p.endswith(".py") or p.endswith(".json"))
    ) or p in ("control_allocation.py", "provenance.py", "scenario_config.py")]
    hashes = {}
    for name in selected:
        raw = subprocess.check_output(["git", "show", f"{BASELINE}:{name}"], cwd=REPO)
        target = SNAPSHOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    sys.path.insert(0, str(SNAPSHOT))
    return hashes


def inputs(cfg, seed, duration, noise, wind):
    from hex_research.observation import Sensors, OUWind
    streams = np.random.SeedSequence(seed).spawn(2)
    sensor = Sensors(cfg, streams[0]); wind_process = OUWind(cfg, streams[1], wind)
    c = cfg["sensor"]; dt = 1 / c["rate_hz"]
    values = np.zeros((round(duration/dt)+2, 24))
    ba = np.zeros(3); bg = np.zeros(3)
    for k in range(len(values)):
        values[k, :3] = wind_process.update(k*dt)
        if not noise:
            continue
        ba += c["accel_bias_diffusion"]*np.sqrt(dt)*sensor.rng[0].normal(size=3)
        bg += c["gyro_bias_diffusion"]*np.sqrt(dt)*sensor.rng[1].normal(size=3)
        values[k, 3:6] = ba+c["accel_noise_density"]/np.sqrt(dt)*sensor.rng[2].normal(size=3)+c["additional_accel_sample_std"]*sensor.rng[6].normal(size=3)
        values[k, 6:9] = bg+c["gyro_noise_density"]/np.sqrt(dt)*sensor.rng[3].normal(size=3)
        values[k, 9:12] = c["position_sample_std_m"]*sensor.rng[4].normal(size=3)
        values[k, 12:15] = c["velocity_sample_std_m_s"]*sensor.rng[4].normal(size=3)
        values[k, 15:18] = c["attitude_sample_std_rad"]*sensor.rng[4].normal(size=3)
        values[k, 18:24] = c["rpm_sample_std_rad_s"]*sensor.rng[5].normal(size=6)
    return values


def main():
    hashes = snapshot()
    from hex_research.configuration import load
    from hex_research.simulation import simulate, FaultSchedule
    from hex_research.commands import ParameterizedTask, LegacyTask
    cfg = load(); cases = {}; summaries = []
    plans = [(f"normal_{family}", i, 0, 1., 3., np.inf, 8., False) for i, family in enumerate(("hover", "line", "circle", "vertical"))]
    plans += [(f"partial_rotor_{m}", 0, m, .8, 2., np.inf, 5., False) for m in range(1, 7)]
    plans += [
        ("severe_rotor_1", 0, 1, .2, 2., np.inf, 5., False),
        ("full_rotor_6", 0, 6, 0., 2., np.inf, 5., False),
        ("partial_recovery", 1, 1, .8, 2., 4., 8., False),
        ("off_grid_fault", 0, 1, 0., .0150000000001, .035, .06, False),
        ("saturation_recovery", 6, 0, 1., 3., np.inf, 8., False),
        ("noisy_delayed_wind", 2, 2, .8, 2., np.inf, 3., True),
    ]
    for name, family, motor, factor, start, end, duration, stochastic in plans:
        seed = 7 if stochastic else cfg["seed"]
        delay = .03 if stochastic else cfg["sensor"]["feedback_delay_s"]
        cfg["sensor"]["feedback_delay_s"] = delay
        if family <= 3:
            spec = {"family": ("hover", "line", "circle", "vertical")[family],
                    "duration_s": duration, "motion_start_s": min(1., duration/2),
                    "motion_end_s": min(7., duration), "origin_m": [0., 0., 2.5],
                    "heading_rad": 0., "yaw_start_rad": 0., "yaw_change_rad": 0.,
                    "displacement_m": 1.2, "radius_m": .3, "turn_rad": 2*np.pi,
                    "limits": {"speed_m_s": 100., "acceleration_m_s2": 10000., "yaw_rate_rad_s": 100.,
                               "horizontal_radius_m": 100., "altitude_m": [-100., 100.]}}
            provider = ParameterizedTask(spec)
        else:
            provider = LegacyTask("saturation")
        provider = PositionVelocityTask(provider)
        schedule = FaultSchedule(motor-1, factor, start if motor else None, end if np.isfinite(end) else None)
        run = simulate(cfg, duration, schedule, seed=seed, feedback="observed_feedback" if stochastic else "ideal_feedback",
                       noise=stochastic, wind=stochastic, task_provider=provider)
        data = []
        for row in run["rows"]:
            c = row["control"]; obs = row["measurement"]; r = row["reference"]
            observation = np.r_[obs["x"], obs["v"], obs["q"], obs["w"], obs["rotor_measurement"]]
            audit = np.r_[c["desired_wrench"], c["nominal_allocated_wrench"], c["nominal_residual_wrench"],
                          c["velocity_integral"], c["rate_integral"], c["position_error"], c["velocity_error"],
                          c["rate_error"], c["q_desired"], c["rate_reference"],
                          c["allocator_success"], c["allocator_bound_count"], c["allocator_feasible"],
                          c["velocity_integral_applied"], c["rate_integral_applied"]]
            data.append(np.r_[row["time"], row["state"], r["x"], r["x_dot"],
                              c["cmd_motor_speeds"], obs["accel_sensor"], obs["gyro_sensor"], observation,
                              row["wind"], row["effectiveness"], audit])
        cases[name] = dict(task_family=float(family), fault_motor=float(motor), fault_factor=factor,
                           fault_start=start, fault_end=end, duration=duration, observed_feedback=stochastic,
                           noise_enabled=stochastic, wind_enabled=stochastic, feedback_delay=delay, seed=float(seed),
                           motion_start=min(1., min(7., duration)/2), motion_end=min(7., duration),
                           exogenous=inputs(cfg, seed, duration, stochastic, stochastic), data=np.asarray(data),
                           termination=run["termination"])
        summaries.append(dict(name=name, samples=len(data), termination=run["termination"], actual_duration_s=run["actual_duration_s"]))
        print(name, len(data), run["termination"], flush=True)
    fixture = OUT / "python_pv_reference.mat"
    savemat(fixture, {"cases": cases, "baseline": BASELINE, "reference_mode": REFERENCE_MODE}, do_compression=True)
    receipt = {"baseline": BASELINE, "reference_mode": REFERENCE_MODE,
               "input_adapter": "zero acceleration feedforward; fixed yaw=0; zero yaw-rate feedforward",
               "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "source_sha256": hashes, "cases": summaries,
               "reference_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest()}
    (OUT / "python_pv_reference_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
