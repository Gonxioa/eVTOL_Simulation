"""Check current tuned SLX traces against the frozen Python equations.

The frozen controller receives the current explicit gains and a PV adapter.
Original Python sources and historical migration fixtures are not changed.
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "baseline_tuning"
SNAPSHOT = ROOT / "output" / "validation" / "python_baseline"
sys.path.insert(0, str(SNAPSHOT))

from hex_research.commands import ParameterizedTask
from hex_research.simulation import simulate


class PVTask:
    def __init__(self, provider, yaw):
        self.provider = provider
        self.name = provider.name
        self.spec = dict(provider.spec, reference_mode="position_velocity_fixed_heading")
        self.yaw = yaw

    def reset(self):
        self.provider.reset()

    def at(self, t):
        return dict(self.provider.at(t), x_ddot=np.zeros(3), yaw=self.yaw, yaw_dot=0.)


def main():
    cfg = json.loads((ROOT / "configs" / "research.json").read_text(encoding="utf-8"))
    cases = []
    for family in ("hover", "line", "circle", "vertical"):
        spec = dict(family=family, duration_s=8., motion_start_s=1., motion_end_s=7.,
                    origin_m=[0., 0., 2.5], heading_rad=0., yaw_start_rad=0.,
                    yaw_change_rad=0., displacement_m=1.2, radius_m=.3, turn_rad=2*np.pi,
                    limits=dict(speed_m_s=100., acceleration_m_s2=10000., yaw_rate_rad_s=100.,
                                horizontal_radius_m=100., altitude_m=[-100., 100.]))
        provider = PVTask(ParameterizedTask(spec), cfg["control"]["yaw_hold_rad"])
        run = simulate(cfg, duration=8., feedback="ideal_feedback", noise=False,
                       wind=False, task_provider=provider)
        expected = np.asarray([np.r_[row["time"], row["state"], row["reference"]["x"],
                                    row["reference"]["x_dot"], row["control"]["cmd_motor_speeds"]]
                               for row in run["rows"]])
        actual = np.loadtxt(OUT / f"{family}.csv", delimiter=",")
        assert actual.shape == expected.shape, (family, actual.shape, expected.shape)
        delta = np.abs(actual-expected)
        case = dict(name=family, samples=len(actual),
                    max_position_difference_m=float(delta[:, 1:4].max()),
                    max_state_difference=float(delta[:, 1:20].max()),
                    max_reference_difference=float(delta[:, 20:26].max()),
                    max_command_difference_rad_s=float(delta[:, 26:32].max()))
        case["passed"] = bool(case["max_position_difference_m"] < 1e-6 and
                              case["max_state_difference"] < 1e-5 and
                              case["max_reference_difference"] < 1e-10 and
                              case["max_command_difference_rad_s"] < 1e-4)
        cases.append(case)
        print(json.dumps(case), flush=True)
    report = dict(source_commit="a6b9f1dd3d839416e469755490b8ff7733acd4ee",
                  parameter_profile=cfg["control_tuning_version"], control=cfg["control"],
                  adapter="zero acceleration and yaw-rate feedforward; fixed internal yaw",
                  cases=cases, passed=all(item["passed"] for item in cases))
    (ROOT / "validation" / "BASELINE_PYTHON_PARITY.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    assert report["passed"], "Current-gain Python/SLX parity failed"


if __name__ == "__main__":
    main()
