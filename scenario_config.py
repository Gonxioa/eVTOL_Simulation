"""Single source of truth for the 19 hexarotor health scenarios."""

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np

from hex_physical_characteristics import HEX_PHYSICAL_CHARACTERISTICS


NUM_ROTORS = int(HEX_PHYSICAL_CHARACTERISTICS["num_rotors"])
SEVERITY_LEVELS = (
    ("full", "Full", 0.0),
    ("severe", "Sev", 0.2),
    ("partial", "Part", 0.8),
)
# RotorPy cannot retain enough stable post-fault data for motors 4/5 at exactly
# zero effectiveness.  The residual factors are explicit assumptions, not
# hidden changes to the label.
FULL_FAULT_FACTOR_OVERRIDES = {4: 0.05, 5: 0.05}


@dataclass(frozen=True)
class Scenario:
    name: str
    label: int
    fault_motor: int
    severity: str
    nominal_factor: float
    simulated_factor: float

    @property
    def fault_motors(self) -> List[int]:
        return [] if self.fault_motor < 0 else [self.fault_motor]

    @property
    def label_name(self) -> str:
        if self.label == 0:
            return "Normal"
        short = next(x[1] for x in SEVERITY_LEVELS if x[0] == self.severity)
        return f"M{self.fault_motor}-{short}"

    def effectiveness(self, num_rotors=NUM_ROTORS) -> np.ndarray:
        values = np.ones(int(num_rotors), dtype=np.float32)
        if self.fault_motor >= 0:
            values[self.fault_motor] = self.simulated_factor
        return values


def make_scenarios(num_rotors=NUM_ROTORS) -> List[Scenario]:
    scenarios = [Scenario("normal", 0, -1, "normal", 1.0, 1.0)]
    label = 1
    for severity, _short, nominal_factor in SEVERITY_LEVELS:
        for motor in range(int(num_rotors)):
            simulated = (
                FULL_FAULT_FACTOR_OVERRIDES.get(motor, nominal_factor)
                if severity == "full"
                else nominal_factor
            )
            scenarios.append(
                Scenario(
                    name=f"motor{motor}_{severity}",
                    label=label,
                    fault_motor=motor,
                    severity=severity,
                    nominal_factor=float(nominal_factor),
                    simulated_factor=float(simulated),
                )
            )
            label += 1
    return scenarios


SCENARIOS = tuple(make_scenarios())
SCENARIO_BY_NAME: Dict[str, Scenario] = {item.name: item for item in SCENARIOS}
SCENARIO_BY_LABEL: Dict[int, Scenario] = {item.label: item for item in SCENARIOS}
LABEL_NAMES = {item.label: item.label_name for item in SCENARIOS}


def effectiveness_table(num_rotors=NUM_ROTORS):
    return {
        int(item.label): item.effectiveness(num_rotors).tolist()
        for item in SCENARIOS
    }


def serializable_scenarios():
    return [asdict(item) | {"label_name": item.label_name} for item in SCENARIOS]
