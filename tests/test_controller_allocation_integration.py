import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from data_process import process_source
from hex_params import hex_params
from main import (
    FaultAwareSE3Control,
    FaultInjectionMultirotor,
    existing_run_matches,
    export_to_csv,
)
from scenario_config import SCENARIO_BY_NAME
from validate_hex_data import (
    EXPECTED_RATE,
    EXPECTED_ROWS,
    EXPECTED_WARMUP,
    inspect_file,
)


def hover_state_and_flat_output():
    count = int(hex_params["num_rotors"])
    hover_speed = np.sqrt(
        hex_params["mass"] * 9.81 / (count * hex_params["k_eta"])
    )
    state = {
        "x": np.array([0.0, 0.0, 2.5]),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.full(count, hover_speed),
    }
    flat = {
        "x": state["x"].copy(),
        "x_dot": np.zeros(3),
        "x_ddot": np.zeros(3),
        "x_dddot": np.zeros(3),
        "x_ddddot": np.zeros(3),
        "yaw": 0.0,
        "yaw_dot": 0.0,
    }
    return state, flat


class ControllerAllocationIntegrationTests(unittest.TestCase):
    def test_fault_truth_does_not_change_nominal_allocator_command(self):
        state, flat = hover_state_and_flat_output()
        controller = FaultAwareSE3Control(
            hex_params,
            fault_time=8.0,
            fault_motor_indices=[2],
            fault_factor=0.2,
        )
        before = controller.update(7.99, state, flat)
        after = controller.update(8.0, state, flat)

        self.assertEqual(set(before), set(after))
        np.testing.assert_allclose(
            before["cmd_motor_speeds"], after["cmd_motor_speeds"], atol=0.0
        )
        np.testing.assert_allclose(before["plant_effectiveness"], np.ones(6))
        expected = np.ones(6)
        expected[2] = 0.2
        np.testing.assert_allclose(after["plant_effectiveness"], expected)
        self.assertEqual(before["fault_active"], 0.0)
        self.assertEqual(after["fault_active"], 1.0)
        self.assertTrue(np.all(before["cmd_motor_speeds"] >= 0.0))
        self.assertTrue(
            np.all(before["cmd_motor_speeds"] <= hex_params["rotor_speed_max"])
        )
        self.assertEqual(before["allocator_success"], 1.0)
        self.assertEqual(after["allocator_success"], 1.0)

    def test_vehicle_applies_effectiveness_once_to_motor_speed_target(self):
        state, flat = hover_state_and_flat_output()
        controller = FaultAwareSE3Control(
            hex_params,
            fault_time=8.0,
            fault_motor_indices=[2],
            fault_factor=0.2,
        )
        vehicle = FaultInjectionMultirotor(
            hex_params,
            state,
            fault_time=8.0,
            fault_motor_indices=[2],
            fault_factor=0.2,
        )
        control = controller.update(8.0, state, flat)
        plant_target = vehicle.get_cmd_motor_speeds(state, control)
        expected = control["cmd_motor_speeds"] * np.sqrt(
            control["plant_effectiveness"]
        )
        np.testing.assert_allclose(plant_target, expected, rtol=0.0, atol=1e-12)

        command_thrust = hex_params["k_eta"] * control["cmd_motor_speeds"] ** 2
        plant_thrust = hex_params["k_eta"] * plant_target**2
        np.testing.assert_allclose(
            plant_thrust,
            control["plant_effectiveness"] * command_thrust,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            plant_thrust,
            control["predicted_plant_motor_thrusts"],
            rtol=0.0,
            atol=1e-12,
        )

    def test_exported_allocator_audit_schema_passes_validator(self):
        state, flat = hover_state_and_flat_output()
        scenario = SCENARIO_BY_NAME["motor2_severe"]
        controller = FaultAwareSE3Control(
            hex_params,
            fault_time=8.0,
            fault_motor_indices=[2],
            fault_factor=scenario.simulated_factor,
        )
        times = EXPECTED_WARMUP + np.arange(EXPECTED_ROWS) / EXPECTED_RATE
        controls = [controller.update(t, state, flat) for t in times]
        control_history = {
            key: np.asarray([control[key] for control in controls])
            for key in controls[0]
        }
        count = len(times)
        state_history = {
            "x": np.repeat(state["x"][None, :], count, axis=0),
            "v": np.repeat(state["v"][None, :], count, axis=0),
            "q": np.repeat(state["q"][None, :], count, axis=0),
            "w": np.repeat(state["w"][None, :], count, axis=0),
            "wind": np.repeat(state["wind"][None, :], count, axis=0),
            "rotor_speeds": np.asarray([
                control["cmd_motor_speeds"]
                * np.sqrt(control["plant_effectiveness"])
                for control in controls
            ]),
        }
        results = {
            "time": times,
            "state": state_history,
            "control": control_history,
        }
        imu = {"acc": np.zeros((count, 3)), "gyro": np.zeros((count, 3))}

        with TemporaryDirectory() as directory:
            export_to_csv(
                results,
                imu,
                filename="motor2_severe__run001.csv",
                fault_label=scenario.label,
                fault_motor=scenario.fault_motor,
                fault_factor=scenario.simulated_factor,
                warmup=EXPECTED_WARMUP,
                output_dir=directory,
                run_metadata={
                    "fault_time": 8.0,
                    "scenario_name": scenario.name,
                    "run_id": 1,
                    "random_seed": 123,
                    "trajectory_id": "unit_test",
                },
            )
            exported = Path(directory) / "motor2_severe__run001.csv"
            record = inspect_file(exported)
            self.assertNotEqual(record["status"], "FAIL", record["reason"])
            self.assertTrue(
                existing_run_matches(
                    exported,
                    scenario.name,
                    1,
                    scenario.label,
                    123,
                )
            )
            truncated = Path(directory) / "truncated.csv"
            lines = exported.read_text(encoding="utf-8").splitlines()
            truncated.write_text("\n".join(lines[:20]) + "\n", encoding="utf-8")
            self.assertFalse(
                existing_run_matches(
                    truncated,
                    scenario.name,
                    1,
                    scenario.label,
                    123,
                )
            )
            arrays, source = process_source(
                exported,
                matched_start=9.0,
            )
            np.testing.assert_array_equal(np.unique(arrays["groups"]), [1])
            self.assertEqual(source["group_id"], 1)
            self.assertGreater(len(arrays["X"]), 0)


if __name__ == "__main__":
    unittest.main()
