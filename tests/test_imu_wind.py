import unittest

import numpy as np
from scipy.spatial.transform import Rotation
from rotorpy.environments import Environment

from main import (
    RealisticIMU, TurbulentWind, build_scenario, generate_imu_data,
    make_run_variation, make_truth_imu,
)


class ImuTruthTests(unittest.TestCase):
    def test_truth_sensor_uses_body_origin_and_has_no_builtin_error(self):
        sensor = make_truth_imu(100)
        np.testing.assert_array_equal(sensor.p_BS, np.zeros(3))
        np.testing.assert_array_equal(sensor.R_BS, np.eye(3))
        for name in (
            "accel_bias", "gyro_bias", "accel_random_walk", "gyro_random_walk",
            "accel_variance", "gyro_variance",
        ):
            np.testing.assert_array_equal(getattr(sensor, name), np.zeros(3))

    def test_specific_force_matches_independent_body_frame_calculation(self):
        sensor = make_truth_imu(100)
        gravity_world = np.array([0.0, 0.0, -9.81])
        cases = (
            (Rotation.identity(), np.zeros(3), np.array([0.0, 0.0, 9.81])),
            (Rotation.identity(), gravity_world, np.zeros(3)),
            (Rotation.from_euler("xyz", [0.3, -0.4, 0.2]),
             np.array([1.0, -2.0, 0.5]), None),
        )
        for rotation, acceleration_world, expected_level in cases:
            with self.subTest(acceleration_world=acceleration_world):
                state = {"q": rotation.as_quat(), "w": np.array([0.2, -0.1, 0.3])}
                derivatives = {"vdot": acceleration_world, "wdot": np.zeros(3)}
                truth = sensor.measurement(state, derivatives, with_noise=False)
                expected = rotation.as_matrix().T @ (acceleration_world - gravity_world)
                np.testing.assert_allclose(truth["accel"], expected, atol=1e-12)
                np.testing.assert_allclose(truth["gyro"], state["w"], atol=0.0)
                if expected_level is not None:
                    np.testing.assert_allclose(truth["accel"], expected_level, atol=1e-12)

    @staticmethod
    def results(accel, gyro):
        return {
            "time": np.arange(len(accel), dtype=float) / 100,
            "imu_gt": {"accel": accel, "gyro": gyro},
        }

    def test_zero_custom_noise_returns_truth_without_mutating_inputs(self):
        accel = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
        gyro = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]])
        accel_before, gyro_before = accel.copy(), gyro.copy()
        sensor = RealisticIMU(
            acc_noise_std=0, acc_bias_walk=0, gyro_noise_std=0,
            gyro_bias_walk=0, vibration_std=0, rng=np.random.default_rng(5),
        )
        measured = generate_imu_data(self.results(accel, gyro), sensor)
        np.testing.assert_array_equal(measured["acc"], accel)
        np.testing.assert_array_equal(measured["gyro"], gyro)
        np.testing.assert_array_equal(accel, accel_before)
        np.testing.assert_array_equal(gyro, gyro_before)

    def test_custom_noise_is_repeatable_with_same_seed(self):
        results = self.results(np.zeros((10, 3)), np.zeros((10, 3)))
        first = generate_imu_data(results, RealisticIMU(rng=np.random.default_rng(91)))
        second = generate_imu_data(results, RealisticIMU(rng=np.random.default_rng(91)))
        np.testing.assert_array_equal(first["acc"], second["acc"])
        np.testing.assert_array_equal(first["gyro"], second["gyro"])
        self.assertFalse(np.all(first["acc"] == 0))

    def test_missing_malformed_and_nonfinite_truth_fail_loudly(self):
        valid = self.results(np.zeros((2, 3)), np.zeros((2, 3)))
        variants = (
            ({"time": valid["time"]}, "imu_gt"),
            ({**valid, "imu_gt": {"gyro": valid["imu_gt"]["gyro"]}}, "accel"),
            (self.results(np.zeros((1, 3)), np.zeros((2, 3))), "shape"),
            (self.results(np.zeros((2, 2)), np.zeros((2, 3))), "shape"),
            (self.results(np.full((2, 3), np.nan), np.zeros((2, 3))), "finite"),
        )
        for results, message in variants:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    generate_imu_data(results, RealisticIMU())

    def test_short_rotorpy_run_exposes_body_frame_truth(self):
        vehicle, controller, trajectory, wind, world, _, _ = build_scenario(
            use_wind=False, run_variation=make_run_variation(123)
        )
        env = Environment(
            vehicle=vehicle, controller=controller, trajectory=trajectory,
            wind_profile=wind, world=world, sim_rate=100,
            imu=make_truth_imu(100), safety_margin=0.0,
        )
        results = env.run(
            t_final=0.02, use_mocap=False, terminate=False, plot=False,
            plot_mocap=False, plot_estimator=False, plot_imu=False,
        )
        count = len(results["time"])
        self.assertGreaterEqual(count, 3)
        self.assertEqual(results["imu_gt"]["accel"].shape, (count, 3))
        self.assertEqual(results["imu_gt"]["gyro"].shape, (count, 3))
        np.testing.assert_allclose(
            results["imu_gt"]["gyro"], results["state"]["w"], atol=0.0
        )

        state = {key: value[0] for key, value in results["state"].items()}
        control = {key: value[0] for key, value in results["control"].items()}
        derivatives = vehicle.statedot(state, control, 0.01)
        rotation = Rotation.from_quat(state["q"]).as_matrix()
        expected = rotation.T @ (
            derivatives["vdot"] - np.array([0.0, 0.0, -9.81])
        )
        np.testing.assert_allclose(results["imu_gt"]["accel"][0], expected, atol=1e-10)


class TurbulentWindTests(unittest.TestCase):
    def test_zero_turbulence_matches_configured_sinusoid(self):
        position = np.zeros(3)
        for amplitude in (0.0, 0.5, 1.0):
            wind = TurbulentWind(base_amplitude=amplitude, turbulence_std=0)
            for t in (0.0, 0.125, 0.25, 0.5):
                with self.subTest(amplitude=amplitude, time=t):
                    expected = np.full(3, amplitude * np.sin(2 * np.pi * t))
                    np.testing.assert_allclose(wind.update(t, position), expected, atol=1e-15)

    def test_turbulence_is_seeded_and_independent_of_base_amplitude(self):
        position = np.zeros(3)
        first = TurbulentWind(0.5, 0.2, np.random.default_rng(31))
        replay = TurbulentWind(0.5, 0.2, np.random.default_rng(31))
        larger_base = TurbulentWind(1.0, 0.2, np.random.default_rng(31))
        for t in (0.0, 0.125, 0.4):
            wind = first.update(t, position)
            np.testing.assert_array_equal(wind, replay.update(t, position))
            np.testing.assert_allclose(
                larger_base.update(t, position) - wind,
                np.full(3, 0.5 * np.sin(2 * np.pi * t)), atol=1e-15,
            )

    def test_internal_type_error_is_not_masked(self):
        wind = TurbulentWind()

        class BrokenBase:
            def update(self, t, position):
                raise TypeError("internal wind failure")

        wind.base = BrokenBase()
        with self.assertRaisesRegex(TypeError, "internal wind failure"):
            wind.update(0.1, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
