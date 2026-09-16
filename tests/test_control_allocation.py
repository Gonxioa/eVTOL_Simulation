import unittest

import numpy as np

from control_allocation import (
    BoundedControlAllocator,
    predict_effective_response,
    scheduled_effectiveness,
    validate_effectiveness,
)
from hex_physical_characteristics import HEX_PHYSICAL_CHARACTERISTICS


P = HEX_PHYSICAL_CHARACTERISTICS
NUM_ROTORS = int(P["num_rotors"])
K_ETA = float(P["k_eta"])
ROTOR_MAX = float(P["rotor_speed_max"])
F_MAX = K_ETA * ROTOR_MAX**2
MG = float(P["mass"]) * 9.81


def make_matrix():
    positions = P["rotor_pos"]
    arms = np.hstack([
        np.cross(positions[key], np.array([0.0, 0.0, 1.0]))
        .reshape(-1, 1)[0:2]
        for key in positions
    ])
    yaw = (
        float(P["k_m"]) / K_ETA
        * np.asarray(P["rotor_directions"], dtype=float)
    ).reshape(1, -1)
    return np.vstack((np.ones((1, NUM_ROTORS)), arms, yaw))


A = make_matrix()


def make_allocator(priority=(2.0, 1.0, 1.0, 0.2), regularization=0.0):
    return BoundedControlAllocator(
        A,
        thrust_min=0.0,
        thrust_max=F_MAX,
        wrench_priority=priority,
        regularization=regularization,
    )


class ControlAllocationTests(unittest.TestCase):
    def assert_physical_result(self, result):
        self.assertEqual(result.commanded_thrusts.shape, (NUM_ROTORS,))
        self.assertTrue(np.isfinite(result.commanded_thrusts).all())
        self.assertTrue(np.isfinite(result.allocated_wrench).all())
        self.assertTrue(np.all(result.commanded_thrusts >= -1e-10))
        self.assertTrue(np.all(result.commanded_thrusts <= F_MAX + 1e-9))
        np.testing.assert_allclose(
            result.allocated_wrench,
            A @ result.commanded_thrusts,
            rtol=0.0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            result.residual_wrench,
            result.desired_wrench - result.allocated_wrench,
            rtol=0.0,
            atol=1e-10,
        )

    def test_allocation_matrix_matches_hex_geometry(self):
        expected = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [
                0.0,
                0.23815698604072064,
                0.23815698604072064,
                0.0,
                -0.23815698604072064,
                -0.23815698604072064,
            ],
            [-0.275, -0.1375, 0.1375, 0.275, 0.1375, -0.1375],
            [
                0.03949730700179534,
                -0.03949730700179534,
                0.03949730700179534,
                -0.03949730700179534,
                0.03949730700179534,
                -0.03949730700179534,
            ],
        ])
        self.assertEqual(A.shape, (4, 6))
        self.assertEqual(np.linalg.matrix_rank(A), 4)
        np.testing.assert_allclose(A, expected, rtol=0.0, atol=1e-12)

    def test_normal_hover_is_exact_symmetric_and_inside_speed_limit(self):
        result = make_allocator().allocate(np.array([MG, 0.0, 0.0, 0.0]))
        self.assertTrue(result.success)
        self.assertTrue(result.wrench_feasible)
        self.assertTrue(result.used_unconstrained_solution)
        self.assert_physical_result(result)
        np.testing.assert_allclose(
            result.commanded_thrusts,
            np.full(NUM_ROTORS, MG / NUM_ROTORS),
            rtol=0.0,
            atol=1e-10,
        )
        speeds = np.sqrt(result.commanded_thrusts / K_ETA)
        self.assertTrue(np.all(speeds < ROTOR_MAX))
        np.testing.assert_allclose(
            result.allocated_wrench, result.desired_wrench, atol=1e-10
        )

    def test_random_feasible_wrenches_remain_physical(self):
        for seed in range(8):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                source = rng.uniform(0.15 * F_MAX, 0.75 * F_MAX, NUM_ROTORS)
                desired = A @ source
                result = make_allocator().allocate(desired)
                self.assertTrue(result.success)
                self.assertTrue(result.wrench_feasible)
                self.assert_physical_result(result)
                np.testing.assert_allclose(result.allocated_wrench, desired, atol=1e-7)

    def test_excess_collective_returns_best_bounded_solution(self):
        desired = np.array([10.0 * MG, 0.0, 0.0, 0.0])
        result = make_allocator().allocate(desired)
        self.assertTrue(result.success)
        self.assertFalse(result.wrench_feasible)
        self.assert_physical_result(result)
        np.testing.assert_allclose(
            result.commanded_thrusts, np.full(NUM_ROTORS, F_MAX), atol=1e-7
        )
        self.assertTrue(np.all(result.upper_bound_active))
        self.assertEqual(result.bound_active_count, NUM_ROTORS)

    def test_negative_collective_returns_zero_without_signed_sqrt(self):
        result = make_allocator().allocate(np.array([-MG, 0.0, 0.0, 0.0]))
        self.assertTrue(result.success)
        self.assertFalse(result.wrench_feasible)
        self.assert_physical_result(result)
        np.testing.assert_allclose(result.commanded_thrusts, 0.0, atol=1e-8)
        self.assertTrue(np.all(result.lower_bound_active))

    def test_diagnostic_priority_sacrifices_yaw_first(self):
        desired = np.array([MG, 0.0, 0.0, 10.0])
        diagnostic = make_allocator(priority=(4.0, 3.0, 3.0, 0.5)).allocate(desired)
        yaw_first = make_allocator(priority=(0.5, 0.5, 0.5, 4.0)).allocate(desired)
        self.assertTrue(diagnostic.success and yaw_first.success)
        self.assert_physical_result(diagnostic)
        self.assert_physical_result(yaw_first)
        scale = make_allocator().wrench_scale
        protected_diagnostic = np.linalg.norm(
            diagnostic.residual_wrench[:3] / scale[:3]
        )
        protected_yaw = np.linalg.norm(yaw_first.residual_wrench[:3] / scale[:3])
        self.assertLess(protected_diagnostic, protected_yaw)
        self.assertGreaterEqual(
            abs(diagnostic.residual_wrench[3]),
            abs(yaw_first.residual_wrench[3]) - 1e-8,
        )

    def test_schedule_changes_exactly_at_fault_time(self):
        before = scheduled_effectiveness(7.999, NUM_ROTORS, 8.0, [2], 0.2)
        at_fault = scheduled_effectiveness(8.0, NUM_ROTORS, 8.0, [2], 0.2)
        after = scheduled_effectiveness(8.001, NUM_ROTORS, 8.0, [2], 0.2)
        np.testing.assert_array_equal(before, np.ones(NUM_ROTORS))
        expected = np.ones(NUM_ROTORS)
        expected[2] = 0.2
        np.testing.assert_array_equal(at_fault, expected)
        np.testing.assert_array_equal(after, expected)

    def test_effectiveness_is_applied_once_in_plant_prediction(self):
        result = make_allocator().allocate(np.array([MG, 0.0, 0.0, 0.0]))
        for factor in (1.0, 0.8, 0.2, 0.0):
            with self.subTest(factor=factor):
                eta = np.ones(NUM_ROTORS)
                eta[3] = factor
                effective, predicted = predict_effective_response(
                    A, result.commanded_thrusts, eta
                )
                np.testing.assert_allclose(effective, eta * result.commanded_thrusts)
                np.testing.assert_allclose(predicted, A @ effective)
                command_speeds = np.sqrt(result.commanded_thrusts / K_ETA)
                plant_speeds = np.sqrt(eta) * command_speeds
                np.testing.assert_allclose(
                    K_ETA * plant_speeds**2, effective, atol=1e-12
                )

    def test_allocator_is_deterministic(self):
        desired = np.array([MG, 1.2, -0.8, 2.0])
        first = make_allocator(regularization=1e-8).allocate(desired)
        second = make_allocator(regularization=1e-8).allocate(desired)
        np.testing.assert_allclose(
            first.commanded_thrusts, second.commanded_thrusts, atol=1e-12
        )
        np.testing.assert_allclose(
            first.allocated_wrench, second.allocated_wrench, atol=1e-12
        )

    def test_invalid_inputs_fail_loudly(self):
        invalid_effectiveness = (
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, -0.1],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.1],
            [1.0, 1.0, 1.0, 1.0, 1.0, np.nan],
        )
        for effectiveness in invalid_effectiveness:
            with self.subTest(effectiveness=effectiveness):
                with self.assertRaises(ValueError):
                    validate_effectiveness(effectiveness, NUM_ROTORS)

        allocator = make_allocator()
        with self.assertRaises(ValueError):
            allocator.allocate(np.zeros(3))
        with self.assertRaises(ValueError):
            allocator.allocate(np.array([MG, 0.0, 0.0, np.nan]))
        with self.assertRaises(ValueError):
            BoundedControlAllocator(
                A, 0.0, F_MAX, wrench_priority=[1.0, 1.0, 1.0, 0.0]
            )
        with self.assertRaises(ValueError):
            BoundedControlAllocator(A, 0.0, F_MAX, regularization=-1.0)


if __name__ == "__main__":
    unittest.main()
