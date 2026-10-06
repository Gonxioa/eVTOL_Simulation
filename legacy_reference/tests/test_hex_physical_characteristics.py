import unittest

import numpy as np


NUMPY_INV_BEFORE_PROJECT_IMPORT = np.linalg.inv

from hex_physical_characteristics import (  # noqa: E402
    HEX_PHYSICAL_CHARACTERISTICS,
    physical_review_summary,
)
from hex_params import hex_params  # noqa: E402
from main import make_init  # noqa: E402
from rotorpy.controllers.quadrotor_control import SE3Control  # noqa: E402
from rotorpy.vehicles.multirotor import Multirotor  # noqa: E402


class HexPhysicalCharacteristicsTests(unittest.TestCase):
    def test_source_backed_parameter_values(self):
        params = HEX_PHYSICAL_CHARACTERISTICS
        self.assertAlmostEqual(params["Ixx"], 0.0476)
        self.assertAlmostEqual(params["Iyy"], 0.0476)
        self.assertAlmostEqual(params["Izz"], 0.1094)
        self.assertAlmostEqual(params["rotor_radius"], 0.12065)
        self.assertAlmostEqual(params["k_eta"], 7.45e-6)
        self.assertAlmostEqual(params["k_m"] / params["k_eta"], 0.01315)
        self.assertAlmostEqual(params["tau_m"], 1.0 / 20.06)

        review = physical_review_summary()
        self.assertAlmostEqual(review["max_rotor_speed_rpm"], 10_500.0)
        self.assertGreater(review["max_thrust_to_weight"], 1.0)
        self.assertLess(review["hover_speed_rpm"], 10_500.0)

    def test_rotorpy_patch_is_hex_compatible_without_persistent_numpy_patch(self):
        init = make_init(np.array([0.0, 0.0, 1.0]))
        vehicle = Multirotor(hex_params, init)
        controller = SE3Control(hex_params)

        self.assertIs(np.linalg.inv, NUMPY_INV_BEFORE_PROJECT_IMPORT)
        self.assertEqual(vehicle.f_to_TM.shape, (4, 6))
        self.assertEqual(vehicle.TM_to_f.shape, (6, 4))
        self.assertEqual(controller.TM_to_f.shape, (6, 4))
        np.testing.assert_allclose(
            vehicle.f_to_TM @ vehicle.TM_to_f,
            np.eye(4),
            atol=1e-12,
        )

        packed = Multirotor._pack_state(init)
        self.assertEqual(packed.shape, (22,))


if __name__ == "__main__":
    unittest.main()
