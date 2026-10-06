"""RotorPy adapter assembled from physical and non-physical parameters.

The aircraft's physical characteristics live in
``hex_physical_characteristics.py``.  This compatibility module adds only the
controller and simulator-specific fields expected by the existing code, so
``main.py`` can keep importing ``hex_params`` without changing its behavior.
"""

from copy import deepcopy

import numpy as np

from hex_physical_characteristics import (
    ARM_LENGTH,
    HEX_PHYSICAL_CHARACTERISTICS,
)


hex_params = deepcopy(HEX_PHYSICAL_CHARACTERISTICS)
hex_params.update(
    {
        # Simulator-specific motor disturbance setting, not an airframe property.
        "motor_noise_std": 0.0,

        # -------------------------------------------------------------------
        # Multirotor-internal low-level gains. These belong to RotorPy's
        # Multirotor class and are only consulted for control_abstraction
        # in {'cmd_ctbr', 'cmd_vel', 'cmd_ctatt'}. This project always runs
        # control_abstraction='cmd_motor_speeds', so today these are inert.
        # Kept only in case a future experiment switches abstraction mode.
        # DO NOT reuse these names for SE3Control's outer-loop gains below —
        # that reuse is exactly what caused Bug A.
        # -------------------------------------------------------------------
        "k_w": 1,
        "k_v": 10,
        "kp_att": 544,
        "kd_att": 46.64,

        # -------------------------------------------------------------------
        # SE3Control outer-loop gains. These are the ones that actually drive
        # cmd_motor_speeds in this project (FaultAwareSE3Control.update ->
        # base SE3Control.update computes u2 from kp_att/kd_att regardless of
        # control_abstraction, then TM_to_f turns [u1, u2] into motor speeds).
        # Deliberately distinct key names so FaultAwareSE3Control.__init__ can
        # apply them without ever colliding with the Multirotor keys above.
        #
        # Values below start EQUAL to RotorPy's own hardcoded SE3Control
        # defaults on purpose: step 1 is only "make the config path real and
        # verified", not "retune for this airframe" (that's a separate,
        # later step once the unconstrained-allocation issue is fixed too).
        # -------------------------------------------------------------------
        "se3_kp_pos": np.array([6.5, 6.5, 15.0]),
        "se3_kd_pos": np.array([4.0, 4.0, 9.0]),
        "se3_kp_att": 544.0,
        "se3_kd_att": 46.64,

        # -------------------------------------------------------------------
        # Conventional bounded control allocation (Problem D).
        # Axis order is [collective thrust, roll, pitch, yaw].  Values are
        # soft priorities after each axis is normalized by the airframe's
        # available wrench authority.  Yaw is intentionally least important:
        # under saturation, preserving lift and roll/pitch is safer and is the
        # usual choice for a diagnosis platform.  The allocator always uses
        # the healthy nominal matrix; ground-truth fault effectiveness is
        # applied only by the simulated vehicle, never as controller oracle
        # knowledge.
        # -------------------------------------------------------------------
        "allocator_wrench_priority": np.array([2.0, 1.0, 1.0, 0.2]),
        # No command-bias regularizer: the motor first-order model already
        # smooths commands, while zero regularization preserves every feasible
        # wrench exactly instead of introducing a tiny artificial residual.
        "allocator_regularization": 0.0,
        "allocator_solver_tolerance": 1e-10,
        "allocator_feasibility_tolerance": 1e-6,
    }
)
