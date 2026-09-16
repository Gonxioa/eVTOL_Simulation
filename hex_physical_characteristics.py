"""Hexarotor airframe properties only.

This module deliberately excludes trajectories, wind, sensor noise, fault
injection, controller gains, and machine-learning settings.  It is the single
code source for the aircraft properties documented in
``docs/HEXAROTOR_PHYSICAL_CHARACTERISTICS.md``.
"""

import numpy as np


ARM_LENGTH = 0.275

HEX_PHYSICAL_CHARACTERISTICS = {
    "mass": 3.4,
    "Ixx": 2.15e-2,
    "Iyy": 2.15e-2,
    "Izz": 4.06e-2,
    "Ixy": 0.0,
    "Iyz": 0.0,
    "Ixz": 0.0,
    "num_rotors": 6,
    "rotor_radius": 0.127,
    "rotor_pos": {
        "r1": ARM_LENGTH * np.array([1.0, 0.0, 0.0]),
        "r2": ARM_LENGTH * np.array([0.5, np.sqrt(3.0) / 2.0, 0.0]),
        "r3": ARM_LENGTH * np.array([-0.5, np.sqrt(3.0) / 2.0, 0.0]),
        "r4": ARM_LENGTH * np.array([-1.0, 0.0, 0.0]),
        "r5": ARM_LENGTH * np.array([-0.5, -np.sqrt(3.0) / 2.0, 0.0]),
        "r6": ARM_LENGTH * np.array([0.5, -np.sqrt(3.0) / 2.0, 0.0]),
    },
    "rotor_directions": np.array([1, -1, 1, -1, 1, -1]),
    "rI": np.zeros(3),
    "c_Dx": 0.5e-2,
    "c_Dy": 0.5e-2,
    "c_Dz": 1.0e-2,
    "k_eta": 5.57e-6,
    "k_m": 2.20e-7,
    "k_d": 1.19e-4,
    "k_z": 2.32e-4,
    "k_h": 3.39e-3,
    "k_flap": 0.0,
    "tau_m": 0.02,
    "rotor_speed_min": 0.0,
    "rotor_speed_max": 1500.0,
}


def physical_review_summary(gravity=9.81):
    """Return simple consistency checks without introducing sim settings."""
    p = HEX_PHYSICAL_CHARACTERISTICS
    hover = np.sqrt(p["mass"] * gravity / (p["num_rotors"] * p["k_eta"]))
    max_per_rotor = p["k_eta"] * p["rotor_speed_max"] ** 2
    return {
        "hover_speed_rad_s": float(hover),
        "max_thrust_per_rotor_n": float(max_per_rotor),
        "max_total_thrust_n": float(max_per_rotor * p["num_rotors"]),
        "max_thrust_to_weight": float(
            max_per_rotor * p["num_rotors"] / (p["mass"] * gravity)
        ),
    }
