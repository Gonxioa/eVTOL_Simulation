"""Hexarotor airframe properties only.

This module deliberately excludes trajectories, wind, sensor noise, fault
injection, controller gains, and machine-learning settings.  It is the single
code source for the aircraft properties documented in
``docs/HEXAROTOR_PHYSICAL_CHARACTERISTICS.md``.
"""

import numpy as np


ARM_LENGTH = 0.275
PROPELLER_DIAMETER_IN = 9.5
AIR_GEAR_350_MAX_RPM = 10_500.0

# AIR2213/920KV + T9545, 4S static-thrust bench data, fitted with the
# RotorPy model T = k_eta * omega**2 through the physical zero intercept.
# The 35 samples span 300--10,500 rpm (R^2 = 0.9985).
AIR_GEAR_350_K_ETA = 7.45e-6

# No trustworthy torque curve for the legacy AIR Gear 350/T9545 combination
# was found.  The original workbook's torque tab contains placeholder values,
# so it must not be treated as a measurement.  Until this exact propulsion
# unit is tested, use the measured Q/T ratio from T-Motor's successor
# AIR2213-KV1000/MF9545 data.  This 0.01315 m ratio is also close to the
# 0.016 m AscTec Firefly value and is explicitly a provisional proxy.
PROVISIONAL_YAW_MOMENT_TO_THRUST = 1.315e-2

# The experimentally identified actuator pole for the modified F550 was
# p = 20.06 rad/s.  RotorPy's first-order motor state uses tau_m = 1 / p.
F550_ACTUATOR_POLE_RAD_S = 20.06

HEX_PHYSICAL_CHARACTERISTICS = {
    "mass": 3.4,
    # Closed-loop grey-box identification on a modified DJI F550:
    # Jx = Jy = 0.0476 kg m^2 and Jz = 0.1094 kg m^2.
    "Ixx": 4.76e-2,
    "Iyy": 4.76e-2,
    "Izz": 1.094e-1,
    "Ixy": 0.0,
    "Iyz": 0.0,
    "Ixz": 0.0,
    "num_rotors": 6,
    "rotor_radius": PROPELLER_DIAMETER_IN * 0.0254 / 2.0,
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
    "k_eta": AIR_GEAR_350_K_ETA,
    "k_m": AIR_GEAR_350_K_ETA * PROVISIONAL_YAW_MOMENT_TO_THRUST,
    "k_d": 1.19e-4,
    "k_z": 2.32e-4,
    "k_h": 3.39e-3,
    "k_flap": 0.0,
    "tau_m": 1.0 / F550_ACTUATOR_POLE_RAD_S,
    "rotor_speed_min": 0.0,
    "rotor_speed_max": AIR_GEAR_350_MAX_RPM * 2.0 * np.pi / 60.0,
}


def physical_review_summary(gravity=9.81):
    """Return simple consistency checks without introducing sim settings."""
    p = HEX_PHYSICAL_CHARACTERISTICS
    hover = np.sqrt(p["mass"] * gravity / (p["num_rotors"] * p["k_eta"]))
    max_per_rotor = p["k_eta"] * p["rotor_speed_max"] ** 2
    return {
        "hover_speed_rad_s": float(hover),
        "hover_speed_rpm": float(hover * 60.0 / (2.0 * np.pi)),
        "max_rotor_speed_rpm": float(
            p["rotor_speed_max"] * 60.0 / (2.0 * np.pi)
        ),
        "max_thrust_per_rotor_n": float(max_per_rotor),
        "max_total_thrust_n": float(max_per_rotor * p["num_rotors"]),
        "max_thrust_to_weight": float(
            max_per_rotor * p["num_rotors"] / (p["mass"] * gravity)
        ),
        "yaw_moment_to_thrust_m": float(p["k_m"] / p["k_eta"]),
    }
