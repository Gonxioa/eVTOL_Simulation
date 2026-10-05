"""Newton--Euler core in NED/FRD; quaternion wxyz maps body to world.

Equation evidence: docs/uavfd_f550/MODEL_SPEC.md E1--E6.
No aircraft-specific parameters are embedded here.
"""
from dataclasses import dataclass
import numpy as np


def qmul(a, b):
    aw, av, bw, bv = a[0], a[1:], b[0], b[1:]
    return np.r_[aw*bw-av@bv, aw*bv+bw*av+np.cross(av, bv)]


def rotation(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid quaternion")
    w, x, y, z = q/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def from_euler(roll, pitch, yaw):
    cr, cp, cy = np.cos(np.array([roll, pitch, yaw])/2)
    sr, sp, sy = np.sin(np.array([roll, pitch, yaw])/2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def rpm_to_rad_s(rpm):
    return np.asarray(rpm)*2*np.pi/60


@dataclass
class Body:
    mass: float
    inertia: np.ndarray
    rotor_positions: np.ndarray
    spin_sign: np.ndarray  # +1 is clockwise seen from above; FRD +z.
    kT: np.ndarray
    kQ: np.ndarray
    gravity: float
    status: str

    def __post_init__(self):
        self.inertia = np.asarray(self.inertia, dtype=float)
        self.rotor_positions = np.asarray(self.rotor_positions, dtype=float)
        self.spin_sign = np.asarray(self.spin_sign, dtype=float)
        self.kT, self.kQ = np.asarray(self.kT, dtype=float), np.asarray(self.kQ, dtype=float)
        n = len(self.spin_sign)
        if not np.isfinite(self.mass) or not np.isfinite(self.gravity) or self.mass <= 0 or self.gravity <= 0 or self.inertia.shape != (3, 3):
            raise ValueError("Positive mass/gravity and 3x3 inertia required")
        if not np.allclose(self.inertia, self.inertia.T) or np.min(np.linalg.eigvalsh(self.inertia)) <= 0:
            raise ValueError("Inertia must be symmetric positive definite")
        if self.rotor_positions.shape != (n, 3) or self.kT.shape != (n,) or self.kQ.shape != (n,):
            raise ValueError("Rotor parameter dimensions disagree")
        if not np.all(np.isin(self.spin_sign, [-1, 1])) or np.any(self.kT < 0) or np.any(self.kQ < 0):
            raise ValueError("Invalid spin signs or aerodynamic coefficients")
        if not all(np.all(np.isfinite(x)) for x in (self.inertia, self.rotor_positions, self.kT, self.kQ)):
            raise ValueError("Nonfinite body parameters")
        if self.status not in ("demo", "source_confirmed", "flight_identified"):
            raise ValueError("An explicit evidence status is required")

    def wrench(self, Omega_rotor, kT_scale=None, kQ_scale=None):
        Omega = np.asarray(Omega_rotor, dtype=float)
        if Omega.shape != self.kT.shape or np.any(Omega < 0) or not np.all(np.isfinite(Omega)):
            raise ValueError("Nonnegative finite rotor angular speeds required")
        kt = np.ones_like(Omega) if kT_scale is None else np.asarray(kT_scale)
        kq = np.ones_like(Omega) if kQ_scale is None else np.asarray(kQ_scale)
        if kt.shape != Omega.shape or kq.shape != Omega.shape or not np.all(np.isfinite(kt)) or not np.all(np.isfinite(kq)) or np.any(kt < 0) or np.any(kq < 0):
            raise ValueError("Explicit nonnegative thrust/torque scales required")
        force_each = np.zeros((len(Omega), 3))
        force_each[:, 2] = -self.kT*kt*Omega**2
        torque_each = np.cross(self.rotor_positions, force_each)
        torque_each[:, 2] -= self.spin_sign*self.kQ*kq*Omega**2
        return force_each.sum(axis=0), torque_each.sum(axis=0)

    def derivative(self, state, Omega_rotor, extra_force_body=None, extra_torque_body=None):
        x = np.asarray(state, dtype=float)
        if x.shape != (13,) or not np.all(np.isfinite(x)):
            raise ValueError("State must be finite [position3,velocity3,qwxyz4,omega_body3]")
        force, torque = self.wrench(Omega_rotor)
        if extra_force_body is not None:
            force += np.asarray(extra_force_body)
        if extra_torque_body is not None:
            torque += np.asarray(extra_torque_body)
        R, w = rotation(x[6:10]), x[10:13]
        acceleration = np.array([0, 0, self.gravity])+R@force/self.mass
        alpha = np.linalg.solve(self.inertia, torque-np.cross(w, self.inertia@w))
        return np.r_[x[3:6], acceleration, .5*qmul(x[6:10], np.r_[0., w]), alpha]


def rk4(body, state, rotor_input, t, dt, extra=None):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive integration step required")
    def f(ti, xi):
        force, torque = (None, None) if extra is None else extra(ti)
        return body.derivative(xi, rotor_input(ti), force, torque)
    k1 = f(t, state)
    k2 = f(t+dt/2, state+dt*k1/2)
    k3 = f(t+dt/2, state+dt*k2/2)
    k4 = f(t+dt, state+dt*k3)
    result = state+dt*(k1+2*k2+2*k3+k4)/6
    result[6:10] /= np.linalg.norm(result[6:10])
    return result


def specific_force(state, state_derivative, gravity, r_sensor, a_elastic=None):
    R, w = rotation(state[6:10]), state[10:13]
    alpha = state_derivative[10:13]
    r = np.asarray(r_sensor, dtype=float)
    f = R.T@(state_derivative[3:6]-np.array([0, 0, gravity]))
    f += np.cross(alpha, r)+np.cross(w, np.cross(w, r))
    if a_elastic is not None:
        f += np.asarray(a_elastic)
    return f


def imu_observe(state, derivative, gravity, r_sensor, a_elastic, local_angular_rate,
                accel_bias, gyro_bias, accel_noise_std, gyro_noise_std, rng):
    # Logged IMU filtering must be handled outside this physical observation equation.
    accel = specific_force(state, derivative, gravity, r_sensor, a_elastic)
    gyro = state[10:13]+np.asarray(local_angular_rate)
    return (accel+accel_bias+rng.normal(0, accel_noise_std, 3),
            gyro+gyro_bias+rng.normal(0, gyro_noise_std, 3))


def first_order_step(Omega, steady_Omega, tau, dt):
    if not np.isfinite(tau) or not np.isfinite(dt) or tau <= 0 or dt <= 0:
        raise ValueError("Positive actuator time constant and timestep required")
    return steady_Omega+(Omega-steady_Omega)*np.exp(-dt/tau)


def body_from_config(config, allow_demo=False):
    """Unknown/provisional aircraft values cannot silently instantiate a Body."""
    required = ("mass", "inertia", "rotor_positions", "spin_sign", "kT", "kQ", "gravity")
    missing = [k for k in required if k not in config or config[k].get("value") is None]
    if missing:
        raise ValueError("Unknown physical parameters: " + ", ".join(missing))
    accepted = {"source_confirmed", "flight_identified", "standard_gravity_convention"}
    if allow_demo:
        accepted.add("demo")
    unsupported = [k for k in required if config[k].get("status") not in accepted]
    if unsupported:
        raise ValueError("Unsupported parameter evidence: " + ", ".join(unsupported))
    status = "demo" if any(config[k]["status"] == "demo" for k in required) else "flight_identified"
    return Body(**{k: config[k]["value"] for k in required}, status=status)


def old_enu_flu_to_ned_frd(position, velocity, R, omega):
    world = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    body = np.diag([1, -1, -1])
    return world@position, world@velocity, world@R@body, body@omega
