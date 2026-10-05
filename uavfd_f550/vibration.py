"""Rotating imbalance and local elastic modes, E7--E9 in MODEL_SPEC.

mu is a signed rotating mass first moment, not a blade-loss percentage.
Positive mu: mass excess at theta; support reaction is radially outward.
Tangential reaction is retained when an explicit angular acceleration is known.
"""
import numpy as np
from scipy.linalg import expm


def imbalance_force(mu, Omega, theta, spin_sign, Omega_dot=0.):
    if not np.all(np.isfinite([mu, Omega, theta, Omega_dot])) or Omega < 0 or spin_sign not in (-1, 1):
        raise ValueError("Invalid rotor speed/spin")
    radial = np.array([np.cos(theta), np.sin(theta), 0.])
    tangential = np.array([-np.sin(theta), np.cos(theta), 0.])
    return mu*(Omega**2*radial-spin_sign*Omega_dot*tangential)


def integrate_phase(t, Omega, spin_sign, initial_phase):
    from scipy.integrate import cumulative_trapezoid
    if np.any(np.diff(t) <= 0):
        raise ValueError("Strictly increasing phase timeline required")
    return initial_phase+spin_sign*cumulative_trapezoid(Omega, t, initial=0)


class LocalModes:
    """Local displacement relative to rigid body, not total body motion.

    M=1 normalized per mode; input_gain combines forcing and modal participation.
    output_shape maps q_ddot into sensor local elastic acceleration.
    Neither gain nor mu is individually identifiable from this single observation.
    """
    def __init__(self, frequency_hz, damping_ratio, input_gain, output_shape):
        self.frequency = np.asarray(frequency_hz, dtype=float)
        self.damping = np.asarray(damping_ratio, dtype=float)
        self.input_gain = np.asarray(input_gain, dtype=float)
        self.output_shape = np.asarray(output_shape, dtype=float)
        n = len(self.frequency)
        if not all(np.all(np.isfinite(x)) for x in (self.frequency, self.damping, self.input_gain, self.output_shape)):
            raise ValueError("Finite explicit modal parameters required")
        if np.any(self.frequency <= 0) or np.any(self.damping < 0) or self.damping.shape != (n,):
            raise ValueError("Invalid modal frequencies/damping")
        if self.input_gain.ndim != 2 or self.input_gain.shape[0] != n or self.output_shape.shape != (3, n):
            raise ValueError("Invalid modal participation dimensions")
        wn = 2*np.pi*self.frequency
        self.A = np.block([[np.zeros((n, n)), np.eye(n)], [-np.diag(wn**2), -np.diag(2*self.damping*wn)]])
        self.B = np.vstack([np.zeros_like(self.input_gain), self.input_gain])
        self.n = n

    def simulate(self, t, force, initial=None):
        t, force = np.asarray(t), np.asarray(force)
        dt = np.diff(t)
        if len(t) < 2 or np.any(dt <= 0) or not np.allclose(dt, dt[0], rtol=1e-7):
            raise ValueError("Uniform increasing internal modal timeline required")
        if force.shape != (len(t), self.B.shape[1]) or not np.all(np.isfinite(force)):
            raise ValueError("Force timeline dimensions disagree")
        m = self.B.shape[1]
        augmented = np.block([[self.A, self.B], [np.zeros((m, 2*self.n+m))]])
        discrete = expm(augmented*dt[0])
        Ad, Bd = discrete[:2*self.n, :2*self.n], discrete[:2*self.n, 2*self.n:]
        x = np.zeros(2*self.n) if initial is None else np.asarray(initial).copy()
        if x.shape != (2*self.n,) or not np.all(np.isfinite(x)):
            raise ValueError("Finite explicit initial modal state required")
        states = np.empty((len(t), 2*self.n))
        accel = np.empty((len(t), 3))
        for k in range(len(t)):
            states[k] = x
            accel[k] = self.output_shape@(self.A@x+self.B@force[k])[self.n:]
            if k+1 < len(t):
                x = Ad@x+Bd@force[k]  # Exact ZOH discretization of linear modes.
        return states, accel


def checked_sample(t_internal, signal, t_observation, highest_retained_hz):
    from scipy.signal import butter, sosfiltfilt
    fs = 1/np.median(np.diff(t_internal))
    target = 1/np.median(np.diff(t_observation))
    if not 0 < highest_retained_hz < .45*min(fs, target):
        raise ValueError("Retained frequencies violate internal/observation Nyquist margin")
    filtered = sosfiltfilt(butter(6, highest_retained_hz, fs=fs, output="sos"), signal, axis=0)
    if t_observation[0] < t_internal[0] or t_observation[-1] > t_internal[-1]:
        raise ValueError("No observation extrapolation allowed")
    return np.column_stack([np.interp(t_observation, t_internal, col) for col in filtered.T])


def base_firmware_vibe(t, accel_raw):
    """Historical-base VIBE algorithm reference (not customized log reconstruction).

    5Hz first-order floor -> squared residual -> 2Hz first-order mean -> sqrt.
    Requires pre-log raw acceleration and original per-sensor dt, unavailable here.
    Initial filter states are explicit zero, matching a fresh reference start.
    """
    t, accel_raw = np.asarray(t), np.asarray(accel_raw)
    if len(t) < 2 or np.any(np.diff(t) <= 0) or accel_raw.shape != (len(t), 3):
        raise ValueError("Raw acceleration timeline required")
    floor, variance = np.zeros(3), np.zeros(3)
    values = np.empty_like(accel_raw, dtype=float)
    for i in range(len(t)):
        dt = t[1]-t[0] if i == 0 else t[i]-t[i-1]
        floor += dt/(dt+1/(2*np.pi*5))*(accel_raw[i]-floor)
        difference_squared = (accel_raw[i]-floor)**2
        variance += dt/(dt+1/(2*np.pi*2))*(difference_squared-variance)
        values[i] = np.sqrt(np.maximum(variance, 0))
    return values
