"""Strict observation-only packet and shared feature white lists.

Time and validity control buffering; neither is an input feature. Frame FLU,
observed Hamilton xyzw body-to-ENU quaternion; command means rad/s, not RPM.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = 'diagnostic_packet_v1'
IMU = ['ax_body', 'ay_body', 'az_body', 'gx_body', 'gy_body', 'gz_body']
ROTATION = ['R11', 'R21', 'R31', 'R12', 'R22', 'R32']
COMMAND = [f'cmd_motor{i}' for i in range(6)]
FEATURE_SETS = {'imu_only': IMU, 'imu_attitude': IMU+ROTATION,
                'imu_attitude_command': IMU+ROTATION+COMMAND}
QUATERNION = ['observed_qx', 'observed_qy', 'observed_qz', 'observed_qw']


def array(value, n, name):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError('Invalid observable ' + name)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ObservationPacket:
    timestamp: float
    specific_force_body: np.ndarray
    gyro_body: np.ndarray
    attitude_xyzw: np.ndarray | None = None
    motor_command_rad_s: np.ndarray | None = None
    valid: bool = True
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA or not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError('Invalid diagnostic packet schema or time')
        object.__setattr__(self, 'specific_force_body', array(self.specific_force_body, 3, 'specific force'))
        object.__setattr__(self, 'gyro_body', array(self.gyro_body, 3, 'gyro'))
        if self.attitude_xyzw is not None:
            q = array(self.attitude_xyzw, 4, 'observed attitude')
            if abs(np.linalg.norm(q)-1) > 1e-6:
                raise ValueError('Observed attitude must be a unit quaternion')
            object.__setattr__(self, 'attitude_xyzw', q)
        if self.motor_command_rad_s is not None:
            command = array(self.motor_command_rad_s, 6, 'motor command')
            if np.any(command < 0):
                raise ValueError('Unsigned motor command required')
            object.__setattr__(self, 'motor_command_rad_s', command)
        if type(self.valid) is not bool:
            raise ValueError('Validity must be a boolean')


def from_observation(observation, command):
    # This constructor explicitly selects observation fields; no plant state,
    # reference, schedule, label, group ID or measured rotor speed is accepted.
    return ObservationPacket(float(observation['time']), observation['specific_force_body'],
                             observation['gyro_body'], observation['q'], command)


def from_csv_row(row):
    q = [row[n] for n in QUATERNION] if all(n in row for n in QUATERNION) else None
    cmd = [row[n] for n in COMMAND] if all(n in row for n in COMMAND) else None
    return ObservationPacket(float(row['time']), [row[n] for n in IMU[:3]],
                             [row[n] for n in IMU[3:]], q, cmd,
                             bool(int(row['data_valid'])), str(row['packet_schema']))


def extract(packet, feature_set='imu_only'):
    if not isinstance(packet, ObservationPacket):
        raise TypeError('Diagnostic feature input must be an ObservationPacket')
    if feature_set not in FEATURE_SETS:
        raise ValueError('Unknown diagnostic feature set')
    parts = [packet.specific_force_body, packet.gyro_body]
    if feature_set != 'imu_only':
        if packet.attitude_xyzw is None:
            raise ValueError('Feature set requires observed attitude')
        rotation = Rotation.from_quat(packet.attitude_xyzw).as_matrix()
        parts += [rotation[:, 0], rotation[:, 1]]
    if feature_set == 'imu_attitude_command':
        if packet.motor_command_rad_s is None:
            raise ValueError('Feature set requires motor commands')
        parts.append(packet.motor_command_rad_s)
    return np.concatenate(parts).astype(np.float32)


def packets_from_frame(frame):
    return [from_csv_row(row) for row in frame.to_dict('records')]
