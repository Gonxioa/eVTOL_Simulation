import os
import csv
import time as _time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from scipy.spatial.transform import Rotation as R

from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.world import World

from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.wind.default_winds import NoWind, SinusoidWind
from control_allocation import (
    BoundedControlAllocator,
    predict_effective_response,
    scheduled_effectiveness,
    validate_effectiveness,
)
from hex_params import hex_params as quad_params
from provenance import code_hashes, runtime_info, sha256_file, stable_hash, write_json
from scenario_config import NUM_ROTORS, SCENARIOS, serializable_scenarios


def patch_rotorpy_for_redundant_multirotors():
    """
    RotorPy's dynamics and SE3 controller are parameterized by num_rotors, but
    this installed version still has two quadrotor assumptions:
    - allocation uses inverse(), while a hexarotor needs a pseudoinverse;
    - state packing is hardcoded to 16 + 4 states.
    Patch them process-locally so six rotors run without editing site-packages.
    """
    original_inv = getattr(np.linalg.inv, "_original_inv", np.linalg.inv)

    def safe_inv(a):
        arr = np.asarray(a)
        if arr.shape[-2] == arr.shape[-1]:
            return original_inv(arr)
        return np.linalg.pinv(arr)

    safe_inv._original_inv = original_inv
    np.linalg.inv = safe_inv

    def pack_state_dynamic(cls, state):
        rotor_speeds = np.asarray(state["rotor_speeds"])
        s = np.zeros((16 + rotor_speeds.size,))
        s[0:3] = state["x"]
        s[3:6] = state["v"]
        s[6:10] = state["q"]
        s[10:13] = state["w"]
        s[13:16] = state["wind"]
        s[16:] = rotor_speeds
        return s

    Multirotor._pack_state = classmethod(pack_state_dynamic)


patch_rotorpy_for_redundant_multirotors()

# =============================================================================
# ★ 全局配置
# =============================================================================

# ---------- 运行模式 ----------
# True  = 批量采集所有故障场景，自动导出 CSV（建议用于数据采集）
# False = 单场景调试，使用下方 SCENARIO / FAULT_* 参数
BATCH_MODE = os.getenv("HEX_BATCH_MODE", "1") != "0"

# ---------- 单场景参数（BATCH_MODE=False 时生效）----------
SCENARIO     = 'normal'   # 场景名，见 build_scenario()
FAULT_TIME   = 8.0              # 故障注入时刻 (s)
FAULT_MOTORS = []              # 失效电机编号列表
FAULT_FACTOR = 1.0              # 剩余推力比例 (0=完全失效, 0.8=轻微失效)

# ---------- 仿真参数 ----------
SIM_RATE  = int(os.getenv("HEX_SIM_RATE", "100"))
DURATION  = float(os.getenv("HEX_DURATION", "20.0"))
WARMUP    = float(os.getenv("HEX_WARMUP", "2.0"))  # 丢弃起飞瞬态的时间 (s)

# 正式数据不再是“每类一个CSV”。每类至少4次独立仿真，使4折分组评估
# 有真实独立来源可留出。旧 output/ 不会被覆盖。
PROJECT_ROOT = Path(__file__).resolve().parent
HEX_OUTPUT_DIR = Path(
    os.getenv("HEX_OUTPUT_DIR", str(PROJECT_ROOT / "output_hex_v3"))
).expanduser().resolve()
HEX_REPEATS = max(1, int(os.getenv("HEX_REPEATS", "4")))
HEX_BASE_SEED = int(os.getenv("HEX_BASE_SEED", "20260601"))
HEX_SAVE_PLOTS = os.getenv("HEX_SAVE_PLOTS", "0") == "1"
HEX_GENERATION_RESUME = os.getenv("HEX_GENERATION_RESUME", "1") != "0"

# ---------- 安全包络（仅用于统计输出，不终止仿真）----------
MAX_ROLL_DEG  = 75.0
MAX_PITCH_DEG = 75.0
MIN_Z, MAX_Z  = 0.1, 8.0
MAX_SPEED     = 15.0

# ---------- ★ IMU 噪声参数（参考 MPU-6000 数据手册）----------
# 加速度计：白噪声标准差 (m/s²)，随机游走标准差 (m/s²/step)
ACC_NOISE_STD  = 0.035          # 加速度计白噪声
ACC_BIAS_WALK  = 0.0002         # 加速度计随机游走
# 陀螺仪：白噪声标准差 (rad/s)，随机游走标准差 (rad/s/step)
GYRO_NOISE_STD = 0.003          # 陀螺仪白噪声
GYRO_BIAS_WALK = 0.000002       # 陀螺仪随机游走

# ---------- ★ 振动噪声参数（机架震动，填补 Domain Gap）----------
VIBRATION_STD  = 0.008          # 叠加在 IMU 上的高频振动噪声标准差 (m/s²)

# 场景标签、故障电机、严重度和效能因子只允许在 scenario_config.py 定义。
BATCH_SCENARIOS = [
    (item.name, item.fault_motors, item.simulated_factor, item.label)
    for item in SCENARIOS
]


# =============================================================================
# 故障注入载体
# =============================================================================
class FaultAwareSE3Control(SE3Control):
    """
    SE3 controller with conventional bounded nominal control allocation.

    The historical class name is retained for compatibility, but the
    allocator is intentionally *not* fault-aware.  The ground-truth fault
    schedule is emitted only as simulation-plant metadata; it is never used
    to compute motor commands.  Otherwise a diagnosis experiment would give
    the controller the answer in advance and suppress the fault signature.
    """
    # SE3Control.__init__ hardcodes kp_pos/kd_pos/kp_att/kd_att/kp_vel and
    # ignores whatever is in `params` for these keys (confirmed against
    # RotorPy 2.1.2 source). These are the outer-loop gains we expect to be
    # able to configure per-airframe; anything not in this set either isn't
    # exposed as an attribute by this RotorPy version, or doesn't affect the
    # cmd_motor_speeds path this project uses.
    _REQUIRED_GAIN_OVERRIDES = {
        "kp_pos": "se3_kp_pos",
        "kd_pos": "se3_kd_pos",
        "kp_att": "se3_kp_att",
        "kd_att": "se3_kd_att",
    }
    _OPTIONAL_GAIN_OVERRIDES = {
        "kp_vel": "se3_kp_vel",  # not used by cmd_motor_speeds; kept for completeness
    }

    def __init__(self, params, fault_time=None, fault_motor_indices=None, fault_factor=1.0):
        super().__init__(params)
        self._apply_gain_overrides(params)
        self.fault_time = fault_time
        self.fault_motor_indices = tuple(fault_motor_indices or ())
        self.fault_factor = float(fault_factor)

        # Validate the complete schedule at construction time.  Calling the
        # helper at t=0 also catches invalid indices/factors before a run can
        # create any output files.
        scheduled_effectiveness(
            0.0,
            self.num_rotors,
            self.fault_time,
            self.fault_motor_indices,
            self.fault_factor,
        )

        thrust_min = self.k_eta * float(self.rotor_speed_min) ** 2
        thrust_max = self.k_eta * float(self.rotor_speed_max) ** 2
        self.control_allocator = BoundedControlAllocator(
            self.f_to_TM,
            thrust_min=thrust_min,
            thrust_max=thrust_max,
            wrench_priority=params.get(
                "allocator_wrench_priority", np.array([2.0, 1.0, 1.0, 0.2])
            ),
            regularization=params.get("allocator_regularization", 0.0),
            solver_tolerance=params.get("allocator_solver_tolerance", 1e-10),
            feasibility_tolerance=params.get(
                "allocator_feasibility_tolerance", 1e-6
            ),
        )

    def _apply_gain_overrides(self, params):
        """Re-apply SE3 outer-loop gains from params, overriding whatever
        SE3Control.__init__ just hardcoded. Fails loudly instead of silently
        keeping RotorPy's defaults, so this bug class cannot silently recur."""
        applied = {}
        missing_keys = []
        for attr, key in self._REQUIRED_GAIN_OVERRIDES.items():
            if key not in params:
                missing_keys.append(key)
                continue
            value = params[key]
            value = np.asarray(value, dtype=float) if hasattr(value, "__len__") else float(value)
            setattr(self, attr, value)
            applied[attr] = value
        if missing_keys:
            raise ValueError(
                "FaultAwareSE3Control: missing required SE3 gain override key(s) "
                f"{missing_keys} in params. Add them to hex_params.py (see "
                "se3_kp_pos/se3_kd_pos/se3_kp_att/se3_kd_att) -- refusing to fall "
                "back to RotorPy's hardcoded defaults silently."
            )
        for attr, key in self._OPTIONAL_GAIN_OVERRIDES.items():
            if key in params and hasattr(self, attr):
                value = params[key]
                value = np.asarray(value, dtype=float) if hasattr(value, "__len__") else float(value)
                setattr(self, attr, value)
                applied[attr] = value

        # Verify every override actually stuck (catches typos in attribute
        # names, or a future RotorPy version that stops exposing one of these
        # as a plain instance attribute).
        for attr, expected in applied.items():
            actual = getattr(self, attr)
            if not np.allclose(actual, expected):
                raise RuntimeError(
                    f"FaultAwareSE3Control: override of '{attr}' did not take "
                    f"effect (expected {expected}, got {actual}). SE3Control's "
                    "internals may have changed -- do not trust this run's gains."
                )
        print(
            "  [增益校验] SE3外环增益已显式生效: "
            f"kp_pos={self.kp_pos}, kd_pos={self.kd_pos}, "
            f"kp_att={self.kp_att}, kd_att={self.kd_att}"
        )

    def update(self, t, state, flat_output):
        control_input = super().update(t, state, flat_output)

        desired_wrench = np.array([
            control_input['cmd_thrust'],
            control_input['cmd_moment'][0],
            control_input['cmd_moment'][1],
            control_input['cmd_moment'][2],
        ], dtype=float)

        # Problem D fix: every time step, including normal flight, uses the
        # same nonnegative, speed-bounded allocator.  No pseudoinverse result
        # can reach the vehicle with a negative thrust or excessive speed.
        allocation = self.control_allocator.allocate(desired_wrench)
        cmd_rotor_thrusts = allocation.commanded_thrusts
        cmd_motor_speeds = np.sqrt(
            np.maximum(cmd_rotor_thrusts, 0.0) / self.k_eta
        )

        # Simulation truth for the plant and for audit logging only.  It is
        # intentionally computed *after* nominal allocation and cannot affect
        # the motor command above.
        plant_effectiveness = scheduled_effectiveness(
            t,
            self.num_rotors,
            self.fault_time,
            self.fault_motor_indices,
            self.fault_factor,
        )
        predicted_plant_thrusts, predicted_plant_wrench = \
            predict_effective_response(
                self.f_to_TM, cmd_rotor_thrusts, plant_effectiveness
            )
        plant_residual = desired_wrench - predicted_plant_wrench

        control_input.update({
            'cmd_motor_thrusts': cmd_rotor_thrusts,
            'cmd_motor_speeds': cmd_motor_speeds,
            'allocator_desired_wrench': allocation.desired_wrench,
            'allocator_allocated_wrench': allocation.allocated_wrench,
            'allocator_residual_wrench': allocation.residual_wrench,
            'allocator_lower_bound_active': allocation.lower_bound_active,
            'allocator_upper_bound_active': allocation.upper_bound_active,
            'allocator_success': float(allocation.success),
            'allocator_wrench_feasible': float(allocation.wrench_feasible),
            'allocator_cost': float(allocation.cost),
            'allocator_weighted_residual_norm': float(
                allocation.weighted_residual_norm
            ),
            'allocator_bound_active_count': float(
                allocation.bound_active_count
            ),
            'allocator_used_unconstrained_solution': float(
                allocation.used_unconstrained_solution
            ),
            'plant_effectiveness': plant_effectiveness,
            'predicted_plant_motor_thrusts': predicted_plant_thrusts,
            'predicted_plant_wrench': predicted_plant_wrench,
            'plant_wrench_residual': plant_residual,
            'plant_weighted_residual_norm': float(
                self.control_allocator.weighted_residual_norm(plant_residual)
            ),
            'fault_active': float(np.any(plant_effectiveness < 1.0)),
        })
        return control_input


class FaultInjectionMultirotor(Multirotor):

    def __init__(self, params, initial_state=None,
                 fault_time=None, fault_motor_indices=None, fault_factor=0.5):
        super().__init__(params, initial_state)
        self.fault_time          = fault_time
        self.fault_motor_indices = tuple(fault_motor_indices or ())
        self.fault_factor        = float(fault_factor)
        self.fault_active        = False
        scheduled_effectiveness(
            0.0,
            self.num_rotors,
            self.fault_time,
            self.fault_motor_indices,
            self.fault_factor,
        )

    def get_cmd_motor_speeds(self, state, control):
        """Apply plant degradation to RotorPy's nominal motor command.

        Both ``step`` and ``statedot`` call this method, so the physical fault
        now has one consistent path.  The old ``step``-only implementation
        missed ``statedot`` and activated one integration step earlier than
        the controller's timestamp.
        """
        nominal_speeds = np.asarray(
            super().get_cmd_motor_speeds(state, control), dtype=float
        )
        if 'plant_effectiveness' not in control:
            raise KeyError(
                "FaultInjectionMultirotor requires control['plant_effectiveness'] "
                "to keep plant fault timing synchronized"
            )
        effectiveness = validate_effectiveness(
            control['plant_effectiveness'], self.num_rotors
        )
        if np.any(effectiveness < 1.0):
            expected = np.ones(self.num_rotors)
            expected[list(self.fault_motor_indices)] = self.fault_factor
            if not np.allclose(effectiveness, expected, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    "controller/vehicle plant-effectiveness schedules disagree"
                )
        return nominal_speeds * np.sqrt(effectiveness)

    def step(self, state, control, t_step):
        effectiveness = validate_effectiveness(
            control['plant_effectiveness'], self.num_rotors
        )
        if np.any(effectiveness < 1.0) and not self.fault_active:
            self.fault_active = True
            activation_time = (
                f"{float(self.fault_time):.3f}s"
                if self.fault_time is not None else "当前步"
            )
            print(f"  [故障注入] t={activation_time} | "
                  f"电机{self.fault_motor_indices} 推力→{self.fault_factor*100:.0f}%")
        return super().step(state, control, t_step)


# =============================================================================
# ★ IMU 噪声模拟（手动实现，兼容所有 RotorPy 版本）
# =============================================================================
class RealisticIMU:
    """
    模拟真实 IMU 的测量噪声：
      - 加速度计白噪声 + 随机游走偏差
      - 陀螺仪白噪声 + 随机游走偏差
      - 机架振动高频噪声
    """
    def __init__(self,
                 acc_noise_std=ACC_NOISE_STD,
                 acc_bias_walk=ACC_BIAS_WALK,
                 gyro_noise_std=GYRO_NOISE_STD,
                 gyro_bias_walk=GYRO_BIAS_WALK,
                 vibration_std=VIBRATION_STD,
                 rng=None):
        self.acc_noise_std  = acc_noise_std
        self.acc_bias_walk  = acc_bias_walk
        self.gyro_noise_std = gyro_noise_std
        self.gyro_bias_walk = gyro_bias_walk
        self.vibration_std  = vibration_std
        self.rng = rng if rng is not None else np.random.default_rng()
        # 初始偏差
        self.acc_bias  = np.zeros(3)
        self.gyro_bias = np.zeros(3)

    def measure(self, true_acc, true_gyro):
        """
        输入真实加速度 (m/s²) 和角速度 (rad/s)，返回带噪声的 IMU 读数。
        """
        # 随机游走更新偏差
        self.acc_bias  += self.rng.normal(0, self.acc_bias_walk,  3)
        self.gyro_bias += self.rng.normal(0, self.gyro_bias_walk, 3)

        # 加速度计：白噪声 + 偏差 + 振动
        acc_meas = (true_acc
                    + self.rng.normal(0, self.acc_noise_std,  3)
                    + self.acc_bias
                    + self.rng.normal(0, self.vibration_std,  3))

        # 陀螺仪：白噪声 + 偏差
        gyro_meas = (true_gyro
                     + self.rng.normal(0, self.gyro_noise_std, 3)
                     + self.gyro_bias)

        return acc_meas, gyro_meas


# =============================================================================
# ★ 湍流风场（SinusoidWind + 随机高频扰动）
# =============================================================================
class TurbulentWind:
    """
    在 SinusoidWind 基础上叠加随机高频扰动，模拟轻度大气湍流。
    接口与 RotorPy 风场类兼容。
    """
    def __init__(self, base_amplitude=0.5, turbulence_std=0.2, rng=None):
        self.base   = SinusoidWind()
        self.std    = turbulence_std
        self.amp    = base_amplitude
        self.rng    = rng if rng is not None else np.random.default_rng()

    def update(self, t, position):
        # 基础正弦风
        try:
            base_wind = self.base.update(t, position)
        except TypeError:
            base_wind = self.base.update(t)
        # 叠加随机湍流扰动
        turbulence = self.rng.normal(0, self.std, 3)
        return base_wind + turbulence


# =============================================================================
# 工具函数
# =============================================================================
def quat_to_euler(q):
    return R.from_quat([q[0], q[1], q[2], q[3]]).as_euler('xyz', degrees=False)


def make_init(pos, euler=None):
    n_rotors = quad_params['num_rotors']
    hover_speed = np.sqrt(quad_params['mass'] * 9.81 / (n_rotors * quad_params['k_eta']))
    quaternion = (
        R.from_euler('xyz', np.asarray(euler, dtype=float)).as_quat()
        if euler is not None else np.array([0., 0., 0., 1.])
    )
    return {
        'x'           : np.array(pos, dtype=float),
        'v'           : np.zeros(3),
        'q'           : quaternion,
        'w'           : np.zeros(3),
        'wind'        : np.zeros(3),
        'rotor_speeds': np.array([hover_speed] * n_rotors),
    }


def check_safety_envelope(euler_deg, z_pos, v_norm, t_arr, warmup=WARMUP):
    """只在 warmup 秒之后检查安全包络。"""
    mask = t_arr >= warmup
    if not np.any(mask):
        return False, None
    idxs = np.where(mask)[0]
    exceed = (
        (np.abs(euler_deg[idxs, 0]) > MAX_ROLL_DEG)  |
        (np.abs(euler_deg[idxs, 1]) > MAX_PITCH_DEG) |
        (z_pos[idxs] < MIN_Z) | (z_pos[idxs] > MAX_Z) |
        (v_norm[idxs] > MAX_SPEED)
    )
    if np.any(exceed):
        return True, int(idxs[np.where(exceed)[0][0]])
    return False, None


# =============================================================================
# ★ CSV 导出（含 IMU 噪声列）
# =============================================================================
def export_to_csv(results, imu_data, filename="flight_data.csv",
                  fault_label=0, fault_motor=-1, fault_factor=1.0,
                  warmup=WARMUP, output_dir=HEX_OUTPUT_DIR,
                  run_metadata=None):
    """
    导出 CSV，包含：
      基础状态列：time, x, y, z, vx, vy, vz, roll, pitch, yaw, p, q_body, r
      电机列：     cmd_motor{i}, actual_motor{i}, cmd_motor_thrust{i}
      分配审计列：期望/名义分配/故障后预测扳手、残差、边界状态
      ★ IMU 噪声列：ax, ay, az, gx, gy, gz
      标签列：     fault_label, fault_motor, fault_factor
    """
    if results is None:
        print("  [导出] 跳过（results=None）"); return

    t_key = 'time' if 'time' in results else ('t' if 't' in results else None)
    if t_key is None:
        print("  [导出] 跳过（无时间键）"); return

    run_metadata = dict(run_metadata or {})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / Path(filename).name

    t, state, ctrl = results[t_key], results['state'], results['control']
    euler = np.array([quat_to_euler(q) for q in state['q']])
    n_m   = state['rotor_speeds'].shape[1]

    required_allocator_keys = (
        'cmd_motor_thrusts',
        'allocator_allocated_wrench',
        'allocator_residual_wrench',
        'allocator_lower_bound_active',
        'allocator_upper_bound_active',
        'allocator_success',
        'allocator_wrench_feasible',
        'allocator_cost',
        'allocator_weighted_residual_norm',
        'allocator_bound_active_count',
        'plant_effectiveness',
        'predicted_plant_motor_thrusts',
        'predicted_plant_wrench',
        'plant_wrench_residual',
        'plant_weighted_residual_norm',
        'fault_active',
    )
    missing_allocator_keys = [
        key for key in required_allocator_keys if key not in ctrl
    ]
    if missing_allocator_keys:
        raise RuntimeError(
            "拒绝导出缺少分配器审计字段的数据: "
            f"{missing_allocator_keys}"
        )

    # ★ 从 warmup 秒开始截取数据，丢弃起飞瞬态
    start_idx = np.searchsorted(t, warmup)

    rows = []
    for i in range(start_idx, len(t)):
        row = {
            'time'        : round(t[i], 4),
            'x'           : state['x'][i, 0],
            'y'           : state['x'][i, 1],
            'z'           : state['x'][i, 2],
            'vx'          : state['v'][i, 0],
            'vy'          : state['v'][i, 1],
            'vz'          : state['v'][i, 2],
            'roll'        : euler[i, 0],
            'pitch'       : euler[i, 1],
            'yaw'         : euler[i, 2],
            'p'           : state['w'][i, 0],
            'q_body'      : state['w'][i, 1],
            'r'           : state['w'][i, 2],
        }
        cmd_thrust = ctrl.get('cmd_thrust')
        cmd_moment = ctrl.get('cmd_moment')
        row['cmd_thrust'] = float(cmd_thrust[i]) if cmd_thrust is not None else np.nan
        for axis, key in enumerate(('cmd_moment_x', 'cmd_moment_y', 'cmd_moment_z')):
            row[key] = float(cmd_moment[i, axis]) if cmd_moment is not None else np.nan

        allocated = ctrl['allocator_allocated_wrench'][i]
        allocation_residual = ctrl['allocator_residual_wrench'][i]
        predicted_plant = ctrl['predicted_plant_wrench'][i]
        plant_residual = ctrl['plant_wrench_residual'][i]
        for values, names in (
            (allocated, (
                'allocated_thrust', 'allocated_moment_x',
                'allocated_moment_y', 'allocated_moment_z',
            )),
            (allocation_residual, (
                'allocation_residual_thrust', 'allocation_residual_moment_x',
                'allocation_residual_moment_y', 'allocation_residual_moment_z',
            )),
            (predicted_plant, (
                'predicted_plant_thrust', 'predicted_plant_moment_x',
                'predicted_plant_moment_y', 'predicted_plant_moment_z',
            )),
            (plant_residual, (
                'plant_residual_thrust', 'plant_residual_moment_x',
                'plant_residual_moment_y', 'plant_residual_moment_z',
            )),
        ):
            for axis, name in enumerate(names):
                row[name] = float(values[axis])

        row['allocator_success'] = int(ctrl['allocator_success'][i] >= 0.5)
        row['allocator_wrench_feasible'] = int(
            ctrl['allocator_wrench_feasible'][i] >= 0.5
        )
        row['allocator_cost'] = float(ctrl['allocator_cost'][i])
        row['allocator_weighted_residual_norm'] = float(
            ctrl['allocator_weighted_residual_norm'][i]
        )
        row['plant_weighted_residual_norm'] = float(
            ctrl['plant_weighted_residual_norm'][i]
        )
        row['allocator_bound_active_count'] = int(
            round(float(ctrl['allocator_bound_active_count'][i]))
        )

        # 电机列
        for m in range(n_m):
            row[f'cmd_motor{m}']    = ctrl['cmd_motor_speeds'][i, m]
            row[f'actual_motor{m}'] = state['rotor_speeds'][i, m]
            row[f'cmd_motor_thrust{m}'] = ctrl['cmd_motor_thrusts'][i, m]
            row[f'predicted_plant_motor_thrust{m}'] = \
                ctrl['predicted_plant_motor_thrusts'][i, m]
            row[f'effectiveness_motor{m}'] = \
                ctrl['plant_effectiveness'][i, m]
            row[f'allocator_lower_active_motor{m}'] = int(
                ctrl['allocator_lower_bound_active'][i, m]
            )
            row[f'allocator_upper_active_motor{m}'] = int(
                ctrl['allocator_upper_bound_active'][i, m]
            )

        # ★ IMU 噪声列
        row['ax'] = imu_data['acc'][i, 0]
        row['ay'] = imu_data['acc'][i, 1]
        row['az'] = imu_data['acc'][i, 2]
        row['gx'] = imu_data['gyro'][i, 0]
        row['gy'] = imu_data['gyro'][i, 1]
        row['gz'] = imu_data['gyro'][i, 2]

        # 标签列
        row['fault_label']  = fault_label
        row['fault_motor']  = fault_motor
        row['fault_factor'] = fault_factor
        row['fault_time']   = run_metadata.get('fault_time', np.nan)
        row['scenario_name'] = run_metadata.get('scenario_name', '')
        row['run_id'] = int(run_metadata.get('run_id', -1))
        row['random_seed'] = int(run_metadata.get('random_seed', -1))
        row['trajectory_id'] = run_metadata.get('trajectory_id', '')
        # fault_label is run/scenario metadata and remains constant for group
        # splitting. These two columns make the per-sample timing explicit;
        # data_process.py still uses only the post-fault stable segment.
        row['sample_fault_active'] = int(ctrl['fault_active'][i] >= 0.5)
        row['sample_fault_label'] = (
            int(fault_label) if row['sample_fault_active'] else 0
        )

        rows.append(row)

    if not rows:
        print("  [导出] 跳过（无有效数据行）"); return

    def _write(path):
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    try:
        _write(filepath)
        print(f"  [导出] → {filepath}  ({len(rows)} 行)")
    except PermissionError as exc:
        raise PermissionError(
            f"无法原子写入 {filepath}；请关闭占用该文件的程序后重试，"
            "正式数据禁止静默改名。"
        ) from exc


# =============================================================================
# ★ IMU 数据后处理：对整段结果生成噪声 IMU 序列
# =============================================================================
def generate_imu_data(results, imu: RealisticIMU):
    """
    遍历仿真结果，为每个时间步生成带噪声的 IMU 读数。
    真实加速度用状态导数近似，角速度直接取 state['w']。
    """
    t_key = 'time' if 'time' in results else 't'
    t     = results[t_key]
    state = results['state']
    n     = len(t)

    acc_out  = np.zeros((n, 3))
    gyro_out = np.zeros((n, 3))

    # 用速度差分近似加速度（加上重力补偿，模拟 IMU 比力）
    v = state['v']
    g_vec = np.array([0., 0., 9.81])  # 重力（世界系）

    for i in range(n):
        if i == 0:
            true_acc = g_vec.copy()
        else:
            dt = t[i] - t[i-1]
            if dt < 1e-9:
                dt = 1.0 / SIM_RATE
            dv = (v[i] - v[i-1]) / dt
            true_acc = dv + g_vec   # IMU 测量的是比力（含重力）

        true_gyro = state['w'][i]
        acc_out[i], gyro_out[i] = imu.measure(true_acc, true_gyro)

    return {'acc': acc_out, 'gyro': gyro_out}


# =============================================================================
# 绘图工具
# =============================================================================
def plot_motor_speeds(results, fault_time=None, title="电机转速", save_path=None):
    t_key = 'time' if 'time' in results else 't'
    t = results[t_key]
    cmd = results['control']['cmd_motor_speeds']
    act = results['state']['rotor_speeds']
    n_m = cmd.shape[1]
    n_cols = min(3, n_m)
    n_rows = int(np.ceil(n_m / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    fig.suptitle(title)
    axes_flat = np.atleast_1d(axes).flat
    for i, ax in enumerate(axes_flat):
        if i >= n_m:
            ax.axis('off')
            continue
        ax.plot(t, cmd[:, i], 'b--', lw=1.2, label='指令')
        ax.plot(t, act[:, i], 'r-', lw=1.2, label='实际')
        if fault_time:
            ax.axvline(fault_time, color='k', ls=':', lw=1.5, label='故障')
        ax.set_title(f'Motor {i}');
        ax.set_xlabel('t (s)')
        ax.set_ylabel('rad/s');
        ax.legend(fontsize=8);
        ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # 必须关闭 fig 释放内存，否则批量运行时会内存泄漏
    else:
        plt.show()


def plot_euler(results, fault_time=None, title="姿态角", save_path=None):
    t_key = 'time' if 'time' in results else 't'
    t = results[t_key]
    euler_deg = np.rad2deg(np.array([quat_to_euler(q) for q in results['state']['q']]))
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for i, (ax, lbl) in enumerate(zip(axes, ['Roll°', 'Pitch°', 'Yaw°'])):
        ax.plot(t, euler_deg[:, i], lw=1.2)
        if fault_time:
            ax.axvline(fault_time, color='r', ls='--', lw=1)
        ax.set_ylabel(lbl);
        ax.grid(alpha=0.3)
    axes[0].set_title(title);
    axes[-1].set_xlabel('t (s)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_imu(imu_data, t, title="IMU 噪声数据", save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    labels_acc = [r'ax (m/s$^2$)', r'ay (m/s$^2$)', r'az (m/s$^2$)']  # 顺便修复了之前上标2报错的问题
    labels_gyro = ['gx (rad/s)', 'gy (rad/s)', 'gz (rad/s)']
    for j, lbl in enumerate(labels_acc):
        axes[0].plot(t, imu_data['acc'][:, j], lw=0.8, label=lbl)
    for j, lbl in enumerate(labels_gyro):
        axes[1].plot(t, imu_data['gyro'][:, j], lw=0.8, label=lbl)
    axes[0].set_title(title);
    axes[0].legend(fontsize=8);
    axes[0].grid(alpha=0.3)
    axes[1].legend(fontsize=8);
    axes[1].grid(alpha=0.3)
    axes[-1].set_xlabel('t (s)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# =============================================================================
# 场景构建
# =============================================================================
def make_run_variation(random_seed):
    """Create bounded nuisance variation for one independent simulation."""
    rng = np.random.default_rng(int(random_seed))
    return {
        "trajectory_id": f"seed_{int(random_seed)}",
        "A": float(rng.uniform(1.30, 1.70)),
        "B": float(rng.uniform(1.30, 1.70)),
        "delta": float(np.pi / 2 + rng.uniform(-0.18, 0.18)),
        "x_offset": float(5.0 + rng.uniform(-0.25, 0.25)),
        "y_offset": float(5.0 + rng.uniform(-0.25, 0.25)),
        "height": float(2.5 + rng.uniform(-0.15, 0.15)),
        "init_offset": rng.uniform(-0.12, 0.12, size=3),
        "init_euler": np.deg2rad(rng.uniform(-2.0, 2.0, size=3)),
        "wind_std": float(rng.uniform(0.10, 0.22)),
    }


def build_scenario(fault_motors=None, fault_factor=1.0, fault_time=FAULT_TIME,
                   use_wind=True, run_variation=None, wind_rng=None):
    """
    统一构建场景。
    fault_motors=[]  且 fault_factor=1.0 → 正常飞行
    否则            → 故障注入
    use_wind=True   → 使用湍流风场（更真实）
    """
    if fault_motors is None:
        fault_motors = []

    world = World({'bounds': {'extents': [0, 15, 0, 15, 0, 10]}, 'blocks': []})
    ctrl  = FaultAwareSE3Control(
        quad_params,
        fault_time=fault_time,
        fault_motor_indices=fault_motors,
        fault_factor=fault_factor,
    )

    variation = run_variation or make_run_variation(HEX_BASE_SEED)
    # 始终使用同一类8字形任务，但让幅值、相位、位置和初始状态小幅变化。
    # 这些变化与标签无关，用来构造真正独立的重复运行。
    traj = TwoDLissajous(
        A=variation["A"], B=variation["B"],
        a=1, b=2,
        delta=variation["delta"],
        x_offset=variation["x_offset"],
        y_offset=variation["y_offset"],
        height=variation["height"],
        yaw_bool=False,
    )

    init_pos = np.array([6.5, 5.0, variation["height"]]) + variation["init_offset"]
    init_pos[2] = max(1.5, init_pos[2])
    init = make_init(init_pos, variation["init_euler"])

    # 正常飞行 vs 故障飞行
    if len(fault_motors) == 0 or fault_factor >= 1.0:
        vehicle = Multirotor(quad_params, init)
        actual_fault_time = None
    else:
        vehicle = FaultInjectionMultirotor(
            quad_params, init,
            fault_time=fault_time,
            fault_motor_indices=fault_motors,
            fault_factor=fault_factor)
        actual_fault_time = fault_time

    # ★ 湍流风场
    wind = (
        TurbulentWind(
            base_amplitude=0.5,
            turbulence_std=variation["wind_std"],
            rng=wind_rng,
        )
        if use_wind else NoWind()
    )

    return vehicle, ctrl, traj, wind, world, DURATION, actual_fault_time


# =============================================================================
# 单次仿真
# =============================================================================
def run_one(fault_motors, fault_factor, fault_label, scenario_name,
            plot=True, save_plots=False, use_wind=True, run_id=1,
            random_seed=HEX_BASE_SEED, variation_seed=None,
            output_dir=HEX_OUTPUT_DIR):

    print(f"\n  场景: {scenario_name}  |  电机: {fault_motors}  "
          f"|  推力比: {fault_factor:.0%}  |  label={fault_label}")

    variation = make_run_variation(
        random_seed if variation_seed is None else variation_seed
    )
    wind_rng = np.random.default_rng(int(random_seed) + 100_000)
    imu_rng = np.random.default_rng(int(random_seed) + 200_000)
    vehicle, ctrl, traj, wind, world, duration, fault_time = \
        build_scenario(
            fault_motors,
            fault_factor,
            use_wind=use_wind,
            run_variation=variation,
            wind_rng=wind_rng,
        )

    env = Environment(
        vehicle       = vehicle,
        controller    = ctrl,
        trajectory    = traj,
        world         = world,
        wind_profile  = wind,
        sim_rate      = SIM_RATE,
        imu           = None,   # 我们自己实现 IMU 噪声，不用 RotorPy 内置
        mocap         = None,
        estimator     = None,
        safety_margin = 0.0,
    )

    results = env.run(
        t_final        = duration,
        use_mocap      = False,
        terminate      = False,
        plot           = False,   # 批量模式下关闭自动绘图
        plot_mocap     = False,
        plot_estimator = False,
        plot_imu       = False,
        animate_bool   = False,
        animate_wind   = False,
        verbose        = False,
        fname          = None,
    )

    if results is None:
        print("  [错误] 仿真返回 None"); return

    t_key = 'time' if 'time' in results else ('t' if 't' in results else None)
    if t_key is None:
        print(f"  [错误] 无时间键"); return

    t         = results[t_key]
    x_hist    = results['state']['x']
    v_hist    = results['state']['v']
    euler_rad = np.array([quat_to_euler(q) for q in results['state']['q']])
    euler_deg = np.rad2deg(euler_rad)
    v_norm    = np.linalg.norm(v_hist, axis=1)

    # 安全包络（从 warmup 后检查）
    exceeded, idx = check_safety_envelope(euler_deg, x_hist[:, 2], v_norm, t)
    if exceeded:
        print(f"  [WARN] 安全包络违反 @ t={t[idx]:.2f}s  "
              f"roll={euler_deg[idx,0]:.1f}°  z={x_hist[idx,2]:.2f}m")
    else:
        print(f"  [OK] 全程安全  |  终点={np.round(x_hist[-1], 2)}")

    # ★ 生成带噪声的 IMU 数据
    imu_sensor = RealisticIMU(rng=imu_rng)
    imu_data   = generate_imu_data(results, imu_sensor)

    # 导出 CSV
    fault_motor_id = fault_motors[0] if len(fault_motors) > 0 else -1
    export_to_csv(
        results, imu_data,
        filename=f"{scenario_name}__run{int(run_id):03d}.csv",
        fault_label=fault_label,
        fault_motor=fault_motor_id,
        fault_factor=fault_factor,
        output_dir=output_dir,
        run_metadata={
            "scenario_name": scenario_name,
            "run_id": int(run_id),
            "random_seed": int(random_seed),
            "trajectory_id": variation["trajectory_id"],
            "fault_time": fault_time if fault_time is not None else np.nan,
        },
    )

    # 绘图（单场景显示 或 批量自动保存）
    if plot or save_plots:
        save_dir = None
        if save_plots:
            # 在 output 目录下新建 plots 文件夹，并按场景名创建子文件夹
            save_dir = os.path.join(
                str(output_dir), "plots", scenario_name, f"run_{int(run_id):03d}"
            )
            os.makedirs(save_dir, exist_ok=True)

        plot_euler(results, fault_time, f"[{scenario_name}] 姿态角",
                   save_path=os.path.join(save_dir, "euler.png") if save_plots else None)

        plot_imu(imu_data, t, f"[{scenario_name}] IMU 噪声数据",
                 save_path=os.path.join(save_dir, "imu.png") if save_plots else None)

        plot_motor_speeds(results, fault_time, f"[{scenario_name}] 电机转速",
                save_path=os.path.join(save_dir, "motors.png") if save_plots else None)


def existing_run_matches(path, scenario_name, run_id, label, random_seed):
    """Fully validate an existing CSV before generation-resume skips it.

    A first-row metadata check is insufficient: an interrupted write can leave
    a perfectly plausible header and first row.  Export is now atomic, and the
    resume gate additionally checks every row plus the expected time coverage.
    """
    try:
        required_columns = {
            "time", "scenario_name", "run_id", "fault_label", "random_seed",
            "allocator_success", "predicted_plant_thrust",
            *[f"cmd_motor{i}" for i in range(NUM_ROTORS)],
            *[f"effectiveness_motor{i}" for i in range(NUM_ROTORS)],
        }
        row_count = 0
        first_time = None
        previous_time = None
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                return False
            for row in reader:
                current_time = float(row["time"])
                if not np.isfinite(current_time):
                    return False
                if previous_time is not None and current_time <= previous_time:
                    return False
                if (
                    row.get("scenario_name") != scenario_name
                    or int(row.get("run_id", -1)) != int(run_id)
                    or int(row.get("fault_label", -1)) != int(label)
                    or int(row.get("random_seed", -1)) != int(random_seed)
                    or int(row.get("allocator_success", 0)) != 1
                ):
                    return False
                if first_time is None:
                    first_time = current_time
                previous_time = current_time
                row_count += 1

        expected_rows = int(round((DURATION - WARMUP) * SIM_RATE)) + 1
        time_tolerance = max(1e-4, 0.51 / SIM_RATE)
        return (
            row_count == expected_rows
            and first_time is not None
            and abs(first_time - WARMUP) <= time_tolerance
            and abs(previous_time - DURATION) <= time_tolerance
        )
    except Exception:
        return False


# =============================================================================
# 主程序
# =============================================================================
def main():
    sep = '=' * 60

    if BATCH_MODE:
        # ── 批量模式：自动遍历所有场景 ─────────────────────────────
        print(f"\n{sep}")
        print(
            f"  批量采集模式  |  {len(BATCH_SCENARIOS)} 个场景 × "
            f"{HEX_REPEATS} 次独立运行"
        )
        print(f"  风场: 湍流  |  IMU噪声: 开启  |  数据截取: t>={WARMUP}s")
        print(f"  输出目录: {HEX_OUTPUT_DIR}")
        print(sep)

        t0 = _time.time()
        HEX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        generation_plan = {
            "dataset_version": "hex_v3_bounded_nominal_allocator",
            "repeats_per_class": HEX_REPEATS,
            "base_seed": HEX_BASE_SEED,
            "sim_rate_hz": SIM_RATE,
            "duration_s": DURATION,
            "warmup_s": WARMUP,
            "fault_time_s": FAULT_TIME,
            "scenarios": serializable_scenarios(),
            "code_sha256": code_hashes(
                PROJECT_ROOT,
                [
                    "main.py", "scenario_config.py", "hex_params.py",
                    "hex_physical_characteristics.py", "control_allocation.py",
                ],
            ),
        }
        generation_plan["plan_signature"] = stable_hash(generation_plan)
        plan_path = HEX_OUTPUT_DIR / "generation_plan.json"
        if plan_path.exists():
            old_plan = __import__("json").loads(plan_path.read_text(encoding="utf-8"))
            if old_plan.get("plan_signature") != generation_plan["plan_signature"]:
                raise RuntimeError(
                    "输出目录中的generation_plan.json与当前代码/参数不一致。"
                    "请改用新的HEX_OUTPUT_DIR，禁止混合续跑。"
                )
        else:
            if any(HEX_OUTPUT_DIR.glob("*.csv")):
                raise RuntimeError(
                    "输出目录已有CSV但没有generation_plan.json，无法证明其配置来源。"
                    "请改用新的HEX_OUTPUT_DIR。"
                )
            write_json(plan_path, generation_plan)
        completed_manifest_path = HEX_OUTPUT_DIR / "dataset_generation_manifest.json"
        if completed_manifest_path.exists():
            completed_manifest = __import__("json").loads(
                completed_manifest_path.read_text(encoding="utf-8")
            )
            if completed_manifest.get("plan_signature") != generation_plan["plan_signature"]:
                raise RuntimeError(
                    "既有数据集清单与当前生成计划不一致，禁止续跑。"
                )
            recorded_hashes = completed_manifest.get("csv_sha256", {})
            changed = [
                name for name, digest in recorded_hashes.items()
                if not (HEX_OUTPUT_DIR / name).is_file()
                or sha256_file(HEX_OUTPUT_DIR / name) != digest
            ]
            if changed:
                raise RuntimeError(
                    f"既有CSV在完成清单之后缺失或被修改: {changed}"
                )
        expected_names = {
            f"{name}__run{run_id:03d}.csv"
            for run_id in range(1, HEX_REPEATS + 1)
            for name, _motors, _factor, _label in BATCH_SCENARIOS
        }
        if completed_manifest_path.exists() and set(recorded_hashes) != expected_names:
            raise RuntimeError(
                "既有完成清单的CSV集合与当前期望集合不一致，禁止续跑。"
            )
        extra_csv = sorted(path.name for path in HEX_OUTPUT_DIR.glob("*.csv") if path.name not in expected_names)
        if extra_csv:
            raise RuntimeError(
                f"输出目录含不属于当前配置的CSV: {extra_csv}。"
                "请改用新的HEX_OUTPUT_DIR，避免不同数据版本混合。"
            )
        for run_id in range(1, HEX_REPEATS + 1):
            print(f"\n--- 独立运行 {run_id}/{HEX_REPEATS} ---")
            for name, motors, factor, label in BATCH_SCENARIOS:
                # 每个文件独立随机流；公式固定，便于复现且避免标签共享噪声。
                seed = HEX_BASE_SEED + run_id * 10_000 + label * 101
                variation_seed = HEX_BASE_SEED + run_id * 10_000
                destination = HEX_OUTPUT_DIR / f"{name}__run{run_id:03d}.csv"
                if destination.exists():
                    if HEX_GENERATION_RESUME and existing_run_matches(
                        destination, name, run_id, label, seed
                    ):
                        print(f"  [续跑] 已存在且元数据匹配，跳过 {destination.name}")
                        continue
                    raise RuntimeError(
                        f"{destination} 已存在但元数据不匹配或禁止续跑。"
                        "请改用新的HEX_OUTPUT_DIR，避免覆盖。"
                    )
                run_one(
                    motors,
                    factor,
                    label,
                    name,
                    plot=False,
                    save_plots=HEX_SAVE_PLOTS,
                    use_wind=True,
                    run_id=run_id,
                    random_seed=seed,
                    variation_seed=variation_seed,
                    output_dir=HEX_OUTPUT_DIR,
                )

        csv_files = sorted(HEX_OUTPUT_DIR.glob("*.csv"))
        if {path.name for path in csv_files} != expected_names:
            raise RuntimeError(
                "生成循环结束但CSV集合不完整；拒绝写入完成清单。"
            )
        incomplete_or_invalid = []
        for run_id in range(1, HEX_REPEATS + 1):
            for name, _motors, _factor, label in BATCH_SCENARIOS:
                seed = HEX_BASE_SEED + run_id * 10_000 + label * 101
                path = HEX_OUTPUT_DIR / f"{name}__run{run_id:03d}.csv"
                if not existing_run_matches(path, name, run_id, label, seed):
                    incomplete_or_invalid.append(path.name)
        if incomplete_or_invalid:
            raise RuntimeError(
                "生成循环结束但以下CSV未通过完整性门禁；拒绝写入完成清单: "
                f"{incomplete_or_invalid}"
            )
        manifest = {
            **runtime_info(),
            **generation_plan,
            "output_dir": str(HEX_OUTPUT_DIR),
            "repeats_per_class": HEX_REPEATS,
            "base_seed": HEX_BASE_SEED,
            "sim_rate_hz": SIM_RATE,
            "duration_s": DURATION,
            "warmup_s": WARMUP,
            "fault_time_s": FAULT_TIME,
            "scenarios": serializable_scenarios(),
            "code_sha256": code_hashes(
                PROJECT_ROOT,
                [
                    "main.py",
                    "scenario_config.py",
                    "hex_params.py",
                    "hex_physical_characteristics.py",
                    "control_allocation.py",
                ],
            ),
            "csv_sha256": {
                path.name: sha256_file(path) for path in csv_files
            },
        }
        write_json(HEX_OUTPUT_DIR / "dataset_generation_manifest.json", manifest)

        elapsed = _time.time() - t0
        print(f"\n{sep}")
        print(f"  批量采集完成  |  耗时 {elapsed:.1f}s")
        print(f"  CSV 文件位于: {HEX_OUTPUT_DIR}")
        print("  下一步先运行 validate_hex_data.py，不要直接训练")
        print(sep)

    else:
        # ── 单场景调试模式 ────────────────────────────────────────────────
        print(f"\n{sep}")
        print(f"  单场景模式  |  {SCENARIO}")
        print(f"  电机: {FAULT_MOTORS}  推力比: {FAULT_FACTOR:.0%}  "
              f"故障时刻: t={FAULT_TIME}s")
        print(sep)

        # 根据参数自动匹配 label
        label = 0
        for name, motors, factor, lbl in BATCH_SCENARIOS:
            if motors == FAULT_MOTORS and abs(factor - FAULT_FACTOR) < 0.01:
                label = lbl; break

        run_one(
            fault_motors  = FAULT_MOTORS,
            fault_factor  = FAULT_FACTOR,
            fault_label   = label,
            scenario_name = SCENARIO,
            plot          = True,
            use_wind      = True,
            run_id        = 1,
            random_seed   = HEX_BASE_SEED,
            variation_seed= HEX_BASE_SEED,
            output_dir    = HEX_OUTPUT_DIR,
        )

        print(f"\n{sep}  完成\n{sep}\n")


if __name__ == "__main__":
    main()
