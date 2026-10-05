"""Time-aware segments and offline anti-aliased observations.

All filters here belong to analysis, not a claim to recover customized firmware.
They are acausal and must not be used as an online fault detector.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from .data import Record, author_imu_average
from .dynamics import rpm_to_rad_s


CHANNELS = ["PARM", "IMU_0", "IMU_1", "IMU_2", "ESC_0", "ESC_1", "ESC_2", "ESC_3", "ESC_4", "ESC_5",
            "RCOU", "ATT", "ARM", "MODE", "XKF1_0"]


def interpolate(tab, fields, t, angular=False):
    src = tab["TimeUS"]*1e-6
    if len(src) < 2 or np.any(np.diff(src) <= 0):
        raise ValueError("Interpolation requires ordered unique timestamps")
    value = np.column_stack([tab[k] for k in fields]).astype(float)
    if angular:
        value = np.unwrap(np.deg2rad(value), axis=0)
    return np.column_stack([np.interp(t, src, col, left=np.nan, right=np.nan) for col in value.T])


def support(tab, t, max_gap):
    src = tab["TimeUS"]*1e-6
    idx = np.searchsorted(src, t, side="right")-1
    safe = np.clip(idx, 0, len(src)-2)
    return (t >= src[0]) & (t < src[-1]) & ((src[safe+1]-src[safe]) <= max_gap)


def pieces(mask, t, maximum_gap_s):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    breaks = np.flatnonzero((np.diff(idx) != 1) | (np.diff(t[idx]) > maximum_gap_s))+1
    return [part for part in np.split(idx, breaks) if len(part) > 1]


def segments(data_dir, name, config, vibration=False, role="development"):
    r = Record(data_dir / name, CHANNELS, role=role)
    t, y, imu_each = author_imu_average(r)
    params = r.parameters()
    mapping = {int(v)-33: f"C{k[5:k.index('_')]}" for k, v in params.items()
               if k.startswith("SERVO") and k.endswith("_FUNCTION") and 33 <= v <= 38}
    if set(mapping) != set(range(6)):
        raise ValueError("Cannot establish six motor PWM mapping")
    pwm_tab = r.table("RCOU")
    pwm = interpolate(pwm_tab, [mapping[i] for i in range(6)], t)
    esc_tabs = [r.table(f"ESC_{i}") for i in range(6)]
    rpm = np.column_stack([interpolate(tab, ["RPM"], t)[:, 0] for tab in esc_tabs])
    voltage = np.column_stack([interpolate(tab, ["Volt"], t)[:, 0] for tab in esc_tabs])
    att_tab = r.table("ATT")
    angles = interpolate(att_tab, ["Roll", "Pitch", "Yaw"], t, angular=True)
    vel_tab = r.table("XKF1_0")
    velocity = interpolate(vel_tab, ["VN", "VE", "VD"], t)
    # The authors' sync_zoh_exa.m uses mean PWM >=1450; this is only a powered
    # candidate rule, not proof of airborne/hover state or fault onset.
    mask = (np.mean(pwm, axis=1) >= 1450) & np.all(rpm > 0, axis=1)
    mask &= support(pwm_tab, t, .1) & support(att_tab, t, .1) & support(vel_tab, t, .1)
    for tab in esc_tabs:
        mask &= support(tab, t, config["max_rpm_gap_s"])
    # Before the first ARM event state remains unknown, explicitly recorded.
    arm = r.table("ARM")
    ia = np.searchsorted(arm["TimeUS"]*1e-6, t, side="right")-1
    mask &= (ia < 0) | (arm["ArmState"][np.maximum(ia, 0)] == 1)
    rate = config["vibration_rate_hz"] if vibration else config["low_rate_hz"]
    result = []
    for idx in pieces(mask, t, .03):
        ti = t[idx]
        trim = config["edge_trim_s"]
        if ti[-1]-ti[0] < config["segment_min_s"]+2*trim:
            continue
        native_hz = int(round(1/np.median(np.diff(ti))))
        if vibration and config["vibration_lowpass_hz"] >= .45*min(native_hz, rate):
            raise ValueError(f"Vibration sampling inadequate: {name}, native {native_hz}Hz")
        tu = np.arange(ti[0], ti[-1], 1/native_hz)
        source = np.column_stack([y[idx], rpm_to_rad_s(rpm[idx]), voltage[idx], angles[idx], velocity[idx]])
        dense = np.column_stack([np.interp(tu, ti, col) for col in source.T])
        cutoff = config["vibration_lowpass_hz"] if vibration else config["lowpass_hz"]
        # Interpolation at native log rate is a jitter regularization approximation;
        # filtering occurs BEFORE decimation. Data gaps were split beforehand.
        dense = sosfiltfilt(butter(config["filter_order"], cutoff, fs=native_hz, output="sos"), dense, axis=0)
        if vibration:
            dense[:, :6] = sosfiltfilt(butter(config["filter_order"], config["vibration_highpass_hz"],
                                             fs=native_hz, output="sos", btype="highpass"), dense[:, :6], axis=0)
            dense[:, 6:] = sosfiltfilt(butter(config["filter_order"], config["lowpass_hz"],
                                              fs=native_hz, output="sos"), dense[:, 6:], axis=0)
        target_hz = int(rate)
        resampled = resample_poly(dense, target_hz, native_hz, axis=0)
        tout = tu[0]+np.arange(len(resampled))/rate
        keep = (tout >= tu[0]+trim) & (tout <= tu[-1]-trim)
        z = resampled[keep]
        if len(z) < config["segment_min_s"]*rate:
            continue
        result.append({"t": tout[keep], "imu": z[:, :6], "Omega": z[:, 6:12], "voltage": z[:, 12:18],
                       "angles": z[:, 18:21], "velocity": z[:, 21:24],
                       "selection": {"start_s": float(tout[keep][0]), "end_s": float(tout[keep][-1]),
                                      "native_log_rate_regularized_hz": native_hz,
                                      "arm_state_unknown_fraction": float(np.mean(ia[idx] < 0)),
                                      "imu_source": "exact_timestamp_intersection_three_IMU_mean",
                                      "firmware_filter_reconstruction": "not_recovered",
                                      "flight_state": "powered_candidate_not_certified_hover"}})
    if not result:
        raise ValueError(f"No valid contiguous powered segments in {name}; do not fabricate data")
    return result
