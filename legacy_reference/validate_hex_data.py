"""Validate repeated hexarotor CSVs before preprocessing or ML training."""

from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from hex_physical_characteristics import HEX_PHYSICAL_CHARACTERISTICS
from provenance import runtime_info, sha256_file, stable_hash, write_json
from scenario_config import NUM_ROTORS, SCENARIO_BY_LABEL, SCENARIOS


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("HEX_OUTPUT_DIR", PROJECT_ROOT / "output_hex_v3")).resolve()
REPORT_DIR = Path(
    os.getenv("HEX_VALIDATION_DIR", PROJECT_ROOT / "plots" / "hex" / "data_validation")
).resolve()
EXPECTED_RATE = float(os.getenv("HEX_SIM_RATE", "100"))
EXPECTED_DURATION = float(os.getenv("HEX_DURATION", "20.0"))
EXPECTED_WARMUP = float(os.getenv("HEX_WARMUP", "2.0"))
EXPECTED_ROWS = int(round((EXPECTED_DURATION - EXPECTED_WARMUP) * EXPECTED_RATE)) + 1
MIN_REPEATS = max(1, int(os.getenv("HEX_MIN_GROUPS_PER_CLASS", "4")))
MAX_ROLL_PITCH_DEG = 75.0
ROTOR_MAX = float(HEX_PHYSICAL_CHARACTERISTICS["rotor_speed_max"])
ROTOR_MIN = float(HEX_PHYSICAL_CHARACTERISTICS["rotor_speed_min"])
K_ETA = float(HEX_PHYSICAL_CHARACTERISTICS["k_eta"])
MAX_THRUST = K_ETA * ROTOR_MAX**2
BOUND_ATOL = max(1e-8, 1e-7 * max(1.0, MAX_THRUST))


def _allocation_matrix():
    params = HEX_PHYSICAL_CHARACTERISTICS
    positions = params["rotor_pos"]
    moment_arms = np.hstack([
        np.cross(positions[key], np.array([0.0, 0.0, 1.0]))
        .reshape(-1, 1)[0:2]
        for key in positions
    ])
    yaw = (
        float(params["k_m"]) / float(params["k_eta"])
        * np.asarray(params["rotor_directions"], dtype=float)
    ).reshape(1, -1)
    return np.vstack((np.ones((1, NUM_ROTORS)), moment_arms, yaw))


ALLOCATION_MATRIX = _allocation_matrix()
WRENCH_SCALE = np.sum(np.abs(ALLOCATION_MATRIX) * MAX_THRUST, axis=1)


def inspect_file(
    path,
    expected_rate=EXPECTED_RATE,
    expected_duration=EXPECTED_DURATION,
    expected_warmup=EXPECTED_WARMUP,
):
    df = pd.read_csv(path)
    expected_rate = float(expected_rate)
    expected_duration = float(expected_duration)
    expected_warmup = float(expected_warmup)
    expected_rows = int(
        round((expected_duration - expected_warmup) * expected_rate)
    ) + 1
    required = [
        "time", "roll", "pitch", "z", "vx", "vy", "vz", "cmd_thrust",
        "cmd_moment_x", "cmd_moment_y", "cmd_moment_z",
        "fault_label", "fault_motor", "fault_factor", "fault_time",
        "scenario_name", "run_id", "random_seed", "sample_fault_active",
        "sample_fault_label", "trajectory_id", "allocator_success",
        "allocator_wrench_feasible", "allocator_cost",
        "allocator_weighted_residual_norm", "plant_weighted_residual_norm",
        "allocator_bound_active_count",
    ]
    for prefix in (
        "allocated", "allocation_residual", "predicted_plant", "plant_residual"
    ):
        required.extend([
            f"{prefix}_thrust", f"{prefix}_moment_x",
            f"{prefix}_moment_y", f"{prefix}_moment_z",
        ])
    for motor in range(NUM_ROTORS):
        required.extend([
            f"cmd_motor{motor}", f"actual_motor{motor}",
            f"cmd_motor_thrust{motor}",
            f"predicted_plant_motor_thrust{motor}",
            f"effectiveness_motor{motor}",
            f"allocator_lower_active_motor{motor}",
            f"allocator_upper_active_motor{motor}",
        ])
    missing = [name for name in required if name not in df]
    if missing:
        return {"file": path.name, "status": "FAIL", "reason": f"missing={missing}"}
    label_values = df["fault_label"].unique()
    scenario_values = df["scenario_name"].unique()
    if len(label_values) != 1 or len(scenario_values) != 1:
        return {"file": path.name, "status": "FAIL", "reason": "mixed label/scenario"}
    label, scenario_name = int(label_values[0]), str(scenario_values[0])
    if label not in SCENARIO_BY_LABEL or scenario_name != SCENARIO_BY_LABEL[label].name:
        return {"file": path.name, "status": "FAIL", "reason": "scenario config mismatch"}
    expected_scenario = SCENARIO_BY_LABEL[label]
    constant_columns = ("fault_motor", "fault_factor", "run_id", "random_seed", "trajectory_id")
    if any(len(df[name].dropna().unique()) != 1 for name in constant_columns):
        return {"file": path.name, "status": "FAIL", "reason": "mixed run metadata"}
    if (
        int(df["fault_motor"].iloc[0]) != expected_scenario.fault_motor
        or not np.isclose(
            float(df["fault_factor"].iloc[0]),
            expected_scenario.simulated_factor,
            rtol=0.0,
            atol=1e-12,
        )
    ):
        return {"file": path.name, "status": "FAIL", "reason": "fault metadata mismatch"}

    # fault_time is intentionally empty for Normal and finite for fault runs.
    # Do not mistake the semantic "no fault" marker for corrupt numeric data.
    finite_columns = [
        name for name in df.select_dtypes(include=[np.number]).columns
        if name != "fault_time"
    ]
    if not np.isfinite(df[finite_columns].to_numpy()).all():
        return {"file": path.name, "status": "FAIL", "reason": "NaN/Inf in required numeric data"}
    fault_times = df["fault_time"].dropna().unique()
    if label == 0 and len(fault_times) != 0:
        return {"file": path.name, "status": "FAIL", "reason": "Normal must not contain fault_time"}
    if label != 0 and (
        len(fault_times) != 1 or not np.isfinite(float(fault_times[0]))
    ):
        return {"file": path.name, "status": "FAIL", "reason": "fault run needs one finite fault_time"}

    time_values = df["time"].to_numpy(float)
    dt = np.diff(time_values)
    if len(dt) == 0 or not np.all(dt > 0.0):
        return {"file": path.name, "status": "FAIL", "reason": "time not strictly increasing"}
    time_tolerance = max(1e-4, 0.51 / expected_rate)
    if (
        len(df) != expected_rows
        or abs(float(time_values[0]) - expected_warmup) > time_tolerance
        or abs(float(time_values[-1]) - expected_duration) > time_tolerance
    ):
        return {
            "file": path.name,
            "status": "FAIL",
            "reason": (
                "truncated/unexpected time coverage: "
                f"rows={len(df)} expected={expected_rows}, "
                f"time=[{time_values[0]}, {time_values[-1]}]"
            ),
        }
    speed = np.linalg.norm(df[["vx", "vy", "vz"]].to_numpy(float), axis=1)
    roll_pitch = np.rad2deg(np.abs(df[["roll", "pitch"]].to_numpy(float)))
    command = df[[f"cmd_motor{i}" for i in range(NUM_ROTORS)]].to_numpy(float)
    command_thrust = df[
        [f"cmd_motor_thrust{i}" for i in range(NUM_ROTORS)]
    ].to_numpy(float)
    effectiveness = df[
        [f"effectiveness_motor{i}" for i in range(NUM_ROTORS)]
    ].to_numpy(float)
    predicted_motor_thrust = df[
        [f"predicted_plant_motor_thrust{i}" for i in range(NUM_ROTORS)]
    ].to_numpy(float)
    lower_active = df[
        [f"allocator_lower_active_motor{i}" for i in range(NUM_ROTORS)]
    ].to_numpy(int).astype(bool)
    upper_active = df[
        [f"allocator_upper_active_motor{i}" for i in range(NUM_ROTORS)]
    ].to_numpy(int).astype(bool)

    if np.any(command < ROTOR_MIN - 1e-7) or np.any(command > ROTOR_MAX + 1e-7):
        return {"file": path.name, "status": "FAIL", "reason": "motor command outside physical bounds"}
    if np.any(command_thrust < -1e-8) or np.any(command_thrust > MAX_THRUST + 1e-7):
        return {"file": path.name, "status": "FAIL", "reason": "motor thrust outside physical bounds"}
    if not np.allclose(command_thrust, K_ETA * command**2, rtol=1e-8, atol=1e-8):
        return {"file": path.name, "status": "FAIL", "reason": "speed/thrust conversion mismatch"}
    if np.any(effectiveness < 0.0) or np.any(effectiveness > 1.0):
        return {"file": path.name, "status": "FAIL", "reason": "effectiveness outside [0,1]"}
    if not np.allclose(
        predicted_motor_thrust,
        effectiveness * command_thrust,
        rtol=1e-8,
        atol=1e-8,
    ):
        return {"file": path.name, "status": "FAIL", "reason": "predicted plant thrust mismatch"}

    expected_effectiveness = np.ones((len(df), NUM_ROTORS))
    if label != 0:
        motor = int(df["fault_motor"].iloc[0])
        fault_time_value = float(df["fault_time"].dropna().iloc[0])
        active = time_values >= fault_time_value
        expected_effectiveness[active, motor] = float(df["fault_factor"].iloc[0])
    else:
        active = np.zeros(len(df), dtype=bool)
    if not np.allclose(effectiveness, expected_effectiveness, rtol=0.0, atol=1e-12):
        return {"file": path.name, "status": "FAIL", "reason": "effectiveness timing/value mismatch"}

    expected_sample_active = active.astype(int)
    expected_sample_label = np.where(active, label, 0)
    if not np.array_equal(df["sample_fault_active"].to_numpy(int), expected_sample_active):
        return {"file": path.name, "status": "FAIL", "reason": "sample_fault_active mismatch"}
    if not np.array_equal(df["sample_fault_label"].to_numpy(int), expected_sample_label):
        return {"file": path.name, "status": "FAIL", "reason": "sample_fault_label mismatch"}
    if not np.all(df["allocator_success"].to_numpy(int) == 1):
        return {"file": path.name, "status": "FAIL", "reason": "allocator solver failure"}

    desired = df[
        ["cmd_thrust", "cmd_moment_x", "cmd_moment_y", "cmd_moment_z"]
    ].to_numpy(float)
    allocated_export = df[
        ["allocated_thrust", "allocated_moment_x", "allocated_moment_y", "allocated_moment_z"]
    ].to_numpy(float)
    allocation_residual_export = df[
        [
            "allocation_residual_thrust", "allocation_residual_moment_x",
            "allocation_residual_moment_y", "allocation_residual_moment_z",
        ]
    ].to_numpy(float)
    predicted_export = df[
        [
            "predicted_plant_thrust", "predicted_plant_moment_x",
            "predicted_plant_moment_y", "predicted_plant_moment_z",
        ]
    ].to_numpy(float)
    plant_residual_export = df[
        [
            "plant_residual_thrust", "plant_residual_moment_x",
            "plant_residual_moment_y", "plant_residual_moment_z",
        ]
    ].to_numpy(float)
    allocated = command_thrust @ ALLOCATION_MATRIX.T
    predicted = predicted_motor_thrust @ ALLOCATION_MATRIX.T
    allocation_residual = desired - allocated
    plant_residual = desired - predicted
    diagnostic_pairs = (
        (allocated_export, allocated, "allocated wrench"),
        (allocation_residual_export, allocation_residual, "allocation residual"),
        (predicted_export, predicted, "predicted plant wrench"),
        (plant_residual_export, plant_residual, "plant residual"),
    )
    for exported, recomputed, name in diagnostic_pairs:
        if not np.allclose(exported, recomputed, rtol=1e-7, atol=1e-7):
            return {"file": path.name, "status": "FAIL", "reason": f"{name} identity mismatch"}

    expected_lower = np.isclose(command_thrust, 0.0, rtol=0.0, atol=BOUND_ATOL)
    expected_upper = np.isclose(
        command_thrust, MAX_THRUST, rtol=0.0, atol=BOUND_ATOL
    )
    if not np.array_equal(lower_active, expected_lower):
        return {"file": path.name, "status": "FAIL", "reason": "lower-bound flags mismatch"}
    if not np.array_equal(upper_active, expected_upper):
        return {"file": path.name, "status": "FAIL", "reason": "upper-bound flags mismatch"}
    expected_bound_count = np.count_nonzero(lower_active | upper_active, axis=1)
    if not np.array_equal(
        df["allocator_bound_active_count"].to_numpy(int), expected_bound_count
    ):
        return {"file": path.name, "status": "FAIL", "reason": "bound-active count mismatch"}

    upper_bound_fraction = float(np.mean(upper_active))
    rows_with_upper_bound = float(np.mean(np.any(upper_active, axis=1)))
    lower_bound_fraction = float(np.mean(lower_active))
    normalized_allocation_residual = np.linalg.norm(
        allocation_residual / WRENCH_SCALE[None, :], axis=1
    )
    normalized_plant_residual = np.linalg.norm(
        plant_residual / WRENCH_SCALE[None, :], axis=1
    )
    post_mask = active if label != 0 else np.ones(len(df), dtype=bool)
    post_allocation_p95 = float(np.percentile(normalized_allocation_residual[post_mask], 95))
    post_plant_median = float(np.median(normalized_plant_residual[post_mask]))
    post_plant_p95 = float(np.percentile(normalized_plant_residual[post_mask], 95))

    attenuation_ratio = np.nan
    attenuation_expected = np.nan
    if label != 0:
        motor = int(df["fault_motor"].iloc[0])
        fault_time = float(df["fault_time"].dropna().iloc[0])
        mask = time_values > fault_time + 1.0
        valid = mask & (command[:, motor] > 0.05 * ROTOR_MAX)
        if np.any(valid):
            actual = df[f"actual_motor{motor}"].to_numpy(float)
            attenuation_ratio = float(np.median(actual[valid] / command[valid, motor]))
            attenuation_expected = float(np.sqrt(float(df["fault_factor"].iloc[0])))

    warnings = []
    if abs(float(np.median(dt)) - 1.0 / expected_rate) > 0.002:
        warnings.append("unexpected dt")
    if roll_pitch.max() > MAX_ROLL_PITCH_DEG:
        warnings.append("attitude envelope")
    if rows_with_upper_bound > 0.10:
        warnings.append("rows with upper motor bound >10%")
    if post_allocation_p95 > 0.05:
        warnings.append("nominal allocation residual p95 >5%")
    if post_plant_p95 > 0.20:
        warnings.append("post-fault predicted plant residual p95 >20%")
    if np.mean(df["allocator_wrench_feasible"].to_numpy(int) == 0) > 0.10:
        warnings.append("requested wrench infeasible >10%")
    if np.isfinite(attenuation_ratio) and abs(attenuation_ratio - attenuation_expected) > 0.25:
        warnings.append("fault attenuation mismatch")
    return {
        "file": path.name, "status": "WARN" if warnings else "OK",
        "reason": "; ".join(warnings), "label": label,
        "scenario": scenario_name, "run_id": int(df["run_id"].iloc[0]),
        "seed": int(df["random_seed"].iloc[0]), "rows": len(df),
        "duration_s": float(df["time"].iloc[-1] - df["time"].iloc[0]),
        "median_dt_s": float(np.median(dt)), "max_roll_pitch_deg": float(roll_pitch.max()),
        "min_z_m": float(df["z"].min()), "max_z_m": float(df["z"].max()),
        "max_speed_m_s": float(speed.max()),
        "upper_bound_motor_fraction": upper_bound_fraction,
        "rows_with_any_upper_bound": rows_with_upper_bound,
        "lower_bound_motor_fraction": lower_bound_fraction,
        "post_nominal_allocation_normalized_residual_p95": post_allocation_p95,
        "post_plant_normalized_residual_median": post_plant_median,
        "post_plant_normalized_residual_p95": post_plant_p95,
        "fault_speed_ratio": attenuation_ratio, "expected_fault_speed_ratio": attenuation_expected,
    }


def main():
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"{DATA_DIR} 中没有CSV")
    manifest_path = DATA_DIR / "dataset_generation_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"缺少 {manifest_path.name}；数据集可能尚未完整生成，拒绝正式验证。"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != "hex_v3_bounded_nominal_allocator":
        raise SystemExit(
            "数据集版本不是 hex_v3_bounded_nominal_allocator，拒绝混用旧分配逻辑。"
        )
    plan_path = DATA_DIR / "generation_plan.json"
    if not plan_path.exists():
        raise SystemExit("缺少 generation_plan.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_signature = plan.get("plan_signature")
    unsigned_plan = dict(plan)
    unsigned_plan.pop("plan_signature", None)
    if (
        plan_signature is None
        or stable_hash(unsigned_plan) != plan_signature
        or manifest.get("plan_signature") != plan_signature
    ):
        raise SystemExit("generation_plan.json签名无效或与完成清单不一致")
    expected_hashes = manifest.get("csv_sha256")
    if not isinstance(expected_hashes, dict):
        raise SystemExit("dataset_generation_manifest.json 缺少 csv_sha256")
    actual_names = {path.name for path in files}
    if actual_names != set(expected_hashes):
        missing_from_disk = sorted(set(expected_hashes) - actual_names)
        untracked = sorted(actual_names - set(expected_hashes))
        raise SystemExit(
            "CSV集合与生成清单不一致；"
            f"磁盘缺失={missing_from_disk}，清单外文件={untracked}"
        )
    hash_mismatches = [
        path.name
        for path in files
        if sha256_file(path) != expected_hashes[path.name]
    ]
    if hash_mismatches:
        raise SystemExit(f"CSV哈希与生成清单不一致: {hash_mismatches}")
    manifest_rate = float(manifest["sim_rate_hz"])
    manifest_duration = float(manifest["duration_s"])
    manifest_warmup = float(manifest["warmup_s"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records = [
        inspect_file(
            path,
            expected_rate=manifest_rate,
            expected_duration=manifest_duration,
            expected_warmup=manifest_warmup,
        )
        for path in files
    ]
    table = pd.DataFrame(records)
    table.to_csv(REPORT_DIR / "validation_report.csv", index=False, encoding="utf-8-sig")
    html = (
        "<html><head><meta charset='utf-8'><title>Hex data validation</title></head><body>"
        "<h1>六旋翼仿真数据质量审查</h1>" + table.to_html(index=False) + "</body></html>"
    )
    (REPORT_DIR / "validation_report.html").write_text(html, encoding="utf-8")

    valid = [r for r in records if r["status"] != "FAIL"]
    label_counts = Counter(int(r["label"]) for r in valid)
    missing = sorted(set(item.label for item in SCENARIOS) - set(label_counts))
    too_few = {label: n for label, n in label_counts.items() if n < MIN_REPEATS}
    summary = {
        **runtime_info(), "data_dir": str(DATA_DIR), "files": len(files),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "csv_hashes_verified": True,
        "status_counts": dict(Counter(r["status"] for r in records)),
        "valid_files_per_label": dict(label_counts), "missing_labels": missing,
        "labels_below_minimum_repeats": too_few, "minimum_repeats": MIN_REPEATS,
    }
    write_json(REPORT_DIR / "validation_summary.json", summary)
    print(table[["file", "status", "reason"]].to_string(index=False))
    print(f"\n报告: {REPORT_DIR / 'validation_report.html'}")
    if any(r["status"] == "FAIL" for r in records) or missing or too_few:
        raise SystemExit("数据未达到正式预处理条件；请先处理FAIL/缺失/重复次数不足。")
    print("结构与独立重复次数检查通过；WARN项需结合报告说明后再决定是否纳入。")


if __name__ == "__main__":
    main()
