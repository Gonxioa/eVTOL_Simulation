"""Leakage-safe preprocessing for repeated hexarotor simulations.

Formal rules: one CSV is one class source, while the 19 class files sharing a
``run_id`` form one paired nuisance-condition group.  A fold therefore holds
out the same trajectory/initial-condition realization across every class.
Fault CSVs use only their post-fault stable interval; windows overlap by 50%;
raw windows are authoritative and scaling is fold-local.
"""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from provenance import code_hashes, runtime_info, sha256_file, stable_hash, write_json
from scenario_config import LABEL_NAMES, NUM_ROTORS, SCENARIO_BY_LABEL, SCENARIOS


PROJECT_ROOT = Path(__file__).resolve().parent


def _path_env(name, default):
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else Path(default).resolve()


DATA_DIR = _path_env("HEX_OUTPUT_DIR", PROJECT_ROOT / "output_hex_v3")
FEATURE_SET = os.getenv("FEATURE_SET", "combined").strip().lower()
SAVE_DIR = _path_env(
    "DATA_PROCESS_DIR_OVERRIDE",
    PROJECT_ROOT / "plots" / "hex" / "data_process" / FEATURE_SET,
)
WINDOW = max(2, int(os.getenv("HEX_WINDOW", "100")))
STEP = max(1, int(os.getenv("HEX_STEP", "50")))
BUFFER = max(0.0, float(os.getenv("HEX_FAULT_BUFFER", "1.0")))
MIN_GROUPS_PER_CLASS = max(2, int(os.getenv("HEX_MIN_GROUPS_PER_CLASS", "4")))
ALLOW_INCOMPLETE = os.getenv("HEX_ALLOW_INCOMPLETE", "0") == "1"

STATE_COLS = ["roll", "pitch", "yaw", "p", "q_body", "r"]
IMU_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]
COMMAND_COLS = [f"cmd_motor{i}" for i in range(NUM_ROTORS)]
FEATURE_SETS = {
    "imu": IMU_COLS,
    "state": STATE_COLS,
    "command": COMMAND_COLS,
    "response": STATE_COLS + IMU_COLS,
    "combined": STATE_COLS + IMU_COLS + COMMAND_COLS,
}
if FEATURE_SET not in FEATURE_SETS:
    raise ValueError(f"FEATURE_SET={FEATURE_SET!r} 无效；可选: {sorted(FEATURE_SETS)}")
FEATURE_COLS = FEATURE_SETS[FEATURE_SET]


def sliding_windows(values, target_thrust, window=WINDOW, step=STEP):
    windows, thrust_windows = [], []
    for start in range(0, len(values) - window + 1, step):
        stop = start + window
        windows.append(values[start:stop])
        thrust_windows.append(target_thrust[start:stop])
    if not windows:
        return (
            np.empty((0, window, values.shape[1]), dtype=np.float32),
            np.empty((0, window), dtype=np.float32),
        )
    return np.asarray(windows, dtype=np.float32), np.asarray(thrust_windows, dtype=np.float32)


def _constant_value(df, column, path):
    values = df[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{path.name}: {column} 必须在单个CSV内保持唯一，实际={values}")
    return values[0]


def process_source(path, matched_start):
    df = pd.read_csv(path)
    required = FEATURE_COLS + [
        "time", "cmd_thrust", "predicted_plant_thrust", "fault_label",
        "fault_motor", "fault_factor", "scenario_name", "run_id",
        "random_seed", "trajectory_id",
    ]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: 缺少正式数据列 {missing}")

    label = int(_constant_value(df, "fault_label", path))
    scenario_name = str(_constant_value(df, "scenario_name", path))
    run_id = int(_constant_value(df, "run_id", path))
    group_id = run_id
    random_seed = int(_constant_value(df, "random_seed", path))
    trajectory_id = str(_constant_value(df, "trajectory_id", path))
    fault_motor = int(_constant_value(df, "fault_motor", path))
    fault_factor = float(_constant_value(df, "fault_factor", path))
    if label not in SCENARIO_BY_LABEL:
        raise ValueError(f"{path.name}: 未定义的 fault_label={label}")
    expected = SCENARIO_BY_LABEL[label]
    if (
        scenario_name != expected.name
        or fault_motor != expected.fault_motor
        or not np.isclose(fault_factor, expected.simulated_factor, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            f"{path.name}: CSV元数据与 scenario_config.py 不一致: "
            f"label={label}, scenario={scenario_name}, motor={fault_motor}, "
            f"factor={fault_factor}"
        )

    numeric = df[
        FEATURE_COLS + ["cmd_thrust", "predicted_plant_thrust", "time"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path.name}: 正式输入列含 NaN/Inf")
    if not np.all(np.diff(df["time"].to_numpy(dtype=float)) > 0):
        raise ValueError(f"{path.name}: time 不是严格递增")

    if label == 0:
        selected = df.loc[df["time"] > matched_start]
        segment = f"matched_normal_t_gt_{matched_start:g}"
    else:
        fault_times = df["fault_time"].dropna().unique() if "fault_time" in df else []
        if len(fault_times) != 1:
            raise ValueError(f"{path.name}: 故障文件必须提供唯一 fault_time")
        stable_start = float(fault_times[0]) + BUFFER
        selected_start = max(stable_start, matched_start)
        selected = df.loc[df["time"] > selected_start]
        segment = f"post_fault_matched_t_gt_{selected_start:g}"

    values = selected[FEATURE_COLS].to_numpy(dtype=np.float32)
    desired_thrust = selected["cmd_thrust"].to_numpy(dtype=np.float32)
    physics_thrust = selected["predicted_plant_thrust"].to_numpy(dtype=np.float32)
    X, physics_thrust_windows = sliding_windows(values, physics_thrust)
    _, desired_thrust_windows = sliding_windows(values, desired_thrust)
    if len(X) == 0:
        raise ValueError(f"{path.name}: 选定区间只有{len(selected)}行，小于窗口{WINDOW}")

    n = len(X)
    arrays = {
        "X": X,
        "y": np.full(n, label, dtype=np.int64),
        "groups": np.full(n, group_id, dtype=np.int64),
        # Keep both quantities distinct. cmd_thrust is the SE3 controller's
        # request; physics_thrust is what eta*k_eta*omega_cmd^2 predicts the
        # degraded plant can actually receive and is the valid GAN target.
        "cmd_thrust": desired_thrust_windows,
        "physics_thrust": physics_thrust_windows,
        "fault_motor": np.full(n, fault_motor, dtype=np.int16),
        "fault_factor": np.full(n, fault_factor, dtype=np.float32),
        "run_id": np.full(n, run_id, dtype=np.int16),
        "random_seed": np.full(n, random_seed, dtype=np.int64),
        "source_file": np.full(n, path.name, dtype=f"<U{max(1, len(path.name))}"),
    }
    record = {
        "group_id": int(group_id), "source_file": path.name,
        "source_sha256": sha256_file(path), "scenario_name": scenario_name,
        "label": label, "label_name": LABEL_NAMES[label], "run_id": run_id,
        "random_seed": random_seed, "fault_motor": fault_motor,
        "fault_factor": fault_factor, "trajectory_id": trajectory_id,
        "selected_segment": segment,
        "physics_target_column": "predicted_plant_thrust",
        "rows_selected": int(len(selected)), "windows": n,
    }
    return arrays, record


def _validate_group_coverage(y, groups):
    counts = {
        int(label): int(np.unique(groups[y == label]).size)
        for label in sorted(np.unique(y))
    }
    missing_labels = sorted(set(LABEL_NAMES) - set(counts))
    too_few = {label: count for label, count in counts.items() if count < MIN_GROUPS_PER_CLASS}
    if (missing_labels or too_few) and not ALLOW_INCOMPLETE:
        raise RuntimeError(
            "数据不是正式分组数据集。"
            f"缺失类别={missing_labels}，独立组不足={too_few}；"
            f"每类至少需要{MIN_GROUPS_PER_CLASS}个CSV。先运行 main.py，"
            "或仅排错时显式设置 HEX_ALLOW_INCOMPLETE=1。"
        )
    return counts


def _save_overview(y, group_counts):
    labels = sorted(np.unique(y))
    names = [LABEL_NAMES[int(label)] for label in labels]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    axes[0].bar(names, [int(np.sum(y == label)) for label in labels], color="#4472c4")
    axes[0].set_title("Window count per class")
    axes[0].set_ylabel("windows")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(names, [group_counts[int(label)] for label in labels], color="#70ad47")
    axes[1].axhline(MIN_GROUPS_PER_CLASS, color="red", linestyle="--", label="formal minimum")
    axes[1].set_title("Independent source groups per class")
    axes[1].set_ylabel("CSV groups")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    fig.savefig(SAVE_DIR / "data_overview.png", dpi=150)
    plt.close(fig)


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"{DATA_DIR} 中没有CSV。正式数据需先运行 main.py。")
    manifest_path = DATA_DIR / "dataset_generation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"缺少 {manifest_path}；数据集可能未完整生成，禁止预处理。"
        )
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("dataset_version") != "hex_v3_bounded_nominal_allocator":
        raise RuntimeError("拒绝预处理非 hex_v3_bounded_nominal_allocator 数据")
    expected_hashes = dataset_manifest.get("csv_sha256", {})
    if set(expected_hashes) != {path.name for path in files}:
        raise RuntimeError("CSV集合与dataset_generation_manifest.json不一致")
    changed_files = [
        path.name for path in files
        if sha256_file(path) != expected_hashes[path.name]
    ]
    if changed_files:
        raise RuntimeError(f"CSV在生成清单之后发生变化: {changed_files}")
    generation_plan_path = DATA_DIR / "generation_plan.json"
    if not generation_plan_path.exists():
        raise FileNotFoundError(
            f"缺少 {generation_plan_path}；不能确定跨类别匹配的故障时刻。"
        )
    generation_plan = json.loads(generation_plan_path.read_text(encoding="utf-8"))
    recorded_plan_signature = generation_plan.get("plan_signature")
    unsigned_plan = dict(generation_plan)
    unsigned_plan.pop("plan_signature", None)
    if (
        recorded_plan_signature is None
        or stable_hash(unsigned_plan) != recorded_plan_signature
        or recorded_plan_signature != dataset_manifest.get("plan_signature")
    ):
        raise RuntimeError(
            "generation_plan.json自身签名无效或与完成清单不一致"
        )
    matched_start = float(dataset_manifest["fault_time_s"]) + BUFFER

    print("=" * 68)
    print("六旋翼正式预处理：独立来源分组 + 单一标签 + Fold内缩放")
    print(f"输入: {DATA_DIR}")
    print(f"特征方案: {FEATURE_SET} ({len(FEATURE_COLS)}维) = {FEATURE_COLS}")
    print(f"窗口={WINDOW}, 步长={STEP}, 重叠率={1 - STEP / WINDOW:.0%}")
    print(
        f"所有类别统一使用 t>{matched_start:g}s；"
        "故障类同时满足故障后稳定缓冲，避免轨迹阶段混杂"
    )
    print("=" * 68)

    buckets = defaultdict(list)
    source_records = []
    for path in files:
        arrays, record = process_source(path, matched_start)
        for key, value in arrays.items():
            buckets[key].append(value)
        source_records.append(record)
        print(
            f"[OK] {path.name:<34} label={record['label']:>2} "
            f"group={record['group_id']:>2} windows={record['windows']:>3}"
        )

    expected_labels = set(LABEL_NAMES)
    for group_id in sorted({record["group_id"] for record in source_records}):
        group_records = [
            record for record in source_records if record["group_id"] == group_id
        ]
        group_labels = [record["label"] for record in group_records]
        trajectory_ids = {record["trajectory_id"] for record in group_records}
        if (
            len(group_labels) != len(set(group_labels))
            or set(group_labels) != expected_labels
            or len(trajectory_ids) != 1
        ):
            raise RuntimeError(
                f"run_id={group_id} 不是完整的一次19类配对实验；"
                f"labels={sorted(group_labels)}, trajectory_ids={sorted(trajectory_ids)}"
            )

    merged = {key: np.concatenate(values, axis=0) for key, values in buckets.items()}
    X_raw = merged.pop("X").astype(np.float32, copy=False)
    y = merged.pop("y").astype(np.int64, copy=False)
    groups = merged["groups"].astype(np.int64, copy=False)
    group_counts = _validate_group_coverage(y, groups)

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_raw.reshape(-1, X_raw.shape[-1])).reshape(X_raw.shape)
    np.save(SAVE_DIR / "X_raw.npy", X_raw)
    np.save(SAVE_DIR / "X_all.npy", X_all.astype(np.float32))
    np.save(SAVE_DIR / "y_all.npy", y)
    for name, values in merged.items():
        np.save(SAVE_DIR / f"{name}.npy", values)
    with (SAVE_DIR / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    write_json(SAVE_DIR / "label_names.json", LABEL_NAMES)
    write_json(SAVE_DIR / "feature_names.json", FEATURE_COLS)
    write_json(SAVE_DIR / "source_manifest.json", source_records)

    metadata = {
        **runtime_info(), "dataset_type": "hexarotor_simulation_only",
        "formal_protocol": True, "data_dir": str(DATA_DIR),
        "feature_set": FEATURE_SET, "feature_names": FEATURE_COLS,
        "window": WINDOW, "step": STEP, "overlap_fraction": 1 - STEP / WINDOW,
        "fault_buffer_s": BUFFER,
        "matched_analysis_start_s": matched_start,
        "normal_policy": "standalone normal runs, time-matched to fault classes",
        "fault_policy": "post-fault stable segment, time-matched across classes",
        "desired_thrust_array": "cmd_thrust.npy",
        "physics_target_array": "physics_thrust.npy",
        "physics_target_column": "predicted_plant_thrust",
        "group_policy": "all classes sharing run_id form one paired holdout group",
        "groups_per_class": group_counts,
        "class_counts": {int(k): int(v) for k, v in Counter(y.tolist()).items()},
        "focus_labels": [item.label for item in SCENARIOS if item.severity == "partial"],
        "focus_name": "Partial Fault",
        "source_manifest_sha256": stable_hash(source_records),
        "generation_plan_sha256": (
            sha256_file(DATA_DIR / "generation_plan.json")
            if (DATA_DIR / "generation_plan.json").exists() else None
        ),
        "code_sha256": code_hashes(
            PROJECT_ROOT, ["data_process.py", "scenario_config.py", "provenance.py"]
        ),
        "warning": "X_all/scaler.pkl仅为兼容产物；正式CV只读取X_raw.npy。",
    }
    write_json(SAVE_DIR / "preprocess_metadata.json", metadata)
    _save_overview(y, group_counts)

    print(f"\n输出: {SAVE_DIR}")
    print(f"X_raw={X_raw.shape}, y={y.shape}, groups={len(np.unique(groups))}")
    print(f"每类独立来源数: {group_counts}")
    print("下一步运行 classify.py；它会强制读取 groups.npy 并在每折内缩放。")


if __name__ == "__main__":
    main()
