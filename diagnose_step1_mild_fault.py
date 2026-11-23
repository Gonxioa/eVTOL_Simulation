"""Step 1 diagnostic: mild-fault (80%) physical SNR, validation probability autopsy,
and causal feature probe. Strictly uses development flights (train + validation only);
never loads or reads held-out test records.
"""
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

from simulink_data import FEATURES, read
from simulink_sequence_data import load_flights

SEV_MAP = {0: "healthy", 1: "full_0pct", 2: "severe_20pct", 3: "partial_80pct"}


def severity_bucket(label: int) -> str:
    if label == 0:
        return "healthy"
    return ("full_0pct", "severe_20pct", "partial_80pct")[(label - 1) // 6]


def opposite_label(label: int) -> int:
    if label == 0:
        return 0
    sev = (label - 1) // 6
    motor = (label - 1) % 6
    return 1 + sev * 6 + ((motor + 3) % 6)


def augment_channels(x: np.ndarray):
    """Strictly causal point-wise physical channel combinations (no future info)."""
    rpm = x[:, 6:12]
    rpm_mean = rpm.mean(axis=1, keepdims=True)
    rpm_rel = rpm - rpm_mean  # 6 dims: single-motor deviation from common lift
    rpm_opp = np.stack([rpm[:, i] - rpm[:, (i + 3) % 6] for i in range(6)], axis=1)  # 6 dims
    rpm_parity = (rpm[:, [0, 2, 4]].mean(axis=1, keepdims=True)
                  - rpm[:, [1, 3, 5]].mean(axis=1, keepdims=True))  # 1 dim: CW vs CCW balance
    names = (
        list(FEATURES["imu_rpm"])
        + [f"rpm_rel{i}" for i in range(6)]
        + [f"rpm_opp{i}" for i in range(6)]
        + ["rpm_parity"]
    )
    return np.concatenate([x, rpm_rel, rpm_opp, rpm_parity], axis=1), names


def load_dev_flights(results_root: Path, dataset_root: Path | None = None):
    cache = results_root / "development_flights.pt"
    if cache.exists():
        flights = torch.load(cache, weights_only=False)
    else:
        if dataset_root is None:
            raise FileNotFoundError(f"Missing {cache}; pass --dataset-root.")
        flights = load_flights([dataset_root / "development_sources.json"], roles=("train", "validation"))
    if any(f.role not in ("train", "validation") for f in flights):
        raise PermissionError("Refusing to run Step 1 on non-development (test) flights.")
    return flights


def analyze_physical_snr(flights):
    """Compare paired fault-minus-healthy residuals vs cross-group healthy natural std."""
    fault_starts = [f.fault_start_s for f in flights if f.label > 0 and f.fault_start_s is not None]
    onset = float(np.median(fault_starts))
    windows = [
        ("pre_-1_0s", -1.0, 0.0),
        ("0.0_0.1s", 0.0, 0.1),
        ("0.1_0.3s", 0.1, 0.3),
        ("0.3_1.0s", 0.3, 1.0),
        ("1.0_3.0s", 1.0, 3.0),
        ("3.0_10.0s", 3.0, 10.0),
        ("10.0_21.0s", 10.0, 21.0),
    ]

    # Index augmented trajectories by (group, label)
    by_gl = {}
    ch_names = None
    for f in flights:
        aug, ch_names = augment_channels(f.x)
        by_gl[(f.group, f.label)] = (f, aug)

    groups = sorted({f.group for f in flights})
    rows = []
    for wname, w0, w1 in windows:
        # Cross-group natural fluctuation among healthy flights in this time window
        h_means, h_stds, h_point_vals = [], [], []
        for g in groups:
            hf, haug = by_gl[(g, 0)]
            mask = (hf.time >= onset + w0 - 1e-9) & (hf.time < onset + w1 - 1e-9)
            seg = haug[mask]
            h_means.append(seg.mean(axis=0))
            h_stds.append(seg.std(axis=0))
            h_point_vals.append(seg)
        h_means = np.asarray(h_means)  # [G, C]
        cross_group_mean_std = np.maximum(h_means.std(axis=0), 1e-8)  # online window-mean noise floor
        cross_group_point_std = np.maximum(np.concatenate(h_point_vals, axis=0).std(axis=0), 1e-8)

        for sev_idx, sev_name in [(1, "full_0pct"), (2, "severe_20pct"), (3, "partial_80pct")]:
            labels = list(range(1 + (sev_idx - 1) * 6, 1 + sev_idx * 6))
            # Track both: (a) faulted motor's own RPM / rel / opp channel, (b) max across IMU channels
            fault_motor_rpm_abs, fault_motor_rel_abs, fault_motor_opp_abs = [], [], []
            fault_motor_rpm_snr, fault_motor_rel_snr, fault_motor_opp_snr = [], [], []
            fault_motor_rpm_pt_snr, fault_motor_rel_pt_snr = [], []
            imu_acc_snr, imu_gyro_snr, parity_snr = [], [], []

            for g in groups:
                hf, haug = by_gl[(g, 0)]
                for lbl in labels:
                    ff, faug = by_gl[(g, lbl)]
                    t0 = ff.fault_start_s
                    mask = (ff.time >= t0 + w0 - 1e-9) & (ff.time < t0 + w1 - 1e-9)
                    diff = faug[mask] - haug[mask]
                    mean_abs_diff = np.abs(diff.mean(axis=0))
                    rms_diff = np.sqrt((diff ** 2).mean(axis=0))

                    m = (lbl - 1) % 6
                    rpm_c, rel_c, opp_c, par_c = 6 + m, 12 + m, 18 + m, 24

                    fault_motor_rpm_abs.append(mean_abs_diff[rpm_c])
                    fault_motor_rel_abs.append(mean_abs_diff[rel_c])
                    fault_motor_opp_abs.append(mean_abs_diff[opp_c])

                    fault_motor_rpm_snr.append(mean_abs_diff[rpm_c] / cross_group_mean_std[rpm_c])
                    fault_motor_rel_snr.append(mean_abs_diff[rel_c] / cross_group_mean_std[rel_c])
                    fault_motor_opp_snr.append(mean_abs_diff[opp_c] / cross_group_mean_std[opp_c])

                    fault_motor_rpm_pt_snr.append(rms_diff[rpm_c] / cross_group_point_std[rpm_c])
                    fault_motor_rel_pt_snr.append(rms_diff[rel_c] / cross_group_point_std[rel_c])

                    imu_acc_snr.append(np.max(mean_abs_diff[0:3] / cross_group_mean_std[0:3]))
                    imu_gyro_snr.append(np.max(mean_abs_diff[3:6] / cross_group_mean_std[3:6]))
                    parity_snr.append(mean_abs_diff[par_c] / cross_group_mean_std[par_c])

            rows.append(dict(
                severity=sev_name,
                window=wname,
                motor_rpm_abs_diff_rad_s=float(np.mean(fault_motor_rpm_abs)),
                motor_rel_abs_diff_rad_s=float(np.mean(fault_motor_rel_abs)),
                motor_opp_abs_diff_rad_s=float(np.mean(fault_motor_opp_abs)),
                snr_win_mean_rpm=float(np.mean(fault_motor_rpm_snr)),
                snr_win_mean_rpm_rel=float(np.mean(fault_motor_rel_snr)),
                snr_win_mean_rpm_opp=float(np.mean(fault_motor_opp_snr)),
                snr_win_mean_acc_max=float(np.mean(imu_acc_snr)),
                snr_win_mean_gyro_max=float(np.mean(imu_gyro_snr)),
                snr_win_mean_parity=float(np.mean(parity_snr)),
                snr_point_rpm=float(np.mean(fault_motor_rpm_pt_snr)),
                snr_point_rpm_rel=float(np.mean(fault_motor_rel_pt_snr)),
            ))
    return pd.DataFrame(rows)


def analyze_validation_probabilities(results_root: Path, flights):
    """Autopsy saved validation probabilities for all formal models (no test access)."""
    val_flights = [f for f in flights if f.role == "validation"]
    formal_dir = results_root / "formal"
    if not formal_dir.exists():
        return pd.DataFrame()

    rows = []
    for job_dir in sorted(formal_dir.iterdir()):
        npz_path = job_dir / "validation" / "state_probabilities.npz"
        if not npz_path.exists():
            continue
        probs_map = np.load(npz_path)
        name = job_dir.name

        for phase_name, t_lo, t_hi in [("0_1s", 0.0, 1.0), ("1_3s", 1.0, 3.0), ("after_3s", 3.0, 99.0)]:
            # 1) Mild fault (80%, labels 13..18) probability breakdown
            p_healthy_list, p_fault_list, p_true_list, p_any80_list = [], [], [], []
            top1_hit, top3_hit, fault_gt_03, fault_gt_05 = [], [], [], []
            # 2) Severe fault (20%, labels 7..12) opposite-motor confusion
            sev_correct, sev_opp, sev_other = [], [], []

            for f in val_flights:
                if f.label == 0 or f.source_id not in probs_map:
                    continue
                p = probs_map[f.source_id]
                t0 = f.fault_start_s
                mask = (f.time >= t0 + t_lo - 1e-9) & (f.time < t0 + t_hi - 1e-9)
                seg = p[mask]
                if not len(seg):
                    continue
                pred = seg.argmax(axis=1)
                if 13 <= f.label <= 18:
                    p_healthy_list.append(seg[:, 0])
                    p_fault_list.append(1.0 - seg[:, 0])
                    p_true_list.append(seg[:, f.label])
                    p_any80_list.append(seg[:, 13:19].sum(axis=1))
                    top1_hit.append(pred == f.label)
                    top3 = np.argsort(seg, axis=1)[:, -3:]
                    top3_hit.append(np.any(top3 == f.label, axis=1))
                    fault_gt_03.append((1.0 - seg[:, 0]) >= 0.3)
                    fault_gt_05.append((1.0 - seg[:, 0]) >= 0.5)
                elif 7 <= f.label <= 12:
                    opp = opposite_label(f.label)
                    sev_correct.append(pred == f.label)
                    sev_opp.append(pred == opp)
                    sev_other.append((pred != f.label) & (pred != opp))

            if p_healthy_list:
                pf = np.concatenate(p_fault_list)
                rows.append(dict(
                    job=name,
                    phase=phase_name,
                    mild80_mean_p_healthy=float(np.concatenate(p_healthy_list).mean()),
                    mild80_mean_p_fault=float(pf.mean()),
                    mild80_p95_p_fault=float(np.quantile(pf, 0.95)),
                    mild80_mean_p_true=float(np.concatenate(p_true_list).mean()),
                    mild80_mean_p_any80=float(np.concatenate(p_any80_list).mean()),
                    mild80_frac_pfault_ge_03=float(np.concatenate(fault_gt_03).mean()),
                    mild80_frac_pfault_ge_05=float(np.concatenate(fault_gt_05).mean()),
                    mild80_top1_recall=float(np.concatenate(top1_hit).mean()),
                    mild80_top3_recall=float(np.concatenate(top3_hit).mean()),
                    severe20_correct_rate=float(np.concatenate(sev_correct).mean()),
                    severe20_opposite_motor_rate=float(np.concatenate(sev_opp).mean()),
                    severe20_other_error_rate=float(np.concatenate(sev_other).mean()),
                ))
    return pd.DataFrame(rows)


def extract_causal_window_features(flights, win_samples: int, step_samples: int, mode: str):
    """Build strictly causal window summary features for train->val probe."""
    xs, ys, roles, feat_names = [], [], [], None
    for f in flights:
        if mode == "raw12":
            arr = f.x
            base_names = list(FEATURES["imu_rpm"])
        elif mode == "phys25":
            arr, base_names = augment_channels(f.x)
        else:
            raise ValueError(mode)

        # Causal exponential moving baseline (tau = 2.0s => alpha = 1/200)
        ema = np.empty_like(arr)
        alpha = 1.0 / 200.0
        ema[0] = arr[0]
        for t in range(1, len(arr)):
            ema[t] = (1.0 - alpha) * ema[t - 1] + alpha * arr[t]
        delta_ema = arr - ema

        # Running cumulative mean (strictly causal)
        cum_mean = np.cumsum(arr, axis=0) / np.arange(1, len(arr) + 1)[:, None]
        delta_cum = arr - cum_mean

        current, _ = f.targets()
        if feat_names is None:
            feat_names = (
                [f"{n}_mean" for n in base_names]
                + [f"{n}_std" for n in base_names]
                + [f"{n}_slope" for n in base_names]
            )
            if mode == "phys25":
                feat_names += (
                    [f"{n}_dEMA_mean" for n in base_names]
                    + [f"{n}_dCum_mean" for n in base_names]
                )

        # Sample causally every step_samples once a full window is available
        for end in range(win_samples, len(arr), step_samples):
            seg = arr[end - win_samples:end]
            half = max(1, win_samples // 2)
            parts = [
                seg.mean(axis=0),
                seg.std(axis=0),
                seg[half:].mean(axis=0) - seg[:half].mean(axis=0),
            ]
            if mode == "phys25":
                parts.append(delta_ema[end - win_samples:end].mean(axis=0))
                parts.append(delta_cum[end - win_samples:end].mean(axis=0))
            xs.append(np.concatenate(parts))
            ys.append(int(current[end - 1]))
            roles.append(f.role)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), np.asarray(roles), feat_names


def run_causal_rf_probe(flights):
    """Train Random Forest strictly on train groups, evaluate on validation groups."""
    rows = []
    top_features = {}
    for win_s in (0.5, 1.0, 3.0):
        win_samples = int(round(win_s * 100))
        for mode in ("raw12", "phys25"):
            X, y, roles, names = extract_causal_window_features(flights, win_samples, step_samples=50, mode=mode)
            tr, va = roles == "train", roles == "validation"
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[va])
            macro_f1 = float(f1_score(y[va], pred, labels=np.arange(19), average="macro", zero_division=0))
            per_cls = recall_score(y[va], pred, labels=np.arange(19), average=None, zero_division=0)
            rec_healthy = float(per_cls[0])
            rec_0 = float(per_cls[1:7].mean())
            rec_20 = float(per_cls[7:13].mean())
            rec_80 = float(per_cls[13:19].mean())
            # Also check binary fault recall on 80% samples (pred > 0)
            mask80 = (y[va] >= 13) & (y[va] <= 18)
            det_80 = float((pred[mask80] > 0).mean())
            rows.append(dict(
                window_s=win_s,
                feature_mode=mode,
                n_features=X.shape[1],
                val_macro_f1=macro_f1,
                val_recall_healthy=rec_healthy,
                val_recall_0pct=rec_0,
                val_recall_20pct=rec_20,
                val_recall_80pct_joint=rec_80,
                val_detect_80pct_anyfault=det_80,
            ))
            if win_s == 1.0 and mode == "phys25":
                order = np.argsort(clf.feature_importances_)[::-1][:15]
                top_features["1.0s_phys25_top15"] = [
                    dict(rank=int(i + 1), feature=names[idx], importance=float(clf.feature_importances_[idx]))
                    for i, idx in enumerate(order)
                ]
    return pd.DataFrame(rows), top_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Path to output_realtime_diagnosis_20261123")
    parser.add_argument("--dataset-root", default=None, help="Optional fallback dataset root if cache missing")
    args = parser.parse_args()

    results_root = Path(args.results)
    out_dir = results_root / "step1_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    flights = load_dev_flights(results_root, Path(args.dataset_root) if args.dataset_root else None)
    print(f"[Step 1] Loaded {len(flights)} development flights "
          f"(train={sum(f.role=='train' for f in flights)}, val={sum(f.role=='validation' for f in flights)}). "
          f"Test flights touched: 0.")

    print("\n=== 1. Physical Paired Residual & Online SNR (Development 9 Groups) ===")
    snr_df = analyze_physical_snr(flights)
    snr_df.to_csv(out_dir / "step1_physical_snr.csv", index=False, encoding="utf-8-sig")
    mild_snr = snr_df[snr_df["severity"] == "partial_80pct"]
    print("\n[Partial 80% Remaining Effectiveness]")
    print(mild_snr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n[Severe 20% Remaining Effectiveness]")
    print(snr_df[snr_df["severity"] == "severe_20pct"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n=== 2. Existing Models Validation Probability Autopsy (80% Mild & 20% Opposite Confusion) ===")
    prob_df = analyze_validation_probabilities(results_root, flights)
    if len(prob_df):
        prob_df.to_csv(out_dir / "step1_val_prob_autopsy.csv", index=False, encoding="utf-8-sig")
        summary_prob = prob_df.assign(
            family=prob_df["job"].str.replace(r"_seed\d+$", "", regex=True)
        ).groupby(["family", "phase"], as_index=False).mean(numeric_only=True)
        cols = [
            "family", "phase", "mild80_mean_p_fault", "mild80_p95_p_fault",
            "mild80_mean_p_any80", "mild80_top1_recall", "mild80_top3_recall",
            "severe20_correct_rate", "severe20_opposite_motor_rate",
        ]
        print(summary_prob[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        print("No validation state_probabilities.npz found under formal/*/validation/.")

    print("\n=== 3. Causal Feature Probe (Random Forest: Train 6 Groups -> Val 3 Groups) ===")
    rf_df, top_feats = run_causal_rf_probe(flights)
    rf_df.to_csv(out_dir / "step1_causal_rf_probe.csv", index=False, encoding="utf-8-sig")
    print(rf_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n[Top-15 Most Important Causal Features (1.0s phys25)]")
    for item in top_feats.get("1.0s_phys25_top15", []):
        print(f"  {item['rank']:2d}. {item['feature']:28s}  importance={item['importance']:.4f}")

    summary = dict(
        test_touched=False,
        development_flights=len(flights),
        top_features=top_feats,
    )
    (out_dir / "step1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Step 1 Complete] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()