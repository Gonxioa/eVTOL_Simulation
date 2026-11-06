"""Whole-flight, measured-only inputs for sequence/event diagnosis.

One Flight is one source record. Computational chunks never become independent
labelled samples. Fault metadata supplies supervision/evaluation only.
"""
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import hashlib
import json

import numpy as np
from sklearn.preprocessing import StandardScaler

from provenance import sha256_file
from simulink_data import ROOT, FEATURES, checked_plan, observable_features, read, validate_run


@dataclass
class Flight:
    source_id: str
    role: str
    group: int
    seed: int
    x: np.ndarray
    time: np.ndarray
    label: int
    fault_start_s: float | None = None
    fault_end_s: float | None = None
    csv_sha256: str = ""
    metadata_sha256: str = ""

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float32)
        self.time = np.asarray(self.time, dtype=np.float64)
        if self.x.ndim != 2 or self.x.shape[1] != 12 or len(self.x) != len(self.time):
            raise ValueError("Expected a complete [time, 12 measured channels] flight")
        if len(self.time) < 2 or not np.isfinite(self.x).all() or not np.isfinite(self.time).all():
            raise ValueError("Empty/nonfinite flight")
        if np.any(np.diff(self.time) <= 0) or not 0 <= self.label < 19:
            raise ValueError("Invalid chronology/record label")
        if self.role not in ("train", "validation", "test"):
            raise ValueError("Unknown source role")
        if self.label and (self.fault_start_s is None or not np.isfinite(self.fault_start_s)):
            raise ValueError("Fault records need an onset for supervision/evaluation")
        if self.label and not self.time[0] <= self.fault_start_s <= self.time[-1]:
            raise ValueError("The labelled fault must occur within the flight")
        if self.fault_end_s is not None and self.fault_end_s <= self.fault_start_s:
            raise ValueError("Fault end precedes onset")

    def targets(self):
        """Current physical state and fault-ever-seen prefix targets, respectively."""
        current = np.zeros(len(self.time), dtype=np.int64)
        seen = current.copy()
        if self.label:
            after = self.time >= self.fault_start_s - 1e-10
            active = after if self.fault_end_s is None else after & (self.time < self.fault_end_s - 1e-10)
            current[active] = self.label
            seen[after] = self.label
        return current, seen

    def identity(self):
        return dict(source_id=self.source_id, role=self.role, group=self.group, seed=self.seed,
                    label=self.label, points=len(self.x), start_s=float(self.time[0]),
                    end_s=float(self.time[-1]), fault_start_s=self.fault_start_s,
                    fault_end_s=self.fault_end_s, csv_sha256=self.csv_sha256,
                    metadata_sha256=self.metadata_sha256,
                    measured_sha256=hashlib.sha256(np.ascontiguousarray(self.x).tobytes()).hexdigest())


def validate_partitions(flights):
    if len({f.source_id for f in flights}) != len(flights):
        raise ValueError("Repeated source: split whole flights, never overlapping chunks")
    for attr in ("group", "seed"):
        owners = {}
        for f in flights:
            key = getattr(f, attr)
            if key in owners and owners[key] != f.role:
                raise ValueError(f"{attr} {key} crosses source roles")
            owners[key] = f.role


def _source_path(value, source_root):
    direct = Path(value)
    if direct.exists():
        return direct
    # Registries collected on Windows contain absolute paths. Preserve their
    # project-relative output_simulink_gan suffix when the project is relocated.
    parts = PureWindowsPath(str(value)).parts
    if "output_simulink_gan" in parts:
        candidate = Path(source_root).joinpath(*parts[parts.index("output_simulink_gan"):])
        if candidate.exists():
            return candidate
    candidate = Path(source_root) / str(value)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Native source missing: {value}. Set --source-root to its project root.")


def load_flights(registry_paths, roles=("train", "validation"), source_root=ROOT):
    """Read only requested roles; numerical test data is not opened by training."""
    records = []
    for path in registry_paths:
        payload = read(path)
        for row in payload["records"]:
            records.append(row)
    seen = set()
    owners = {"group_id": {}, "seed": {}}
    for r in records:
        if r["source_id"] in seen:
            raise ValueError("Duplicate source in input registries")
        seen.add(r["source_id"])
        for name in owners:
            value = r["group_id"] if name == "group_id" else r["run"]["seed"]
            if value in owners[name] and owners[name][value] != r["role"]:
                raise ValueError("Registry condition crosses training/validation/test roles")
            owners[name][value] = r["role"]
    plans = {}
    output = []
    for r in records:
        if r["role"] not in roles:
            continue
        folder = _source_path(r["source_path"], source_root)
        plan_path = _source_path(r["plan_path"], source_root)
        if plan_path not in plans:
            plans[plan_path] = checked_plan(plan_path.parent)
        meta, df = validate_run(folder, r["run"], plans[plan_path])
        for filename, key in (("signals.csv", "csv_sha256"), ("run_metadata.json", "metadata_sha256")):
            if sha256_file(folder / filename) != r[key]:
                raise ValueError("Registry/source identity mismatch: " + r["source_id"])
        values = observable_features(df, meta, "imu_rpm")
        time = df.time_s.to_numpy()
        if not np.allclose(np.diff(time), .01, atol=1e-9, rtol=0):
            raise ValueError("This protocol expects the native 100 Hz measured chain")
        label = int(r["run"]["label"])
        end = meta.get("fault_end_s")
        end = None if end in (None, "infinity", "Inf", "inf") else float(end)
        if end is not None and not np.isfinite(end):
            end = None
        output.append(Flight(r["source_id"], r["role"], int(r["group_id"]), int(r["run"]["seed"]),
                             values, time, label, float(meta["fault_start_s"]) if label else None,
                             end if label else None, r["csv_sha256"], r["metadata_sha256"]))
    if not output:
        raise ValueError("No complete native flights for the requested role")
    validate_partitions(output)
    return output


def fit_scaler(flights):
    if not flights or any(f.role != "train" for f in flights):
        raise PermissionError("Fit normalization on training flights only")
    scaler = StandardScaler()
    for f in flights:
        scaler.partial_fit(f.x)  # Each actual time sample appears once.
    return scaler


def collate(flights, scaler, device):
    import torch
    lengths = np.asarray([len(f.x) for f in flights], dtype=np.int64)
    x = np.zeros((len(flights), max(lengths), 12), dtype=np.float32)
    current = np.zeros(x.shape[:2], dtype=np.int64)
    seen = current.copy()
    for i, f in enumerate(flights):
        n = lengths[i]
        x[i, :n] = scaler.transform(f.x).astype(np.float32)
        current[i, :n], seen[i, :n] = f.targets()
    return tuple(torch.as_tensor(a, device=device) for a in (x, lengths, current, seen,
                 np.asarray([f.label for f in flights], dtype=np.int64)))


def label_name(label):
    if label is None or label == -1:
        return "Unresolved"
    if label == 0:
        return "Healthy"
    return f"M{(label-1)%6+1}-remaining-{(0,20,80)[(label-1)//6]}pct"


def source_receipt(flights):
    return dict(unit="complete_flight", feature_names=FEATURES["imu_rpm"],
                independent_groups={role: len({f.group for f in flights if f.role == role})
                                    for role in sorted({f.role for f in flights})},
                records=[f.identity() for f in flights])
