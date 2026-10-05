"""Full integrity/config/timing audit; never computes held-out signal amplitudes."""
from pathlib import Path
import numpy as np
from .common import digest, split, write_json, manifest
from .data import Record, inventory, timing


def run(data_dir, out, release):
    data_dir, out = Path(data_dir), Path(out)
    groups = split()  # Freeze/validate before any signal analysis.
    summary = []
    hashes = {}
    for role in ("calibration", "development", "locked_test"):
        for name in groups[role]:
            path = data_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Frozen record missing: {path}")
            inv = inventory(path)
            channels = [k for k, shape, _ in inv if not k.endswith("_label") and
                        k.startswith(("IMU_", "ESC_", "RCOU", "ATT", "RATE", "PID", "XKF1", "VIBE", "MODE", "ARM", "VER", "BAT"))]
            r = Record(path, channels + ["PARM", "MSG", "UNIT", "FMT", "FMTU", "MULT"], metadata_only=True)
            params = r.parameters()
            info = {"name": name, "split": role, "bytes": path.stat().st_size,
                    "md5": digest(path, "md5"), "sha256": digest(path), "variables": inv,
                    "loader_warnings": r.warnings, "timing": {}, "parameters_count": len(params),
                    "parameters": params, "label_only": [],
                    "duplicate_variable_names": sorted({k for k, _, _ in inv if sum(n == k for n, _, _ in inv) > 1})}
            published = release.get("files", {}).get(name)
            info["published"] = published
            info["release_match"] = (info["md5"] == published["checksum"].split(":")[-1] and
                                      info["bytes"] == published["size"]) if published else None
            if info["release_match"] is False:
                raise ValueError(f"Release integrity mismatch: {name}")
            for ch in channels:
                if np.asarray(r.raw[ch]).size:
                    tab = r.table(ch)
                    info["timing"][ch] = timing(tab["TimeUS"])
                    if ch in ("VER", "MODE", "ARM"):
                        info[ch] = {k: v.tolist() for k, v in tab.items()}
            imu_t = [r.table(f"IMU_{i}")["TimeUS"] for i in range(3)]
            info["imu_timestamps_identical"] = all(np.array_equal(imu_t[0], t) for t in imu_t[1:])
            intersection = np.intersect1d(np.intersect1d(*imu_t[:2]), imu_t[2])
            info["imu_common_n"] = len(intersection)
            info["firmware_string_preservation"] = "FWS/MSG/UNIT/FMT string fields not recovered from numeric MAT export; base GitHash only"
            for ch, field in (("VER", "FWS"), ("MSG", "Message"), ("UNIT", "Label"), ("FMT", "Name")):
                if ch in r.raw and np.asarray(r.raw[ch]).size:
                    tab = r.table(ch)
                    if field in tab:
                        info[ch + "_string_field_numeric_export"] = {"field": field,
                             "dtype": str(np.asarray(tab[field]).dtype), "nonzero_n": int(np.count_nonzero(tab[field]))}
            labels = [k for k, _, _ in inv if k.endswith("_label")]
            bases = {k.rsplit("_", 1)[0] if k.rsplit("_", 1)[-1].isdigit() else k for k in channels}
            info["label_only"] = [k for k in labels if k[:-6] not in bases]
            info["motor_output_map"] = {f"Motor{int(v)-32}": f"C{k[5:k.index('_')]}"
                                        for k, v in params.items() if k.startswith("SERVO") and
                                        k.endswith("_FUNCTION") and 33 <= v <= 38}
            hashes[name] = {k: info[k] for k in ("bytes", "md5", "sha256", "release_match")}
            summary.append(info)
            print(f"audit {name}: checksum={info['release_match']}, parameters={len(params)}", flush=True)
    scripts = {}
    for p in data_dir.glob("*.m"):
        published = release.get("files", {}).get(p.name)
        md5 = digest(p, "md5")
        scripts[p.name] = {"md5": md5, "sha256": digest(p), "published": published,
                           "release_match": md5 == published["checksum"].split(":")[-1] and p.stat().st_size == published["size"] if published else None}
    write_json(out / "audit.json", {"records": summary, "scripts": scripts,
               "signal_statistics_on_locked_test": False, "manifest": manifest(data_dir, list(hashes))})
    write_json(out / "data_hashes.json", hashes)
    return summary
