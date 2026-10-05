"""MAT decoding through embedded labels; no hard-coded signal columns."""
from pathlib import Path
import re
import warnings
import numpy as np
from scipy.io import loadmat, whosmat
from .common import permit


def scalar(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    if isinstance(value, np.ndarray) and value.dtype.kind in "US":
        return "".join(value.ravel().tolist())
    return value


def inventory(path):
    if is_hdf5(path):
        import h5py
        with h5py.File(path) as f:
            return [(k, tuple(reversed(v.shape)), "hdf5") for k, v in f.items() if not k.startswith("#") and hasattr(v, "shape")]
    try:
        return whosmat(path)
    except NotImplementedError:
        import h5py
        with h5py.File(path) as f:
            return [(k, tuple(reversed(v.shape)), "hdf5") for k, v in f.items() if not k.startswith("#")]


def is_hdf5(path):
    with Path(path).open("rb") as f:
        if f.read(8) == b"\x89HDF\r\n\x1a\n":
            return True
        f.seek(512)
        return f.read(8) == b"\x89HDF\r\n\x1a\n"


def load_variables(path, names):
    names = list(names)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if is_hdf5(path):
                raise NotImplementedError("MATLAB HDF5")
            raw = loadmat(path, variable_names=names)
        except NotImplementedError:
            import h5py
            with h5py.File(path) as f:
                def decode(ds):
                    cls = ds.attrs.get("MATLAB_class", b"")
                    x = np.array(ds).T
                    if cls == b"char":
                        return "".join(chr(int(c)) for c in x.ravel() if c)
                    if cls == b"cell":
                        out = np.empty(x.shape, dtype=object)
                        for idx in np.ndindex(x.shape):
                            out[idx] = decode(f[x[idx]])
                        return out
                    return x
                raw = {k: decode(f[k]) for k in names if k in f}
    return raw, [str(w.message) for w in caught]


class Record:
    def __init__(self, path, channels, role="development", metadata_only=False):
        self.path = Path(path)
        if not metadata_only:
            permit(self.path.name, role)
        wanted = set(channels)
        available = inventory(path)
        labels = {re.sub(r"_\d+$", "", k) + "_label" for k in wanted}
        counts = {k: sum(n == k for n, _, _ in available) for k in wanted | labels}
        if any(v > 1 for v in counts.values()):
            raise ValueError(f"Ambiguous duplicate requested MAT variable: {counts}")
        self.raw, self.warnings = load_variables(path, wanted | labels)

    def table(self, channel):
        if channel not in self.raw or np.asarray(self.raw[channel]).size == 0:
            raise KeyError(f"No records for {channel}; a label alone is insufficient")
        label = channel + "_label"
        if label not in self.raw:
            label = re.sub(r"_\d+$", "", channel) + "_label"
        if label not in self.raw:
            raise KeyError(f"No embedded label for {channel}")
        names = [str(scalar(v)) for v in np.asarray(self.raw[label]).ravel()]
        x = np.asarray(self.raw[channel])
        if x.ndim != 2 or x.shape[1] != len(names) or len(set(names)) != len(names):
            raise ValueError(f"Invalid table/labels for {channel}: {x.shape}/{names}")
        return {name: x[:, i] for i, name in enumerate(names)}

    def parameters(self):
        x = self.raw.get("PARM")
        if x is None:
            raise KeyError("PARM unavailable")
        return {str(scalar(row[0])): float(scalar(row[1])) for row in x}


def timing(t_us):
    t = np.asarray(t_us, dtype=float) * 1e-6
    dt = np.diff(t)
    positive = dt[dt > 0]
    return {"n": len(t), "start_s": float(t[0]), "end_s": float(t[-1]),
            "mean_rate_hz": float((len(t)-1)/(t[-1]-t[0])) if t[-1] > t[0] else None,
            "median_interval_s": float(np.median(positive)) if len(positive) else None,
            "p01_p50_p99_interval_s": np.quantile(positive, [.01, .5, .99]).tolist() if len(positive) else None,
            "nonincreasing_intervals": int(np.sum(dt <= 0)),
            "gap_gt_2_median": int(np.sum(dt > 2*np.median(positive))) if len(positive) else None}


def author_imu_average(record):
    tables = [record.table(f"IMU_{i}") for i in range(3)]
    common = tables[0]["TimeUS"]
    for tab in tables[1:]:
        common = np.intersect1d(common, tab["TimeUS"])
    if len(common) == 0:
        raise ValueError("No three-IMU timestamp intersection")
    values = []
    for tab in tables:
        t = tab["TimeUS"]
        if np.any(np.diff(t) <= 0):
            raise ValueError("IMU timestamps require explicit duplicate/reorder handling")
        idx = np.searchsorted(t, common)
        values.append(np.column_stack([tab[k][idx] for k in ("AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ")]))
    values = np.array(values, dtype=float)
    return common * 1e-6, values.mean(axis=0), values
