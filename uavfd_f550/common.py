from pathlib import Path
import hashlib
import json
import platform
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "configs"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n", encoding="utf-8")


def manifest(data_dir, records, config=None):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    import scipy
    return {"code_head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted(Path(__file__).parent.rglob("*.py"))},
            "configuration_sha256": {p.name: digest(p) for p in sorted(CONFIG.glob("*.json"))},
            "split_sha256": digest(CONFIG / "split.json"), "config": config,
            "records": {name: digest(Path(data_dir) / name) for name in records},
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "seed": (config or {}).get("seed"), "evaluation_role": "development_not_final_test"}


def split():
    value = read_json(CONFIG / "split.json")
    names = sum((value[k] for k in ("calibration", "development", "locked_test")), [])
    if len(names) != 18 or len(set(names)) != 18 or value["test_enabled"]:
        raise ValueError("Invalid or unlocked split. Final test is closed in this implementation.")
    return value


def permit(name, role):
    s = split()
    allowed = s["calibration"] if role == "fit" else s["calibration"] + s["development"]
    if name not in allowed:
        raise PermissionError(f"Signal access denied for {name} ({role}); locked test must remain untouched.")
