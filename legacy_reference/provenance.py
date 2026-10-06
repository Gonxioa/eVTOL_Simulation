"""Small provenance helpers shared by generation, preprocessing and training."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys


def sha256_file(path):
    path = Path(path)
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hashes(project_root, filenames):
    root = Path(project_root)
    return {
        name: sha256_file(root / name)
        for name in filenames
        if (root / name).is_file()
    }


def stable_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def runtime_info():
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(path)
