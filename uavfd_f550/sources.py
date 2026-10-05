"""Acquire primary-source snapshots. Network failures are evidence, not defaults."""
from pathlib import Path
import json
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from .common import digest, write_json

URLS = {
    "zenodo": "https://zenodo.org/api/records/7648996",
    "systemid": "https://ardupilot.org/copter/docs/systemid-model-development.html",
    "logmessages_current_reference_only": "https://ardupilot.org/copter/docs/logmessages.html",
    "vibration_2026": "https://arxiv.org/html/2604.07281v1",
    "diagnosis_2024": "https://arxiv.org/html/2410.05447v1",
    "logged_commit": "https://api.github.com/repos/ArduPilot/ardupilot/commits/232083c4",
    "icuas2023_metadata": "https://api.crossref.org/works/10.1109/ICUAS57906.2023.10156060",
    "original_paper_iris": "https://iris.univpm.it/handle/11566/320071",
    "original_paper_ieee": "https://ieeexplore.ieee.org/document/10156060",
    "f550_manual": "https://dl.djicdn.com/downloads/flamewheel/en/F550_User_Manual_v2.0_en.pdf",
    "air_gear_350_candidate": "https://store.tmotor.com/product/air-gear-350.html",
    "standard_gravity": "https://physics.nist.gov/cgi-bin/cuu/Value?gn"
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "UAVFD-model-evidence/0.1", "Accept": "application/json,text/html,*/*"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.read(), response.status, response.geturl()


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    bodies = {}
    for name, url in URLS.items():
        info = {"url": url, "accessed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
        try:
            body, status, final = get(url)
            extension = ".pdf" if url.endswith(".pdf") else ".json" if url.startswith(("https://api.", "https://zenodo.org/api")) else ".html"
            path = out / (name + extension)
            path.write_bytes(body)
            info.update(status=status, resolved_url=final, sha256=digest(path), archive=str(path))
            bodies[name] = body
        except Exception as exc:
            info.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        entries[name] = info
        print(f"source {name}: {info['status']}", flush=True)
    if "zenodo" in bodies:
        record = json.loads(bodies["zenodo"])
        write_json(out.parent / "release.json", {"id": record["id"], "doi": record.get("doi"),
                   "metadata": record["metadata"], "files": {f["key"]: {"size": f["size"], "checksum": f["checksum"],
                   "url": f["links"]["self"]} for f in record["files"]}})
    if "logged_commit" in bodies:
        commit = json.loads(bodies["logged_commit"])
        sha = commit["sha"]
        entries["logged_commit"]["sha"] = sha
        paths = ["libraries/AP_Motors/AP_MotorsMatrix.cpp", "libraries/AP_Logger/LogStructure.h",
                 "libraries/AP_InertialSensor/AP_InertialSensor.cpp",
                 "libraries/AP_InertialSensor/AP_InertialSensor_Backend.cpp",
                 "libraries/AP_ESC_Telem/AP_ESC_Telem.cpp", "libraries/AP_BLHeli/AP_BLHeli.cpp",
                 "ArduCopter/Log.cpp", "libraries/AP_Motors/AP_Motors_Class.h",
                 "libraries/SRV_Channel/SRV_Channel.h", "libraries/AP_HAL/Util.cpp",
                 "libraries/AP_InertialSensor/LogStructure.h", "libraries/AP_ESC_Telem/LogStructure.h",
                 "libraries/AP_InertialSensor/AP_InertialSensor_Logging.cpp", "libraries/AP_Logger/LogFile.cpp",
                 "libraries/AP_Common/AP_FWVersion.h", "libraries/AP_Motors/AP_MotorsMatrix.h",
                 "libraries/AP_HAL/RCOutput.h", "ArduCopter/Copter.cpp"]
        for path in paths:
            name = path.replace("/", "__")
            url = f"https://raw.githubusercontent.com/ArduPilot/ardupilot/{sha}/{path}"
            info = {"url": url, "accessed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "sha": sha}
            try:
                body, status, _ = get(url)
                local = out / name
                local.write_bytes(body)
                info.update(status=status, sha256=digest(local), archive=str(local))
            except Exception as exc:
                info.update(status="unavailable", error=str(exc))
            entries[name] = info
            print(f"historical {path}: {info['status']}", flush=True)
    write_json(out.parent / "source_fetch.json", entries)
    return entries
