"""Package the completed experiment without duplicating raw native data."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from provenance import sha256_file,write_json
from simulink_data import read


def package(output):
    output=Path(output)
    read(output/"evaluation_complete.json")
    if read(output/"actual_execution_audit.json")["status"]!="passed":
        raise RuntimeError("Actual execution audit has not passed")
    versions={name:importlib.metadata.version(name) for name in ("torch","numpy","scipy","scikit-learn","pandas","matplotlib")}
    write_json(output/"runtime_environment.json",dict(python=sys.version,platform=platform.platform(),
        versions=versions,torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        training_core_git_commit=subprocess.check_output(["git","rev-list","-1","HEAD","--","run_simulink_realtime.py","simulink_realtime.py","simulink_realtime_training.py"],cwd=ROOT,text=True).strip(),
        controller_base_commit="362646775e38183381867ba37aa07f729e4195b5"))
    code=output/"training_code"
    code.mkdir(exist_ok=True)
    protocol=read(output/"protocol.json")
    for name,digest in protocol["code"].items():
        if sha256_file(ROOT/name)!=digest:
            raise ValueError("Training core changed during experiment")
        target=code/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    for name in ("stream_fault_diagnosis.py","benchmark_realtime_diagnosis.py","docs/REALTIME_USAGE.md","docs/REALTIME_EXPERIMENT_PLAN.md"):
        target=code/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    provenance=output/"data_provenance"
    provenance.mkdir(exist_ok=True)
    dataset=Path(protocol["config"]["dataset_root"])
    for name in ("source_registry.json","master_collection_plan.json"):
        shutil.copy2(dataset/name,provenance/name)
    shutil.copy2(dataset/"collection_receipt.json",provenance/"collection_receipt_at_collection.json")
    logs=output/"logs"
    logs.mkdir(exist_ok=True)
    for name in ("realtime_execution_final_20261123.log","realtime_final_tests_20261123.log","realtime_range_test_20261123.log",
                 "realtime_formal_20261123.log","realtime_development_range_draft_20261123.log",
                 "realtime_validation_stream_preflight_20261123.log","realtime_actual_audit_20261123.log",
                 "realtime_latency_20261123.log","realtime_report_20261123.log",
                 "realtime_pilot_20261123.log","realtime_smoke_20261123.log","realtime_audit_20261123.log"):
        source=Path(protocol["config"]["dataset_root"]).parents[1]/name
        if source.exists():shutil.copy2(source,logs/name)
    tracked=ROOT/"docs/validation/realtime_20261123"
    tracked.mkdir(parents=True,exist_ok=True)
    for source in output.iterdir():
        if source.is_file() and source.suffix in (".json",".csv",".md"):
            if source.name not in ("formal_training_summaries.json",):
                shutil.copy2(source,tracked/source.name)
    for name in ("figures","logs","data_provenance"):
        shutil.copytree(output/name,tracked/name,dirs_exist_ok=True)
    for folder in (output/"formal").iterdir():
        if not folder.is_dir():continue
        destination=tracked/"formal"/folder.name
        destination.mkdir(parents=True,exist_ok=True)
        for name in ("summary.json","history.json","job.json"):
            shutil.copy2(folder/name,destination/name)
        for role in ("validation","test"):
            target=destination/role
            target.mkdir(exist_ok=True)
            for name in ("metrics.json","per_record.csv","by_group_motor_severity.csv","legacy_prefix_cases.csv"):
                if (folder/role/name).exists():shutil.copy2(folder/role/name,target/name)
    models=ROOT/"model_artifacts/realtime_20261123"
    shutil.copytree(output/"portable_weights",models,dirs_exist_ok=True)
    bundle=output.parent/"实时故障诊断交付_20261123.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(output.rglob("*")):
            if not path.is_file():continue
            rel=path.relative_to(output)
            if path.suffix in (".npz",".pt") and rel.parts[0]!="portable_weights":continue
            archive.write(path,rel.as_posix())
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:raise AssertionError("Corrupted delivery archive")
    manifest=dict(bundle=str(bundle),bundle_sha256=sha256_file(bundle),bundle_bytes=bundle.stat().st_size,
        tracked_reports=str(tracked),portable_models=str(models),models=12,
        raw_dataset_duplicated=False,raw_resume_checkpoints_location=str(output/"formal"))
    write_json(output/"delivery_receipt.json",manifest)
    write_json(tracked/"delivery_receipt.json",manifest)
    print("DELIVERY_READY",bundle)
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",required=True)
    args=parser.parse_args()
    package(args.results)
