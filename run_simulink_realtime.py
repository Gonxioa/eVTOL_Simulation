"""Audit, develop, train, freeze and evaluate the measured-only realtime baselines."""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from provenance import code_hashes, runtime_info, sha256_file, stable_hash, write_json
from simulink_data import FEATURES, checked_plan, observable_features, read
from simulink_sequence_data import load_flights, source_receipt, validate_partitions
from simulink_sequence_events import EventPolicy, infer_flight as legacy_infer
from simulink_realtime import RealtimeDiagnoser, SEMANTICS, replay_flight
from simulink_realtime_evaluation import (choose_policy, decisions_from_probabilities,
    probability_sequences, score_events, score_states)
from simulink_realtime_training import train_job, load_realtime_checkpoint

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT/"configs/simulink_realtime.json"
CODE = ["simulink_realtime.py", "simulink_realtime_training.py", "simulink_realtime_evaluation.py",
        "run_simulink_realtime.py", "simulink_sequence_model.py", "simulink_sequence_data.py",
        "simulink_sequence_events.py", "simulink_data.py", "provenance.py", "configs/simulink_realtime.json"]


def initialize(config, output):
    output.mkdir(parents=True, exist_ok=True)
    dataset = Path(config["dataset_root"])
    identity = dict(config=config, code=code_hashes(ROOT,CODE),
        dataset_registry_sha256=sha256_file(dataset/"source_registry.json"),
        collection_receipt_sha256=sha256_file(dataset/"collection_receipt.json"),
        development_registry_sha256=sha256_file(dataset/"development_sources.json"),
        test_registry_sha256=sha256_file(dataset/"confirmation_sources.json"),
        plan_sha256=sha256_file(ROOT/"docs/REALTIME_EXPERIMENT_PLAN.md"))
    signature = stable_hash(identity)
    if (output/"protocol.json").exists():
        if read(output/"protocol.json")["signature"] != signature:
            raise ValueError("Protocol/source/code changed; new run directory is required")
    else:
        write_json(output/"protocol.json", dict(signature=signature, **identity, runtime=runtime_info()))
    return identity


def audit(config, output):
    dataset = Path(config["dataset_root"])
    rows = read(dataset/"source_registry.json")["records"]
    if len(rows) != 228 or len({r["source_id"] for r in rows}) != 228:
        raise ValueError("Expected 228 distinct sources")
    roles, seeds, groups, records = {}, {}, {}, []
    for r in rows:
        folder = Path(r["source_path"])
        role, group = r["role"], r["group_id"]
        for owners, key in ((roles, group), (seeds, r["run"]["seed"])):
            if key in owners and owners[key] != role:
                raise ValueError("Paired source crosses roles")
            owners[key] = role
        groups.setdefault(group,[]).append(r["run"]["label"])
        meta = read(folder/"run_metadata.json")
        p = meta["parameters"]
        expected = dict(control_tuning_version=config["controller_version"], observed_feedback=True,
                        wind_enabled=True,noise_enabled=True,feedback_delay=.03,task_family=2,duration=24)
        if any(p.get(k) != v for k,v in expected.items()) or meta["dataset_version"] != config["dataset_version"]:
            raise ValueError("Unexpected native conditions/controller")
        plan = checked_plan(Path(r["plan_path"]).parent)
        if plan["plan_signature"] != meta["plan_signature"] or meta["run"] != r["run"] or plan["sources"] != meta["sources"]:
            raise ValueError("Untrusted collection provenance")
        for filename,key in (("signals.csv","csv_sha256"),("result.mat","mat_sha256"),("run_metadata.json","metadata_sha256")):
            if sha256_file(folder/filename) != r[key]:
                raise ValueError("Native hash changed: "+str(folder/filename))
        # Structural CSV audit without parsing held-out measurement values.
        with (folder/"signals.csv").open(encoding="utf-8-sig") as handle:
            columns = next(handle).strip().split(",")
            points = sum(1 for _ in handle)
        if points != 2401 or not set([f"feedback_state_{i}" for i in range(14,20)]).issubset(columns):
            raise ValueError("Incomplete measured trajectory")
        if meta["status"] != "completed" or meta["termination"] != "completed":
            raise ValueError("Incomplete trajectory retained in registry")
        records.append(dict(source_id=r["source_id"],role=role,group=group,seed=r["run"]["seed"],
            label=r["run"]["label"],points=points,channels=12,hashes_verified=True,controller=p["control_tuning_version"]))
    if len(groups) != 12 or any(sorted(v) != list(range(19)) for v in groups.values()):
        raise ValueError("Every paired group must contain all 19 classes")
    expected_counts = {"train":114,"validation":57,"test":57}
    counts = {role:sum(r["role"]==role for r in rows) for role in expected_counts}
    if counts != expected_counts:
        raise ValueError("Role counts changed")
    development = load_flights([dataset/"development_sources.json"], source_root=ROOT)
    if len(development) != 171:
        raise AssertionError("Development role filter failed")
    # Verify forbidden source fields cannot influence measured features.
    r = next(r for r in rows if r["role"]=="train" and r["run"]["label"]==1)
    df = pd.read_csv(Path(r["source_path"])/"signals.csv")
    meta = read(Path(r["source_path"])/"run_metadata.json")
    original = observable_features(df,meta,"imu_rpm")
    altered = df.copy()
    allowed = [f"accelerometer_sensor_m_s2_{i}" for i in range(1,4)]+[f"gyro_sensor_rad_s_{i}" for i in range(1,4)]+[f"feedback_state_{i}" for i in range(14,20)]
    for column in altered:
        if column not in allowed+["time_s"]:
            altered[column] = 123456.
    if not np.array_equal(original,observable_features(altered,meta,"imu_rpm")):
        raise AssertionError("Truth/command/internal metadata leaked into features")
    delay = np.maximum(np.arange(2401)-3,0)
    np.testing.assert_allclose(original[:,6:],df[[f"feedback_state_{i}" for i in range(14,20)]].values,rtol=1e-6)
    np.testing.assert_allclose(original[:,:6],df[allowed[:6]].values[delay],rtol=1e-6)
    pd.DataFrame(records).to_csv(output/"source_audit.csv",index=False,encoding="utf-8-sig")
    receipt = dict(status="passed",counts=counts,groups=12,source_records=228,healthy_records=12,
        fault_records=216,all_hashes_verified=True,points_per_record=2401,feature_names=FEATURES["imu_rpm"],
        input_columns=allowed,forbidden_field_mutation_test_passed=True,IMU_delay_samples=3,
        measured_RPM_additional_delay_samples=0,development_full_native_validation_passed=171,
        test_signal_values_loaded=False,test_full_native_numeric_validation="Deferred until frozen test access",
        source_groups_disjoint=True,controller_version=config["controller_version"],runtime=runtime_info())
    write_json(output/"data_audit.json",receipt)
    torch.save(development,output/"development_flights.pt")
    write_json(output/"development_sources.json",source_receipt(development))
    return receipt


def development_flights(config,output):
    if not (output/"development_flights.pt").exists():
        audit(config,output)
    flights = torch.load(output/"development_flights.pt",weights_only=False)
    if any(f.role not in ("train","validation") for f in flights):
        raise PermissionError("Test source in development cache")
    if stable_hash(source_receipt(flights)) != stable_hash(read(output/"development_sources.json")):
        raise ValueError("Development cache signature mismatch")
    return flights


def smoke(config,output,identity,device):
    flights = development_flights(config,output)
    selected = [f for f in flights if f.group in (0,6)]
    rows=[]
    for family in config["models"]:
        cfg=dict(config,model=family,seed=42,objective="current_state",segment_samples=100,early_weight=1.,
                 max_epochs=1,validation_every_epochs=1)
        folder=output/"smoke"/family
        result=train_job(cfg,selected,folder,identity,device,resume=True)
        model,scaler,saved=load_realtime_checkpoint(folder/"best.pt",device)
        f=next(f for f in selected if f.role=="validation" and f.label==7)
        a=replay_flight(model,scaler,f.x,f.time,device=device,arrival_sizes=(1,))
        b=replay_flight(model,scaler,f.x,f.time,device=device,arrival_sizes=(3,17,1,23))
        if a["trace"] != b["trace"] or a["events"] != b["events"]:
            raise AssertionError("Real-source arrival batching changed diagnosis")
        probabilities=probability_sequences(model,scaler,[f],device)[0]
        np.testing.assert_allclose([r["probabilities"] for r in a["trace"]],probabilities[9::10],atol=1e-4,rtol=1e-4)
        rows.append(dict(model=family,train_records=result["train_records"],validation_records=result["validation_records"],
            optimizer_updates=result["optimizer_updates"],real_flight_streaming_checks_passed=True))
    write_json(output/"smoke_receipt.json",dict(status="passed",runs=rows,synthetic=False,test_used=False))
    return rows


def pilot(config,output,identity,device):
    if not (output/"smoke_receipt.json").exists():
        raise RuntimeError("Run smoke before pilot")
    flights=development_flights(config,output)
    rows=[]
    for segment in config["segment_candidates"]:
        for weight in config["early_weight_candidates"]:
            for family in config["models"]:
                cfg=dict(config,model=family,seed=config["pilot_seed"],objective="current_state",
                    segment_samples=segment,early_weight=weight,max_epochs=config["pilot_epochs"])
                folder=output/"pilot"/f"{family}_segment{segment}_early{weight:g}"
                print("PILOT_TRAIN",folder.name,flush=True)
                result=train_job(cfg,flights,folder,identity,device,resume=True)
                rows.append(dict(model=family,segment_samples=segment,early_weight=weight,
                    rank=result["selected_validation"]["selected"]["rank"],
                    state_macro_f1=result["selected_validation"]["states"]["all"]["macro_f1"],
                    localization_1s=result["selected_validation"]["selected"]["metrics"]["localization_within_1s"],
                    optimizer_updates=result["optimizer_updates"]))
                write_json(output/"pilot_progress.json",rows)
    for family in config["models"]:
        ordered=sorted([r for r in rows if r["model"]==family],key=lambda r:tuple(r["rank"]),reverse=True)
        for position,row in enumerate(ordered):
            row["within_model_rank"] = 1+sum(tuple(other["rank"])>tuple(row["rank"]) for other in ordered)
    candidates=[]
    for segment in config["segment_candidates"]:
        for weight in config["early_weight_candidates"]:
            subset=[r for r in rows if r["segment_samples"]==segment and r["early_weight"]==weight]
            candidates.append(dict(segment_samples=segment,early_weight=weight,
                                  mean_rank=float(np.mean([r["within_model_rank"] for r in subset]))))
    selected=min(candidates,key=lambda r:(r["mean_rank"],r["early_weight"],-r["segment_samples"]))
    result=dict(selected=selected,candidates=candidates,runs=rows,test_used=False,training_seed=config["pilot_seed"])
    write_json(output/"development_selection.json",result)
    return result


def formal_training(config,output,identity,device):
    choice=read(output/"development_selection.json")["selected"]
    flights=development_flights(config,output)
    jobs=[]
    for objective in ("current_state","legacy_record"):
        for family in config["models"]:
            for seed in config["training_seeds"]:
                cfg=dict(config,**{k:choice[k] for k in ("segment_samples","early_weight")},
                         model=family,seed=seed,objective=objective)
                folder=output/"formal"/f"{objective}_{family}_seed{seed}"
                write_json(output/"progress.json",dict(stage="formal_training",job=folder.name,completed=len(jobs),total=12))
                print("FORMAL_TRAIN",folder.name,flush=True)
                summary=train_job(cfg,flights,folder,identity,device,resume=True)
                jobs.append(dict(folder=str(folder),**summary))
                write_json(output/"formal_training_summaries.json",jobs)
    return jobs


def freeze(config,output,identity):
    jobs=read(output/"formal_training_summaries.json")
    expected={(o,m,s) for o in ("current_state","legacy_record") for m in config["models"] for s in config["training_seeds"]}
    if {(r["objective"],r["model"],r["seed"]) for r in jobs} != expected:
        raise RuntimeError("All 12 runs must finish before test unlocking")
    if identity["code"] != code_hashes(ROOT,CODE):
        raise ValueError("Training code changed before freeze")
    files={str(Path(r["folder"])/name):sha256_file(Path(r["folder"])/name)
           for r in jobs for name in ("best.pt","last.pt","job.json","summary.json")}
    payload=dict(protocol_signature=stable_hash(identity),files=files,code=identity["code"],
        development_selection_sha256=sha256_file(output/"development_selection.json"),
        selected_policies={Path(r["folder"]).name:r["selected_policy"] for r in jobs},
        test_registry_sha256=identity["test_registry_sha256"],test_used_for_selection=False,
        test_exposure_note="Prior physical descriptive analysis existed; no classification results used for this selection")
    signature=stable_hash(payload)
    if (output/"freeze_receipt.json").exists() and read(output/"freeze_receipt.json")["signature"] != signature:
        raise ValueError("Frozen experiment identity changed")
    write_json(output/"freeze_receipt.json",dict(signature=signature,**payload,runtime=runtime_info()))
    return signature


def authorize_test(config,output,identity):
    receipt=read(output/"freeze_receipt.json")
    unsigned={k:v for k,v in receipt.items() if k not in ("signature","runtime")}
    if stable_hash(unsigned) != receipt["signature"] or receipt["protocol_signature"] != stable_hash(identity):
        raise PermissionError("Freeze signature invalid")
    if receipt["code"] != code_hashes(ROOT,CODE):
        raise PermissionError("Code changed after model freeze")
    for path,digest in receipt["files"].items():
        if sha256_file(path) != digest:
            raise PermissionError("Frozen model/scaler/config changed")
    registry=Path(config["dataset_root"])/"confirmation_sources.json"
    if sha256_file(registry) != receipt["test_registry_sha256"]:
        raise PermissionError("Test sources changed")
    access=dict(freeze_signature=receipt["signature"],classification_evaluation_authorized=True,
        authorizing_request="2026-11-23 user request: evaluate after all models and policies frozen",
        purpose="One unified final classification evaluation; no model/threshold selection",runtime=runtime_info())
    if not (output/"test_access_receipt.json").exists():
        write_json(output/"test_access_receipt.json",access)
    return access


def _legacy_control_predictions(model,scaler,flights,policy,device):
    return [legacy_infer(model,scaler,f.x,f.time,10,EventPolicy(**policy),device) for f in flights]


def evaluate_job(folder,flights,device,output,formal_streaming=True):
    model,scaler,saved=load_realtime_checkpoint(folder/"best.pt",device,allow_legacy_control=True)
    cfg=saved["config"]
    probabilities=probability_sequences(model,scaler,flights,device,cfg["batch_flights"])
    states=score_states(flights,probabilities)
    policy=EventPolicy(**saved["best_policy"])
    predictions=[]
    errors=[]
    for index,(flight,p) in enumerate(zip(flights,probabilities)):
        if formal_streaming:
            result=replay_flight(model,scaler,flight.x,flight.time,policy,cfg["decision_samples"],cfg["sample_hz"],device,
                arrival_sizes=(7,1,23,9),record_head_trained=cfg["objective"]=="legacy_record")
            error=float(np.max(np.abs(np.asarray([r["probabilities"] for r in result["trace"]])-p[9::10])))
            if error>2e-4:
                raise AssertionError(f"Actual streaming differs from causal full forward: {error}")
            cached=decisions_from_probabilities(flight.time,p,policy,10)
            def boundaries(result):
                return [(e["detected_at_s"],e["cleared_at_s"],[(u["time_s"],u["label"]) for u in e["class_updates"]]) for e in result["events"]]
            if boundaries(result) != boundaries(cached):
                raise AssertionError("Cached and actual streaming event boundaries differ")
            errors.append(error)
        else:
            result=decisions_from_probabilities(flight.time,p,policy,10)
        predictions.append(result)
    metrics,cases=score_events(flights,predictions)
    original_predictions=[decisions_from_probabilities(f.time,p,EventPolicy(**cfg["event_policy"]),10) for f,p in zip(flights,probabilities)]
    original_score,_=score_events(flights,original_predictions)
    result=dict(events=metrics,states=states,original_untuned_policy=original_score,
                selected_policy=asdict(policy),model=cfg["model"],seed=cfg["seed"],objective=cfg["objective"],
                source_role=flights[0].role,streaming_all_records_verified=formal_streaming,
                max_full_vs_stream_probability_error=max(errors) if errors else None)
    if cfg["objective"]=="legacy_record":
        old=_legacy_control_predictions(model,scaler,flights,cfg["event_policy"],device)
        legacy_score,legacy_cases=score_events(flights,old)
        result["legacy_prefix_localization_original_policy"]=legacy_score
        pd.DataFrame(legacy_cases).to_csv(output/"legacy_prefix_cases.csv",index=False,encoding="utf-8-sig")
    write_json(output/"metrics.json",result)
    pd.DataFrame(cases).to_csv(output/"per_record.csv",index=False,encoding="utf-8-sig")
    write_json(output/"events.json",[dict(source_id=f.source_id,**{k:v for k,v in p.items() if k not in ("trace","processing_seconds")})
                                   for f,p in zip(flights,predictions)])
    traces=[]
    for f,p in zip(flights,predictions):
        for r in p["trace"]:
            traces.append(dict(source_id=f.source_id,**{k:v for k,v in r.items() if k!="probabilities"}))
    pd.DataFrame(traces).to_csv(output/"decision_trace.csv",index=False,encoding="utf-8-sig")
    np.savez_compressed(output/"state_probabilities.npz",**{f.source_id:p for f,p in zip(flights,probabilities)})
    grouped=[]
    frame=pd.DataFrame(cases)
    for axis in ("group","motor","remaining_effectiveness"):
        for value,subset in frame.groupby(axis):
            if axis!="group":
                subset=subset[subset.label>0]
            faults=subset[subset.label>0]
            row=dict(axis=axis,value=value,records=len(subset),faults=len(faults))
            for point in (.5,1.,3.):
                for name in ("detection","localization"):
                    column=f"{name}_within_{point:g}s"
                    row[column]=float(faults[column].mean()) if len(faults) else None
            row.update(first_wrong=int(faults.first_localization_wrong.fillna(False).sum()),
                       always_missed=int(faults.detection_delay_s.isna().sum()),
                       never_localized=int(faults.localization_delay_s.isna().sum()))
            for name in ("detection","localization"):
                vals=faults[name+"_delay_s"].dropna()
                row[name+"_delay_median_s"]=float(vals.median()) if len(vals) else None
                row[name+"_delay_p95_s"]=float(vals.quantile(.95)) if len(vals) else None
            grouped.append(row)
    pd.DataFrame(grouped).to_csv(output/"by_group_motor_severity.csv",index=False,encoding="utf-8-sig")
    return result


def final_evaluation(config,output,identity,device):
    authorize_test(config,output,identity)
    development=development_flights(config,output)
    test=load_flights([Path(config["dataset_root"])/"confirmation_sources.json"],roles=("test",))
    validate_partitions(development+test)
    write_json(output/"test_source_receipt.json",source_receipt(test))
    jobs=read(output/"formal_training_summaries.json")
    rows=[]
    for job in jobs:
        folder=Path(job["folder"])
        for role,flights in (("validation",[f for f in development if f.role=="validation"]),("test",test)):
            destination=folder/role
            destination.mkdir(exist_ok=True)
            print("FINAL_EVAL",folder.name,role,flush=True)
            result=(read(destination/"metrics.json") if (destination/"metrics.json").exists() else
                    evaluate_job(folder,flights,device,destination,formal_streaming=True))
            rows.append(dict(model=job["model"],objective=job["objective"],seed=job["seed"],role=role,
                state_macro_f1=result["states"]["all"]["macro_f1"],
                healthy_state_accuracy=result["states"]["healthy"]["accuracy"],
                early_state_accuracy=result["states"]["early_0_3s"]["accuracy"],
                sustained_state_accuracy=result["states"]["sustained_after_3s"]["accuracy"],
                **{k:v for k,v in result["events"].items() if not isinstance(v,(dict,list))},
                detection_delay_median_s=result["events"]["detection_delay_s"]["median"],
                detection_delay_p95_s=result["events"]["detection_delay_s"]["p95"],
                localization_delay_median_s=result["events"]["localization_delay_s"]["median"],
                localization_delay_p95_s=result["events"]["localization_delay_s"]["p95"]))
            pd.DataFrame(rows).to_csv(output/"per_seed_results.csv",index=False,encoding="utf-8-sig")
    write_json(output/"evaluation_complete.json",dict(status="completed",formal_training_runs=12,
        test_records=57,train_records=114,validation_records=57,original_native_records=228,
        new_simulations_this_round=0,GAN_used=False,ROS_used=False,runtime=runtime_info()))
    write_json(output/"progress.json",dict(stage="completed",formal_training_runs=12,test_records=57))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("audit","smoke","pilot","train","freeze","evaluate","all"))
    parser.add_argument("--config",default=str(CONFIG))
    parser.add_argument("--output")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu",choices=("cpu","cuda"))
    args=parser.parse_args()
    config=read(args.config)
    output=Path(args.output or config["results_root"])
    identity=initialize(config,output)
    stages=("audit","smoke","pilot","train","freeze","evaluate") if args.stage=="all" else (args.stage,)
    for stage in stages:
        print("REALTIME_STAGE",stage,flush=True)
        function={"audit":audit,"smoke":smoke,"pilot":pilot,"train":formal_training,"freeze":freeze,"evaluate":final_evaluation}[stage]
        if stage=="audit":
            result=function(config,output)
        elif stage=="freeze":
            result=function(config,output,identity)
        else:
            result=function(config,output,identity,args.device)
        print("REALTIME_STAGE_COMPLETE",stage,flush=True)


if __name__=="__main__":
    main()
