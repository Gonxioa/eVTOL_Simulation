"""Audit actual optimizer counters, training exposure and trained-model causality."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from provenance import sha256_file,write_json
from simulink_data import read
from simulink_realtime import replay_flight
from simulink_realtime_training import load_realtime_checkpoint
from simulink_sequence_data import fit_scaler
from simulink_sequence_events import EventPolicy
from simulink_sequence_model import build_model


def audit(output):
    output=Path(output)
    jobs=read(output/"formal_training_summaries.json")
    flights=torch.load(output/"development_flights.pt",weights_only=False)
    train=[f for f in flights if f.role=="train"]
    expected=fit_scaler(train)
    counters=[]
    initials={}
    for job in jobs:
        folder=Path(job["folder"])
        last=torch.load(folder/"last.pt",map_location="cpu",weights_only=False)
        best=torch.load(folder/"best.pt",map_location="cpu",weights_only=False)
        steps={int(state["step"]) for state in last["optimizer"]["state"].values()}
        if steps != {job["optimizer_updates"]} or last["updates"] != job["optimizer_updates"]:
            raise AssertionError("Actual optimizer step counters disagree")
        for saved in (last,best):
            if int(saved["scaler"].n_samples_seen_) != 273714:
                raise AssertionError("Scaler exposure count changed")
            np.testing.assert_array_equal(saved["scaler"].mean_,expected.mean_)
            np.testing.assert_array_equal(saved["scaler"].scale_,expected.scale_)
        if sum(row["forward_points"] for row in last["history"]) != job["total_valid_forward_points"]:
            raise AssertionError("Training point count mismatch")
        if any(row["forward_points"] != 273714 or row["record_terminals"] != 114 for row in last["history"]):
            raise AssertionError("Missing/repeated actual training points")
        initials.setdefault((job["model"],job["seed"]),set()).add(job["initial_parameters_sha256"])
        torch.manual_seed(job["seed"])
        initial_model=build_model(last["config"])
        delta=max(float((last["model"][key]-value).abs().max()) for key,value in initial_model.state_dict().items())
        if delta<=0:
            raise AssertionError("Optimizer counters advanced but model parameters did not change")
        counters.append(dict(job=folder.name,actual_optimizer_steps=list(steps)[0],
            record_traversals=job["record_traversals"],valid_points=job["total_valid_forward_points"],
            scaler_points=int(expected.n_samples_seen_),checkpoint_selected_epoch=best["epoch"],
            trainable_parameter_tensors_with_optimizer_state=len(last["optimizer"]["state"]),
            maximum_parameter_change=delta))
    if any(len(v)!=1 for v in initials.values()):
        raise AssertionError("Paired targets did not use identical model initialization")
    torch.set_num_threads(1)
    causality=[]
    example=next(f for f in train if f.group==0 and f.label==7)
    for job in jobs:
        if job["objective"]!="current_state":continue
        model,scaler,saved=load_realtime_checkpoint(Path(job["folder"])/"best.pt","cpu")
        policy=EventPolicy(**saved["best_policy"])
        altered=example.x.copy()
        altered[600:]+=100.
        a=replay_flight(model,scaler,example.x,example.time,policy,arrival_sizes=(1,))
        b=replay_flight(model,scaler,altered,example.time,policy,arrival_sizes=(17,3,1,21))
        if a["trace"][:60]!=b["trace"][:60]:
            raise AssertionError("Future data changed past trained-model decisions")
        c=replay_flight(model,scaler,example.x,example.time,policy,arrival_sizes=(13,8,1))
        if a["trace"]!=c["trace"] or a["events"]!=c["events"]:
            raise AssertionError("Arrival batches changed a trained-model diagnosis")
        single=replay_flight(model,scaler,example.x,example.time,policy,decision_samples=1)
        with torch.inference_mode():
            full=model(torch.as_tensor(scaler.transform(example.x)[None],dtype=torch.float32))
            probabilities=full["current_logits"][0].softmax(-1).numpy()
        error=float(np.max(np.abs(probabilities-np.asarray([r["probabilities"] for r in single["trace"]]))))
        if error>2e-4:
            raise AssertionError("Per-sample streaming disagrees with full causal forward")
        causality.append(dict(model=job["model"],seed=job["seed"],future_perturbation_start_s=6.,
            past_decisions_exactly_unchanged=True,irregular_arrival_sequence_exact=True,
            per_sample_vs_full_probability_max_error=error))
    original=read(output/"original_workspace_protection.json")
    for name,digest in original["hashes"].items():
        if sha256_file(Path(original["root"])/name)!=digest:
            raise AssertionError("Original user file changed: "+name)
    receipt=dict(status="passed",actual_training_counters=counters,trained_model_causality=causality,
        matched_initializations=True,scaler_train_only_verified=True,
        original_uncommitted_files_unchanged=list(original["hashes"]),
        test_not_used_for_additional_selection=True)
    write_json(output/"actual_execution_audit.json",receipt)
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",required=True)
    args=parser.parse_args()
    audit(args.results)
    print("ACTUAL_EXECUTION_AUDIT_PASSED")
