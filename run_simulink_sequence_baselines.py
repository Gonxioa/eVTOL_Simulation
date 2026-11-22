"""Run GAN-free whole-flight CNN/LSTM baselines and historical evaluation."""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from provenance import code_hashes, runtime_info, sha256_file, stable_hash, write_json
from simulink_data import ROOT, read
from simulink_sequence_data import collate, load_flights, source_receipt, validate_partitions
from simulink_sequence_events import EventPolicy, infer_flight, prediction_from_posteriors
from simulink_sequence_model import decision_mask
from run_simulink_sequence import CODE, export_predictions, load_checkpoint, train_model

CONFIG = ROOT / "configs/simulink_sequence_baselines.json"
BASELINE_CODE = CODE+["run_simulink_sequence_baselines.py"]


def batched_predictions(model, scaler, flights, config, device):
    """Efficient replay of causal logits; no past output uses future samples.

    Whole-record causal forward passes avoid launching a GPU operation for
    every decision. Both encoders are separately checked against true streaming.
    """
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(flights), config["batch_flights"]):
            batch = flights[start:start+config["batch_flights"]]
            x, lengths, *_ = collate(batch, scaler, device)
            output = model(x, lengths=lengths)
            mask = decision_mask(lengths, x.shape[1], config["decision_samples"])
            for i, flight in enumerate(batch):
                positions = mask[i].nonzero().flatten()
                current = output["current_logits"][i, positions].softmax(-1).cpu().numpy()
                prefix = output["prefix_logits"][i, positions].softmax(-1).cpu().numpy()
                predictions.append(prediction_from_posteriors(flight.time, positions.cpu().numpy(),
                    current, prefix, EventPolicy(**config["event_policy"])))
    return predictions


def detailed_metrics(flights, predictions):
    cases = []
    for flight, prediction in zip(flights, predictions):
        truth, _ = flight.targets()
        estimated = np.zeros(len(truth), dtype=int)
        for span in prediction["annotations"]:
            estimated[span["start_index"]:span["end_index"]+1] = span["label"]
        confirmed = np.zeros(len(truth), dtype=bool)
        for event in prediction["events"]:
            mask = flight.time >= event["start_s"]-1e-10
            if not event["ongoing_at_record_end"]:
                mask &= flight.time < event["end_s"]-1e-10
            confirmed |= mask
        predicted, label = prediction["record_label"], flight.label
        row = dict(source_id=flight.source_id, group=flight.group, seed=flight.seed, true_label=label,
                   predicted_label=predicted, record_correct=predicted == label,
                   fault_record=label > 0, record_event_consistency=prediction["record_event_consistency"],
                   confirmed_events=len(prediction["events"]),
                   detection_delay_s=None, localization_delay_s=None,
                   confirmed_fault_interval_iou=None, correct_class_interval_iou=None)
        if label:
            onset, end = flight.fault_start_s, flight.fault_end_s or float("inf")
            events = [e for e in prediction["events"]
                      if e["start_s"] >= onset-1e-10 and onset <= e["detected_at_s"] < end]
            correct_updates = [u for e in events for u in e["class_updates"]
                               if onset <= u["time_s"] < end and u["label"] == label]
            active, correct = truth > 0, estimated == label
            row.update(motor_correct=bool(predicted > 0 and (predicted-1)%6 == (label-1)%6),
                       severity_correct=bool(predicted > 0 and (predicted-1)//6 == (label-1)//6),
                       remaining_effectiveness=(0, .2, .8)[(label-1)//6],
                       detection_delay_s=None if not events else events[0]["detected_at_s"]-onset,
                       localization_delay_s=None if not correct_updates else min(u["time_s"] for u in correct_updates)-onset,
                       confirmed_fault_interval_iou=float((confirmed & active).sum()/(confirmed | active).sum()),
                       correct_class_interval_iou=float((correct & active).sum()/(correct | active).sum()))
        cases.append(row)
    faults = [c for c in cases if c["fault_record"]]
    result = dict(fault_record_motor_accuracy=float(np.mean([c["motor_correct"] for c in faults])),
                  fault_record_severity_accuracy=float(np.mean([c["severity_correct"] for c in faults])),
                  fault_record_joint_accuracy=float(np.mean([c["record_correct"] for c in faults])),
                  mean_confirmed_fault_interval_iou=float(np.mean([c["confirmed_fault_interval_iou"] for c in faults])),
                  mean_correct_class_interval_iou=float(np.mean([c["correct_class_interval_iou"] for c in faults])))
    result["severity_groups"] = {str(eta): dict(records=len(group),
        joint_accuracy=float(np.mean([c["record_correct"] for c in group])),
        motor_accuracy=float(np.mean([c["motor_correct"] for c in group])))
        for eta in (0, .2, .8) if (group := [c for c in faults if c["remaining_effectiveness"] == eta])}
    return result, cases


def evaluate_run(model, scaler, flights, config, folder, device, scope):
    folder = Path(folder)
    if (folder / "baseline_metrics.json").exists():
        return read(folder / "baseline_metrics.json")
    predictions = batched_predictions(model, scaler, flights, config, device)
    checks = []
    # Check healthy and M1 severe trajectories against actual stateful inference.
    for label in (0, 7):
        index = next(i for i, f in enumerate(flights) if f.label == label)
        flight, fast = flights[index], predictions[index]
        streamed = infer_flight(model, scaler, flight.x, flight.time, config["decision_samples"],
                                 EventPolicy(**config["event_policy"]), device)
        error = float(np.max(np.abs(np.asarray(streamed["record_probabilities"])-fast["record_probabilities"])))
        if error > 1e-4 or streamed["record_label"] != fast["record_label"]:
            raise AssertionError("Batched causal replay and persistent streaming disagree")
        event_view = lambda p: [(e["start_s"], e["detected_at_s"], e["end_s"], e["label"]) for e in p["events"]]
        if event_view(streamed) != event_view(fast):
            raise AssertionError("Batched and streaming event boundaries disagree")
        checks.append(dict(source_id=flight.source_id, final_probability_max_error=error,
                           event_boundaries_match=True))
    metrics = export_predictions(model, scaler, flights, config, folder, device,
                                  computed_predictions=predictions)
    extra, cases = detailed_metrics(flights, predictions)
    metrics.update(extra, evaluation_scope=scope, streaming_checks=checks,
                   model_family=config["model"], training_seed=config["seed"], GAN_used=False,
                   event_policy=config["event_policy"],
                   event_policy_status="Same fixed, untuned working point for both baseline families")
    pd.DataFrame(cases).to_csv(folder / "per_record_cases.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics["confusion_matrix"]).to_csv(folder / "record_confusion_matrix.csv", index=False)
    write_json(folder / "baseline_metrics.json", metrics)
    return metrics


def flatten_run(model, seed, training, validation, historical=None):
    row = dict(model=model, training_seed=seed, parameters=training["parameters"],
               selected_epoch=training["selected_epoch"], epochs_completed=training["epochs_completed"],
               optimizer_updates=training["optimizer_updates"],
               training_and_validation_wall_s=training["training_and_validation_wall_s"],
               train_record_accuracy=training["selected_train"]["record_accuracy"],
               train_record_macro_f1=training["selected_train"]["record_macro_f1"])
    for role, score in (("validation", validation), ("historical_test", historical)):
        if score is None:
            continue
        for key in ("record_accuracy", "record_macro_f1", "detection_coverage", "detected", "missed",
                    "first_joint_localization_coverage", "eventual_confirmed_joint_localization_coverage",
                    "healthy_false_alarm_events", "healthy_false_alarm_events_per_minute", "normal_observation_s",
                    "fault_record_motor_accuracy", "fault_record_severity_accuracy", "fault_record_joint_accuracy",
                    "mean_confirmed_fault_interval_iou", "mean_correct_class_interval_iou"):
            row[role+"_"+key] = score[key]
        for key in ("detection_delay_s", "localization_delay_s"):
            for q in ("median", "p95"):
                row[role+"_"+key+"_"+q] = None if score[key] is None else score[key][q]
    return row


def run(config_path, output=None, resume=False, device="cuda"):
    protocol = read(config_path)
    base = read(ROOT / protocol["base_config"])
    output = Path(output) if output else ROOT / protocol["results_root"]
    registry = ROOT / base["development_registry"]
    test_registry = ROOT / protocol["historical_test_registry"]
    identity = dict(protocol=protocol, base_config=base, code=code_hashes(ROOT, BASELINE_CODE),
                    development_registry_sha256=sha256_file(registry),
                    historical_registry_sha256=sha256_file(test_registry))
    signature = stable_hash(identity)
    if output.exists() and any(output.iterdir()):
        if not resume or not (output / "protocol.json").exists():
            raise FileExistsError("Use a new baseline directory or --resume for an identical protocol")
        if read(output / "protocol.json")["signature"] != signature:
            raise ValueError("Baseline protocol/code/source registry changed")
    else:
        write_json(output / "protocol.json", dict(signature=signature, **identity, runtime=runtime_info()))
    flights = load_flights([registry], roles=("train", "validation"))
    write_json(output / "development_sources.json", source_receipt(flights))
    validation_flights = [f for f in flights if f.role == "validation"]
    jobs = []
    for family in protocol["models"]:
        for seed in protocol["training_seeds"]:
            config = copy.deepcopy(base)
            config.update(model=family, seed=seed, cnn_kernels=protocol["cnn_kernels"],
                          cnn_dilations=protocol["cnn_dilations"],
                          allow_tf32=protocol["allow_tf32"],
                          profile=protocol["profile"], baseline_protocol_signature=signature)
            folder = output / f"{family}_seed{seed}"
            write_json(output / "progress.json", dict(stage="training", model=family, seed=seed, completed=len(jobs)))
            print("BASELINE_TRAIN", family, seed, flush=True)
            training = train_model(config, flights, folder, device, resume=resume)
            model, scaler, trained = load_checkpoint(folder / "best.pt", device)
            score = evaluate_run(model, scaler, validation_flights, trained, folder / "validation", device,
                                  "Whole-flight development validation; not a blind test")
            jobs.append(dict(model=family, seed=seed, folder=folder, training=training, validation=score))
            pd.DataFrame([flatten_run(j["model"], j["seed"], j["training"], j["validation"]) for j in jobs]).to_csv(
                output / "development_results.csv", index=False, encoding="utf-8-sig")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # All six model/working-point choices are immutable before historical test
    # measurements are first opened by this new baseline experiment.
    frozen_files = {str(j["folder"] / name): sha256_file(j["folder"] / name)
                    for j in jobs for name in ("best.pt", "last.pt", "job.json")}
    freeze = dict(protocol_signature=signature, code=identity["code"], files=frozen_files,
                   working_point=base["event_policy"],
                   scope="Historical exposed held-out sources; no new blind-test claim")
    freeze_signature = stable_hash(freeze)
    if (output / "freeze.json").exists():
        if read(output / "freeze.json")["signature"] != freeze_signature:
            raise ValueError("Frozen model identity changed")
    else:
        write_json(output / "freeze.json", dict(signature=freeze_signature, **freeze, runtime=runtime_info()))
    for path, digest in frozen_files.items():
        if sha256_file(path) != digest:
            raise ValueError("Frozen checkpoint changed: "+path)
    if code_hashes(ROOT, BASELINE_CODE) != identity["code"]:
        raise ValueError("Code changed while training; historical evaluation is blocked")
    write_json(output / "historical_test_access.json", dict(freeze_signature=freeze_signature,
        purpose="Historical held-out evaluation, not new unseen confirmation", runtime=runtime_info()))
    historical = load_flights([test_registry], roles=("test",))
    validate_partitions(flights+historical)
    write_json(output / "historical_sources.json", source_receipt(historical))
    rows = []
    for index, job in enumerate(jobs):
        folder = job["folder"]
        write_json(output / "progress.json", dict(stage="historical_evaluation", model=job["model"],
            seed=job["seed"], completed=index, total=len(jobs)))
        print("BASELINE_HISTORICAL_EVAL", job["model"], job["seed"], flush=True)
        model, scaler, trained = load_checkpoint(folder / "best.pt", device)
        test_score = evaluate_run(model, scaler, historical, trained, folder / "historical_test", device,
                                  "Historical already-exposed held-out comparison, not a new blind test")
        rows.append(flatten_run(job["model"], job["seed"], job["training"], job["validation"], test_score))
        pd.DataFrame(rows).to_csv(output / "per_seed_results.csv", index=False, encoding="utf-8-sig")
        del model
    frame = pd.DataFrame(rows)
    aggregates = []
    for family, group in frame.groupby("model", sort=False):
        row = dict(model=family, training_repeats=len(group))
        for column in frame.select_dtypes(include=[np.number]).columns:
            if column == "training_seed":
                continue
            finite = group[column].dropna()
            row[column+"_mean"] = None if finite.empty else float(finite.mean())
            row[column+"_std"] = None if finite.empty else float(finite.std(ddof=0))
            row[column+"_valid_repeats"] = len(finite)
        aggregates.append(row)
    pd.DataFrame(aggregates).to_csv(output / "summary_by_model.csv", index=False, encoding="utf-8-sig")
    result = dict(status="completed", profile=protocol["profile"], GAN_used=False,
        unit="complete_flight", training_seeds=protocol["training_seeds"],
        train_records=sum(f.role == "train" for f in flights), validation_records=len(validation_flights),
        historical_test_records=len(historical), total_original_records=len(flights)+len(historical),
        historical_test_scope="Already-exposed source groups; not new blind confirmation",
        event_policy=base["event_policy"], event_policy_tuned=False,
        protocol_signature=signature, freeze_signature=freeze_signature,
        statistics_scope="Mean and population std over training seeds; independent source groups unchanged",
        results=rows, aggregates=aggregates, runtime=runtime_info())
    write_json(output / "results.json", result)
    write_json(output / "progress.json", dict(stage="completed", runs=len(jobs), GAN_used=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run(args.config, args.output, args.resume, args.device)
    print("BASELINES_COMPLETE", result["total_original_records"], "records", flush=True)
