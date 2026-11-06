"""Train whole flights; export one record label and causal fault-event segments."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch

from provenance import code_hashes, runtime_info, stable_hash, write_json
from simulink_data import ROOT, read
from simulink_sequence_data import Flight, collate, fit_scaler, load_flights, source_receipt, validate_partitions
from simulink_sequence_events import EventPolicy, infer_flight, metrics
from simulink_sequence_model import FlightClassifier, loss_for_flights

CONFIG = ROOT / "configs/simulink_sequence.json"
CODE = ["simulink_sequence_data.py", "simulink_sequence_model.py", "simulink_sequence_events.py",
        "run_simulink_sequence.py", "simulink_data.py", "provenance.py"]


def architecture(config):
    return FlightClassifier(config["hidden"], config["layers"], config["dropout"])


def _checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def record_scores(model, scaler, flights, config, device):
    """Exactly one prediction and one primary target per complete flight."""
    model.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, len(flights), config["batch_flights"]):
            x, length, *_ = collate(flights[start:start+config["batch_flights"]], scaler, device)
            logits.append(model(x, lengths=length)["record_logits"].cpu())
    logits = torch.cat(logits)
    y = np.asarray([f.label for f in flights])
    predicted = logits.argmax(1).numpy()
    return dict(records=len(flights), record_accuracy=float(accuracy_score(y, predicted)),
                record_macro_f1=float(f1_score(y, predicted, labels=np.arange(19), average="macro", zero_division=0)),
                record_CE=float(torch.nn.functional.cross_entropy(logits, torch.as_tensor(y))),
                predicted=predicted.tolist())


def train_model(config, flights, output, device="cpu", resume=False):
    device = torch.device(device)
    torch.set_num_threads(4)
    validate_partitions(flights)
    if any(f.role == "test" for f in flights):
        raise PermissionError("Do not pass numerical test flights to training")
    train = [f for f in flights if f.role == "train"]
    validation = [f for f in flights if f.role == "validation"]
    if not validation or {f.label for f in train} != set(range(19)):
        raise ValueError("Need all 19 training classes and independent validation flights")
    if not config["batch_flights"] > 0 or not config["max_epochs"] > 0 or not config["decision_samples"] > 0:
        raise ValueError("Invalid training/decision geometry")
    identity = dict(config=config, sources=source_receipt(flights), code=code_hashes(ROOT, CODE),
                    task="whole-flight classification with causal prefix/event diagnosis")
    signature = stable_hash(identity)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        if not resume or not (output/"last.pt").exists():
            raise FileExistsError("Use a new output directory, or --resume for the identical job")
        job = read(output/"job.json")
        if job["signature"] != signature:
            raise ValueError("Source/config/code changed; optimizer cannot resume")
        if (output/"summary.json").exists():
            return read(output/"summary.json")
    output.mkdir(parents=True, exist_ok=True)
    if not (output/"job.json").exists():
        write_json(output/"job.json", dict(signature=signature, **identity, runtime=runtime_info()))
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(config["seed"])
    model = architecture(config).to(device)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scaler = fit_scaler(train)
    history = []
    first_epoch = 1
    updates = 0
    best_score = (-float("inf"), -float("inf"))
    wait = 0
    if resume and (output/"last.pt").exists():
        saved = torch.load(output/"last.pt", map_location=device, weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler = saved["scaler"]
        initial = {k: v.cpu() for k, v in saved["initial"].items()}
        first_epoch = saved["epoch"]+1
        history, updates, wait = saved["history"], saved["updates"], saved["wait"]
        best_score = tuple(saved["best_score"])
        rng.bit_generator.state = saved["sampler_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
    for epoch in range(first_epoch, config["max_epochs"]+1):
        if wait >= config["patience"]:
            break
        # Shuffle complete flights; keep EVERY sample inside each flight ordered.
        order = rng.permutation(len(train))
        losses = []
        for start in range(0, len(order), config["batch_flights"]):
            selected = [train[i] for i in order[start:start+config["batch_flights"]]]
            x, lengths, current, seen, labels = collate(selected, scaler, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            result = model(x, lengths=lengths)  # One full-trajectory forward pass.
            loss, components = loss_for_flights(result, lengths, current, seen, labels,
                period=config["decision_samples"], record_weight=config["record_loss_weight"],
                prefix_weight=config["prefix_loss_weight"], current_weight=config["current_loss_weight"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite trajectory loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite trajectory gradient")
            optimizer.step()
            updates += 1
            losses.append(dict(loss=float(loss.detach()), **components))
        train_score = record_scores(model, scaler, train, config, device)
        val_score = record_scores(model, scaler, validation, config, device)
        row = dict(epoch=epoch, updates=updates, train=train_score, validation=val_score,
                   mean_loss={name: float(np.mean([r[name] for r in losses])) for name in losses[0]})
        history.append(row)
        score = (val_score["record_macro_f1"], -val_score["record_CE"])
        improved = score > best_score
        wait = 0 if improved else wait+1
        best_score = max(best_score, score)
        delta = max(float((v.detach().cpu()-initial[k]).abs().max()) for k, v in model.state_dict().items())
        saved = dict(signature=signature, config=config, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scaler=scaler, epoch=epoch, updates=updates, wait=wait, best_score=best_score, history=history,
                     initial=initial, parameter_max_delta=delta, sampler_rng=rng.bit_generator.state,
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        if improved:
            _checkpoint(output/"best.pt", saved)
        _checkpoint(output/"last.pt", saved)
        write_json(output/"history.json", history)
        print("SEQUENCE_EPOCH", epoch, "updates", updates, "train_record_F1", round(train_score["record_macro_f1"], 5),
              "validation_record_F1", round(val_score["record_macro_f1"], 5), flush=True)
    best = torch.load(output/"best.pt", map_location="cpu", weights_only=False)
    summary = dict(signature=signature, status="completed", training_unit="complete_flight",
                   train_records=len(train), validation_records=len(validation),
                   train_time_samples=sum(len(f.x) for f in train), optimizer_updates=updates,
                   epochs_completed=len(history), selected_epoch=best["epoch"],
                   selected_validation=best["history"][-1]["validation"],
                   parameters=sum(p.numel() for p in model.parameters()),
                   event_policy_status="Untuned config defaults; record selection does not validate event thresholds",
                   new_blind_test_performed=False, synthetic_data_used=bool(config.get("synthetic_smoke", False)))
    write_json(output/"summary.json", summary)
    return summary


def load_checkpoint(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = architecture(saved["config"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved["scaler"], saved["config"]


def export_predictions(model, scaler, flights, config, output, device):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Annotation output already exists; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    records = []
    segments = []
    for index, flight in enumerate(flights):
        # Deliberately exclude label/onset/end/role/group from the inference API.
        result = infer_flight(model, scaler, flight.x, flight.time, config["decision_samples"],
                              EventPolicy(**config["event_policy"]), device)
        predictions.append(result)
        records.append(dict(source_id=flight.source_id, role=flight.role, group=flight.group, seed=flight.seed,
                            points=len(flight.x), true_label=flight.label, predicted_label=result["record_label"],
                            predicted_label_name=result["record_label_name"], confidence=result["record_confidence"]))
        segments.extend(dict(source_id=flight.source_id, **a) for a in result["annotations"])
        name = f"{index:04d}_"+re.sub(r"[^A-Za-z0-9_-]", "_", flight.source_id)
        pd.DataFrame(result["trace"]).to_csv(output/(name+"_trace.csv"), index=False)
        write_json(output/(name+"_events.json"), {k: v for k, v in result.items() if k != "trace"})
    pd.DataFrame(records).to_csv(output/"record_predictions.csv", index=False)
    pd.DataFrame(segments).to_csv(output/"segment_annotations.csv", index=False)
    score = metrics(flights, predictions)
    scope = ("Synthetic execution check only, not a simulation-domain result" if config.get("synthetic_smoke")
             else "Development/historical diagnostic, not a new blind confirmation")
    score.update(evaluation_scope=scope, synthetic_data_used=bool(config.get("synthetic_smoke", False)),
                 event_policy_status="Config working point; no reliability acceptance claim",
                 sources=source_receipt(flights), runtime=runtime_info())
    write_json(output/"metrics.json", score)
    return score


def synthetic_flights(seed=19):
    """Small 19-class native-rate fixtures; NOT the user's simulation data."""
    rng = np.random.default_rng(seed)
    flights = []
    time = np.arange(401)/100
    for group in range(3):
        for label in range(19):
            x = rng.normal(0, .03, (len(time), 12)).astype(np.float32)
            onset = 1.+.1*group
            if label:
                after = time >= onset
                x[after, 6+(label-1)%6] -= 1.
                x[after, 0] += .3+((label-1)//6)*.4
            flights.append(Flight(f"synthetic_g{group}_c{label}", "train" if group < 2 else "validation",
                                  group, seed+group, x, time, label, onset if label else None))
    return flights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "train", "annotate", "demo"))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--registry", action="append", help="Native signed source registry; repeat for separate roles")
    parser.add_argument("--source-root", default=str(ROOT), help="Project root when native paths have moved")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--role", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = read(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    output = Path(args.output) if args.output else ROOT/config["results_root"]
    if args.stage == "demo":
        if not args.output:
            parser.error("--output is required for the synthetic demo")
        config.update(hidden=8, layers=1, dropout=0., batch_flights=19, max_epochs=2, patience=2,
                      synthetic_smoke=True, profile="whole_flight_synthetic_execution_check")
        flights = synthetic_flights()
        summary = train_model(config, flights, output, device, args.resume)
        summary.update(synthetic_data_used=True, purpose="Execution smoke test only, not a simulation-domain result")
        write_json(output/"summary.json", summary)
        model, scaler, config = load_checkpoint(output/"best.pt", device)
        export_predictions(model, scaler, [f for f in flights if f.role == "validation"],
                           config, output/"synthetic_annotations", device)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    registries = args.registry or [ROOT/config["development_registry"]]
    roles = (args.role,) if args.stage == "annotate" else ("train", "validation")
    flights = load_flights(registries, roles, args.source_root)
    if args.stage == "audit":
        receipt = source_receipt(flights)
        print(json.dumps(dict(records=len(flights), groups=receipt["independent_groups"],
                             time_samples=sum(len(f.x) for f in flights), unit=receipt["unit"],
                             features=receipt["feature_names"]), ensure_ascii=False, indent=2))
    elif args.stage == "train":
        print(json.dumps(train_model(config, flights, output, device, args.resume), ensure_ascii=False, indent=2))
    else:
        if not args.checkpoint or not args.output:
            parser.error("annotate requires --checkpoint and a new --output directory")
        model, scaler, trained_config = load_checkpoint(args.checkpoint, device)
        # An explicit config can change an event working point without changing
        # the trained model; retain it in every annotation and metric receipt.
        trained_config["event_policy"] = config["event_policy"]
        trained_config["decision_samples"] = config["decision_samples"]
        print(json.dumps(export_predictions(model, scaler, flights, trained_config, output, device),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
