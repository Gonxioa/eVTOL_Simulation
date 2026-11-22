"""Bounded full-flight execution verification, not classifier retraining."""
from pathlib import Path

import numpy as np
import torch

from provenance import code_hashes, runtime_info, stable_hash, write_json
from simulink_data import ROOT
from simulink_sequence_data import collate, fit_scaler, source_receipt, validate_partitions
from simulink_sequence_events import EventPolicy, infer_flight
from simulink_sequence_model import build_model, loss_for_flights


def verify_flights(config, flights, output, device="cpu"):
    """Audit all supplied flights, make ONE update and check causal inference.

    The temporary model starts from random weights. No checkpoint or accuracy
    claim is produced. Every numerical source must be train or validation.
    """
    device = torch.device(device)
    validate_partitions(flights)
    if any(f.role == "test" for f in flights):
        raise PermissionError("Execution verification does not open held-out test records")
    train = [f for f in flights if f.role == "train"]
    validation = [f for f in flights if f.role == "validation"]
    if {f.label for f in train} != set(range(19)) or not validation:
        raise ValueError("Need all 19 training classes and separate validation flights")
    if config["decision_samples"] < 1:
        raise ValueError("Decision period must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new directory for the execution verification")
    output.mkdir(parents=True, exist_ok=True)
    receipt = source_receipt(flights)
    write_json(output / "sources.json", receipt)
    write_json(output / "job.json", dict(
        purpose="One-update complete-flight execution check; not formal training or performance evaluation",
        config=config, source_signature=stable_hash(receipt), runtime=runtime_info(),
        code=code_hashes(ROOT, ["simulink_sequence_data.py", "simulink_sequence_model.py",
                               "simulink_sequence_events.py", "simulink_sequence_verify.py",
                               "run_simulink_sequence.py", "simulink_data.py", "provenance.py"])))
    print("SEQUENCE_VERIFY", len(train), "training flights,", len(validation), "validation flights", flush=True)
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    model = build_model(config).to(device)
    scaler = fit_scaler(train)
    training_points = sum(len(f.x) for f in train)
    if int(scaler.n_samples_seen_) != training_points:
        raise AssertionError("Scaler must consume each training time sample exactly once")

    # Keep whole records: healthy, total loss, M1 severe and M4 severe.
    selected = [next(f for f in train if f.label == label) for label in (0, 1, 7, 10)]
    x, lengths, current, seen, labels = collate(selected, scaler, device)
    x.requires_grad_()
    initial = {k: value.detach().clone() for k, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    model.train()
    result = model(x, lengths=lengths)
    record_loss = torch.nn.functional.cross_entropy(result["record_logits"], labels)
    record_gradient = torch.autograd.grad(record_loss, x, retain_graph=True)[0]
    early_gradient = float(record_gradient[:, :min(20, x.shape[1])].abs().sum())
    if not np.isfinite(early_gradient) or early_gradient <= 0:
        raise AssertionError("Final record supervision must reach early measured samples")
    loss, _ = loss_for_flights(result, lengths, current, seen, labels,
        period=config["decision_samples"], record_weight=config["record_loss_weight"],
        prefix_weight=config["prefix_loss_weight"], current_weight=config["current_loss_weight"])
    if not torch.isfinite(loss):
        raise AssertionError("Nonfinite complete-flight loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
    if not torch.isfinite(norm):
        raise AssertionError("Nonfinite complete-flight gradient")
    optimizer.step()
    parameter_delta = max(float((value.detach()-initial[key]).abs().max())
                          for key, value in model.state_dict().items())
    if parameter_delta <= 0:
        raise AssertionError("Optimizer did not update the temporary model")
    print("SEQUENCE_VERIFY one complete-flight optimization update passed", flush=True)

    # Verify exactly one final result per validation flight, without reporting
    # classification scores from this deliberately untrained execution model.
    model.eval()
    with torch.inference_mode():
        evaluated = 0
        for start in range(0, len(validation), config["batch_flights"]):
            batch = validation[start:start+config["batch_flights"]]
            bx, bl, *_ = collate(batch, scaler, device)
            logits = model(bx, lengths=bl)["record_logits"]
            if logits.shape != (len(batch), 19) or not torch.isfinite(logits).all():
                raise AssertionError("Expected one finite 19-class output per complete validation flight")
            evaluated += len(batch)

    stream_checks = []
    examples = [next(f for f in validation if f.label == label) for label in (0, 7, 10)
                if any(f.label == label for f in validation)]
    for index, flight in enumerate(examples):
        bx, bl, *_ = collate([flight], scaler, device)
        with torch.inference_mode():
            full = model(bx, lengths=bl)
            full_probability = torch.softmax(full["record_logits"][0], 0).cpu().numpy()
            changed = bx.clone()
            cutoff = max(1, len(flight.x)//2)
            changed[:, cutoff:] = changed[:, cutoff:] + 100
            changed_result = model(changed, lengths=bl)
            causal_error = max(float((full[key][:, :cutoff]-changed_result[key][:, :cutoff]).abs().max())
                               for key in ("current_logits", "prefix_logits"))
        if causal_error > 1e-6:
            raise AssertionError("Future measurements altered previous causal predictions")
        errors = []
        for period in (config["decision_samples"], config["decision_samples"]+7):
            prediction = infer_flight(model, scaler, flight.x, flight.time, period,
                                      EventPolicy(**config["event_policy"]), device)
            error = float(np.max(np.abs(full_probability-np.asarray(prediction["record_probabilities"]))))
            if error > 1e-4:
                raise AssertionError("Changing computation chunk boundaries changed the whole-flight output")
            samples = np.asarray([r["samples_seen"] for r in prediction["trace"]])
            expected = np.r_[np.arange(period, len(flight.x), period), len(flight.x)]
            if not np.array_equal(samples, expected):
                raise AssertionError("Streaming skipped or reread time samples")
            spans = prediction["annotations"]
            if spans[0]["start_index"] != 0 or spans[-1]["end_index"] != len(flight.x)-1:
                raise AssertionError("Annotations must cover the entire original recording")
            if any(b["start_index"] != a["end_index"]+1 for a, b in zip(spans, spans[1:])):
                raise AssertionError("Annotations have gaps or overlapping rows")
            errors.append(error)
            if period == config["decision_samples"]:
                write_json(output / f"example_{index:02d}_execution_only.json", dict(
                    scope="Temporary model after ONE update; not diagnostic results",
                    source_id=flight.source_id, **prediction))
        stream_checks.append(dict(source_id=flight.source_id, points=len(flight.x),
                                  periods_checked=[config["decision_samples"], config["decision_samples"]+7],
                                  final_probability_max_errors=errors, future_prefix_max_error=causal_error))
        print("SEQUENCE_VERIFY persistent history and causal annotations:", flight.source_id, flush=True)

    summary = dict(status="passed", purpose="Native-data execution verification, not formal retraining",
        synthetic_data_used=bool(config.get("synthetic_smoke", False)), formal_training_performed=False,
        performance_evaluation_performed=False, test_numeric_data_opened=False,
        records_audited=len(flights), train_records=len(train), validation_records=len(validation),
        independent_groups=receipt["independent_groups"],
        input_shapes=sorted({(len(f.x), f.x.shape[1]) for f in flights}),
        source_signature=stable_hash(receipt), scaler_time_samples=int(scaler.n_samples_seen_),
        expected_training_time_samples=training_points, optimizer_updates=1,
        verification_update_source_ids=[f.source_id for f in selected],
        record_loss_early_input_gradient_l1=early_gradient, parameter_max_change=parameter_delta,
        validation_whole_record_outputs=evaluated, stream_checks=stream_checks,
        fault_onsets_s=sorted({f.fault_start_s for f in flights if f.label}),
        recorded_durations_s=sorted({float(f.time[-1]-f.time[0]) for f in flights}),
        generalization_scope="Fixed-onset saved flights do not establish arbitrary-onset or longer-flight performance",
        device=str(device), hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
    write_json(output / "verification.json", summary)
    return summary
