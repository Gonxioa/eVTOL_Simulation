"""Offline truth is used here for scoring only, never by the incremental object."""
from dataclasses import asdict
import copy

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch

from simulink_sequence_data import collate
from simulink_sequence_events import EventPolicy
from simulink_realtime import CurrentEventTracker, event_record_label, SEMANTICS


def decisions_from_probabilities(time, probabilities, policy, period=10):
    """Validation cache optimization; the one shared online tracker consumes decisions.

    Uses only outputs at the canonical cumulative sample counts. Never flushes an
    incomplete interval into a decision. Public file/live entry is replay_flight.
    """
    tracker = CurrentEventTracker(policy, period*.01)
    trace = []
    for i in range(period-1, len(time), period):
        p = probabilities[i]
        state = tracker.observe(float(time[i]), p)
        trace.append(dict(time_s=float(time[i]), samples_seen=i+1, current_label=int(p.argmax()),
                          fault_probability=float(1-p[0]), probabilities=p.tolist(),
                          alarm=state["alarm"], confirmed_label=state["label"]))
    events = tracker.finish(float(time[-1]))
    return dict(events=events, trace=trace, record_label=event_record_label(events),
                pending_event=tracker.pending, samples=len(time), semantics=SEMANTICS)


def probability_sequences(model, scaler, flights, device="cpu", batch=4):
    """Causal batched evaluation cache; no fault metadata is passed to the model."""
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flights), batch):
            selected = flights[start:start+batch]
            x, lengths, *_ = collate(selected, scaler, device)
            output = model(x, lengths=lengths)
            probabilities = torch.softmax(output["current_logits"], -1).cpu().numpy()
            for i, f in enumerate(selected):
                result.append(probabilities[i, :len(f.x)].copy())
    return result


def _classification(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=int)
    if not len(y):
        return dict(points=0, accuracy=None, macro_f1=None, class_recall=[None]*19)
    cm = confusion_matrix(y, p, labels=np.arange(19))
    return dict(points=len(y), accuracy=float(accuracy_score(y, p)),
        macro_f1=float(f1_score(y, p, labels=np.arange(19), average="macro", zero_division=0)),
        class_recall=(np.diag(cm)/np.maximum(cm.sum(1), 1)).tolist(),
        class_support=cm.sum(1).tolist(), confusion_matrix=cm.tolist())


def _quantiles(values):
    finite = [v for v in values if v is not None]
    return dict(count=len(finite), median=None if not finite else float(np.median(finite)),
                p95=None if not finite else float(np.quantile(finite, .95)),
                scope="Conditional on observed success; all misses separately reported")


def score_events(flights, predictions):
    """Each real fault episode matches its first NEW formal alarm after onset.

    Alarms already active at onset do not become true positives retroactively.
    Further new alarms during the same fault are repeats. Correct localization
    is joint motor+severity; later corrections are allowed but first errors stay.
    """
    if len(flights) != len(predictions) or not flights:
        raise ValueError("One prediction per source flight required")
    cases = []
    for flight, result in zip(flights, predictions):
        events = result["events"]
        trace = result["trace"]
        first = trace[0]["time_s"] if trace else float(flight.time[-1])
        last = trace[-1]["time_s"] if trace else first
        observation = max(0., last-first)
        onset = flight.fault_start_s
        end = float(flight.time[-1])+1e-8 if flight.fault_end_s is None else flight.fault_end_s
        row = dict(source_id=flight.source_id, role=flight.role, group=flight.group, seed=flight.seed,
            label=flight.label, motor=0 if not flight.label else (flight.label-1)%6+1,
            remaining_effectiveness=1. if not flight.label else (0., .2, .8)[(flight.label-1)//6],
            healthy_observation_s=observation if not flight.label else 0.,
            prefault_observation_s=0. if not flight.label else max(0., min(onset, last)-first),
            healthy_false_alarm_events=len(events) if not flight.label else 0,
            prefault_false_alarm_events=0 if not flight.label else sum(e["detected_at_s"] < onset-1e-9 for e in events),
            detection_delay_s=None, localization_delay_s=None, first_localization_wrong=None,
            repeat_alarms=0, wrong_clears=0, record_prediction=result["record_label"],
            record_correct=result["record_label"] == flight.label,
            active_false_alarm_at_onset=False)
        if flight.label:
            row["active_false_alarm_at_onset"] = any(e["detected_at_s"] < onset and
                (e["cleared_at_s"] is None or e["cleared_at_s"] >= onset) for e in events)
            fresh = [e for e in events if onset-1e-9 <= e["detected_at_s"] < end]
            if fresh:
                row["detection_delay_s"] = max(0., fresh[0]["detected_at_s"]-onset)
                row["repeat_alarms"] = len(fresh)-1
                updates = sorted([u for e in fresh for u in e["class_updates"] if onset <= u["time_s"] < end],
                                 key=lambda u: u["time_s"])
                if updates:
                    row["first_localization_wrong"] = updates[0]["label"] != flight.label
                correct = [u for u in updates if u["label"] == flight.label]
                if correct:
                    row["localization_delay_s"] = correct[0]["time_s"]-onset
            row["wrong_clears"] = sum(e["cleared_at_s"] is not None and onset <= e["cleared_at_s"] < end for e in events)
        for point in (.5, 1., 3.):
            for name in ("detection", "localization"):
                delay = row[name+"_delay_s"]
                row[f"{name}_within_{point:g}s"] = delay is not None and delay <= point+1e-9
        cases.append(row)
    faults = [r for r in cases if r["label"]]
    n = len(faults)
    healthy_seconds = sum(r["healthy_observation_s"] for r in cases)
    pre_seconds = sum(r["prefault_observation_s"] for r in cases)
    healthy_fp = sum(r["healthy_false_alarm_events"] for r in cases)
    pre_fp = sum(r["prefault_false_alarm_events"] for r in cases)
    first_known = [r for r in faults if r["first_localization_wrong"] is not None]
    score = dict(records=len(flights), fault_events=n,
        detected=sum(r["detection_delay_s"] is not None for r in faults),
        always_missed=sum(r["detection_delay_s"] is None for r in faults),
        never_joint_localized=sum(r["localization_delay_s"] is None for r in faults),
        detection_delay_s=_quantiles([r["detection_delay_s"] for r in faults]),
        localization_delay_s=_quantiles([r["localization_delay_s"] for r in faults]),
        first_localization_wrong=sum(bool(r["first_localization_wrong"]) for r in first_known),
        first_localization_count=len(first_known),
        first_localization_error_fraction=None if not first_known else float(np.mean([r["first_localization_wrong"] for r in first_known])),
        healthy_false_alarm_events=healthy_fp, healthy_observation_s=healthy_seconds,
        healthy_false_alarms_per_minute=healthy_fp*60/healthy_seconds if healthy_seconds else None,
        prefault_false_alarm_events=pre_fp, prefault_observation_s=pre_seconds,
        prefault_false_alarms_per_minute=pre_fp*60/pre_seconds if pre_seconds else None,
        prefault_exposure_scope="All fault records' prefixes, paired correlated repeats; not added to unique healthy exposure",
        repeat_alarms=sum(r["repeat_alarms"] for r in faults), wrong_clears=sum(r["wrong_clears"] for r in faults),
        active_false_alarm_at_onset=sum(r["active_false_alarm_at_onset"] for r in faults),
        record_accuracy=float(np.mean([r["record_correct"] for r in cases])),
        record_macro_f1=float(f1_score([r["label"] for r in cases], [r["record_prediction"] for r in cases],
                                       labels=np.arange(19), average="macro", zero_division=0)),
        record_summary="last confirmed online event class, unresolved=-1, no alarm=healthy")
    for point in (.5, 1., 3.):
        for name in ("detection", "localization"):
            score[f"{name}_within_{point:g}s"] = sum(r[f"{name}_within_{point:g}s"] for r in faults)/n if n else None
    by_severity = {}
    for eff, sev_name in ((0.0, "full_0pct"), (0.2, "severe_20pct"), (0.8, "partial_80pct")):
        sub = [r for r in faults if abs(r["remaining_effectiveness"] - eff) < 1e-9]
        if not sub:
            continue
        by_severity[sev_name] = dict(
            records=len(sub),
            always_missed=sum(r["detection_delay_s"] is None for r in sub),
            never_joint_localized=sum(r["localization_delay_s"] is None for r in sub),
            detection_within_0_5s=sum(r["detection_within_0.5s"] for r in sub)/len(sub),
            detection_within_1s=sum(r["detection_within_1s"] for r in sub)/len(sub),
            detection_within_3s=sum(r["detection_within_3s"] for r in sub)/len(sub),
            localization_within_0_5s=sum(r["localization_within_0.5s"] for r in sub)/len(sub),
            localization_within_1s=sum(r["localization_within_1s"] for r in sub)/len(sub),
            localization_within_3s=sum(r["localization_within_3s"] for r in sub)/len(sub),
        )
    score["by_severity"] = by_severity
    if by_severity:
        score["min_severity_detection_within_1s"] = min(v["detection_within_1s"] for v in by_severity.values())
        score["min_severity_localization_within_1s"] = min(v["localization_within_1s"] for v in by_severity.values())
        score["min_severity_localization_within_3s"] = min(v["localization_within_3s"] for v in by_severity.values())
    else:
        score["min_severity_detection_within_1s"] = None
        score["min_severity_localization_within_1s"] = None
        score["min_severity_localization_within_3s"] = None
    score["late_detection_over_3s"] = sum(r["detection_delay_s"] is not None and r["detection_delay_s"] > 3+1e-9 for r in faults)
    score["late_localization_over_3s"] = sum(r["localization_delay_s"] is not None and r["localization_delay_s"] > 3+1e-9 for r in faults)
    score["observation_horizons_s"] = [.5, 1., 3.]
    score["safety_deadline_validated"] = False
    return score, cases


def score_states(flights, probabilities):
    stages = {s: ([], []) for s in ("all", "healthy", "early_0_3s", "sustained_after_3s")}
    ces = {s: [] for s in stages}
    for f, p in zip(flights, probabilities):
        y, _ = f.targets()
        predicted = p.argmax(1)
        early = np.zeros(len(y), dtype=bool) if not f.label else (f.time >= f.fault_start_s) & (f.time < f.fault_start_s+3) & (y > 0)
        masks = dict(all=np.ones(len(y), dtype=bool), healthy=y == 0,
                     early_0_3s=early, sustained_after_3s=(y > 0) & ~early)
        loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
        for s, mask in masks.items():
            stages[s][0].extend(y[mask].tolist())
            stages[s][1].extend(predicted[mask].tolist())
            ces[s].extend(loss[mask].tolist())
    return {s: dict(_classification(*values), CE=float(np.mean(ces[s])) if ces[s] else None) for s, values in stages.items()}


def policy_rank(score, state_macro_f1):
    """Predeclared validation rule; prevents selecting severity-blind checkpoints."""
    healthy = score["healthy_false_alarms_per_minute"] or 0.
    pre = score["prefault_false_alarms_per_minute"] or 0.
    feasible = healthy <= 1.+1e-9 and pre <= 1.+1e-9
    excess = max(0., healthy-1)+max(0., pre-1)
    min_det_1s = score.get("min_severity_detection_within_1s") or 0.
    min_loc_1s = score.get("min_severity_localization_within_1s") or 0.
    all_tiers_active = int(min_det_1s > 0.0)
    return (
        int(feasible),
        -excess,
        all_tiers_active,
        score["localization_within_1s"] or 0.,
        min_loc_1s,
        score["localization_within_3s"] or 0.,
        score["detection_within_1s"] or 0.,
        state_macro_f1,
        -(score["first_localization_error_fraction"] or 0.),
        -healthy-pre,
        -(score["localization_delay_s"]["p95"] or 1000.),
    )


def choose_policy(flights, probabilities, policies, period=10):
    state = score_states(flights, probabilities)
    candidates = []
    for index, policy in enumerate(policies):
        cfg = EventPolicy(**policy)
        predictions = [decisions_from_probabilities(f.time, p, cfg, period) for f, p in zip(flights, probabilities)]
        score, _ = score_events(flights, predictions)
        rank = policy_rank(score, state["all"]["macro_f1"])
        candidates.append(dict(index=index, policy=asdict(cfg), metrics=score, rank=list(rank)))
    selected = max(candidates, key=lambda r: (*r["rank"], -r["index"]))
    return dict(selected=copy.deepcopy(selected), candidates=candidates, states=state)