"""Causal event assembly and whole-record inference; no fault truth is an input."""
from dataclasses import asdict, dataclass

import numpy as np
import torch

from simulink_sequence_data import label_name


@dataclass
class EventPolicy:
    fault_threshold: float = .8
    clear_threshold: float = .3
    class_threshold: float = .6
    confirm_s: float = .3
    localize_s: float = .3
    clear_s: float = .5

    def __post_init__(self):
        if not 0 <= self.clear_threshold < self.fault_threshold <= 1:
            raise ValueError("Require clear_threshold < fault_threshold")
        if not 0 <= self.class_threshold <= 1 or min(self.confirm_s, self.localize_s, self.clear_s) < 0:
            raise ValueError("Invalid event confirmation policy")


class EventTracker:
    def __init__(self, policy):
        self.policy = policy
        self.events = []
        self.active = None
        self.candidate_start = None
        self.clear_start = None
        self.class_candidate = None
        self.class_start = None
        self.previous_time = -float("inf")
        self.finished = False
        self.pending = None

    def observe(self, time_s, current_probabilities, prefix_probabilities):
        """Consume one NEW chronological decision, never a labelled window."""
        if self.finished or not np.isfinite(time_s) or time_s <= self.previous_time:
            raise ValueError("Decisions must be strictly chronological within one flight")
        self.previous_time = float(time_s)
        for p in (current_probabilities, prefix_probabilities):
            if np.shape(p) != (19,) or not np.isfinite(p).all() or np.any(np.asarray(p) < 0) or not np.isclose(np.sum(p), 1):
                raise ValueError("Expected finite 19-class probabilities")
        policy = self.policy
        fault_probability = 1-float(current_probabilities[0])
        label = int(np.argmax(prefix_probabilities))
        confidence = float(prefix_probabilities[label])
        positive = fault_probability >= policy.fault_threshold
        class_evidence = label if positive and label > 0 and confidence >= policy.class_threshold else None
        if class_evidence != self.class_candidate:
            self.class_candidate = class_evidence
            self.class_start = float(time_s) if class_evidence is not None else None
        if self.active is None:
            if positive:
                if self.candidate_start is None:
                    self.candidate_start = float(time_s)
                if time_s-self.candidate_start+1e-9 >= policy.confirm_s:
                    self.active = dict(start_s=self.candidate_start, detected_at_s=float(time_s),
                                       end_s=None, cleared_at_s=None, ongoing_at_record_end=False,
                                       label=None, label_name="Unresolved", class_updates=[])
                    self.events.append(self.active)
            else:
                self.candidate_start = None
        if self.active is not None:
            if class_evidence is not None and time_s-self.class_start+1e-9 >= policy.localize_s:
                if self.active["label"] != class_evidence:
                    self.active["class_updates"].append(dict(time_s=float(time_s), label=class_evidence,
                        label_name=label_name(class_evidence), confidence=confidence))
                self.active["label"] = class_evidence
                self.active["label_name"] = label_name(class_evidence)
            if fault_probability <= policy.clear_threshold:
                if self.clear_start is None:
                    self.clear_start = float(time_s)
                if time_s-self.clear_start+1e-9 >= policy.clear_s:
                    self.active.update(end_s=self.clear_start, cleared_at_s=float(time_s))
                    self.active = None
                    self.candidate_start = None
                    self.class_candidate = None
                    self.class_start = None
                    self.clear_start = None
            else:
                self.clear_start = None

    def finish(self, last_time_s):
        if self.finished or not np.isfinite(last_time_s) or last_time_s < self.previous_time:
            raise ValueError("Invalid record end")
        if self.active is not None:
            self.active.update(end_s=float(last_time_s), ongoing_at_record_end=True)
        elif self.candidate_start is not None:
            # Running out of recorded time is not evidence of health. Keep the
            # unconfirmed tail separate from confirmed fault/alarm events.
            self.pending = dict(start_s=self.candidate_start, end_s=float(last_time_s),
                                detected_at_s=None, label=None, label_name="Unresolved",
                                ongoing_at_record_end=True, status="pending_confirmation")
        self.finished = True
        return self.events


def annotations(time, events, pending_event=None):
    """Partition original rows into estimated healthy/fault/unresolved segments.

    A final event class can annotate its cached earlier samples retrospectively;
    detected_at_s/class_updates retain when that evidence became available.
    No estimated boundary is silently replaced with a true injection time.
    """
    labels = np.zeros(len(time), dtype=np.int64)
    for event in [*events, *([] if pending_event is None else [pending_event])]:
        start = int(np.searchsorted(time, event["start_s"], side="left"))
        end = len(time) if event["ongoing_at_record_end"] else int(np.searchsorted(time, event["end_s"], side="left"))
        labels[start:end] = -1 if event["label"] is None else event["label"]
    cuts = np.r_[0, np.flatnonzero(np.diff(labels) != 0)+1, len(labels)]
    return [dict(start_index=int(a), end_index=int(b-1), start_s=float(time[a]), end_s=float(time[b-1]),
                 label=int(labels[a]), label_name=label_name(int(labels[a])), boundary_status="model_estimate")
            for a, b in zip(cuts[:-1], cuts[1:])]


def infer_flight(model, scaler, x, time, decision_samples=10, policy=None, device="cpu"):
    """Only measured samples and their chronology enter inference.

    NEW contiguous samples advance memory once. Reading a computational chunk
    does not discard the previous history or produce an independent sample.
    """
    if decision_samples < 1:
        raise ValueError("decision_samples must be positive")
    x = np.asarray(x, dtype=np.float32)
    time = np.asarray(time, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 12 or len(x) != len(time) or not len(x):
        raise ValueError("Expected one complete measured flight")
    if not np.isfinite(x).all() or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Nonfinite/unordered flight")
    values = scaler.transform(x).astype(np.float32)
    model.eval()
    tracker = EventTracker(policy or EventPolicy())
    memory = None
    trace = []
    with torch.inference_mode():
        for start in range(0, len(values), decision_samples):
            end = min(start+decision_samples, len(values))
            bx = torch.as_tensor(values[None, start:end], device=device)
            result = model(bx, memory)
            memory = result["memory"]
            current = torch.softmax(result["current_logits"][0, -1], dim=0).cpu().numpy()
            prefix = torch.softmax(result["record_logits"][0], dim=0).cpu().numpy()
            t = float(time[end-1])
            tracker.observe(t, current, prefix)
            trace.append(dict(time_s=t, samples_seen=end, current_label=int(current.argmax()),
                              fault_probability=float(1-current[0]),
                              prefix_label=int(prefix.argmax()), prefix_confidence=float(prefix.max())))
    return finish_prediction(time, trace, prefix, tracker)


def finish_prediction(time, trace, prefix, tracker):
    """Shared event/record assembly for streaming and causal batched replay."""
    label = int(np.argmax(prefix))
    events = tracker.finish(float(time[-1]))
    if label == 0:
        consistency = "event_evidence_with_healthy_record_label" if events or tracker.pending else "consistent"
    elif not events:
        consistency = "fault_record_without_confirmed_event"
    elif any(e["label"] is not None and e["label"] != label for e in events):
        consistency = "record_and_event_labels_disagree"
    elif any(e["label"] is None for e in events):
        consistency = "fault_record_with_unresolved_event"
    else:
        consistency = "consistent"
    return dict(unit="complete_flight", samples=len(time), first_decision_time_s=trace[0]["time_s"],
                record_label=label, record_label_name=label_name(label), record_confidence=float(prefix.max()),
                record_probabilities=prefix.tolist(), events=events, pending_event=tracker.pending,
                record_event_consistency=consistency,
                annotations=annotations(time, events, pending_event=tracker.pending),
                trace=trace, policy=asdict(tracker.policy),
                annotation_scope="Final estimated event labels, with causal detection/confirmation times retained")


def prediction_from_posteriors(time, positions, current, prefix, policy):
    """Replay causally computed probabilities; no injection metadata enters."""
    time = np.asarray(time)
    positions = np.asarray(positions, dtype=np.int64)
    if not len(positions) or positions[-1] != len(time)-1 or positions[0] < 0 or np.any(np.diff(positions) <= 0):
        raise ValueError("Decisions must be ordered and include the final recorded sample")
    if np.shape(current) != (len(positions), 19) or np.shape(prefix) != (len(positions), 19):
        raise ValueError("Expected current/prefix probabilities for each decision")
    tracker, trace = EventTracker(policy), []
    for index, cp, pp in zip(positions, current, prefix):
        stamp = float(time[index])
        tracker.observe(stamp, cp, pp)
        trace.append(dict(time_s=stamp, samples_seen=int(index)+1, current_label=int(np.argmax(cp)),
                          fault_probability=float(1-cp[0]), prefix_label=int(np.argmax(pp)),
                          prefix_confidence=float(np.max(pp))))
    return finish_prediction(time, trace, np.asarray(prefix[-1]), tracker)


def metrics(flights, predictions):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    if len(flights) != len(predictions) or not flights:
        raise ValueError("One prediction per full flight is required")
    y = [f.label for f in flights]
    p = [r["record_label"] for r in predictions]
    cm = confusion_matrix(y, p, labels=np.arange(19))
    fault_count = detected = localized = first_correct = false_events = prefault_events = 0
    normal_seconds = 0.
    delays = []
    location_delays = []
    for flight, result in zip(flights, predictions):
        if flight.label == 0:
            false_events += len(result["events"])
            normal_seconds += max(0., float(flight.time[-1]-result["first_decision_time_s"]))
            continue
        fault_count += 1
        onset = flight.fault_start_s
        end = float("inf") if flight.fault_end_s is None else flight.fault_end_s
        prefault_events += sum(e["start_s"] < onset-1e-10 for e in result["events"])
        fresh = [e for e in result["events"]
                 if e["start_s"] >= onset-1e-10 and onset <= e["detected_at_s"] < end]
        if fresh:
            detected += 1
            delays.append(float(fresh[0]["detected_at_s"]-onset))
            updates = sorted((u for e in fresh for u in e["class_updates"]
                              if onset <= u["time_s"] < end), key=lambda u: u["time_s"])
            if updates:
                first_correct += int(updates[0]["label"] == flight.label)
            correct = [u for u in updates if u["label"] == flight.label]
            if correct:
                localized += 1
                location_delays.append(float(correct[0]["time_s"]-onset))

    def quantiles(values):
        return None if not values else {k: float(v) for k, v in zip(
            ("median", "p95"), np.quantile(values, [.5, .95]))}

    return dict(record_count=len(flights), independent_groups=len({f.group for f in flights}),
                record_accuracy=float(accuracy_score(y, p)),
                record_macro_f1=float(f1_score(y, p, labels=np.arange(19), average="macro", zero_division=0)),
                confusion_matrix=cm.tolist(), class_support=cm.sum(1).tolist(),
                class_recall=(np.diag(cm)/np.maximum(cm.sum(1), 1)).tolist(),
                fault_records=fault_count, detected=detected, missed=fault_count-detected,
                detection_coverage=detected/fault_count if fault_count else None,
                first_joint_localization_coverage=first_correct/fault_count if fault_count else None,
                eventual_confirmed_joint_localization_coverage=localized/fault_count if fault_count else None,
                detection_delay_s=quantiles(delays), localization_delay_s=quantiles(location_delays),
                normal_observation_s=normal_seconds, healthy_false_alarm_events=false_events,
                healthy_false_alarm_events_per_minute=false_events/(normal_seconds/60) if normal_seconds else None,
                fault_record_prefault_events=prefault_events,
                records_with_pending_confirmation=sum(r.get("pending_event") is not None for r in predictions),
                record_event_inconsistencies=sum(r.get("record_event_consistency", "consistent") != "consistent"
                                                 for r in predictions),
                detected_within_s={str(t): sum(d <= t+1e-9 for d in delays)/fault_count if fault_count else None
                                   for t in (.5, 1., 3.)},
                delay_scope="Conditional delay quantiles; coverage and all misses are reported separately",
                statistical_unit="independent paired condition group; not chunks or decision ticks")
