"""Measured-only, incremental diagnosis. No label or fault time enters this API.

Semantic version is intentionally incompatible with legacy prefix-localization
checkpoints. Offline replay calls exactly the same object as live ingestion.
"""
from dataclasses import asdict
import copy
import time

import numpy as np
import torch

from simulink_sequence_data import label_name
from simulink_sequence_events import EventPolicy, annotations

SEMANTICS = "current_state_stream_v2"


class CurrentEventTracker:
    """Elapsed observed time, not tick counts, confirms alarms and classes."""
    def __init__(self, policy=None, decision_interval_s=.1):
        self.policy = policy or EventPolicy()
        self.interval = float(decision_interval_s)
        self.events, self.class_confirmations, self.discontinuities = [], [], []
        self.active = None
        self.candidate_start = self.clear_start = None
        self.class_candidate = self.class_start = None
        self.previous_time = None
        self.finished = False
        self.pending = None

    def break_continuity(self, time_s, reason="missing_measurements"):
        self.candidate_start = self.clear_start = None
        self.class_candidate = self.class_start = None
        self.discontinuities.append(dict(time_s=float(time_s), reason=reason))
        # Missing data never proves that an existing alarm has cleared.

    def observe(self, time_s, probabilities):
        p = np.asarray(probabilities, dtype=float)
        if self.finished or not np.isfinite(time_s) or (self.previous_time is not None and time_s <= self.previous_time):
            raise ValueError("Decisions must be finite and strictly increasing")
        if p.shape != (19,) or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
            raise ValueError("Expected normalized finite 19-class current probabilities")
        if self.previous_time is not None and time_s-self.previous_time > self.interval+1e-7:
            self.break_continuity(time_s, "interrupted_decisions")
        self.previous_time = float(time_s)
        cfg = self.policy
        fault = 1-float(p[0])
        # Inclusive mathematical boundaries: e.g. 1-0.7 is 0.30000000000000004.
        # 1e-12 covers double subtraction roundoff, not a practical retuning.
        positive = fault+1e-12 >= cfg.fault_threshold
        label = int(p.argmax())
        evidence = label if positive and label > 0 and p[label]+1e-12 >= cfg.class_threshold else None
        if evidence != self.class_candidate:
            self.class_candidate = evidence
            self.class_start = float(time_s) if evidence is not None else None
        if self.active is None:
            if positive:
                if self.candidate_start is None:
                    self.candidate_start = float(time_s)
                if time_s-self.candidate_start+1e-9 >= cfg.confirm_s:
                    self.active = dict(start_s=self.candidate_start, candidate_at_s=self.candidate_start,
                        detected_at_s=float(time_s), end_s=None, cleared_at_s=None,
                        ongoing_at_record_end=False, label=None, label_name="Unresolved", class_updates=[])
                    self.events.append(self.active)
            else:
                self.candidate_start = None
        if self.active is not None:
            if evidence is not None and time_s-self.class_start+1e-9 >= cfg.localize_s:
                if self.active["label"] != evidence:
                    update = dict(time_s=float(time_s), candidate_at_s=self.class_start,
                                  label=evidence, label_name=label_name(evidence), confidence=float(p[evidence]))
                    self.active["class_updates"].append(update)
                    self.class_confirmations.append(copy.deepcopy(update))
                    self.active.update(label=evidence, label_name=label_name(evidence))
            if fault <= cfg.clear_threshold+1e-12:
                if self.clear_start is None:
                    self.clear_start = float(time_s)
                if time_s-self.clear_start+1e-9 >= cfg.clear_s:
                    self.active.update(end_s=self.clear_start, cleared_at_s=float(time_s))
                    self.active = None
                    self.candidate_start = self.clear_start = None
                    self.class_candidate = self.class_start = None
            else:
                self.clear_start = None
        return self.snapshot()

    def snapshot(self):
        return copy.deepcopy(dict(alarm=self.active is not None,
            label=None if self.active is None else self.active["label"],
            status=("healthy_or_unconfirmed" if self.active is None else
                    "fault_class_pending" if self.active["label"] is None else "fault_localized"),
            candidate_at_s=self.candidate_start, class_candidate=self.class_candidate,
            class_candidate_at_s=self.class_start, clear_candidate_at_s=self.clear_start,
            active_event=self.active))

    def finish(self, last_time_s):
        if self.finished or not np.isfinite(last_time_s) or (self.previous_time is not None and last_time_s < self.previous_time):
            raise ValueError("Invalid stream end")
        if self.active is not None:
            self.active.update(end_s=float(last_time_s), ongoing_at_record_end=True)
        elif self.candidate_start is not None:
            self.pending = dict(start_s=self.candidate_start, end_s=float(last_time_s), detected_at_s=None,
                label=None, label_name="Unresolved", ongoing_at_record_end=True, status="pending_confirmation")
        self.finished = True
        return copy.deepcopy(self.events)


def event_record_label(events):
    """Truth-free summary: last confirmed class; unresolved alarm=-1; no alarm=0."""
    updates = [u for e in events for u in e["class_updates"]]
    return int(updates[-1]["label"]) if updates else (-1 if events else 0)


class RealtimeDiagnoser:
    """Use push(new_x, timestamps), current(), finish(), reset() for one flight.

    Only new, already time-aligned [N,12] measurements are accepted. IMU delay
    alignment belongs to the sensor adapter; measured RPM is never delayed again.
    A canonical decision-sized buffer makes arrival packet sizes irrelevant.
    No decision is invented for an incomplete final decision interval.
    """
    def __init__(self, model, scaler, policy=None, sample_hz=100, decision_samples=10,
                 device="cpu", gap_policy="reject", record_head_trained=False):
        if sample_hz <= 0 or int(decision_samples) != decision_samples or decision_samples < 1:
            raise ValueError("Positive sample rate and integer decision period required")
        if gap_policy not in ("reject", "reset_context"):
            raise ValueError("gap_policy must be reject or reset_context")
        self.model, self.scaler, self.device = model.to(device).eval(), scaler, torch.device(device)
        self.policy = policy or EventPolicy()
        self.sample_hz, self.decision_samples = float(sample_hz), int(decision_samples)
        self.gap_policy, self.record_head_trained = gap_policy, record_head_trained
        self.reset()

    def reset(self):
        self.memory = None
        self.received = self.processed = 0
        self.last_time = None
        self.buffer_x = np.empty((0, 12), dtype=np.float32)
        self.buffer_t = np.empty(0, dtype=np.float64)
        self.tracker = CurrentEventTracker(self.policy, self.decision_samples/self.sample_hz)
        self.trace, self.processing_seconds = [], []
        self.finished = False
        self.last_prefix = None
        return self

    initialize = reset

    def _advance(self, count, decide):
        bx = self.buffer_x[:count]
        ts = float(self.buffer_t[count-1])
        scaled = self.scaler.transform(bx).astype(np.float32)
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        with torch.inference_mode():
            output = self.model(torch.as_tensor(scaled[None], device=self.device), self.memory)
            self.memory = output["memory"]
            p = torch.softmax(output["current_logits"][0, -1], -1).cpu().numpy()
            self.last_prefix = torch.softmax(output["record_logits"][0], -1).cpu().numpy()
        self.processed += count
        self.buffer_x, self.buffer_t = self.buffer_x[count:], self.buffer_t[count:]
        if decide:
            state = self.tracker.observe(ts, p)
            self.trace.append(dict(time_s=ts, samples_seen=self.processed,
                current_label=int(p.argmax()), fault_probability=float(1-p[0]),
                current_confidence=float(p.max()), probabilities=p.tolist(),
                alarm=state["alarm"], confirmed_label=state["label"], status=state["status"]))
        self.processing_seconds.append(time.perf_counter()-started)

    def push(self, new_x, timestamps):
        if self.finished:
            raise RuntimeError("finish() has been called; reset before a new flight")
        x, t = np.asarray(new_x, dtype=np.float32), np.asarray(timestamps, dtype=np.float64)
        if x.ndim == 1:
            x = x[None]
        t = np.atleast_1d(t)
        if x.ndim != 2 or x.shape[1] != 12 or t.shape != (len(x),):
            raise ValueError("Expected [N,12] measured input and N acquisition timestamps")
        if not np.isfinite(x).all() or not np.isfinite(t).all():
            raise ValueError("Nonfinite input rejected without advancing state")
        if not len(t):
            return []
        previous = t if self.last_time is None else np.r_[self.last_time, t]
        dt = np.diff(previous)
        if (dt <= 0).any():
            raise ValueError("Duplicate/reversed timestamps rejected without advancing state")
        expected = 1/self.sample_hz
        if (dt < expected-1e-7).any():
            raise ValueError("Sampling faster than the configured sensor rate")
        gaps = dt > expected+1e-7
        if gaps.any() and self.gap_policy == "reject":
            raise ValueError("Data gap rejected; explicit reset_context policy is required to continue")
        trace_start = len(self.trace)
        # Split only at real interruptions. Default contiguous delivery is vectorized.
        boundaries = np.flatnonzero(gaps)+(0 if self.last_time is not None else 1)
        cuts = sorted(set([0, *boundaries.tolist(), len(t)]))
        for a, b in zip(cuts[:-1], cuts[1:]):
            if a in boundaries:
                if len(self.buffer_t):
                    self._advance(len(self.buffer_t), False)
                self.memory = None
                self.tracker.break_continuity(t[a], "measurement_gap_reset_context")
            self.buffer_x = np.concatenate((self.buffer_x, x[a:b]))
            self.buffer_t = np.concatenate((self.buffer_t, t[a:b]))
            self.received += b-a
            while len(self.buffer_t) >= self.decision_samples-(self.processed % self.decision_samples):
                self._advance(self.decision_samples-(self.processed % self.decision_samples), True)
        self.last_time = float(t[-1])
        return copy.deepcopy(self.trace[trace_start:])

    update = push

    def current(self):
        return dict(samples_received=self.received, samples_processed=self.processed,
                    pending_samples=len(self.buffer_t), last_input_time_s=self.last_time,
                    last_decision=copy.deepcopy(self.trace[-1]) if self.trace else None,
                    **self.tracker.snapshot())

    def finish(self):
        if self.finished or self.last_time is None:
            raise RuntimeError("Cannot finish an empty/already finished stream")
        if len(self.buffer_t):
            self._advance(len(self.buffer_t), False)
        events = self.tracker.finish(self.last_time)
        self.finished = True
        label = event_record_label(events)
        result = dict(semantics=SEMANTICS, unit="complete_flight", samples=self.received,
            record_label=label, record_label_name=label_name(label), record_summary="last_confirmed_online_event_class",
            events=events, pending_event=copy.deepcopy(self.tracker.pending), trace=copy.deepcopy(self.trace),
            policy=asdict(self.policy), discontinuities=copy.deepcopy(self.tracker.discontinuities),
            decision_samples=self.decision_samples, sample_hz=self.sample_hz,
            processing_seconds=list(self.processing_seconds), first_decision_time_s=self.trace[0]["time_s"] if self.trace else None,
            annotation_scope="Retrospective backfill only; online delays use detected_at_s and class_updates.time_s")
        if self.record_head_trained:
            result["auxiliary_legacy_record_probabilities"] = self.last_prefix.tolist()
        return result


def replay_flight(model, scaler, x, timestamps, policy=None, decision_samples=10,
                  sample_hz=100, device="cpu", arrival_sizes=(10,), record_head_trained=False):
    """File replay is delivery to the public incremental object, not a second alarm path."""
    if not arrival_sizes or min(arrival_sizes) < 1:
        raise ValueError("Positive arrival sizes required")
    diagnostic = RealtimeDiagnoser(model, scaler, policy, sample_hz, decision_samples, device,
                                   record_head_trained=record_head_trained)
    start = packet = 0
    while start < len(x):
        end = min(start+arrival_sizes[packet % len(arrival_sizes)], len(x))
        diagnostic.push(x[start:end], timestamps[start:end])
        start, packet = end, packet+1
    result = diagnostic.finish()
    result["annotations"] = annotations(timestamps, result["events"], result["pending_event"])
    return result
