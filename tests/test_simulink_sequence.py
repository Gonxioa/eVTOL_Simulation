"""Task-level checks: whole-flight units, causal history and honest event timing."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from simulink_data import read
from simulink_sequence_data import Flight, collate, fit_scaler, load_flights, validate_partitions
from simulink_sequence_events import EventPolicy, EventTracker, annotations, infer_flight, metrics
from simulink_sequence_model import FlightClassifier, loss_for_flights
from run_simulink_sequence import CONFIG, export_predictions, train_model


def flight(label=0, role="train", group=0, points=81, onset=.3, end=None, source=None):
    rng = np.random.default_rng(100+label+group)
    time = np.arange(points)/100
    x = rng.normal(0, .05, (points, 12)).astype(np.float32)
    if label:
        x[time >= onset, 6+(label-1)%6] -= .8
    return Flight(source or f"{role}_{group}_{label}", role, group, group+100, x, time, label,
                  onset if label else None, end)


def posterior(label):
    p = np.full(19, .001)
    p[label] = 1-.018
    return p


class SequenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.model = FlightClassifier(hidden=6, layers=1, dropout=0).eval()

    def test_complete_flight_and_prefix_targets_keep_healthy_prefix(self):
        f = flight(7, points=2401, onset=3., end=5.)
        current, seen = f.targets()
        self.assertTrue(np.all(current[:300] == 0))
        self.assertTrue(np.all(current[300:500] == 7))
        self.assertTrue(np.all(current[500:] == 0))
        self.assertTrue(np.all(seen[300:] == 7))
        self.assertEqual(f.label, 7)
        scaler = fit_scaler([f])
        x, lengths, *_ = collate([f], scaler, "cpu")
        self.assertEqual(tuple(x.shape), (1, 2401, 12))
        self.assertEqual(lengths.tolist(), [2401])
        self.assertEqual(scaler.n_samples_seen_, 2401)

    def test_24_second_chunked_inference_keeps_whole_flight_history(self):
        x = torch.randn(1, 2401, 12)
        with torch.inference_mode():
            full = self.model(x)
            memory = None
            prefix, current = [], []
            for start in range(0, x.shape[1], 113):
                part = self.model(x[:, start:start+113], memory)
                memory = part["memory"]
                prefix.append(part["prefix_logits"])
                current.append(part["current_logits"])
        torch.testing.assert_close(torch.cat(prefix, 1), full["prefix_logits"], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(torch.cat(current, 1), full["current_logits"], atol=2e-6, rtol=2e-5)
        self.assertEqual(memory.count.item(), 2401)
        torch.testing.assert_close(part["record_logits"], full["record_logits"], atol=2e-6, rtol=2e-5)

    def test_future_data_cannot_change_previous_prefix_predictions(self):
        a = torch.randn(1, 80, 12)
        b = a.clone()
        b[:, 30:] = torch.randn_like(b[:, 30:])*1000
        with torch.inference_mode():
            pa, pb = self.model(a), self.model(b)
        for key in ("current_logits", "prefix_logits"):
            torch.testing.assert_close(pa[key][:, :30], pb[key][:, :30], atol=0, rtol=0)

    def test_padding_is_not_history_and_cannot_change_record_output(self):
        x = torch.randn(2, 80, 12)
        x[0, 40:] = 10000
        with torch.inference_mode():
            batch = self.model(x, lengths=torch.tensor([40, 80]))
            first = self.model(x[:1, :40])
        torch.testing.assert_close(batch["record_logits"][:1], first["record_logits"], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(batch["memory"].hidden[0][:, :1], first["memory"].hidden[0], atol=2e-6, rtol=2e-5)
        self.assertEqual(batch["memory"].count[:, 0].tolist(), [40, 80])

    def test_partitions_and_scaler_reject_validation_and_source_leakage(self):
        a = flight()
        b = flight(role="validation", group=1)
        with self.assertRaises(PermissionError):
            fit_scaler([a, b])
        with self.assertRaises(ValueError):
            validate_partitions([a, copy.deepcopy(a)])
        b.seed = a.seed
        with self.assertRaises(ValueError):
            validate_partitions([a, b])

    def test_native_loader_does_not_open_numerical_test_sources(self):
        def row(name, role, group):
            return dict(source_id=name, role=role, group_id=group,
                        run=dict(label=0, seed=group+100), source_path=name,
                        plan_path=name+"_plan", csv_sha256="hash", metadata_sha256="hash")
        registry = dict(records=[row("train", "train", 0), row("locked_test", "test", 1)])
        def resolve(value, _):
            if "locked_test" in value:
                raise AssertionError("Training opened test data")
            return Path(value)
        class Frame:
            class Times:
                @staticmethod
                def to_numpy(): return np.arange(4)/100
            time_s = Times()
        with patch("simulink_sequence_data.read", return_value=registry), \
             patch("simulink_sequence_data._source_path", side_effect=resolve), \
             patch("simulink_sequence_data.checked_plan", return_value={}), \
             patch("simulink_sequence_data.validate_run", return_value=({"fault_start_s": 3}, Frame())), \
             patch("simulink_sequence_data.sha256_file", return_value="hash"), \
             patch("simulink_sequence_data.observable_features", return_value=np.zeros((4, 12))):
            records = load_flights(["registry.json"], roles=("train",))
        self.assertEqual([f.source_id for f in records], ["train"])

    def test_inference_resets_between_flights_and_never_reads_fault_truth(self):
        a, b = flight(7), flight(10, group=1)
        scaler = fit_scaler([a])
        first = infer_flight(self.model, scaler, a.x, a.time)
        infer_flight(self.model, scaler, b.x, b.time)
        a.label, a.fault_start_s = 0, None
        again = infer_flight(self.model, scaler, a.x, a.time)
        self.assertEqual(first, again)

    def test_events_have_estimated_boundaries_and_later_confirmation_times(self):
        tracker = EventTracker(EventPolicy(confirm_s=.2, localize_s=.2, clear_s=.3))
        time = np.arange(31)/10
        for t in time:
            p = posterior(7 if 1 <= t < 2 else 0)
            tracker.observe(float(t), p, posterior(7 if t >= 1 else 0))
        events = tracker.finish(3.)
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertAlmostEqual(e["start_s"], 1.)
        self.assertAlmostEqual(e["detected_at_s"], 1.2)
        self.assertAlmostEqual(e["end_s"], 2.)
        self.assertAlmostEqual(e["cleared_at_s"], 2.3)
        self.assertEqual(e["label"], 7)
        self.assertFalse(e["ongoing_at_record_end"])
        spans = annotations(time, events)
        self.assertEqual([s["label"] for s in spans], [0, 7, 0])
        self.assertEqual([(s["start_index"], s["end_index"]) for s in spans], [(0, 9), (10, 19), (20, 30)])

    def test_low_class_confidence_keeps_fault_localization_unresolved(self):
        tracker = EventTracker(EventPolicy(confirm_s=0, localize_s=0))
        p = np.full(19, .99/18)
        p[0] = .01
        tracker.observe(0., posterior(7), p)
        events = tracker.finish(1.)
        self.assertIsNone(events[0]["label"])
        self.assertEqual(annotations(np.arange(11)/10, events)[0]["label"], -1)

    def test_new_evidence_can_correct_a_previously_confirmed_class(self):
        tracker = EventTracker(EventPolicy(confirm_s=.1, localize_s=.1))
        for i in range(8):
            tracker.observe(i/10, posterior(7), posterior(7 if i < 4 else 10))
        event = tracker.finish(.7)[0]
        self.assertEqual(event["label"], 10)
        self.assertEqual([u["label"] for u in event["class_updates"]], [7, 10])

    def test_prefault_alarm_is_not_credited_as_new_fault_detection(self):
        f = flight(7, points=301, onset=2.)
        tracker = EventTracker(EventPolicy(confirm_s=.1, localize_s=.1))
        for i in range(31):
            tracker.observe(i/10, posterior(7), posterior(7))
        result = dict(record_label=7, first_decision_time_s=0., events=tracker.finish(3.))
        score = metrics([f], [result])
        self.assertEqual(score["record_count"], 1)
        self.assertEqual(score["record_accuracy"], 1.)
        self.assertEqual(score["missed"], 1)
        self.assertEqual(score["fault_record_prefault_events"], 1)
        self.assertIsNone(score["detection_delay_s"])

    def test_flight_loss_reaches_early_samples_without_independent_chunks(self):
        f = flight(7)
        scaler = fit_scaler([f])
        x, lengths, current, seen, labels = collate([f], scaler, "cpu")
        x.requires_grad_()
        loss, _ = loss_for_flights(self.model(x, lengths=lengths), lengths, current, seen, labels,
                                  prefix_weight=0, current_weight=0)
        loss.backward()
        self.assertGreater(float(x.grad[:, :20].abs().sum()), 0)

    def test_whole_flight_training_checkpoint_resume_and_annotation_exports(self):
        records = [flight(c, role, group, points=31, onset=.1)
                   for role, group in (("train", 0), ("validation", 1)) for c in range(19)]
        cfg = dict(read(CONFIG), hidden=4, layers=1, dropout=.2, batch_flights=19,
                   max_epochs=2, patience=4, decision_samples=5)
        with tempfile.TemporaryDirectory() as folder:
            full, interrupted = Path(folder)/"full", Path(folder)/"interrupted"
            summary = train_model(cfg, records, full)
            from run_simulink_sequence import write_json as real_write
            def stop_after_first_epoch(path, payload):
                real_write(path, payload)
                if Path(path).name == "history.json" and len(payload) == 1:
                    raise RuntimeError("simulate interruption")
            with patch("run_simulink_sequence.write_json", side_effect=stop_after_first_epoch):
                with self.assertRaisesRegex(RuntimeError, "simulate interruption"):
                    train_model(cfg, records, interrupted)
            resumed = train_model(cfg, records, interrupted, resume=True)
            a = torch.load(full/"last.pt", weights_only=False)
            b = torch.load(interrupted/"last.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            self.assertGreater(a["parameter_max_delta"], 0)
            self.assertEqual(summary["train_records"], 19)
            self.assertEqual(resumed["optimizer_updates"], 2)
            model = FlightClassifier(4, 1, .2)
            model.load_state_dict(a["model"])
            score = export_predictions(model, a["scaler"], records[19:], cfg, Path(folder)/"annotations", "cpu")
            self.assertEqual(score["record_count"], 19)
            self.assertEqual(sum(map(sum, score["confusion_matrix"])), 19)
            import pandas as pd
            table = pd.read_csv(Path(folder)/"annotations/record_predictions.csv")
            self.assertEqual(len(table), 19)
            spans = pd.read_csv(Path(folder)/"annotations/segment_annotations.csv")
            self.assertEqual(len(set(spans.source_id)), 19)


if __name__ == "__main__":
    unittest.main()
