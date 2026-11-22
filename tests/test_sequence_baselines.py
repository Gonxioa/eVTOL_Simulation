"""Whole-flight CNN causality, streaming equivalence and baseline semantics."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from run_simulink_sequence import CONFIG, architecture, load_checkpoint, train_model
from run_simulink_sequence_baselines import batched_predictions, detailed_metrics
from simulink_data import read
from simulink_sequence_data import Flight, fit_scaler
from simulink_sequence_events import EventPolicy, infer_flight
from simulink_sequence_model import CausalFlightCNN, FlightClassifier


def fixture(label=0, group=0, role="train", points=81):
    x = np.random.default_rng(10+group+label).normal(0, .1, (points, 12)).astype(np.float32)
    return Flight(f"{role}_{group}_{label}", role, group, group+100, x,
                  np.arange(points)/100, label, .1 if label else None)


class SequenceBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.model = CausalFlightCNN(6, (5, 5, 5, 5), (1, 2, 4, 8), 0).eval()

    def test_architecture_defaults_disable_tf32_for_chunk_consistency(self):
        previous = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            architecture(read(CONFIG))
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        finally:
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = previous

    def test_cnn_preserves_complete_24_second_history_across_chunks(self):
        x = torch.randn(1, 2401, 12)
        with torch.inference_mode():
            full = self.model(x)
            memory, logits = None, []
            for start in range(0, 2401, 113):
                part = self.model(x[:, start:start+113], memory)
                memory = part["memory"]
                logits.append(part["prefix_logits"])
        torch.testing.assert_close(torch.cat(logits, 1), full["prefix_logits"], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(part["record_logits"], full["record_logits"], atol=2e-6, rtol=2e-5)
        self.assertEqual(memory.count.item(), 2401)
        self.assertEqual(self.model.receptive_field_samples, 61)

    def test_cnn_future_cannot_change_past_current_or_prefix_outputs(self):
        a = torch.randn(1, 100, 12)
        b = a.clone()
        b[:, 30:] += 1000
        with torch.inference_mode():
            first, second = self.model(a), self.model(b)
        for key in ("current_logits", "prefix_logits"):
            torch.testing.assert_close(first[key][:, :30], second[key][:, :30], atol=0, rtol=0)

    def test_cnn_padding_does_not_contaminate_cached_continuation(self):
        x, new = torch.randn(2, 80, 12), torch.randn(2, 20, 12)
        x[0, 40:] = 10000
        with torch.inference_mode():
            batch = self.model(x, lengths=torch.tensor([40, 80]))
            continuation = self.model(new, batch["memory"])
            separate = self.model(torch.cat((x[:1, :40], new[:1]), 1))
        torch.testing.assert_close(continuation["record_logits"][:1], separate["record_logits"], atol=2e-6, rtol=2e-5)
        self.assertEqual(continuation["memory"].count[:, 0].tolist(), [60, 100])

    def test_cnn_record_loss_reaches_early_inputs(self):
        x = torch.randn(1, 200, 12, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(self.model(x)["record_logits"], torch.tensor([7]))
        loss.backward()
        self.assertGreater(float(x.grad[:, :20].abs().sum()), 0)

    def test_causal_batched_replay_matches_true_streaming_for_both_families(self):
        flights = [fixture(0), fixture(7), fixture(10)]
        scaler = fit_scaler(flights)
        config = dict(read(CONFIG), batch_flights=2)
        for model in (self.model, FlightClassifier(6, 1, 0).eval()):
            fast = batched_predictions(model, scaler, flights, config, "cpu")
            for flight, prediction in zip(flights, fast):
                slow = infer_flight(model, scaler, flight.x, flight.time, config["decision_samples"],
                                    EventPolicy(**config["event_policy"]), "cpu")
                np.testing.assert_allclose(prediction["record_probabilities"], slow["record_probabilities"], atol=1e-6)
                self.assertEqual(prediction["annotations"], slow["annotations"])
                self.assertEqual([r["samples_seen"] for r in prediction["trace"]],
                                 [r["samples_seen"] for r in slow["trace"]])

    def test_motor_and_severity_accuracy_are_not_joint_accuracy(self):
        flights = [fixture(7), fixture(7, group=1)]
        predictions = [dict(record_label=label, events=[], record_event_consistency="consistent",
                            annotations=[dict(start_index=0, end_index=80, label=0)]) for label in (1, 10)]
        score, cases = detailed_metrics(flights, predictions)
        self.assertEqual(score["fault_record_motor_accuracy"], .5)
        self.assertEqual(score["fault_record_severity_accuracy"], .5)
        self.assertEqual(score["fault_record_joint_accuracy"], 0)
        self.assertEqual(score["mean_correct_class_interval_iou"], 0)
        self.assertTrue(all(c["detection_delay_s"] is None for c in cases))

    def test_cnn_checkpoint_uses_architecture_identity_and_complete_flights(self):
        config = dict(read(CONFIG), model="cnn", hidden=4, cnn_kernels=[3, 3], cnn_dilations=[1, 2],
                      dropout=0., batch_flights=19, max_epochs=1, patience=2)
        flights = [fixture(c, g, role, 31) for role, g in (("train", 0), ("validation", 1)) for c in range(19)]
        with tempfile.TemporaryDirectory() as folder:
            summary = train_model(config, flights, folder)
            model, _, saved = load_checkpoint(Path(folder)/"best.pt", "cpu")
            self.assertIsInstance(model, CausalFlightCNN)
            self.assertEqual(saved["model"], "cnn")
            self.assertEqual(summary["optimizer_updates"], 1)
            self.assertEqual(summary["validation_records"], 19)
        with self.assertRaises(ValueError):
            architecture(dict(config, model="unregistered"))


if __name__ == "__main__":
    unittest.main()
