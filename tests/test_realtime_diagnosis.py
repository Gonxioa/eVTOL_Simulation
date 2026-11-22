"""Actual failure-mode regression tests for the public incremental API and TBPTT."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from simulink_realtime import CurrentEventTracker, RealtimeDiagnoser, replay_flight
from simulink_realtime_evaluation import decisions_from_probabilities, score_events
from simulink_realtime_training import segmented_backward, sample_weights, detach_memory, load_realtime_checkpoint
from simulink_sequence_data import Flight, collate
from simulink_sequence_events import EventPolicy
from simulink_sequence_model import CausalFlightCNN, FlightClassifier
from stream_fault_diagnosis import MeasurementAligner, diagnose_packets


def posterior(label, confidence=.99):
    p = np.full(19, (1-confidence)/18)
    p[label] = confidence
    return p


class RealtimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.models = [FlightClassifier(6, 2, 0).eval(), CausalFlightCNN(6, (3,3), (1,2), 0).eval()]
        self.x = np.random.default_rng(42).normal(0, .2, (141,12)).astype(np.float32)
        self.t = np.arange(len(self.x))/100
        self.scaler = StandardScaler().fit(self.x)

    def test_arrival_sizes_do_not_change_decisions_or_events(self):
        for model in self.models:
            base = replay_flight(model, self.scaler, self.x, self.t, arrival_sizes=(10,))
            for sizes in ((1,), (7,23,1,4,38)):
                other = replay_flight(model, self.scaler, self.x, self.t, arrival_sizes=sizes)
                self.assertEqual(base["trace"], other["trace"])
                self.assertEqual(base["events"], other["events"])
            self.assertEqual([r["samples_seen"] for r in base["trace"]], list(range(10,141,10)))
            self.assertEqual(base["samples"], 141)

    def test_future_inputs_cannot_change_past_decisions_or_alarms(self):
        for model in self.models:
            diagnostic = RealtimeDiagnoser(model, self.scaler)
            past = diagnostic.push(self.x[:70], self.t[:70])
            altered = self.x.copy()
            altered[70:] += 1e4
            result = replay_flight(model, self.scaler, altered, self.t)
            self.assertEqual(past, result["trace"][:7])

    def test_causal_full_forward_and_chunked_inference_agree(self):
        for model in self.models:
            with torch.inference_mode():
                full = model(torch.as_tensor(self.scaler.transform(self.x)[None], dtype=torch.float32))
                p = full["current_logits"].softmax(-1)[0].numpy()
            fast = decisions_from_probabilities(self.t, p, EventPolicy())
            actual = replay_flight(model, self.scaler, self.x, self.t)
            np.testing.assert_allclose([r["probabilities"] for r in fast["trace"]],
                                       [r["probabilities"] for r in actual["trace"]], atol=2e-6, rtol=2e-5)
            self.assertEqual(fast["events"], actual["events"])

    def test_reset_between_records_and_finish_does_not_create_a_decision(self):
        for model in self.models:
            diagnostic = RealtimeDiagnoser(model, self.scaler)
            diagnostic.push(self.x, self.t)
            one = diagnostic.finish()
            with self.assertRaises(RuntimeError):
                diagnostic.push(self.x, self.t)
            diagnostic.reset().push(self.x, self.t)
            two = diagnostic.finish()
            self.assertEqual(one["trace"], two["trace"])
            self.assertEqual(len(one["trace"]), 14)
            self.assertEqual(diagnostic.processed, 141)

    def test_repeat_reverse_nonfinite_and_gap_are_atomic_rejections(self):
        diagnostic = RealtimeDiagnoser(self.models[0], self.scaler)
        diagnostic.push(self.x[:10], self.t[:10])
        before = diagnostic.current()
        bad = [(self.x[:1], [.09]), (self.x[:2], [.11,.1]),
               (np.full((1,12), np.nan), [.1]), (self.x[:1], [.2])]
        for x, t in bad:
            with self.assertRaises(ValueError):
                diagnostic.push(x,t)
            self.assertEqual(before, diagnostic.current())

    def test_one_point_period_and_explicit_gap_resume(self):
        diagnostic = RealtimeDiagnoser(self.models[0], self.scaler, decision_samples=1, gap_policy="reset_context")
        diagnostic.push(self.x[:3], self.t[:3])
        diagnostic.push(self.x[4:8], self.t[4:8])
        self.assertEqual(len(diagnostic.trace), 7)
        self.assertTrue(diagnostic.tracker.discontinuities)
        self.assertEqual(diagnostic.processed, 7)

    def test_four_decisions_spanning_point_three_seconds_required(self):
        tracker = CurrentEventTracker()
        for t in (.09,.19,.29):
            tracker.observe(t, posterior(7))
            self.assertIsNone(tracker.active)
        tracker.observe(.39, posterior(7))
        self.assertEqual(tracker.active["detected_at_s"], .39)
        self.assertEqual(tracker.active["class_updates"][0]["time_s"], .39)

    def test_threshold_interruption_restarts_confirmation_and_class_change(self):
        tracker = CurrentEventTracker()
        for t in (.09,.19):
            tracker.observe(t, posterior(7))
        tracker.observe(.29, posterior(0))
        for t in (.39,.49,.59):
            tracker.observe(t, posterior(7))
            self.assertIsNone(tracker.active)
        tracker.observe(.69, posterior(7))
        for t in (.79,.89):
            tracker.observe(t, posterior(8))
        tracker.observe(.99, posterior(9))
        for t in (1.09,1.19):
            tracker.observe(t, posterior(9))
        self.assertEqual(tracker.active["label"], 7)
        tracker.observe(1.29, posterior(9))
        self.assertEqual(tracker.active["label"], 9)
        self.assertEqual(len(tracker.class_confirmations), 2)

    def test_unresolved_alarm_then_clear_boundary_and_repeated_event(self):
        tracker = CurrentEventTracker()
        p = np.r_[.05, np.repeat(.95/18,18)]
        for t in (.09,.19,.29,.39):
            tracker.observe(t,p)
        self.assertEqual(tracker.snapshot()["status"], "fault_class_pending")
        for t in (.49,.59,.69,.79,.89):
            tracker.observe(t,posterior(0))
            self.assertIsNotNone(tracker.active)
        tracker.observe(.99,posterior(0))
        self.assertIsNone(tracker.active)
        self.assertEqual(tracker.events[0]["cleared_at_s"], .99)
        for t in (1.09,1.19,1.29,1.39):
            tracker.observe(t,posterior(1))
        self.assertEqual(len(tracker.events),2)

    def test_gap_breaks_timers_but_does_not_clear_active_alarm(self):
        tracker = CurrentEventTracker()
        for t in (.09,.19,.29):
            tracker.observe(t,posterior(1))
        tracker.observe(.69,posterior(1))
        self.assertIsNone(tracker.active)
        for t in (.79,.89,.99):
            tracker.observe(t,posterior(1))
        tracker.observe(1.09,posterior(0))
        tracker.observe(2.,posterior(0))
        self.assertIsNotNone(tracker.active)

    def test_inclusive_probability_threshold_boundaries(self):
        tracker = CurrentEventTracker(EventPolicy(confirm_s=0,localize_s=0,clear_s=0))
        p=np.zeros(19)
        p[0],p[7],p[8]=.2,.6,.2
        tracker.observe(.09,p)
        self.assertIsNotNone(tracker.active)
        self.assertEqual(tracker.active["label"],7)
        p=np.zeros(19)
        p[0],p[7]=.7,.3
        tracker.observe(.19,p)
        self.assertIsNone(tracker.active)
        self.assertEqual(tracker.events[0]["cleared_at_s"],.19)

    def test_fault_lifecycle_not_stability_defines_state_targets(self):
        f = Flight("life","train",0,42,self.x,self.t,7,.3,.9)
        y, seen = f.targets()
        self.assertTrue(np.all(y[30:90] == 7))
        self.assertTrue(np.all(y[90:] == 0))
        self.assertTrue(np.all(seen[30:] == 7))

    def test_padding_last_short_segment_and_weights_count_once(self):
        flights = [Flight("a","train",0,42,self.x[:31],self.t[:31],7,.1),
                   Flight("b","train",0,42,self.x[:101],self.t[:101],0)]
        x,lengths,current,seen,labels = collate(flights,self.scaler,"cpu")
        cfg = dict(segment_samples=24,current_loss_weight=1.,prefix_loss_weight=0.,record_loss_weight=0.,
                   early_span_s=.2,early_weight=3.)
        weights = sample_weights(flights,x.shape[1],cfg,"cpu")
        self.assertTrue(torch.all(weights[0,31:] == 0))
        self.assertEqual(float(weights[0,10]),3.)
        for model in self.models:
            model.zero_grad()
            stats = segmented_backward(model,x,lengths,current,seen,labels,weights,cfg)
            self.assertEqual(stats["valid_forward_points"],132)
            self.assertEqual(stats["current_supervised_points"],132)
            self.assertEqual(stats["record_terminals"],2)
            self.assertAlmostEqual(stats["weighted_mass"],1.,places=6)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0.)

    def test_cross_segment_memory_is_detached(self):
        for model in self.models:
            out = model(torch.randn(1,15,12,requires_grad=True))
            memory = detach_memory(out["memory"])
            for value in memory.__dict__.values():
                for v in value if isinstance(value,tuple) else (value,):
                    self.assertFalse(v.requires_grad)

    def test_prefault_alarm_cannot_count_as_a_fresh_fault_detection(self):
        f = Flight("f","test",0,42,self.x,self.t,7,.5)
        p = np.repeat(posterior(7)[None],len(self.x),axis=0)
        result = decisions_from_probabilities(self.t,p,EventPolicy(),10)
        score, rows = score_events([f],[result])
        self.assertEqual(score["detected"],0)
        self.assertEqual(score["prefault_false_alarm_events"],1)
        self.assertEqual(score["always_missed"],1)
        self.assertTrue(rows[0]["active_false_alarm_at_onset"])

    def test_legacy_checkpoint_cannot_silently_use_new_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"old.pt"
            torch.save(dict(config={}),path)
            with self.assertRaisesRegex(ValueError,"Legacy checkpoint"):
                load_realtime_checkpoint(path)

    def test_live_sensor_alignment_delays_only_imu_across_packets(self):
        aligner = MeasurementAligner(3)
        pieces = [aligner.push(self.x[:1]),aligner.push(self.x[1:17]),aligner.push(self.x[17:])]
        result = np.concatenate(pieces)
        np.testing.assert_array_equal(result[:,:6],self.x[np.maximum(np.arange(len(self.x))-3,0),:6])
        np.testing.assert_array_equal(result[:,6:],self.x[:,6:])
        aligner.reset()
        np.testing.assert_array_equal(aligner.push(self.x),result)

    def test_live_generator_emits_before_requesting_future_packet(self):
        diagnostic = RealtimeDiagnoser(self.models[0],self.scaler)
        requested = []
        def packets():
            requested.append(0)
            yield self.x[:10],self.t[:10]
            requested.append(1)
            yield self.x[10:20],self.t[10:20]
        stream = diagnose_packets(diagnostic,packets())
        decision = next(stream)
        self.assertEqual(decision["samples_seen"],10)
        self.assertEqual(requested,[0])

    def test_first_wrong_localization_and_later_correction_are_separate(self):
        t = np.arange(201)/100
        x = np.zeros((201,12),dtype=np.float32)
        f = Flight("fault","test",0,42,x,t,7,.3)
        p = np.repeat(posterior(0)[None],201,axis=0)
        p[30:110] = posterior(8)
        p[110:] = posterior(7)
        prediction = decisions_from_probabilities(t,p,EventPolicy(),10)
        score,cases = score_events([f],[prediction])
        self.assertEqual(score["first_localization_wrong"],1)
        self.assertEqual(score["always_missed"],0)
        self.assertEqual(score["never_joint_localized"],0)
        self.assertAlmostEqual(cases[0]["detection_delay_s"],.39)
        self.assertAlmostEqual(cases[0]["localization_delay_s"],1.19)

    def test_test_access_requires_signed_complete_freeze(self):
        from run_simulink_realtime import authorize_test
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                authorize_test({},Path(folder),{})
            (Path(folder)/"freeze_receipt.json").write_text('{"signature":"forged","protocol_signature":"bad"}',encoding="utf-8")
            with self.assertRaises(PermissionError):
                authorize_test({},Path(folder),{})


if __name__ == "__main__":
    unittest.main()
