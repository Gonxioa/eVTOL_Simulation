import json
import unittest
from copy import deepcopy
from pathlib import Path
import numpy as np
from diagnostics.features import ObservationPacket, extract, FEATURE_SETS
from diagnostics.continuous import CausalWindow, causal_windows, replay, offline_labels
from hex_research.commands import task_library, ParameterizedTask, LegacyTask
from hex_research.configuration import load
from hex_research.simulation import simulate, FaultSchedule

SETTINGS = json.loads((Path(__file__).resolve().parents[1]/'hex_research/configs/continuous.json').read_text())


def packets(n=90):
    return [ObservationPacket(i*.01, [0, 0, 9.81+i*.001], [0, .01*i, 0], [0, 0, 0, 1], np.full(6, 800.)) for i in range(n)]


class TaskAndDiagnosticChecks(unittest.TestCase):
    def test_consistent_derivatives_and_smooth_endpoints(self):
        for item in task_library(SETTINGS, 1):
            task = ParameterizedTask(item['task'])
            for t in (1., 7.):
                np.testing.assert_allclose(task.at(t)['x_dot'], 0, atol=1e-12)
                np.testing.assert_allclose(task.at(t)['x_ddot'], 0, atol=1e-12)
            for t in (1.4, 3.2, 5.8):
                h = 1e-4; left, mid, right = task.at(t-h), task.at(t), task.at(t+h)
                np.testing.assert_allclose((right['x']-left['x'])/(2*h), mid['x_dot'], atol=2e-8)
                np.testing.assert_allclose((right['x_dot']-left['x_dot'])/(2*h), mid['x_ddot'], atol=2e-8)
        self.assertEqual(task_library(SETTINGS), task_library(SETTINGS))

    def test_provider_single_sampling_and_logged_reference(self):
        class CountingTask(LegacyTask):
            def reset(self): self.calls = []
            def at(self, t):
                self.calls.append(t); r = super().at(t); r['yaw'] = len(self.calls)*.0001; return r
        provider = CountingTask('hover'); seen = []
        run = simulate(load(), .2, task_provider=provider, diagnostic_sink=seen.append)
        self.assertEqual(len(provider.calls), 21); self.assertEqual(len(seen), 21)
        for row in run['rows']:
            for key in ('x', 'x_dot', 'x_ddot', 'yaw', 'yaw_dot'):
                np.testing.assert_array_equal(row['reference'][key], row['control_reference'][key])
            self.assertEqual(row['time'], row['control_reference_time'])
        self.assertFalse(hasattr(seen[0], 'fault_label')); self.assertFalse(hasattr(seen[0], 'reference'))

    def test_feature_contract_dimensions_and_no_duplicate_gyro(self):
        packet = packets(1)[0]
        self.assertEqual([len(extract(packet, name)) for name in FEATURE_SETS], [6, 12, 18])
        self.assertEqual(len(set(FEATURE_SETS['imu_attitude_command'])), 18)
        with self.assertRaises(TypeError): extract({'truth_x': 1})
        with self.assertRaises(ValueError): packet.gyro_body[0] = 7

    def test_causal_offline_streaming_and_future_independence(self):
        settings = SETTINGS['diagnostics']; data = packets(); X, times = causal_windows(data, settings)
        buffer = CausalWindow(settings); values = [v for packet in data if (v := buffer.push(packet)) is not None]
        np.testing.assert_array_equal(X, np.stack([v[0] for v in values]))
        np.testing.assert_array_equal(times, np.stack([v[1] for v in values]))
        prefix, prefix_times = causal_windows(data[:65], settings)
        np.testing.assert_array_equal(prefix, X[:len(prefix)])
        np.testing.assert_array_equal(prefix_times, times[:len(prefix_times)])
        self.assertAlmostEqual(times[0, -1], .49)
        labels = offline_labels(times, 7, .555)
        self.assertEqual(labels[0], 0); self.assertEqual(labels[1], -1)
        np.testing.assert_array_equal(causal_windows(data, settings)[0], X)

    def test_gap_invalid_and_early_stop_do_not_fill(self):
        settings = SETTINGS['diagnostics']; data = packets(110)
        missing = data[:30]+data[31:]
        _, t = causal_windows(missing, settings)
        self.assertAlmostEqual(t[0, 0], .31); self.assertAlmostEqual(t[0, -1], .80)
        self.assertEqual(len(causal_windows(data[:49], settings)[0]), 0)
        invalid = ObservationPacket(.3, [0, 0, 9.81], [0, 0, 0], valid=False)
        _, t = causal_windows(data[:30]+[invalid]+data[31:], settings)
        self.assertAlmostEqual(t[0, 0], .31)
        with self.assertRaises(ValueError): causal_windows([data[0], data[0]], settings)

    def test_alarm_persistence_latch_and_truth_not_needed(self):
        predictor = lambda X: np.r_[.05, .95, np.zeros(17)]
        result = replay(packets(100), predictor, SETTINGS['diagnostics'])
        alarms = [p for p in result if p['alarm']]
        self.assertEqual(len(alarms), 1); self.assertAlmostEqual(alarms[0]['time'], .69)

    def test_paired_random_conditions_have_identical_healthy_prefix(self):
        cfg = load(); streams = {'sensor': 310, 'wind': 911}
        normal = simulate(cfg, .3, seed_streams=streams, noise=True, wind=True, feedback='observed_feedback')
        fault = simulate(cfg, .3, schedule=FaultSchedule(4, 0, .205), seed_streams=streams, noise=True, wind=True, feedback='observed_feedback')
        for left, right in zip(normal['rows'], fault['rows']):
            if left['time'] < .205:
                np.testing.assert_array_equal(left['state'], right['state'])
                np.testing.assert_array_equal(left['measurement']['gyro_body'], right['measurement']['gyro_body'])
            else: break


if __name__ == '__main__': unittest.main()
