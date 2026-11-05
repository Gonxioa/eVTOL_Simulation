"""V3 input isolation, prefix endpoints, source matching and test authorization."""
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from provenance import stable_hash, sha256_file, write_json
from simulink_data import FEATURES, observable_features, read
from simulink_v3_data import (CONFIG, EXPECTED, make_windows, pool, root, array_hash,
                              authorize_test, immutable_json, scope_masks)
from simulink_v3_experiment import (fit_scalers, augmentation_counts, bank_additions,
                                    ros_additions, scoped_metrics, causal_window_predictions,
                                    alarm_metrics, select_ratio)


def signals():
    t = np.arange(2401)/100
    df = pd.DataFrame(dict(time_s=t))
    for prefix, count in [('accelerometer_sensor_m_s2_', 3), ('gyro_sensor_rad_s_', 3),
                           ('rotor_command_rad_s_', 6), ('feedback_state_', 19), ('rotor_actual_rad_s_', 6)]:
        for i in range(1, count+1): df[f'{prefix}{i}'] = t*10+i
    return df


class V3Tests(unittest.TestCase):
    def test_existing_plain_GAN_supports_12D_without_command_channels(self):
        import WGAN
        import torch
        keys = ('N_FEATURES', 'N_CLASSES', 'SEQ_LEN', 'HIDDEN_DIM', 'N_LAYERS', 'NOISE_DIM', 'EMBED_DIM', 'DEVICE')
        saved = {k: getattr(WGAN, k) for k in keys}
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            WGAN.DEVICE = torch.device('cpu'); WGAN.HIDDEN_DIM = 8; WGAN.N_LAYERS = 1
            WGAN.NOISE_DIM = 4; WGAN.EMBED_DIM = 4; WGAN.configure_dimensions(12, 19, 8)
            g = WGAN.Generator(); d = WGAN.Discriminator()
            y = torch.tensor([0, 7]); fake = g(WGAN.sample_noise(2), y)
            self.assertEqual(tuple(fake.shape), (2, 8, 12)); self.assertFalse(g.command_indices)
            score = d(fake, y); score.sum().backward()
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in g.parameters()))
        finally:
            for k, v in saved.items(): setattr(WGAN, k, v)
            torch.set_num_threads(threads)

    def test_main_features_ignore_commands_truth_metadata(self):
        a = signals(); meta = dict(feedback_delay_s=.03, parameters=dict(sensor_dt=.01))
        b = a.copy()
        for key in b:
            if key.startswith(('rotor_command', 'rotor_actual')): b[key] = 1e9
        b['fault_start_s'] = -4; b['estimated_effectiveness'] = 0
        x = observable_features(a, meta, 'imu_rpm'); y = observable_features(b, meta, 'imu_rpm')
        np.testing.assert_array_equal(x, y); self.assertEqual(x.shape, (2401, 12))
        self.assertEqual(FEATURES['imu_rpm'], ['ax', 'ay', 'az', 'gx', 'gy', 'gz']+[f'rpm_feedback{i}' for i in range(6)])
        np.testing.assert_array_equal(x[:, 6:], a[[f'feedback_state_{i}' for i in range(14, 20)]].to_numpy().astype(np.float32))
        self.assertAlmostEqual(x[10, 0], a.accelerometer_sensor_m_s2_1.iloc[7], places=5)
        with self.assertRaises(ValueError): observable_features(a.drop(columns=['feedback_state_14']), meta, 'imu_rpm')
        self.assertFalse(np.array_equal(observable_features(a, meta, 'imu_cmd_rpm'), observable_features(b, meta, 'imu_cmd_rpm')))

    def test_prefix_counts_endpoints_and_labels(self):
        times = np.arange(2401)/100; values = times[:, None]
        for s, (healthy, fault, per_group) in EXPECTED.items():
            a = make_windows(values, times, 0, prefix_s=s)
            b = make_windows(values, times, 18, prefix_s=s)
            self.assertEqual((len(a['y']), len(b['y'])), (healthy, fault))
            self.assertEqual(healthy+18*fault, per_group)
            self.assertEqual(a['start_s'][0], 0.); self.assertEqual(b['start_s'][0], 2.5)
            self.assertEqual(b['end_s'][0], 3.49); self.assertEqual(a['end_s'][-1], s-.01)
            self.assertTrue(np.all(b['y'] == 18)); self.assertTrue(np.all(a['y'] == 0))
            self.assertTrue(np.all(b['ends'] < s*100)); self.assertTrue(np.all(b['ends'] >= 300))
            self.assertEqual(np.sum(b['stages'] == 'transition'), 3)
        self.assertEqual(6*EXPECTED[6][2], 714)

    def test_time_grid_is_checked_and_no_padding(self):
        times = np.arange(2401)/100; values = times[:, None]
        bad = times.copy(); bad[10] += .001
        with self.assertRaises(ValueError): make_windows(values, bad, 0, prefix_s=6)
        with self.assertRaises(ValueError): make_windows(values[:500], times[:500], 0, prefix_s=6)
        out = make_windows(values, times, 1, prefix_s=6)
        np.testing.assert_array_equal(out['X'][0, :, 0], values[250:350, 0].astype(np.float32))

    def test_E_source_count_and_scaler_only_selected_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(read(CONFIG), results_root=directory)
            records = [dict(role='train', source_id=f's{c}', group_id=0, run=dict(label=c)) for c in range(19)]
            registry = dict(signature='fixture', records=records)
            meta = dict(parameters=dict(duration=24, sensor_dt=.01), fault_start_s=3., feedback_delay_s=.03)
            with patch('simulink_v3_data.load_record', return_value=(meta, signals())):
                d = pool(cfg, registry, 'train', prefix_s=6)
                e = pool(cfg, registry, 'train', prefix_s=24, equal_count=True)
                full = pool(cfg, registry, 'train', prefix_s=24)
            self.assertEqual(len(e['y']), 119)
            self.assertEqual([(s['source_id'], s['count']) for s in d['selection']],
                             [(s['source_id'], s['count']) for s in e['selection']])
            self.assertTrue(any(s >= 1200 for r in e['selection'] for s in r['starts']))
            for r in e['selection']: self.assertEqual(len(r['starts']), len(set(r['starts'])))
            cls, gan, scale_dir = fit_scalers(cfg, e)
            expected = e['X'].reshape(-1, 12).mean(0, dtype=np.float64)
            np.testing.assert_allclose(cls.mean_, expected, atol=1e-8)
            self.assertFalse(np.allclose(cls.mean_, full['X'].reshape(-1, 12).mean(0)))
            self.assertEqual(read(Path(scale_dir)/'fit_sources.json')['X_sha256'], array_hash(e['X']))
            self.assertEqual(int(cls.n_samples_seen_), 11900)
            self.assertNotEqual(d['signature'], e['signature'])

    def test_main_argmax_is_separate_from_alarm_gate(self):
        values = dict(y=np.array([0, 1]), starts=np.array([0, 1200]), ends=np.array([99, 1299]))
        # Include an early-fault point so every predeclared scope is nonempty.
        values = dict(y=np.array([0, 1, 1, 1, 1]), starts=np.array([0, 250, 400, 600, 1200]),
                      ends=np.array([99, 349, 499, 699, 1299]))
        logits = np.zeros((5, 19)); logits[:, 0] = 1.
        prediction = dict(logits=logits, fault_probability=np.full(5, .9))
        a = scoped_metrics(values, prediction); b = scoped_metrics(values, prediction, .5)
        self.assertEqual(a['prefix_24']['accuracy'], .2); self.assertEqual(b['prefix_24']['accuracy'], .8)

    def test_floor_augmentation_and_matched_ROS(self):
        y = np.r_[np.zeros(66, int), np.repeat(np.arange(1, 19), 36)]
        data = dict(X=np.arange(len(y)*2).reshape(-1, 1, 2), y=y)
        counts = augmentation_counts(y, .25)
        self.assertEqual(counts[0], 16); self.assertEqual(counts[1], 9)
        rx, ry, refs = ros_additions(data, counts, 5)
        gx, gy = bank_additions(data['X'], y, counts)
        np.testing.assert_array_equal(np.bincount(ry), np.bincount(gy))
        np.testing.assert_array_equal(rx, data['X'][refs])

    def test_test_cache_read_needs_freeze_and_unchanged_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(read(CONFIG), results_root=directory); registry = dict(signature='new_test', records=[])
            with self.assertRaises(PermissionError): pool(cfg, registry, 'test')
            path = Path(directory)/'model.bin'; path.write_bytes(b'first')
            frozen = dict(config_signature=stable_hash(cfg), test_registry_signature='new_test',
                          locked_files={str(path): sha256_file(path)})
            frozen['signature'] = stable_hash(frozen); write_json(Path(directory)/'freeze.json', frozen)
            authorize_test(cfg, registry, frozen['signature'])
            with self.assertRaises(PermissionError): authorize_test(cfg, registry, 'bad')
            path.write_bytes(b'changed')
            with self.assertRaises(PermissionError): authorize_test(cfg, registry, frozen['signature'])

    def test_causal_predictions_have_no_fault_time_and_ignore_future(self):
        from sklearn.preprocessing import StandardScaler
        self.assertNotIn('fault_start', inspect.signature(causal_window_predictions).parameters)
        rng = np.random.default_rng(8); a = rng.normal(size=(2401, 12)).astype(np.float32)
        b = a.copy(); b[1200:] = 1e6
        scaler = StandardScaler().fit(a[:600])
        def pred(model, x):
            logits = np.tile(x.mean((1, 2))[:, None], (1, 19))
            return dict(logits=logits, fault_probability=np.zeros(len(x)))
        with patch('classify.predict_classifier', side_effect=pred):
            ends, p = causal_window_predictions(None, scaler, a)
            _, q = causal_window_predictions(None, scaler, b)
        np.testing.assert_array_equal(p['logits'][ends < 1200], q['logits'][ends < 1200])
        self.assertEqual((ends[0], ends[-1], len(ends)), (99, 2399, 231))

    def test_preexisting_alarm_not_new_detection_and_unique_healthy_exposure(self):
        times = np.arange(99, 2400, 10)/100
        rows = [dict(source_id='healthy', group_id=0, label=0, fault_start_s=3,
                     time_s=times, fault_probability=np.zeros(len(times)), fault_class=np.ones(len(times), int)),
                dict(source_id='fault', group_id=0, label=1, fault_start_s=3,
                     time_s=times, fault_probability=np.ones(len(times)), fault_class=np.ones(len(times), int))]
        score = alarm_metrics(rows, .5, 2)
        self.assertEqual(score['detected'], 0); self.assertEqual(score['missed'], 1)
        self.assertEqual(score['normal_observation_s'], 23.)
        self.assertEqual(score['detected_within_0.2s_rate'], 0.)

    def test_fixed_roles_and_ratio_selection_across_seeds(self):
        cfg = read(CONFIG)
        roles = [set(cfg[k]) for k in ('train_groups', 'validation_groups', 'test_groups')]
        self.assertEqual([len(s) for s in roles], [6, 3, 3])
        self.assertFalse(roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
        jobs = [dict(prefix_s=6, feature='imu_rpm', equal_count=False, seed=s, method='raw' if r == 0 else 'gan',
                     ratio=r, signature=f'{r}_{s}', validation={'prefix_24': {'macro_f1': .5}})
                for r in (0., .25, .5, 1.) for s in (42, 43, 44)]
        self.assertEqual(select_ratio(jobs, 'gan', 6)['selected_ratio'], 0.)
        with self.assertRaises(ValueError): select_ratio(jobs[:-1], 'gan', 6)

    def test_evaluation_prefix_views_share_windows(self):
        times = np.arange(2401)/100; values = times[:, None]
        for label in (0, 1):
            whole = make_windows(values, times, label, prefix_s=24)
            for seconds in (6, 12):
                part = make_windows(values, times, label, prefix_s=seconds)
                mask = scope_masks(whole)[f'prefix_{seconds}']
                np.testing.assert_array_equal(part['X'], whole['X'][mask])


if __name__ == '__main__': unittest.main()
