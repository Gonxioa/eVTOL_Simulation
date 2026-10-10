"""Replay safety gates: saved clocks, provenance, and independent predictions."""
import csv
import hashlib
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import urlopen

from hex_viewer.data import ReplayDataset, SERIES
from hex_viewer.server import make_handler


class ReplayChecks(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.csv = self.folder/'g006_circle_c01.csv'
        self.side_path = self.csv.with_suffix('.json')
        self.side = {
            'schema_version': 5, 'group': {'group_id': 6}, 'scenario': {'label': 1},
            'record_id': 'g006_c01', 'requested_duration_s': 8., 'actual_duration_s': 3.8,
            'rows': 2, 'fault_start_s': 2.587, 'termination': 'envelope_exit', 'flight_metrics': {},
            'task': {'family': 'circle', 'origin_m': [0, 0, 2.5], 'motion_start_s': 1.,
                     'motion_end_s': 7., 'heading_rad': 0., 'turn_rad': 6.283185307179586,
                     'radius_m': .3},
            'config': {'frame': 'ENU_world_FLU_body_xyzw_body_to_world',
                       'parameter_status': 'test_fixture',
                       'termination': {'minimum_altitude_m': -2, 'position_norm_m': 100,
                                       'speed_m_s': 50, 'body_rate_rad_s': 100},
                       'airframe': {'arm_radius_m': .275, 'rotor_radius_m': .12065,
                                    'center_radius_m': .1}, 'propulsion': {'omega_max_rad_s': 1200.}}
        }
        fields = ['time', 'reference_yaw', *[c for cols in SERIES.values() for c in cols]]
        self.rows = []
        for t, z in [(0, 2.5), (3.8, -2.01)]:
            row = dict.fromkeys(fields, 0.)
            row.update(time=t, truth_z=z, reference_position_z=2.5)
            row.update({f'effectiveness{i}': 1. for i in range(6)})
            if t: row['effectiveness0'] = 0.
            self.rows.append(row)
        self.save()

    def save(self):
        with self.csv.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader(); writer.writerows(self.rows)
        self.side['csv_sha256'] = hashlib.sha256(self.csv.read_bytes()).hexdigest()
        self.side_path.write_text(json.dumps(self.side), encoding='utf-8')

    def test_early_record_retains_only_saved_samples(self):
        record = ReplayDataset(self.folder).record(6, 1)
        self.assertEqual(record['series']['time'], [0., 3.8])
        self.assertEqual(record['series']['position'][-1], [0., 0., -2.01])
        self.assertEqual(record['meta']['termination_reasons'], ['z < -2 m'])
        self.assertEqual(record['meta']['first_fault_sample_s'], 3.8)
        self.assertEqual(record['meta']['fault_time_s'], 2.587)
        self.assertEqual(record['planned']['time'][-1], 8.)
        self.assertEqual(len(record['series']['time']), 2)  # No filling or extension.
        self.assertEqual(record['planned']['position'][0], [0, 0, 2.5])
        self.assertAlmostEqual(record['planned']['position'][-1][1], 0.)

    def test_tampered_csv_is_rejected_without_rewriting(self):
        self.csv.write_bytes(self.csv.read_bytes()+b'\n')
        before = self.csv.read_bytes()
        with self.assertRaisesRegex(ValueError, '哈希'):
            ReplayDataset(self.folder).record(6, 1)
        self.assertEqual(before, self.csv.read_bytes())

    def test_nonmonotonic_or_nonfinite_samples_are_rejected(self):
        self.rows[1]['time'] = 0.; self.save()
        with self.assertRaisesRegex(ValueError, '递增'):
            ReplayDataset(self.folder).record(6, 1)
        self.rows[1]['time'] = 3.8; self.rows[1]['truth_z'] = float('nan'); self.save()
        with self.assertRaisesRegex(ValueError, '无效值'):
            ReplayDataset(self.folder).record(6, 1)

    def test_identity_and_actual_end_must_match(self):
        self.side['actual_duration_s'] = 8.; self.save()
        with self.assertRaisesRegex(ValueError, '起止时间'):
            ReplayDataset(self.folder).record(6, 1)
        self.side['actual_duration_s'] = 3.8; self.side['group']['group_id'] = 7; self.save()
        with self.assertRaisesRegex(ValueError, '标识'):
            ReplayDataset(self.folder)

    def test_missing_or_invalid_prediction_does_not_become_truth(self):
        source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.folder.iterdir()}
        dataset = ReplayDataset(self.folder, self.folder/'predictions')
        record = dataset.record(6, 1)
        self.assertIsNone(record['diagnostics'])
        self.assertFalse(record['meta']['diagnostics_available'])
        prediction_dir = self.folder/'predictions'; prediction_dir.mkdir()
        bad = [{'time': 8., 'valid': True}]  # Beyond the real early end.
        (prediction_dir/'g006_c01.json').write_text(json.dumps(bad), encoding='utf-8')
        other = ReplayDataset(self.folder, prediction_dir).record(6, 1)
        self.assertIsNone(other['diagnostics'])
        self.assertIn('时间', other['meta']['diagnostics_error'])
        self.assertEqual(other['series'], record['series'])
        self.assertEqual(source, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in self.folder.iterdir() if p.is_file()})

    def test_http_serves_local_catalog_and_rejects_other_files(self):
        dataset = ReplayDataset(self.folder)
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(dataset))
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        url = f'http://127.0.0.1:{server.server_port}'
        with urlopen(url+'/api/catalog') as response:
            self.assertEqual(json.load(response)['count'], 1)
        with urlopen(url+'/api/record?group=6&label=1') as response:
            self.assertEqual(json.load(response)['meta']['actual_end_s'], 3.8)
        with self.assertRaises(HTTPError) as caught:
            urlopen(url+'/../hex_research/configs/default.json')
        self.assertEqual(caught.exception.code, 404)
        with self.assertRaises(HTTPError) as caught:
            urlopen(url+'/api/record?group=6&label=19')
        self.assertEqual(caught.exception.code, 400)


if __name__ == '__main__':
    unittest.main()
