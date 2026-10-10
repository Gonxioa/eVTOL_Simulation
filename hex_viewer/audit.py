"""Audit a saved replay dataset without simulating or changing source records."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from .data import ReplayDataset, SERIES, planned_reference
from .server import ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder, predictions):
    dataset = ReplayDataset(folder, predictions)
    sources = [p for p in dataset.folder.iterdir() if p.is_file() and p.suffix in ('.csv', '.json')]
    if dataset.predictions and dataset.predictions.is_dir():
        sources += sorted(dataset.predictions.glob('*.json'))
    before = {str(p): sha(p) for p in sources}
    checks = []; total_rows = 0; max_reference_error = 0.
    for (group, label), (side_path, side) in sorted(dataset.entries.items()):
        record = dataset.record(group, label); s = record['series']
        with side_path.with_suffix('.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(s['time']) == side['rows']
        for i, row in enumerate(rows):
            assert float(row['time']) == s['time'][i]
            assert float(row['reference_yaw']) == s['reference_yaw'][i]
            for series_name, columns in SERIES.items():
                assert [float(row[c]) for c in columns] == s[series_name][i]
            reference = planned_reference(side['task'], float(row['control_reference_time']))
            max_reference_error = max(max_reference_error, max(abs(a-b) for a,b in zip(reference, s['reference'][i])))
        assert s['time'][-1] == side['actual_duration_s']
        assert record['planned']['time'][-1] == side['requested_duration_s']
        total_rows += len(rows)
        checks.append({'record': side['record_id'], 'rows': len(rows), 'end_s': s['time'][-1],
                       'fault_time_s': record['meta']['fault_time_s'],
                       'first_fault_sample_s': record['meta']['first_fault_sample_s'],
                       'termination': side['termination'], 'reasons': record['meta']['termination_reasons'],
                       'prediction_windows': len(record['diagnostics'] or []),
                       'prediction_error': record['meta']['diagnostics_error']})
    assert max_reference_error < 1e-12, 'Display plan differs from saved control reference'
    after = {str(p): sha(p) for p in sources}
    assert before == after, 'Source records changed during audit'
    # Verify that the simulator/controller signature used for these records still matches.
    plan_path = dataset.folder/'generation_plan.json'
    matching = None; mismatches = []; resolutions = []
    if plan_path.is_file():
        expected = json.loads(plan_path.read_text(encoding='utf-8'))['source_sha256']
        mismatches = [name for name, digest in expected.items()
                      if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
        matching = not mismatches
        registry_path = ROOT/'docs/hex_next/source_snapshot_generation/register.json'
        registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.is_file() else {'entries': []}
        for name in mismatches:
            entry = next((e for e in registry['entries'] if e['original_path'] == name), None)
            snapshot = ROOT/entry['executed_snapshot_path'] if entry else None
            if entry and snapshot.is_file() and (ROOT/name).is_file():
                resolutions.append({'name': name, 'snapshot_matches_generation': sha(snapshot) == expected[name],
                                    'current_matches_registered_fix': sha(ROOT/name) == entry['current_sha256'],
                                    'change': entry['change'], 'snapshot': entry['executed_snapshot_path']})
        resolved = len(resolutions) == len(mismatches) and all(
            r['snapshot_matches_generation'] and r['current_matches_registered_fix'] for r in resolutions)
    else:
        resolved = None
    return {'executed_utc': datetime.now(timezone.utc).isoformat(), 'data_dir': str(dataset.folder),
            'records': len(checks), 'groups': len(dataset.catalog()['groups']), 'samples': total_rows,
            'completed': sum(c['termination'] == 'completed' for c in checks),
            'early_endings': sum(c['termination'] != 'completed' for c in checks),
            'saved_prediction_records': sum(c['prediction_windows'] > 0 for c in checks),
            'prediction_errors': [c for c in checks if c['prediction_error']],
            'source_files_hashed_before_and_after': len(before), 'source_changed_files': 0,
            'all_csv_series_equal_to_source': True, 'padded_flight_samples': 0,
            'maximum_reference_error_m': max_reference_error,
            'generation_code_hashes_match': matching, 'generation_code_mismatches': mismatches,
            'generation_provenance_resolved_with_registered_snapshots': resolved,
            'registered_source_differences': resolutions,
            'source_sha256': before, 'checks': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'output_hex_next_20261009/batch2/dataset_v5')
    parser.add_argument('--predictions-dir', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'output_hex_replay/data_checks.json')
    args = parser.parse_args()
    prediction_dir = args.predictions_dir or args.data_dir.parent/'diagnostics/imu_only_epoch30/predictions'
    target = args.output.resolve()
    if any(target.is_relative_to(path.resolve()) for path in (args.data_dir, prediction_dir)):
        parser.error('Audit output must be outside the source data/predictions folders')
    result = audit(args.data_dir, prediction_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','checks')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
