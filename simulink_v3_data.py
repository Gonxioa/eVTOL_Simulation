"""Independent V3 sources, integer-prefix windows and guarded test reads.

The V1/V2 collectors, physical model and observable alignment remain unchanged.
Metadata is never a neural input. Test authorization precedes even cache reads.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from provenance import stable_hash, sha256_file, write_json, runtime_info
from simulink_data import ROOT, PROFILE, FEATURES, scenarios, read, checked_plan, validate_run, observable_features

CONFIG = ROOT / 'configs/simulink_gan_v3_prefix.json'
EXPECTED = {6: (11, 6, 119), 12: (23, 18, 347), 24: (47, 42, 803)}
_VERIFIED_FILES = {}


def root(config):
    return ROOT / config['results_root']


def immutable_json(path, payload):
    path = Path(path)
    if path.exists():
        if read(path) != payload:
            raise RuntimeError(f'Immutable artifact changed: {path}')
    else:
        write_json(path, payload)
    return payload


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def prepare_test_plan(config):
    """Explicit three-group entry; no duplicated seeds or relaxing the V1 entry."""
    old = read(ROOT / config['v2_registry'])
    used = set()
    scanned = []
    for path in sorted((ROOT / 'output_simulink_gan').rglob('collection_plan.json')):
        if path.parent.resolve() == (ROOT / config['new_test_raw']).resolve():
            continue
        plan = read(path)
        values = sorted({int(r['seed']) for r in plan['runs']})
        used.update(values)
        scanned.append(dict(path=str(path), sha256=sha256_file(path), seeds=values))
    # Also include all known simulation seed settings, including non-GAN flights.
    for path in sorted((ROOT / 'configs').glob('*.json')) + sorted((ROOT / 'simulink').rglob('*.json')):
        if path.resolve() == CONFIG.resolve():
            continue
        def visit(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if 'seed' in k.lower():
                        for s in (v if isinstance(v, list) else [v]):
                            if isinstance(s, int): used.add(s)
                    visit(v)
            elif isinstance(value, list):
                for v in value: visit(v)
        visit(read(path))
    seeds = config['new_test_seeds']
    if len(seeds) != 3 or len(set(seeds)) != 3 or set(seeds) & used:
        raise ValueError('V3 test requires exactly three previously unused seeds')
    reference = checked_plan(Path(old['records'][0]['plan_path']).parent)
    for item in reference['sources']:
        if sha256_file(ROOT / item['path']) != item['sha256']:
            raise ValueError('Physical model differs from the existing V2 sources')
    runs = [dict(**s, group_id=g, run_id=g, seed=int(seed),
                 run_key=f'g{g:03d}_label{s["label"]:02d}')
            for g, seed in enumerate(seeds) for s in scenarios()]
    plan = dict(profile=PROFILE, dataset_version=config['profile'] + '_new_test3',
                task_family=2, sources=reference['sources'], runs=runs, scenarios=scenarios(),
                purpose='Predeclared V3 confirmation test; no feature access until development freeze',
                fault_definition=reference['fault_definition'],
                group_policy='three new paired conditions, 19 records each', rotor_mapping_to_uavfd='unestablished')
    plan['plan_signature'] = stable_hash(plan)
    folder = ROOT / config['new_test_raw']
    immutable_json(folder / 'collection_plan.json', plan)
    immutable_json(root(config) / 'seed_selection_audit.json', dict(
        previously_used_seeds=sorted(used), selected_seeds=seeds, scanned_collection_plans=scanned,
        selection='First three predeclared integers after prior collection seeds; no trial flights or performance selection'))
    return folder / 'collection_plan.json'


def development_registry(config):
    old = read(ROOT / config['v2_registry'])
    records = [r for r in old['records'] if r['role'] in ('train', 'validation')]
    archived = [r for r in old['records'] if r['role'] == 'test']
    if len(records) != 171 or len(archived) != 57:
        raise ValueError('Expected original six train + three validation paired groups')
    expected = dict(train=config['train_groups'], validation=config['validation_groups'])
    source_ids, seeds = set(), {}
    for r in records:
        if r['source_id'] in source_ids or r['group_id'] not in expected[r['role']]:
            raise ValueError('Duplicate source/incorrect development role')
        source_ids.add(r['source_id'])
        for name, key in [('signals.csv', 'csv_sha256'), ('result.mat', 'mat_sha256'),
                          ('run_metadata.json', 'metadata_sha256')]:
            if sha256_file(Path(r['source_path']) / name) != r[key]:
                raise ValueError('Original data changed')
        seeds.setdefault(r['role'], set()).add(r['run']['seed'])
    if seeds['train'] & seeds['validation']:
        raise ValueError('Development seed overlap')
    registry = dict(profile=config['profile'], records=records, roles=expected,
                    original_registry_sha256=sha256_file(ROOT / config['v2_registry']),
                    model_sha256=old['model_sha256'], independent_records=171,
                    independent_paired_groups=9, new_test_plan_sha256=sha256_file(ROOT / config['new_test_raw'] / 'collection_plan.json'))
    registry['signature'] = stable_hash(registry)
    immutable_json(root(config) / 'development_sources.json', registry)
    immutable_json(root(config) / 'old_test_archive.json', dict(
        reason='Previously exposed V2 test; excluded from V3 training, validation and formal test',
        records=archived, count=57, kept_in_place=True))
    immutable_json(root(config) / 'predeclared_protocol.json', dict(config=config, config_signature=stable_hash(config)))
    return registry


def new_test_registry(config, development):
    """Read sidecar metadata/hashes only. Raw feature validation is after freeze."""
    folder = ROOT / config['new_test_raw']; plan = checked_plan(folder)
    records, incomplete = [], []
    for run in plan['runs']:
        path = folder / run['run_key']; meta_path = path / 'run_metadata.json'
        if not meta_path.exists():
            incomplete.append(dict(run=run, error='missing run metadata')); continue
        meta = read(meta_path)
        if meta['status'] != 'completed' or meta.get('termination') != 'completed':
            incomplete.append(dict(run=run, error=meta['status'])); continue
        if meta['plan_signature'] != plan['plan_signature'] or meta['run'] != run or meta['sources'] != plan['sources']:
            raise ValueError('New test sidecar provenance mismatch')
        if meta['last_time_s'] != 24 or meta['parameters']['duration'] != 24 or meta['fault_start_s'] != 3:
            raise ValueError('New test timing differs')
        for filename, key in [('signals.csv', 'csv_sha256'), ('result.mat', 'mat_sha256')]:
            if sha256_file(path / filename) != meta[key]: raise ValueError('New test file hash mismatch')
        records.append(dict(source_id=f'v3_test_{run["run_key"]}', role='test', group_id=9+run['group_id'],
                            source_path=str(path), plan_path=str(folder / 'collection_plan.json'), source_pool='v3_test3',
                            run=run, csv_sha256=meta['csv_sha256'], mat_sha256=meta['mat_sha256'],
                            metadata_sha256=sha256_file(meta_path), flight_acceptance_passed=bool(meta['flight_metrics']['passed'])))
    write_json(root(config) / 'test_collection_status.json', dict(records=len(records), planned=57, incomplete=incomplete))
    if incomplete or len(records) != 57:
        raise RuntimeError(f'New collection incomplete ({len(records)}/57); failures retained, not silently discarded')
    if set(r['run']['seed'] for r in records) & set(r['run']['seed'] for r in development['records']):
        raise ValueError('New test/development seed overlap')
    registry = dict(profile=config['profile'], records=records, roles={'test': config['test_groups']},
                    development_signature=development['signature'], plan_signature=plan['plan_signature'],
                    independent_records=57, independent_paired_groups=3)
    registry['signature'] = stable_hash(registry)
    immutable_json(root(config) / 'new_test_sources.json', registry)
    return registry


def authorize_test(config, registry, token):
    path = root(config) / 'freeze.json'
    if not path.exists() or token is None:
        raise PermissionError('Test features locked until immutable development freeze')
    frozen = read(path)
    unsigned = {k: v for k, v in frozen.items() if k != 'signature'}
    if stable_hash(unsigned) != frozen['signature'] or token != frozen['signature']:
        raise PermissionError('Invalid test authorization')
    if frozen['config_signature'] != stable_hash(config) or frozen['test_registry_signature'] != registry['signature']:
        raise PermissionError('Test/config changed after freeze')
    for filename, digest in frozen['locked_files'].items():
        stat = Path(filename).stat()
        identity = (str(filename), stat.st_size, stat.st_mtime_ns)
        # Avoid rehashing every model for each of 57 causal replay records.
        # A new process always verifies bytes; any file write invalidates this memo.
        if identity not in _VERIFIED_FILES:
            _VERIFIED_FILES[identity] = sha256_file(filename)
        if _VERIFIED_FILES[identity] != digest: raise PermissionError(f'Frozen artifact changed: {filename}')
    return frozen


def load_record(config, registry, record, authorization=None):
    if record['role'] == 'test': authorize_test(config, registry, authorization)
    for filename, key in [('signals.csv', 'csv_sha256'), ('run_metadata.json', 'metadata_sha256')]:
        if sha256_file(Path(record['source_path']) / filename) != record[key]: raise ValueError('Raw source changed')
    return validate_run(record['source_path'], record['run'], checked_plan(Path(record['plan_path']).parent))


def make_windows(values, times, label, fault_start=3., prefix_s=24, window=100, step=50, sample_hz=100):
    """Sample bounds are integers; prefix endpoint is exclusive, e.g. sample 600."""
    times = np.asarray(times)
    expected_times = np.arange(len(times)) / sample_hz
    if not np.allclose(times, expected_times, atol=1e-9, rtol=0): raise ValueError('Time grid must begin at zero')
    limit = int(round(prefix_s * sample_hz)); onset = int(round(fault_start * sample_hz))
    if abs(onset / sample_hz - fault_start) > 1e-9: raise ValueError('Off-grid fault onset')
    if limit > len(times) or window < 1 or step < 1: raise ValueError('Incomplete prefix or invalid window')
    starts = np.arange(0, limit - window + 1, step, dtype=np.int64)
    if label: starts = starts[starts + window - 1 >= onset]
    ends = starts + window - 1
    x = np.asarray([values[s:s+window] for s in starts], dtype=np.float32)
    stages = np.where(starts < onset + sample_hz, 'transition', 'post_fault') if label else np.full(len(starts), 'healthy')
    return dict(X=x, y=np.full(len(starts), label, np.int64), starts=starts, ends=ends,
                start_s=times[starts], end_s=times[ends], stages=stages)


def pool(config, registry, role, feature='imu_rpm', prefix_s=24, equal_count=False, authorization=None):
    if role == 'test': authorize_test(config, registry, authorization)
    if feature not in ('imu_rpm', 'imu_cmd_rpm'): raise ValueError('V3 feature allowlist')
    if equal_count and (role != 'train' or prefix_s != 24 or feature != 'imu_rpm'):
        raise ValueError('E is a training-only 12D full-range sampled pool')
    payload = dict(registry=registry['signature'], role=role, feature=feature, names=FEATURES[feature],
                   prefix_s=prefix_s, window=config['window'], step=config['step'], hz=config['sample_hz'],
                   equal_sampling_seed=config['equal_sampling_seed'] if equal_count else None,
                   alignment='V2 observable_features: IMU delayed once; feedback_state_14..19 already delayed',
                   code_sha256=sha256_file(__file__))
    signature = stable_hash(payload); path = root(config) / 'pools' / f'{role}_{signature}.pkl'
    if path.exists():
        with path.open('rb') as f: result = pickle.load(f)
        if result['signature'] != signature or result['payload'] != payload: raise ValueError('Pool cache signature mismatch')
        return result
    rows = []; selection = []; rng = np.random.default_rng(config['equal_sampling_seed'])
    for index, record in enumerate(registry['records']):
        if record['role'] != role: continue
        meta, df = load_record(config, registry, record, authorization)
        if meta['parameters']['duration'] != 24 or meta['fault_start_s'] != 3 or len(df) != 2401:
            raise ValueError('Expected complete 24s, 100Hz, fault-at-3s native record')
        values = observable_features(df, meta, feature)
        item = make_windows(values, df.time_s.to_numpy(), record['run']['label'], meta['fault_start_s'], prefix_s,
                            config['window'], config['step'], config['sample_hz'])
        expected_count = EXPECTED[prefix_s][int(record['run']['label'] != 0)]
        if len(item['y']) != expected_count: raise AssertionError('Prefix count differs from protocol')
        if equal_count:
            n = EXPECTED[6][int(record['run']['label'] != 0)]
            selected = np.sort(rng.choice(len(item['y']), n, replace=False))
            item = {k: v[selected] for k, v in item.items()}
        selection.append(dict(source_id=record['source_id'], group=record['group_id'], label=record['run']['label'],
                              count=len(item['y']), starts=item['starts'].tolist(), ends=item['ends'].tolist()))
        item.update(groups=np.full(len(item['y']), record['group_id'], np.int64),
                    source_indices=np.full(len(item['y']), index, np.int64),
                    source_ids=np.full(len(item['y']), record['source_id']))
        rows.append(item)
    if not rows: raise ValueError('Empty role pool')
    result = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    result.update(signature=signature, payload=payload, selection=selection, X_sha256=array_hash(result['X']))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f: pickle.dump(result, f)
    path.with_suffix('.tmp').replace(path)
    write_json(path.with_suffix('.json'), dict(signature=signature, payload=payload, window_count=len(result['y']),
                                             X_sha256=result['X_sha256'], selection=selection))
    return result


def scope_masks(values):
    y, starts, ends = values['y'], values['starts'], values['ends']
    return {**{f'prefix_{s}': ends < s*100 for s in (6, 12, 24)},
            'early_fault_3_6': (y != 0) & (ends >= 300) & (ends < 600),
            'middle_fault_6_12': (y != 0) & (ends >= 600) & (ends < 1200),
            'late_fault_12_24': (y != 0) & (starts >= 1200) & (ends < 2400),
            'transition': (y != 0) & (starts < 400), 'post_fault': (y != 0) & (starts >= 400)}
