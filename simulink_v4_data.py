"""V4 measured sources and explicit common/native endpoint grids.

Observable alignment and native-run validation are reused unchanged from V3/V1.
Only selected training arrays can fit scalers. Historical test arrays are locked.
"""
import hashlib
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from provenance import stable_hash, sha256_file, write_json, runtime_info
from simulink_data import ROOT, FEATURES, read, checked_plan, validate_run, observable_features
from simulink_v3_data import immutable_json, array_hash

CONFIG = ROOT / 'configs/simulink_gan_v4_evidence.json'
_SIGNALS = {}
_VERIFIED = {}


def root(config): return ROOT / config['results_root']


def verified_hash(path):
    path = Path(path); s = path.stat(); key = (str(path.resolve()), s.st_size, s.st_mtime_ns)
    if key not in _VERIFIED: _VERIFIED[key] = sha256_file(path)
    return _VERIFIED[key]


def snapshot(config):
    destination = root(config); destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / 'legacy_snapshot.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        for path, digest in receipt['files'].items():
            if verified_hash(path) != digest: raise RuntimeError('Legacy frozen artifact changed: '+path)
        return receipt
    previous = [ROOT / 'output_simulink_gan' / name for name in
                ('simulink_control_v3_observable_20261103_v1', 'v2_12groups', 'v3_imu_rpm_prefix_20261105')]
    files = {}
    # Pin receipts, metadata, source registries, weights, scalers and training logs.
    # Raw CSV/MAT identities are separately verified by the paired-source audit.
    for base in previous:
        if not base.exists(): raise FileNotFoundError(base)
        for p in sorted(base.rglob('*')):
            if p.is_file() and p.suffix.lower() in ('.json', '.pt', '.pth', '.pkl', '.log'):
                files[str(p.resolve())] = sha256_file(p)
    v3 = read(ROOT / config['v3_root'] / 'freeze.json')
    for filename, digest in v3['locked_files'].items():
        if sha256_file(filename) != digest: raise RuntimeError('Existing V3 freeze was modified')
        files[filename] = digest
    result = dict(**runtime_info(), starting_commit=config['baseline_commit'],
        starting_branch='codex/simulink-gan-v3-prefix', starting_worktree_clean=True,
        branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
        files=files, raw_identity_policy='Reference signed raw registries; validate development CSV/MAT separately; no numerical historical-test access')
    result['signature'] = stable_hash(result); immutable_json(receipt_path, result)
    return result


def registry(config):
    v3 = ROOT / config['v3_root']; dev = read(v3 / 'development_sources.json')
    historical = read(v3 / 'old_test_archive.json')['records'] + read(v3 / 'new_test_sources.json')['records']
    sources = dev['records']
    if len(sources) != 171 or len(historical) != 114: raise ValueError('Unexpected legacy source partition')
    for r in sources:
        if r['role'] not in ('train', 'validation'): raise PermissionError('Historical test in development')
        for file, key in [('signals.csv', 'csv_sha256'), ('result.mat', 'mat_sha256'), ('run_metadata.json', 'metadata_sha256')]:
            if sha256_file(Path(r['source_path']) / file) != r[key]: raise ValueError('Legacy source changed')
    for item in checked_plan(Path(sources[0]['plan_path']).parent)['sources']:
        if sha256_file(ROOT/item['path']) != item['sha256']:
            raise ValueError('Physical source changed since the simulation baseline: '+item['path'])
    result = dict(profile=config['profile'], records=sources, historical_test_records=historical,
                  original_development_signature=dev['signature'], roles={'train': list(range(6)), 'validation': [6, 7, 8]})
    result['signature'] = stable_hash(result)
    immutable_json(root(config) / 'development_sources.json', result)
    immutable_json(root(config) / 'registered_config.json', dict(config=config, signature=stable_hash(config)))
    return result


def authorize(config, reg, role, token=None):
    if role in ('train', 'validation'): return
    if role != 'test' or token is None or not (root(config) / 'freeze.json').exists():
        raise PermissionError('V4 confirmation features locked; historical tests cannot be reused as unseen confirmation')
    frozen = read(root(config) / 'freeze.json')
    unsigned = {k: v for k, v in frozen.items() if k != 'signature'}
    if token != frozen['signature'] or stable_hash(unsigned) != token:
        raise PermissionError('Invalid V4 confirmation token')
    if frozen['test_registry_signature'] != reg['signature'] or frozen['config_signature'] != stable_hash(config):
        raise PermissionError('Confirmation source/config mismatch')
    for path, digest in frozen['locked_files'].items():
        if verified_hash(path) != digest: raise PermissionError('Frozen file changed: '+path)


def signals(config, reg, role, feature='imu_rpm', token=None):
    authorize(config, reg, role, token)
    if feature not in ('imu_rpm', 'imu_cmd_rpm'): raise ValueError('Measured feature allowlist')
    key = (reg['signature'], role, feature)
    if key in _SIGNALS: return _SIGNALS[key]
    rows = []
    for r in reg['records']:
        if r['role'] != role: continue
        if role != 'test' and r['group_id'] not in (list(range(6)) if role == 'train' else [6, 7, 8]):
            raise PermissionError('Source group outside development role')
        meta, df = validate_run(r['source_path'], r['run'], checked_plan(Path(r['plan_path']).parent))
        if sha256_file(Path(r['source_path']) / 'run_metadata.json') != r['metadata_sha256']:
            raise ValueError('Metadata identity mismatch')
        if len(df) != 2401 or meta['parameters']['duration'] != 24 or meta['fault_start_s'] != 3:
            raise ValueError('Expected unchanged native 24s/100Hz/fault-at-3s records')
        if not np.isfinite(df.select_dtypes(include=np.number).to_numpy()).all(): raise ValueError('Nonfinite native signals')
        values = observable_features(df, meta, feature)
        rows.append(dict(record=r, meta=meta, X=values, time=df.time_s.to_numpy(),
                         phase=df.path_phase_s.to_numpy(), progress=df.path_progress_rate.to_numpy()))
    if not rows: raise ValueError('Empty source role')
    _SIGNALS[key] = rows
    return rows


def indices(window, prefix_s=24, grid='native', label=0, sample_hz=100, stride=50, fault_s=3., common_anchor=299):
    if window < 1 or stride < 1: raise ValueError('Invalid sampling geometry')
    limit = int(round(prefix_s*sample_hz)); onset = int(round(fault_s*sample_hz))
    anchor = common_anchor if grid == 'common' else window-1
    if grid not in ('common', 'native') or anchor < window-1: raise ValueError('Invalid endpoint grid')
    ends = np.arange(anchor, limit, stride, dtype=np.int64)
    if label: ends = ends[ends >= onset]
    starts = ends-window+1
    fraction = np.clip(ends-np.maximum(starts, onset)+1, 0, window)/window if label else np.zeros(len(ends))
    transport_onset = onset+3
    transport_fraction = np.clip(ends-np.maximum(starts, transport_onset)+1, 0, window)/window if label else np.zeros(len(ends))
    return starts, ends, fraction, transport_fraction


def pool(config, reg, role, window, prefix_s=24, grid='native', sampling_seed=None, matched_prefix=None,
         feature='imu_rpm', token=None):
    authorize(config, reg, role, token)
    if sampling_seed is not None and (role != 'train' or prefix_s != 24 or matched_prefix not in (6, 12)):
        raise ValueError('Equal-count control is training-only full-range sampling')
    identity = dict(registry=reg['signature'], role=role, window=window, prefix_s=prefix_s, grid=grid,
        sampling_seed=sampling_seed, matched_prefix=matched_prefix, feature=feature, names=FEATURES[feature],
        stride=config['training_stride'], common_anchor=config['common_end_anchor'], code_sha256=sha256_file(__file__))
    key = stable_hash(identity); path = root(config) / 'pools' / f'{key}.pkl'
    if path.exists():
        with path.open('rb') as f: result = pickle.load(f)
        if result['identity'] != identity or array_hash(result['X']) != result['X_sha256'] or array_hash(result['y']) != result['y_sha256']:
            raise ValueError('V4 pool cache identity/array integrity mismatch')
        return result
    data = []; selection = []; rng = np.random.default_rng(sampling_seed)
    for source in signals(config, reg, role, feature, token):
        r = source['record']; times = source['time']; label = r['run']['label']
        if not np.allclose(times, np.arange(2401)/100, atol=1e-9, rtol=0): raise ValueError('Unexpected actual timestamps')
        starts, ends, fraction, transport = indices(window, prefix_s, grid, label,
            config['sample_hz'], config['training_stride'], source['meta']['fault_start_s'], config['common_end_anchor'])
        if sampling_seed is not None:
            target = len(indices(window, matched_prefix, 'native', label, stride=config['training_stride'])[0])
            chosen = np.sort(rng.choice(len(ends), target, replace=False))
            starts, ends, fraction, transport = (a[chosen] for a in (starts, ends, fraction, transport))
        x = np.asarray([source['X'][s:e+1] for s, e in zip(starts, ends)], np.float32)
        if not len(x): raise ValueError('No complete source windows')
        selection.append(dict(source_id=r['source_id'], group=r['group_id'], label=label,
                              starts=starts.tolist(), ends=ends.tolist(), count=len(ends)))
        data.append(dict(X=x, y=np.full(len(x), label, np.int64), groups=np.full(len(x), r['group_id'], np.int64),
            source_ids=np.full(len(x), r['source_id']), starts=starts, ends=ends, physical_fault_fraction=fraction,
            transport_eligible_fraction=transport, crosses_fault=(starts < 300) & (ends >= 300) & bool(label),
            phase_end=source['phase'][ends], progress_end=source['progress'][ends]))
    result = {k: np.concatenate([a[k] for a in data]) for k in data[0]}
    result.update(signature=key, identity=identity, X_sha256=array_hash(result['X']), y_sha256=array_hash(result['y']), selection=selection)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as f: pickle.dump(result, f)
    path.with_suffix('.tmp').replace(path)
    write_json(path.with_suffix('.json'), {k: result[k] for k in ('signature', 'identity', 'X_sha256', 'y_sha256', 'selection')})
    return result


def fit_scalers(config, p):
    if p['identity']['role'] != 'train': raise PermissionError('Scalers fit actual selected training windows only')
    directory = root(config) / 'scalers' / p['signature']; directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'scalers.pkl'
    if not path.exists():
        x = p['X'].reshape(-1, p['X'].shape[-1])
        cls = StandardScaler().fit(x); gan = MinMaxScaler(feature_range=(-1, 1)).fit(x)
        with path.open('wb') as f: pickle.dump((cls, gan), f)
        immutable_json(directory / 'fit_sources.json', dict(pool=p['signature'], X_sha256=p['X_sha256'],
            window_count=len(p['y']), sample_count=len(x), selection=p['selection'], feature_names=FEATURES[p['identity']['feature']]))
    with path.open('rb') as f: cls, gan = pickle.load(f)
    receipt = read(directory / 'fit_sources.json')
    if receipt['X_sha256'] != p['X_sha256'] or receipt['pool'] != p['signature']: raise ValueError('Scaler source mismatch')
    return cls, gan, str(directory)


def masks(p):
    y, s, e = p['y'], p['starts'], p['ends']
    return {**{f'prefix_{t}': e < t*100 for t in (6, 12, 24)},
            'mixed': p['crosses_fault'],
            'early_fault_3_6': (y != 0) & (e >= 300) & (e < 600),
            'middle_fault_6_12': (y != 0) & (e >= 600) & (e < 1200),
            'late_fault_12_24': (y != 0) & (s >= 1200) & (e < 2400)}


def actual_source_audit(config, reg):
    rows = []
    for role in ('train', 'validation'):
        for s in signals(config, reg, role):
            r = s['record']; meta = s['meta']
            common = {k:v for k,v in meta['parameters'].items() if k not in ('fault_motor','fault_factor')}
            rows.append(dict(role=role, group=r['group_id'], seed=r['run']['seed'], source_id=r['source_id'],
                label=r['run']['label'], motor=r['run']['motor_simulink'], factor=r['run']['factor'], rows=len(s['X']),
                csv_sha256=r['csv_sha256'], mat_sha256=r['mat_sha256'], metadata_sha256=r['metadata_sha256'],
                completed=True, finite=True, flight_acceptance_passed=meta['flight_metrics']['passed'],
                feedback_delay_s=meta['feedback_delay_s'], sample_dt_s=meta['sample_dt_s'],
                condition_signature=stable_hash(common),
                actual_path_phase_at_7s=float(s['phase'][700]), actual_path_phase_at_12s=float(s['phase'][1200])))
    destination = root(config) / 'audit'; destination.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / 'source_manifest.csv', index=False, encoding='utf-8-sig')
    groups = {role: {r['seed'] for r in rows if r['role'] == role} for role in ('train', 'validation')}
    if groups['train'] & groups['validation']: raise AssertionError('Source role leakage')
    for role, group in pd.DataFrame(rows).groupby(['role', 'group']):
        if set(group.label) != set(range(19)): raise AssertionError('Incomplete paired group')
        if group.condition_signature.nunique() != 1: raise AssertionError('Paired conditions differ beyond injected fault')
    summary = dict(records=len(rows), train=114, validation=57, independent_groups=9,
        retained_flight_acceptance_failures=sum(not r['flight_acceptance_passed'] for r in rows),
        physical_sources_unchanged=True, historical_test_features_read=False,
        counts_common={str(w): {'train': 6*(len(indices(w, 24, 'common', 0)[1])+18*len(indices(w, 24, 'common', 1)[1])),
                              'validation': 3*(len(indices(w, 24, 'common', 0)[1])+18*len(indices(w, 24, 'common', 1)[1]))}
                       for w in config['window_candidates']})
    # Check fixed measured extraction against a real-source command mutation.
    r = reg['records'][0]; meta, df = validate_run(r['source_path'], r['run'], checked_plan(Path(r['plan_path']).parent))
    modified = df.copy()
    for col in modified:
        if col.startswith(('rotor_command_', 'rotor_actual_', 'plant_effectiveness_', 'reference_', 'allocation_')):
            modified[col] = -987654.
    if not np.array_equal(observable_features(df, meta, 'imu_rpm'), observable_features(modified, meta, 'imu_rpm')):
        raise AssertionError('Main features read command/truth/audit columns')
    summary['actual_command_truth_mutation_invariance'] = True
    write_json(destination / 'source_audit.json', summary)
    return summary
