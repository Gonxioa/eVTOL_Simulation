"""Native Simulink collection/provenance adapter. No GAN architecture lives here."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from provenance import code_hashes, sha256_file, stable_hash, write_json, runtime_info

ROOT = Path(__file__).resolve().parent
PROFILE = 'simulink_observable_v1'
VERSION = 'simulink_control_v3_observable_20261103_v1'
IMU = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
CMD = [f'cmd_motor{i}' for i in range(6)]
RPM = [f'rpm_feedback{i}' for i in range(6)]
FEATURES = {'imu': IMU, 'imu_cmd': IMU + CMD, 'imu_rpm': IMU + RPM,
            'imu_cmd_rpm': IMU + CMD + RPM}


def scenarios():
    rows = [dict(label=0, name='normal', motor_python=-1, motor_simulink=0,
                 severity='normal', factor=1., label_name='Normal')]
    for severity, factor in [('full', 0.), ('severe', .2), ('partial', .8)]:
        for i in range(6):
            rows.append(dict(label=len(rows), name=f'motor{i}_{severity}',
                             motor_python=i, motor_simulink=i + 1, severity=severity,
                             factor=factor, label_name=f'M{i+1}-{severity.title()}'))
    return rows


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def prepare_plan(folder, seeds=(20261103, 20261104, 20261105, 20261106), task=2):
    folder = Path(folder).resolve()
    if len(set(seeds)) < 4:
        raise ValueError('At least four distinct paired condition seeds are required.')
    files = [ROOT / 'simulink/hex_evtol.slx', ROOT / 'simulink/configs/research.json']
    for directory in ('', 'blocks', 'functions'):
        files += list((ROOT / 'simulink' / directory).glob('*.m'))
    files += [ROOT / 'simulink/validation/finalize_hex_delivery.m']
    sources = [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256_file(p)) for p in sorted(set(files))]
    runs = []
    for group, seed in enumerate(seeds):
        for scenario in scenarios():
            runs.append(dict(**scenario, group_id=group, run_id=group, seed=int(seed),
                             run_key=f'g{group:03d}_label{scenario["label"]:02d}'))
    plan = dict(profile=PROFILE, dataset_version=VERSION, task_family=task,
                sources=sources, runs=runs, scenarios=scenarios(),
                fault_definition='omega_dot=(sqrt(eta)*command-omega)/tau; thrust=kT*omega^2; eta is steady-state remaining thrust at fixed command',
                purpose='simulation-domain prototype, not real-flight validation',
                group_policy='entire paired 19-class nuisance condition held out',
                rotor_mapping_to_uavfd='unestablished')
    plan['plan_signature'] = stable_hash(plan)
    path = folder / 'collection_plan.json'
    if path.exists() and read(path) != plan:
        raise RuntimeError('Collection plan changed; select a new directory, never overwrite runs.')
    write_json(path, plan)
    return path


def checked_plan(folder):
    plan = read(Path(folder) / 'collection_plan.json')
    unsigned = {k: v for k, v in plan.items() if k != 'plan_signature'}
    if stable_hash(unsigned) != plan['plan_signature'] or plan['profile'] != PROFILE:
        raise ValueError('Invalid collection plan/profile signature')
    if plan['scenarios'] != scenarios():
        raise ValueError('Incompatible fault definitions')
    return plan


def validate_run(folder, run, plan):
    folder = Path(folder)
    meta = read(folder / 'run_metadata.json')
    if (meta['plan_signature'] != plan['plan_signature'] or meta['run'] != run or meta['sources'] != plan['sources']
            or meta['dataset_version'] != plan['dataset_version']):
        raise ValueError(f'{folder}: run provenance does not match signed plan')
    if meta['status'] != 'completed' or meta['termination'] != 'completed':
        raise ValueError(f'{folder}: retained failed/incomplete run: {meta["status"]}')
    for filename, key in [('signals.csv', 'csv_sha256'), ('result.mat', 'mat_sha256')]:
        if sha256_file(folder / filename) != meta[key]:
            raise ValueError(f'{folder}: source hash mismatch: {filename}')
    p = meta['parameters']
    expected = {'seed': run['seed'], 'fault_motor': run['motor_simulink'],
                'fault_factor': run['factor'], 'task_family': plan['task_family'],
                'observed_feedback': True, 'noise_enabled': True, 'wind_enabled': True,
                'feedback_delay': .03, 'control_tuning_version': 'control_v3_20261023'}
    if any(p.get(k) != v for k, v in expected.items()) or not meta['fault_tolerant_control']:
        raise ValueError(f'{folder}: actual parameters disagree with injection/conditions')
    if meta['fault_start_s'] != p['fault_start'] or meta['fault_end_s'] != 'infinity':
        raise ValueError('Untrusted injection timing')
    if meta['frame'] != 'ENU_ground_FLU_body_xyzw_body_to_ground':
        raise ValueError('Unknown coordinate convention')
    df = pd.read_csv(folder / 'signals.csv')
    required = ['time_s'] + [f'plant_effectiveness_{i}' for i in range(1, 7)]
    if set(required) - set(df):
        raise ValueError('Missing time or injection audit columns')
    t = df.time_s.to_numpy()
    if not np.all(np.isfinite(t)) or not np.allclose(np.diff(t), meta['sample_dt_s'], rtol=0, atol=1e-9):
        raise ValueError('Nonuniform/untrusted sample timing')
    if abs(t[0]) > 1e-9 or abs(t[-1] - p['duration']) > 1e-9:
        raise ValueError('Incomplete time series')
    eta = np.ones((len(t), 6))
    if run['motor_simulink']:
        eta[t >= meta['fault_start_s'] - 1e-10, run['motor_python']] = run['factor']
    actual = df[[f'plant_effectiveness_{i}' for i in range(1, 7)]].to_numpy()
    if not np.allclose(actual, eta, atol=1e-12, rtol=0):
        raise ValueError('Injected effectiveness differs from label')
    saved = loadmat(folder/'result.mat',simplify_cells=True)['result']
    actual_p=saved['parameters']
    if any(actual_p[k] != v for k,v in expected.items()):
        raise ValueError('MAT parameter snapshot differs from sidecar')
    wind_columns=[f'wind_enu_m_s_{i}' for i in range(1,4)]
    if not set(wind_columns)<=set(df) or not np.allclose(df[wind_columns],actual_p['exogenous'][:len(t),:3],atol=1e-10,rtol=0):
        raise ValueError('Logged wind differs from the requested random condition')
    delayed=np.maximum(np.arange(len(t))-int(np.ceil(p['feedback_delay']/p['sensor_dt']-1e-10)),0)
    for i in range(1,7):
        observed=f'observation_state_{i+13}'; feedback=f'feedback_state_{i+13}'; truth=f'rotor_actual_rad_s_{i}'
        if not {observed,feedback,truth}<=set(df):
            raise ValueError('Missing measured RPM chain, cannot fall back to truth')
        expected_rpm=df[truth].to_numpy()+actual_p['exogenous'][:len(t),i+17]
        if not np.allclose(df[observed],expected_rpm,atol=1e-8,rtol=0) or not np.allclose(df[feedback],df[observed].to_numpy()[delayed],atol=1e-8,rtol=0):
            raise ValueError('Observed/delayed RPM does not match the actual sensor chain')
    return meta, df


def observable_features(df, meta, feature_set):
    if feature_set not in FEATURES:
        raise ValueError(f'Unsupported observable feature set: {feature_set}')
    raw_imu = [f'accelerometer_sensor_m_s2_{i}' for i in range(1, 4)] + [f'gyro_sensor_rad_s_{i}' for i in range(1, 4)]
    raw_cmd = [f'rotor_command_rad_s_{i}' for i in range(1, 7)]
    raw_rpm = [f'feedback_state_{i}' for i in range(14, 20)]
    columns = raw_imu + (raw_cmd if 'cmd' in feature_set else []) + (raw_rpm if 'rpm' in feature_set else [])
    if set(columns) - set(df):
        raise ValueError(f'Missing measured input columns: {sorted(set(columns)-set(df))}; truth fallback forbidden')
    delay = int(np.ceil(meta['feedback_delay_s'] / meta['parameters']['sensor_dt'] - 1e-10))
    indices = np.maximum(np.arange(len(df)) - delay, 0)
    blocks = [df[raw_imu].to_numpy()[indices]]
    if 'cmd' in feature_set:
        blocks.append(df[raw_cmd].to_numpy())
    if 'rpm' in feature_set:
        blocks.append(df[raw_rpm].to_numpy())
    x = np.concatenate(blocks, axis=1).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite observable input')
    return x


def finalize_dataset(folder):
    folder = Path(folder)
    plan = checked_plan(folder)
    records, failures, conditions = [], [], {}
    for run in plan['runs']:
        path = folder / run['run_key']
        try:
            meta, _ = validate_run(path, run, plan)
            # Full parameters except injected fault must agree within the pair.
            common = {k: v for k, v in meta['parameters'].items() if k not in ('fault_motor', 'fault_factor')}
            signature = stable_hash(common)
            group = run['group_id']
            if group in conditions and conditions[group] != signature:
                raise ValueError('Paired conditions differ beyond injected fault')
            conditions[group] = signature
            records.append(dict(**run, csv_sha256=meta['csv_sha256'], mat_sha256=meta['mat_sha256'],
                                metadata_sha256=sha256_file(path/'run_metadata.json'), condition_signature=signature,
                                flight_acceptance_passed=bool(meta['flight_metrics']['passed'])))
        except Exception as exc:
            failures.append(dict(run_key=run['run_key'], error=str(exc)))
    manifest = dict(profile=PROFILE, dataset_version=VERSION, plan_signature=plan['plan_signature'],
                    records=records, failures=failures, complete=not failures)
    write_json(folder / 'dataset_manifest.json', manifest)
    if failures:
        raise RuntimeError(f'{len(failures)} incomplete/untrusted runs; retained and excluded from formal paired evaluation')
    return manifest


def preprocess(folder, destination, feature_set='imu_cmd_rpm', window=100, step=50, buffer=1.):
    folder, destination = Path(folder).resolve(), Path(destination).resolve()
    plan = checked_plan(folder)
    manifest = finalize_dataset(folder)
    buckets, labels, groups, sources, times, records = [], [], [], [], [], []
    for run in plan['runs']:
        path = folder / run['run_key']
        meta, df = validate_run(path, run, plan)
        x = observable_features(df, meta, feature_set)
        t = df.time_s.to_numpy()
        selected = np.flatnonzero(t > meta['fault_start_s'] + buffer)
        # Retain full record separately for causal replay; stable windows never cross the fault.
        starts = [i for i in selected if i + window <= len(t) and (i-selected[0]) % step == 0]
        if not starts:
            raise ValueError('No complete stable windows')
        buckets.extend(x[i:i+window] for i in starts)
        labels.extend([run['label']] * len(starts)); groups.extend([run['group_id']] * len(starts))
        sources.extend([run['run_key']] * len(starts)); times.extend([t[i+window-1] for i in starts])
        records.append(dict(**run, **{k: meta[k] for k in ('csv_sha256','mat_sha256','sample_dt_s','feedback_delay_s')},
                            metadata_sha256=sha256_file(path/'run_metadata.json'), source_path=str(path),
                            selected_start_exclusive_s=meta['fault_start_s']+buffer, windows=len(starts)))
    y, g = np.asarray(labels, dtype=np.int64), np.asarray(groups, dtype=np.int64)
    for group in set(g):
        if set(y[g == group]) != set(range(19)):
            raise ValueError('Incomplete paired group')
    if len(set(g)) < 4:
        raise ValueError('At least four complete paired groups required')
    destination.mkdir(parents=True, exist_ok=True)
    metadata = dict(profile=PROFILE, dataset_version=VERSION, feature_set=feature_set,
                    window=window, step=step, fault_buffer_s=buffer, segment_mode='post_fault_stable_classification',
                    feature_names=FEATURES[feature_set], label_table=scenarios(),
                    imu='sensor-frame specific force m/s^2 and gyro rad/s, delayed causally with startup hold',
                    command='current controller command rad/s', rpm='delayed noisy feedback_state_14..19 rad/s, not rotor_actual',
                    ground_frame='ENU', body_frame='FLU', quaternion='xyzw body_to_ground (excluded from inputs)',
                    plan_signature=plan['plan_signature'], dataset_manifest_sha256=sha256_file(folder/'dataset_manifest.json'),
                    sources=plan['sources'], grouping='paired 19 classes per condition', physics_targets=None,
                    flight_acceptance_failures=[r['run_key'] for r in manifest['records'] if not r['flight_acceptance_passed']],
                    inclusion_policy='All finite full-duration runs with verified provenance/injection, including recorded control-acceptance failures; none removed for poor recovery.',
                    code_sha256=code_hashes(ROOT,['simulink_data.py','data_process.py','provenance.py']))
    # Never silently reuse preprocessing output for a different source/configuration.
    signature = stable_hash(metadata)
    existing = destination / 'preprocess_metadata.json'
    if existing.exists() and read(existing).get('signature') != signature:
        raise RuntimeError('Incompatible preprocessed directory; choose a new output path')
    metadata['signature'] = signature
    for name, values in [('X_raw',np.asarray(buckets,np.float32)),('y_all',y),('groups',g),
                         ('source_file',np.asarray(sources)),('window_end_s',np.asarray(times))]:
        np.save(destination/f'{name}.npy', values)
    write_json(destination/'feature_names.json', FEATURES[feature_set])
    write_json(destination/'label_names.json', {s['label']:s['label_name'] for s in scenarios()})
    write_json(destination/'source_manifest.json',records)
    write_json(existing,metadata)
    print('SIMULINK_PREPROCESSED', np.asarray(buckets).shape, 'groups', len(set(g)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['plan','finalize','preprocess'])
    parser.add_argument('--data',type=Path,default=ROOT/'output_simulink_gan'/VERSION/'raw')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--feature-set',choices=FEATURES,default='imu_cmd_rpm')
    args=parser.parse_args()
    if args.action=='plan': print(prepare_plan(args.data))
    elif args.action=='finalize': print('DATASET_COMPLETE',len(finalize_dataset(args.data)['records']))
    else: preprocess(args.data,args.output or args.data.parent/'processed'/args.feature_set,args.feature_set)
