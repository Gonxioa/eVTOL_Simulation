"""V5 full-flight paired records; v4 sources are never modified."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from provenance import stable_hash, sha256_file, write_json
from .commands import ParameterizedTask, task_library
from .configuration import physical, rotorpy_fields
from .dynamics import HexPlant
from .simulation import simulate, export, fingerprint, FaultSchedule
from .validation import metric
from diagnostics.features import IMU, QUATERNION, SCHEMA, packets_from_frame
from diagnostics.continuous import causal_windows, offline_labels

VERSION = 'hex_v5_continuous_tasks'


def initial_state(cfg, group):
    origin = np.asarray(group['task']['origin_m']) + group['initial_position_offset_m']
    q = Rotation.from_rotvec(group['initial_attitude_rotvec_rad']).as_quat()
    return HexPlant(cfg).initial(position=origin, q=q)


def plan_for(cfg, settings):
    from scenario_config import serializable_scenarios
    payload = {'dataset_version': VERSION, 'config': cfg, 'settings': settings,
               'groups': task_library(settings), 'scenarios': serializable_scenarios(),
               'source_sha256': fingerprint(cfg)['code_hashes'],
               'formal_protocol': False, 'purpose': 'batches_1_2_development_not_final_test'}
    payload['plan_signature'] = stable_hash(payload)
    return payload


def validate_tasks(cfg, settings, out):
    """Run declared configurations before any fault generation; no retuning."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True); plan = plan_for(cfg, settings)
    acceptance = settings['normal_acceptance']
    write_json(out/'task_predeclared_acceptance.json', {'criteria': acceptance, 'settings': settings,
                                                      'plan_signature': plan['plan_signature'], 'written_before_runs': True})
    results = []; runs_for_plot = []
    for group in plan['groups']:
        for condition in ('ideal', 'wind', 'noise_wind', 'delay_noise_wind'):
            local = deepcopy(cfg)
            if condition == 'delay_noise_wind':
                local['sensor']['feedback_delay_s'] = settings['normal_validation_delay_s']
            run = simulate(local, settings['duration_s'], initial=initial_state(cfg, group),
                           task_provider=ParameterizedTask(group['task']), seed_streams=group['seed_streams'],
                           feedback='ideal_feedback' if condition in ('ideal', 'wind') else 'observed_feedback',
                           noise=condition in ('noise_wind', 'delay_noise_wind'), wind=condition != 'ideal')
            result = metric(run)
            tests = {'completed': result['termination'] == 'completed',
                     'rmse': result['position_rmse_m'] <= acceptance['position_rmse_m'],
                     'final_error': result['final_position_error_m'] <= acceptance['final_position_error_m'],
                     'tilt': result['maximum_tilt_deg'] <= acceptance['maximum_tilt_deg'],
                     'rate': result['maximum_rate_rad_s'] <= acceptance['maximum_body_rate_rad_s'],
                     'saturation': result['nominal_bound_fraction'] <= acceptance['maximum_nominal_bound_fraction']}
            results.append({'group_id': group['group_id'], 'family': group['task']['family'],
                            'condition': condition, 'checks': tests, 'passed': all(tests.values()), **result})
            if group['repeat'] == 0 and condition == 'delay_noise_wind':
                runs_for_plot.append(run)
            print('task normal', group['group_id'], group['task']['family'], condition,
                  'PASS' if all(tests.values()) else 'FAIL', result['position_rmse_m'], flush=True)
    report = {'all_normal_passed': all(r['passed'] for r in results), 'cases': results,
              'plan_signature': plan['plan_signature'], 'config_hash': stable_hash(cfg),
              'acceptance': acceptance, 'real_aircraft_validation': False}
    write_json(out/'task_normal_validation.json', report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, run in zip(axes.flat, runs_for_plot):
        t = [r['time'] for r in run['rows']]
        error = [np.linalg.norm(r['state'][:3]-r['reference']['x']) for r in run['rows']]
        axis.plot(t, error); axis.set(title=run['task'], xlabel='time [s]', ylabel='position error [m]')
    fig.tight_layout(); fig.savefig(out/'task_normal_validation.png', dpi=150); plt.close(fig)
    return report


def export_v5(run, path, scenario, group, signature):
    frame, side = export(run, path, scenario, group['group_id'])
    for axis, name in enumerate(IMU[:3]):
        frame[name] = [r['measurement']['specific_force_body'][axis] for r in run['rows']]
    for axis, name in enumerate(IMU[3:]):
        frame[name] = [r['measurement']['gyro_body'][axis] for r in run['rows']]
    for axis, name in enumerate(QUATERNION):
        frame[name] = [r['measurement']['q'][axis] for r in run['rows']]
    for prefix, key in (('reference_position', 'x'), ('reference_velocity', 'x_dot'), ('reference_acceleration', 'x_ddot')):
        for axis, letter in enumerate('xyz'):
            frame[f'{prefix}_{letter}'] = [r['reference'][key][axis] for r in run['rows']]
    frame['reference_yaw'] = [r['reference']['yaw'] for r in run['rows']]
    frame['reference_yaw_rate'] = [r['reference']['yaw_dot'] for r in run['rows']]
    frame['control_reference_time'] = [r['control_reference_time'] for r in run['rows']]
    frame['data_valid'] = 1; frame['packet_schema'] = SCHEMA
    frame['record_id'] = f"g{group['group_id']:03d}_c{scenario.label:02d}"
    frame['task_family'] = group['task']['family']
    temporary = Path(path).with_suffix('.csv.tmp'); frame.to_csv(temporary, index=False); temporary.replace(path)
    side.update(schema_version=5, dataset_version=VERSION, csv_sha256=sha256_file(path),
                record_id=frame['record_id'].iloc[0], group=group, task=run['task_config'],
                plan_signature=signature, packet_schema=SCHEMA, feature_state_identity='observed_FLU_body_not_truth_or_EKF',
                fault_start_s=run['schedule'].start, fault_end_s=run['schedule'].end,
                initial_state=run['initial_state'], seed_streams=run['seed_streams'])
    write_json(Path(path).with_suffix('.json'), side)
    return frame, side


def generate(cfg, settings, out):
    from scenario_config import SCENARIOS, effectiveness_table
    out = Path(out); plan = plan_for(cfg, settings)
    gate = json.loads((out/'task_normal_validation.json').read_text())
    if not gate['all_normal_passed'] or gate['plan_signature'] != plan['plan_signature']:
        raise RuntimeError('Matching new-task normal gate is missing or failed; generate into a fresh verified output')
    folder = out/'dataset_v5'; folder.mkdir(exist_ok=True)
    plan_path = folder/'generation_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Generation signature changed; old dataset must remain read-only')
    write_json(plan_path, plan); summaries = []; hashes = {}; side_hashes = {}
    for group in plan['groups']:
        for scenario in SCENARIOS:
            name = f"g{group['group_id']:03d}_{group['task']['family']}_c{scenario.label:02d}.csv"
            path = folder/name
            if path.exists():
                side = json.loads(path.with_suffix('.json').read_text())
                if side.get('plan_signature') != plan['plan_signature'] or side.get('schema_version') != 5 or side['csv_sha256'] != sha256_file(path):
                    raise RuntimeError('Partial or incompatible saved source; do not silently overwrite')
                frame = pd.read_csv(path)
            else:
                schedule = FaultSchedule(scenario.fault_motor, scenario.simulated_factor, group['fault_time_s']) if scenario.label else FaultSchedule()
                run = simulate(cfg, settings['duration_s'], schedule=schedule, seed=group['seed_streams']['sensor'],
                               seed_streams=group['seed_streams'], task_provider=ParameterizedTask(group['task']),
                               initial=initial_state(cfg, group), feedback='observed_feedback', noise=True, wind=True)
                run.update(schedule=schedule, matching_fault_time=group['fault_time_s'])
                frame, side = export_v5(run, path, scenario, group, plan['plan_signature'])
                side['flight_metrics'] = metric(run); write_json(path.with_suffix('.json'), side)
            X, times = causal_windows(packets_from_frame(frame), settings['diagnostics'])
            labels = offline_labels(times, scenario.label, side['fault_start_s'], side['fault_end_s'])
            post_samples = int(np.sum(frame['time'].to_numpy() >= group['fault_time_s']))
            summaries.append({'file': name, 'record_id': side['record_id'], 'label': scenario.label,
                              'group_id': group['group_id'], 'repeat': group['repeat'], 'family': group['task']['family'],
                              'termination': side['termination'], 'duration_s': side['actual_duration_s'],
                              'fault_time_s': side['fault_start_s'], 'post_fault_duration_s': max(0., side['actual_duration_s']-group['fault_time_s']) if scenario.label else None,
                              'post_fault_samples': post_samples if scenario.label else None,
                              'causal_windows': len(X), 'healthy_windows': int(np.sum(labels == 0)),
                              'transition_windows': int(np.sum(labels == -1)), 'pure_fault_windows': int(np.sum(labels > 0)),
                              'fault_evaluable': bool(np.any(labels > 0)) if scenario.label else None,
                              'flight_metrics': side['flight_metrics']})
            hashes[name] = sha256_file(path); side_hashes[path.with_suffix('.json').name] = sha256_file(path.with_suffix('.json'))
            print('v5 record', side['record_id'], group['task']['family'], side['termination'],
                  side['actual_duration_s'], 'pure fault windows', int(np.sum(labels > 0)), flush=True)
    fields = rotorpy_fields(cfg)
    manifest = {'dataset_version': VERSION, 'model_version': cfg['model_version'], 'plan_signature': plan['plan_signature'],
                'config_hash': stable_hash(cfg), 'formal_protocol': False, 'purpose': plan['purpose'],
                'csv_sha256': hashes, 'sidecar_sha256': side_hashes, 'runs': summaries,
                'packet_schema': SCHEMA, 'diagnostic_settings': settings['diagnostics'],
                'label_effectiveness': effectiveness_table(), 'physics_parameters': {'mass': fields['mass'], 'k_eta': fields['k_eta'], 'num_rotors': 6, 'gravity': cfg['airframe']['gravity_m_s2'], 'parameter_status': cfg['parameter_status']}}
    write_json(folder/'dataset_generation_manifest.json', manifest)
    validate(folder)
    return manifest


def validate(folder):
    folder = Path(folder); manifest = json.loads((folder/'dataset_generation_manifest.json').read_text())
    if manifest['dataset_version'] != VERSION:
        raise RuntimeError('Continuous validator requires v5; v4 remains a separate contract')
    plan = json.loads((folder/'generation_plan.json').read_text()); unsigned = dict(plan); unsigned.pop('plan_signature')
    if stable_hash(unsigned) != plan['plan_signature'] or plan['plan_signature'] != manifest['plan_signature']:
        raise RuntimeError('Invalid v5 generation signature')
    if set(manifest['csv_sha256']) != {p.name for p in folder.glob('*.csv')}:
        raise RuntimeError('V5 inventory mismatch')
    reference_frames = {}; prefix_errors = []; checked = []
    for record in manifest['runs']:
        path = folder/record['file']; side_path = path.with_suffix('.json')
        if sha256_file(path) != manifest['csv_sha256'][path.name] or sha256_file(side_path) != manifest['sidecar_sha256'][side_path.name]:
            raise RuntimeError('V5 source hash mismatch')
        side = json.loads(side_path.read_text()); frame = pd.read_csv(path)
        if side['schema_version'] != 5 or side['plan_signature'] != plan['plan_signature'] or side['provenance']['config_hash'] != manifest['config_hash']:
            raise RuntimeError('Mixed v5 source identity')
        numeric = frame.select_dtypes(include=np.number).drop(columns=['fault_time'], errors='ignore')
        if len(frame) != side['rows'] or not np.isfinite(numeric.to_numpy()).all():
            raise RuntimeError('Invalid exported samples')
        t = frame['time'].to_numpy(); step = 1/side['config']['timing']['export_hz']
        if not np.allclose(np.diff(t), step, atol=1e-9) or np.any(np.diff(t) <= 0):
            raise RuntimeError('V5 time grid mismatch')
        if side['termination'] == 'completed' and abs(t[-1]-side['requested_duration_s']) > 1e-9:
            raise RuntimeError('False v5 completion')
        p = physical(side['config']); speed = frame[[f'truth_rotor{i}' for i in range(6)]].to_numpy()
        command = frame[[f'cmd_motor{i}' for i in range(6)]].to_numpy()
        eta = frame[[f'effectiveness{i}' for i in range(6)]].to_numpy()
        if np.any(speed < 0) or np.any(speed > p['omega_max']+1e-8) or np.any(command < 0) or np.any(command > p['omega_max']+1e-8):
            raise RuntimeError('V5 actuator range violation')
        if not np.allclose((p['kT']*speed**2).sum(axis=1), frame['dynamic_propulsive_thrust'], atol=1e-8):
            raise RuntimeError('Dynamic thrust must use actual rotor state once')
        if not np.allclose((eta*p['kT']*command**2).sum(axis=1), frame['predicted_plant_thrust'], atol=1e-8):
            raise RuntimeError('Steady prediction identity changed')
        provider = ParameterizedTask(side['task'])
        for row in frame.to_dict('records'):
            reference = provider.at(row['time'])
            for prefix, key in (('reference_position', 'x'), ('reference_velocity', 'x_dot'), ('reference_acceleration', 'x_ddot')):
                np.testing.assert_allclose([row[f'{prefix}_{a}'] for a in 'xyz'], reference[key], atol=1e-12)
            if abs(row['reference_yaw']-reference['yaw']) > 1e-12 or abs(row['control_reference_time']-row['time']) > 1e-12:
                raise RuntimeError('Controller/export task reference mismatch')
        packets_from_frame(frame)
        group_id = side['group_id']
        if record['label'] == 0:
            reference_frames[group_id] = frame
        else:
            normal = reference_frames[group_id]; stop = side['group']['fault_time_s']
            columns = IMU+QUATERNION+[f'cmd_motor{i}' for i in range(6)]
            left = normal.loc[normal.time < stop, columns].to_numpy()
            right = frame.loc[frame.time < stop, columns].to_numpy()
            if not np.array_equal(left, right):
                prefix_errors.append(record['record_id'])
        checked.append(record['record_id'])
    group_ids = {r['group_id'] for r in manifest['runs']}
    for group_id in group_ids:
        if sorted(r['label'] for r in manifest['runs'] if r['group_id'] == group_id) != list(range(19)):
            raise RuntimeError('Incomplete paired 19-class group')
    report = {'all_integrity_passed': not prefix_errors, 'records_checked': len(checked), 'paired_groups': len(group_ids),
              'prefix_mismatch_records': prefix_errors, 'early_stops': sum(r['termination'] != 'completed' for r in manifest['runs']),
              'unevaluable_fault_records': [r['record_id'] for r in manifest['runs'] if r['fault_evaluable'] is False],
              'formal_training_approved': False}
    write_json(folder/'integrity_report.json', report)
    if prefix_errors:
        raise RuntimeError('Paired healthy observation prefix mismatch; see integrity report')
    print('V5 integrity:', report, flush=True)
    return report
