"""Audit actual V3 sources, selected arrays, Adam states and frozen test identity."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from provenance import write_json, sha256_file, stable_hash, runtime_info
from simulink_data import ROOT, read, checked_plan, validate_run, observable_features, FEATURES
from simulink_v3_data import root, authorize_test, array_hash, EXPECTED, scope_masks


def optimizer_steps(state):
    return sorted({int(s['step'].item() if hasattr(s['step'], 'item') else s['step'])
                   for s in state['state'].values()})


def audit(config):
    destination = root(config) / 'audit'; destination.mkdir(parents=True, exist_ok=True)
    dev = read(root(config) / 'development_sources.json')
    test = read(root(config) / 'new_test_sources.json')
    frozen = read(root(config) / 'freeze.json'); authorize_test(config, test, frozen['signature'])
    complete = read(root(config) / 'development_complete.json')
    final = read(root(config) / 'final_test_results.json')
    checks = []; rows = []; conditions = {}; sources = dev['records']+test['records']
    def check(name, success, evidence=None):
        checks.append(dict(check=name, passed=bool(success), evidence=evidence))
        if not success: raise AssertionError(f'V3 audit failed: {name}: {evidence}')
    check('228 formal independent records, 12 paired groups', len(sources) == 228 and len({r['group_id'] for r in sources}) == 12)
    role_seeds = {role: {r['run']['seed'] for r in sources if r['role'] == role} for role in ('train', 'validation', 'test')}
    check('Roles and nuisance seeds disjoint', not (role_seeds['train'] & role_seeds['validation'] or
          role_seeds['train'] & role_seeds['test'] or role_seeds['validation'] & role_seeds['test']),
          {k: sorted(v) for k, v in role_seeds.items()})
    old_test = read(root(config) / 'old_test_archive.json')['records']
    check('Previously exposed test excluded', not ({r['source_id'] for r in sources} & {r['source_id'] for r in old_test}))
    for r in sources:
        path = Path(r['source_path']); plan = checked_plan(Path(r['plan_path']).parent)
        for item in plan['sources']:
            check_name = 'Frozen physical simulation source ' + item['path']
            if not any(c['check'] == check_name for c in checks):
                check(check_name, sha256_file(ROOT / item['path']) == item['sha256'])
        meta, df = validate_run(path, r['run'], plan)
        if len(df) != 2401 or not np.isfinite(df.select_dtypes(include=np.number).to_numpy()).all():
            raise AssertionError('Incomplete/nonfinite formal source')
        common = {k: v for k, v in meta['parameters'].items() if k not in ('fault_motor', 'fault_factor')}
        condition = stable_hash(common)
        key = (r['role'], r['group_id'])
        if key in conditions and conditions[key] != condition: raise AssertionError('Pair conditions differ beyond fault')
        conditions[key] = condition
        rows.append(dict(role=r['role'], group_id=r['group_id'], source_id=r['source_id'], seed=r['run']['seed'],
                         label=r['run']['label'], motor=r['run']['motor_simulink'], factor=r['run']['factor'],
                         rows=len(df), first_time_s=float(df.time_s.iloc[0]), last_time_s=float(df.time_s.iloc[-1]),
                         flight_acceptance_passed=r['flight_acceptance_passed'], csv_sha256=r['csv_sha256'],
                         mat_sha256=r['mat_sha256'], metadata_sha256=r['metadata_sha256'], condition_signature=condition,
                         source_path=str(path)))
    source_df = pd.DataFrame(rows)
    source_df.to_csv(destination / 'source_manifest.csv', index=False, encoding='utf-8-sig')
    check('Every condition has all 19 classes', all(set(g.label) == set(range(19)) for _, g in source_df.groupby(['role', 'group_id'])))
    # Counterfactual column mutation on an actual source, not only a test fixture.
    r = dev['records'][0]; meta, df = validate_run(r['source_path'], r['run'], checked_plan(Path(r['plan_path']).parent))
    changed = df.copy()
    for col in changed:
        if col.startswith(('rotor_command_rad_s_', 'rotor_actual_rad_s_', 'plant_effectiveness_', 'reference_', 'allocation_')):
            changed[col] = -123456.
    check('Actual 12D features invariant to command/truth/audit-column mutation',
          np.array_equal(observable_features(df, meta, 'imu_rpm'), observable_features(changed, meta, 'imu_rpm')))
    check('Exact measured-RPM allowlist', FEATURES['imu_rpm'] == ['ax', 'ay', 'az', 'gx', 'gy', 'gz']+[f'rpm_feedback{i}' for i in range(6)])
    pool_rows = []; pool_objects = {}
    for path in sorted((root(config) / 'pools').glob('*.pkl')):
        with path.open('rb') as f: p = pickle.load(f)
        payload = p['payload']; key = (payload['role'], payload['feature'], payload['prefix_s'], payload['equal_sampling_seed'] is not None)
        pool_objects[key] = p
        check('Pool hash '+path.name, array_hash(p['X']) == p['X_sha256'] and stable_hash(payload) == p['signature'])
        n_groups = len(set(p['groups'])); seconds = 6 if key[3] else payload['prefix_s']
        check('Window count '+path.name, len(p['y']) == n_groups*EXPECTED[seconds][2], len(p['y']))
        check('Dimension '+path.name, p['X'].shape[1:] == (100, len(FEATURES[payload['feature']])))
        for s in p['selection']:
            if s['count'] != EXPECTED[seconds][int(s['label'] != 0)]: raise AssertionError('Per-source count mismatch')
            if len(set(s['starts'])) != len(s['starts']): raise AssertionError('Repeated selected source window')
            for start, end in zip(s['starts'], s['ends']):
                if end-start != 99 or start % 50 or end >= payload['prefix_s']*100: raise AssertionError('Prefix boundary mismatch')
                if s['label'] and end < 300: raise AssertionError('Duplicate healthy prefix retained from fault record')
            pool_rows.append(dict(pool_signature=p['signature'], role=payload['role'], feature=payload['feature'],
                                  prefix_s=payload['prefix_s'], equal_count=key[3], source_id=s['source_id'], group=s['group'],
                                  label=s['label'], count=s['count'], first_start_s=min(s['starts'])/100,
                                  last_end_s=max(s['ends'])/100))
    pd.DataFrame(pool_rows).to_csv(destination / 'window_source_counts.csv', index=False, encoding='utf-8-sig')
    d = pool_objects[('train', 'imu_rpm', 6, False)]; e = pool_objects[('train', 'imu_rpm', 24, True)]
    check('D and E each 714 windows', len(d['y']) == len(e['y']) == 714)
    check('D/E counts match per source/class', [(s['source_id'], s['label'], s['count']) for s in d['selection']] ==
          [(s['source_id'], s['label'], s['count']) for s in e['selection']])
    check('E sampling includes late windows', bool(np.any(e['starts'] >= 1200)))
    for path in (root(config) / 'scalers').glob('*/scalers.pkl'):
        fit = read(path.parent / 'fit_sources.json')
        p = next(p for p in pool_objects.values() if p['signature'] == fit['pool_signature'])
        with path.open('rb') as f: cls, gan = pickle.load(f)
        x = p['X'].reshape(-1, p['X'].shape[-1])
        check('Scaler actual selected-only fit '+path.parent.name, fit['X_sha256'] == p['X_sha256'] and
              int(cls.n_samples_seen_) == len(x) and np.allclose(cls.mean_, np.mean(x, axis=0, dtype=np.float64), atol=1e-8) and
              np.allclose(gan.data_min_, np.min(x, axis=0)) and np.allclose(gan.data_max_, np.max(x, axis=0)))
    budget_rows = []
    for spec in complete['jobs']:
        state = torch.load(Path(spec['directory']) / 'checkpoint_1000.pt', map_location='cpu', weights_only=False)
        check('Classifier Adam updates '+spec['signature'], state['step'] == 1000 and optimizer_steps(state['optimizer']) == [1000])
        check('Classifier actual input '+spec['signature'], state['model']['lstm.weight_ih_l0'].shape[1] == len(FEATURES[spec['feature']]))
        original = np.asarray(spec['class_original']); joined = np.asarray(spec['class_joined'])
        check('Floor augmentation '+spec['signature'], np.array_equal(joined-original, np.floor(original*spec['ratio']).astype(int)))
        check('Three classifier checkpoints '+spec['signature'], [h['step'] for h in state['history']] == [250, 500, 1000])
        check('Classifier completed before freeze '+spec['signature'], spec['completed_utc'] <= frozen['created_utc'])
        budget_rows.append(dict(tag=spec['tag'], seed=spec['seed'], method=spec['method'], prefix_s=spec['prefix_s'],
                                feature=spec['feature'], original_windows=sum(original), joined_windows=sum(joined),
                                augmented_windows=int(sum(joined-original)), classifier_updates=1000, generator_updates=0,
                                critic_updates=0, average_draws=spec['expected_draws_per_training_window'], signature=spec['signature']))
    for spec in complete['generators']:
        state = torch.load(Path(spec['directory']) / 'checkpoint_3000.pt', map_location='cpu', weights_only=False)
        check('GAN actual Adam updates '+spec['signature'], optimizer_steps(state['optimizer_g']) == [3000] and
              optimizer_steps(state['optimizer_d']) == [15000] and state['generator_updates'] == 3000 and state['critic_updates'] == 15000)
        check('GAN plain 12D '+spec['signature'], spec['feature'] == 'imu_rpm' and not state['command_indices'] and state['generator_parameter_max_delta'] > 0)
        p = next(p for p in pool_objects.values() if p['signature'] == spec['train_pool'])
        check('GAN restricted prefix '+spec['signature'], bool(np.all(p['ends'] < spec['prefix_s']*100)))
        budget_rows.append(dict(tag='plain_GAN', seed=spec['seed'], method='generator', prefix_s=spec['prefix_s'], feature='imu_rpm',
                                original_windows=len(p['y']), joined_windows=0, augmented_windows=0, classifier_updates=0,
                                generator_updates=3000, critic_updates=15000, average_draws=None, signature=spec['signature']))
    pd.DataFrame(budget_rows).to_csv(destination / 'actual_training_budgets.csv', index=False, encoding='utf-8-sig')
    check('51 classifier and 6 GAN jobs', len(complete['jobs']) == 51 and len(complete['generators']) == 6)
    identities = []
    for result in final['results']:
        spec = next(j for j in complete['jobs'] if j['signature'] == result['signature'])
        with np.load(Path(spec['directory']) / 'formal_test_predictions.npz') as p:
            identities.append(stable_hash(dict(source=p['source_ids'].tolist(), starts=p['starts'].tolist(), ends=p['ends'].tolist(), y=p['y'].tolist())))
            check('Actual test argmax '+result['signature'], abs(np.mean(p['logits'].argmax(1) == p['y'])-result['test']['prefix_24']['accuracy']) < 1e-12)
        check('Common test prefix counts '+result['signature'], [result['test'][f'prefix_{s}']['window_count'] for s in (6, 12, 24)] == [357, 1041, 2409])
        check('Unique healthy alarm exposure '+result['signature'], abs(result['alarm']['normal_observation_s']-69.) < 1e-9 and
              result['alarm']['fault_records'] == 54)
    check('Every model evaluated on identical source/window/label identities', len(set(identities)) == 1)
    access = read(root(config) / 'test_access_receipt.json')
    check('Test read authorized after freeze', access['created_utc'] >= frozen['created_utc'] and access['freeze_signature'] == frozen['signature'])
    check('Test result uses current freeze', final['freeze_signature'] == frozen['signature'])
    report = dict(**runtime_info(), passed=True, checks=checks, count=len(checks),
                  formal_records=len(sources), original_records_reused=171, new_records_collected=57,
                  old_test_records_archived=57, flight_acceptance_failures_retained=int((~source_df.flight_acceptance_passed).sum()),
                  total_classifier_updates=51000, total_generator_updates=18000, total_critic_updates=90000,
                  freeze_signature=frozen['signature'], distinct_tested_models=final['unique_tested_models'])
    write_json(destination / 'audit.json', report)
    print('V3_AUDIT_PASSED', len(checks), flush=True)
    return report
