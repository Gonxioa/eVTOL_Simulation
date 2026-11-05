"""V3 orchestration: existing LSTM and class-conditional WGAN-GP, new protocol.

No physical-model, loss, network or alarm-state-machine changes live here.
Main window metrics always use the 19-way argmax; alarm-gated metrics are separate.
"""
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import classify
import WGAN
from experiment_protocol import stratified_generation_quality
from provenance import stable_hash, write_json, sha256_file, code_hashes, runtime_info
from replay_diagnostics import select_alarm_policy, score_alarm_records
from simulink_data import ROOT, FEATURES, read, observable_features
from simulink_v3_data import (root, immutable_json, array_hash, pool, scope_masks,
                              load_record, authorize_test)

TRAIN_CODE = ['classify.py', 'WGAN.py', 'experiment_protocol.py', 'replay_diagnostics.py',
              'simulink_data.py', 'simulink_v3_data.py', 'simulink_v3_experiment.py']
_FULL_SIGNALS = {}


def configure(config, feature):
    torch.set_num_threads(4)
    classify.N_CLASSES = 19; classify.N_FEATURES = len(FEATURES[feature]); classify.SEQ_LEN = config['window']
    classify.HIDDEN_DIM = config['classifier_hidden_dim']; classify.N_LAYERS = config['classifier_layers']
    classify.ATTENTION_POOLING = True; classify.FEATURE_NAMES = FEATURES[feature]
    classify.BATCH_SIZE = config['batch_size']; classify.LR = config['classifier_lr']
    WGAN.HIDDEN_DIM = config['generator_hidden_dim']; WGAN.N_LAYERS = config['generator_layers']
    WGAN.NOISE_DIM = config['generator_noise_dim']; WGAN.EMBED_DIM = config['generator_embedding_dim']
    WGAN.N_CRITIC = config['n_critic']; WGAN.LAMBDA_GP = config['gradient_penalty']
    WGAN.LR_G = config['generator_lr']; WGAN.LR_D = config['critic_lr']; WGAN.TEMPORAL_NOISE = True
    WGAN.configure_dimensions(classify.N_FEATURES, 19, config['window'])


def transform(values, scaler):
    return scaler.transform(values.reshape(-1, values.shape[-1])).reshape(values.shape).astype(np.float32)


def metric(y, predicted):
    y = np.asarray(y); predicted = np.asarray(predicted)
    if not len(y): raise ValueError('Empty evaluation scope')
    cm = confusion_matrix(y, predicted, labels=np.arange(19))
    # Fault-only stages do not contain the healthy class: fixed-19 macro F1 is
    # retained as main convention, plus present-class macro F1 for interpretation.
    present = np.unique(y)
    return dict(accuracy=float(accuracy_score(y, predicted)),
                macro_f1=float(f1_score(y, predicted, labels=np.arange(19), average='macro', zero_division=0)),
                present_class_macro_f1=float(f1_score(y, predicted, labels=present, average='macro', zero_division=0)),
                class_recall=recall_score(y, predicted, labels=np.arange(19), average=None, zero_division=0).tolist(),
                confusion_matrix=cm.tolist(), class_support=cm.sum(1).tolist(), window_count=len(y),
                class_labels=list(range(19)))


def scoped_metrics(values, output, threshold=None):
    labels = output['logits'].argmax(1) if threshold is None else np.where(
        output['fault_probability'] >= threshold, output['logits'][:, 1:].argmax(1)+1, 0)
    return {name: metric(values['y'][mask], labels[mask]) for name, mask in scope_masks(values).items()}


def fit_scalers(config, train):
    signature = stable_hash(dict(pool=train['signature'], X=train['X_sha256'], feature=train['payload']['feature']))
    directory = root(config) / 'scalers' / signature
    directory.mkdir(parents=True, exist_ok=True); path = directory / 'scalers.pkl'
    if not path.exists():
        x = train['X'].reshape(-1, train['X'].shape[-1])
        cls = StandardScaler().fit(x); gan = MinMaxScaler(feature_range=(-1, 1)).fit(x)
        with path.open('wb') as f: pickle.dump((cls, gan), f)
        immutable_json(directory / 'fit_sources.json', dict(signature=signature, pool_signature=train['signature'],
                       X_sha256=train['X_sha256'], sample_count=len(x), window_count=len(train['y']),
                       feature_names=FEATURES[train['payload']['feature']], selection=train['selection'],
                       groups=sorted(np.unique(train['groups']).tolist()), fitted_before_augmentation=True))
    with path.open('rb') as f: cls, gan = pickle.load(f)
    audit = read(directory / 'fit_sources.json')
    if audit['X_sha256'] != train['X_sha256'] or audit['pool_signature'] != train['signature']:
        raise ValueError('Scaler source mismatch')
    return cls, gan, str(directory)


def load_classifier(config, spec):
    configure(config, spec['feature'])
    saved = torch.load(Path(spec['directory']) / 'checkpoint_1000.pt', map_location=classify.DEVICE, weights_only=False)
    if saved['signature'] != spec['signature'] or saved['step'] != config['classifier_max_steps']:
        raise ValueError('Classifier identity/update mismatch')
    model = classify.FaultClassifier(dual_head=False).to(classify.DEVICE)
    model.load_state_dict(saved['model']); model.eval()
    return model


def causal_window_predictions(model, scaler, values, window=100, stride=10):
    """No label, source metadata or fault-onset argument is accepted by inference."""
    ends = np.arange(window-1, len(values), stride, dtype=np.int64)
    windows = np.asarray([values[e-window+1:e+1] for e in ends], np.float32)
    output = classify.predict_classifier(model, transform(windows, scaler))
    return ends, output


def full_signals(config, registry, role, feature, authorization=None):
    if role == 'test': authorize_test(config, registry, authorization)
    key = (registry['signature'], role, feature)
    if key not in _FULL_SIGNALS:
        rows = []
        for record in registry['records']:
            if record['role'] != role: continue
            meta, df = load_record(config, registry, record, authorization)
            rows.append((record, meta, df.time_s.to_numpy(), observable_features(df, meta, feature)))
        _FULL_SIGNALS[key] = rows
    return _FULL_SIGNALS[key]


def prediction_records(config, registry, role, spec, model=None, authorization=None):
    if role == 'test': authorize_test(config, registry, authorization)
    identity = dict(registry=registry['signature'], role=role, model_signature=spec['signature'],
                    checkpoint_sha256=sha256_file(Path(spec['directory']) / 'checkpoint_1000.pt'),
                    scaler_sha256=sha256_file(Path(spec['scaler_directory']) / 'scalers.pkl'),
                    feature=spec['feature'], window=config['window'], stride=config['replay_stride'],
                    code_sha256=code_hashes(ROOT, TRAIN_CODE))
    key = stable_hash(identity); path = root(config) / 'prediction_cache' / f'{role}_{key}.pkl'
    if path.exists():
        if read(path.with_suffix('.json')) != identity: raise ValueError('Replay cache identity mismatch')
        with path.open('rb') as f: return pickle.load(f)
    model = model or load_classifier(config, spec)
    with (Path(spec['scaler_directory']) / 'scalers.pkl').open('rb') as f: cls, _ = pickle.load(f)
    records = []
    for record, meta, times, values in full_signals(config, registry, role, spec['feature'], authorization):
        ends, output = causal_window_predictions(model, cls, values, config['window'], config['replay_stride'])
        records.append(dict(source_id=record['source_id'], group_id=record['group_id'], label=record['run']['label'],
                            fault_start_s=meta['fault_start_s'], time_s=times[ends],
                            fault_probability=output['fault_probability'], fault_class=output['logits'][:, 1:].argmax(1)+1))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f: pickle.dump(records, f)
    write_json(path.with_suffix('.json'), identity)
    return records


def alarm_metrics(records, threshold, hold):
    score = score_alarm_records(records, threshold, hold)
    delays = np.asarray([c['detection_delay_s'] for c in score['cases'] if c.get('detected')], float)
    for s in (.2, .5, 1., 3.):
        score[f'detected_within_{s:g}s_rate'] = float(np.sum(delays <= s+1e-9)/max(score['fault_records'], 1))
    score['delay_quantiles_s'] = None if not len(delays) else dict(zip(
        ('p25', 'median', 'p75', 'p95'), np.quantile(delays, (.25, .5, .75, .95)).tolist()))
    return score


def train_job(config, registry, train, validation, seed, tag, method='raw', ratio=0., raw=None, labels=None):
    feature = train['payload']['feature']; configure(config, feature)
    cls, _, scaler_directory = fit_scalers(config, train)
    raw = train['X'] if raw is None else raw; labels = train['y'] if labels is None else labels
    scaled = transform(raw, cls); valx = transform(validation['X'], cls)
    payload = dict(tag=tag, method=method, ratio=ratio, seed=seed, feature=feature,
                   feature_names=FEATURES[feature], prefix_s=train['payload']['prefix_s'],
                   equal_count=train['payload']['equal_sampling_seed'] is not None,
                   train_pool=train['signature'], validation_pool=validation['signature'],
                   X_sha256=array_hash(scaled), y_sha256=array_hash(labels),
                   steps=config['classifier_max_steps'], batch_size=config['batch_size'],
                   lr=config['classifier_lr'], hidden=config['classifier_hidden_dim'], layers=config['classifier_layers'],
                   dual_head=False, attention=True, config_signature=stable_hash(config), code=code_hashes(ROOT, TRAIN_CODE))
    signature = stable_hash(payload)
    directory = root(config) / 'classifiers' / f'{tag}_s{seed}_{signature[:16]}'
    directory.mkdir(parents=True, exist_ok=True)
    immutable_json(directory / 'job.json', dict(**payload, signature=signature, scaler_directory=scaler_directory))
    if (directory / 'result.json').exists():
        spec = read(directory / 'result.json')
        if spec['signature'] != signature or sha256_file(directory / 'checkpoint_1000.pt') != spec['checkpoint_sha256']:
            raise ValueError('Completed job changed')
        return spec
    def callback(model, step):
        out = scoped_metrics(validation, classify.predict_classifier(model, valx))
        compact = {k: {m: r[m] for m in ('accuracy', 'macro_f1', 'window_count')} for k, r in out.items()}
        compact['training_argmax'] = {k: v for k, v in metric(labels, classify.predict_classifier(model, scaled)['logits'].argmax(1)).items()
                                      if k in ('accuracy', 'macro_f1', 'window_count')}
        print('V3_VALIDATION', tag, seed, step, compact['prefix_24'], flush=True)
        return compact
    model = classify.train_classifier_steps(scaled, labels, directory, max_steps=config['classifier_max_steps'],
                checkpoints=config['classifier_checkpoints'], seed=seed, dual_head=False,
                validation_callback=callback, signature=signature, batch_size=config['batch_size'], lr=config['classifier_lr'])
    output = classify.predict_classifier(model, valx)
    spec = dict(**payload, signature=signature, directory=str(directory), scaler_directory=scaler_directory,
                optimizer_updates=model.training_receipt['optimizer_updates'],
                training_window_count=len(labels), original_window_count=len(train['y']),
                class_original=np.bincount(train['y'], minlength=19).tolist(),
                class_joined=np.bincount(labels, minlength=19).tolist(),
                checkpoint_sha256=sha256_file(directory / 'checkpoint_1000.pt'),
                scaler_sha256=sha256_file(Path(scaler_directory) / 'scalers.pkl'),
                completed_utc=runtime_info()['created_utc'],
                expected_draws_per_training_window=config['classifier_max_steps']*config['batch_size']/len(labels),
                validation=scoped_metrics(validation, output), main_metric='19-class argmax')
    records = prediction_records(config, registry, 'validation', spec, model)
    policy = select_alarm_policy(records, config['alarm_thresholds'], config['alarm_holds'])
    write_json(directory / 'alarm_selection.json', policy)
    spec['alarm_policy'] = dict(threshold=policy['threshold'], hold=policy['hold'])
    spec['validation_alarm'] = alarm_metrics(records, **spec['alarm_policy'])
    spec['validation_alarm_gated_window'] = scoped_metrics(validation, output, threshold=policy['threshold'])
    write_json(directory / 'result.json', spec)
    print('V3_JOB_COMPLETE', tag, seed, spec['validation']['prefix_24']['macro_f1'], spec['alarm_policy'], flush=True)
    return spec


def augmentation_counts(labels, ratio):
    return {i: int(np.floor(np.sum(labels == i)*ratio)) for i in range(19)}


def ros_additions(train, counts, seed):
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.choice(np.flatnonzero(train['y'] == c), n, replace=True)
                              for c, n in counts.items()]).astype(np.int64)
    return train['X'][indices], train['y'][indices], indices


def generator_job(config, train, seed):
    if train['payload']['feature'] != 'imu_rpm' or train['X'].shape[-1] != 12:
        raise ValueError('V3 GAN must be plain class-conditioned 12D imu_rpm')
    configure(config, 'imu_rpm'); _, gan, scaler_directory = fit_scalers(config, train)
    x = transform(train['X'], gan)
    payload = dict(feature='imu_rpm', feature_names=FEATURES['imu_rpm'], command_indices=None,
                   prefix_s=train['payload']['prefix_s'], train_pool=train['signature'], X_sha256=array_hash(x),
                   y_sha256=array_hash(train['y']), seed=seed, generator_updates=config['generator_max_steps'],
                   critic_updates=config['generator_max_steps']*config['n_critic'],
                   config_signature=stable_hash(config), code=code_hashes(ROOT, TRAIN_CODE))
    signature = stable_hash(payload)
    directory = root(config) / 'generators' / f'plain12_p{train["payload"]["prefix_s"]}_s{seed}_{signature[:16]}'
    directory.mkdir(parents=True, exist_ok=True)
    immutable_json(directory / 'job.json', dict(**payload, signature=signature, scaler_directory=scaler_directory))
    WGAN.train_generator_steps(x, train['y'], directory, generator_max_steps=config['generator_max_steps'],
          checkpoints=config['generator_checkpoints'], command_indices=None, seed=seed,
          batch_size=config['batch_size'], signature=signature)
    spec = dict(**payload, signature=signature, directory=str(directory), scaler_directory=scaler_directory,
                checkpoint_sha256=sha256_file(directory / 'checkpoint_3000.pt'))
    immutable_json(directory / 'result.json', spec)
    return spec


def synthetic_bank(config, train, validation, generator, raw_classifier):
    path = Path(generator['directory']) / 'synthetic_bank.npz'
    if path.exists():
        receipt = read(path.with_suffix('.json'))
        if receipt['generator_signature'] != generator['signature'] or receipt['sha256'] != sha256_file(path):
            raise ValueError('Synthetic bank identity changed')
        with np.load(path) as a: values, labels = a['X'], a['y']
    else:
        configure(config, 'imu_rpm')
        checkpoint = torch.load(Path(generator['directory']) / 'checkpoint_3000.pt', map_location=WGAN.DEVICE, weights_only=False)
        G = WGAN.Generator().to(WGAN.DEVICE); G.load_state_dict(checkpoint['generator']); G.eval()
        _, gan, _ = fit_scalers(config, train)
        counts = augmentation_counts(train['y'], 1.)
        values, labels, refs = WGAN.generate_from_pool(G, counts, transform(train['X'], gan), train['y'],
                                                       seed=generator['seed']+15000, batch_size=config['batch_size'])
        if values.shape[-1] != 12 or np.any(refs != -1): raise AssertionError('Unexpected command conditioning')
        values = gan.inverse_transform(values.reshape(-1, 12)).reshape(values.shape).astype(np.float32)
        np.savez_compressed(path, X=values, y=labels)
        immutable_json(path.with_suffix('.json'), dict(generator_signature=generator['signature'], sha256=sha256_file(path),
                       counts={str(c): n for c, n in counts.items()}, command_conditioned=False,
                       generated_shape=list(values.shape), seed=generator['seed']+15000))
    if (Path(generator['directory']) / 'quality.json').exists(): return values, labels
    cls, _, _ = fit_scalers(config, train)
    # Quality probes use exactly the trained prefix as nearest-neighbor reference.
    quality_indices = np.concatenate([np.flatnonzero(labels == c)[:config['quality_samples_per_class']] for c in range(19)])
    qx, qy = values[quality_indices], labels[quality_indices]
    quality = stratified_generation_quality(train['X'], train['y'], validation['X'], validation['y'], qx, qy,
                        FEATURES['imu_rpm'], groups=train['groups'], command_conditioned=False)
    model = load_classifier(config, raw_classifier)
    quality['trained_raw_classifier_label_agreement'] = metric(qy, classify.predict_classifier(model, transform(qx, cls))['logits'].argmax(1))
    write_json(Path(generator['directory']) / 'quality.json', quality)
    return values, labels


def bank_additions(values, labels, counts):
    indices = np.concatenate([np.flatnonzero(labels == c)[:n] for c, n in counts.items()]).astype(np.int64)
    return values[indices], labels[indices]


def select_ratio(jobs, method, prefix):
    eligible = [j for j in jobs if j['prefix_s'] == prefix and j['feature'] == 'imu_rpm' and not j['equal_count']]
    rows = []
    for ratio in (0., .25, .5, 1.):
        selected = [j for j in eligible if (j['method'] == method and j['ratio'] == ratio) or
                    (ratio == 0. and j['method'] == 'raw')]
        if len(selected) != 3 or {j['seed'] for j in selected} != {42, 43, 44}:
            raise ValueError('Ratio selection requires all three development seeds')
        scores = [j['validation']['prefix_24']['macro_f1'] for j in selected]
        rows.append(dict(ratio=ratio, mean_macro_f1=float(np.mean(scores)), std_macro_f1=float(np.std(scores)),
                         scores=scores, job_signatures=[j['signature'] for j in selected]))
    chosen = min(rows, key=lambda r: (-round(r['mean_macro_f1'], 8), r['ratio']))
    return dict(selected_ratio=chosen['ratio'], rows=rows, selected=chosen)


def develop(config, registry):
    """Fully predetermined jobs. Test arrays are never passed to this function."""
    if any(r['role'] == 'test' for r in registry['records']): raise ValueError('Test in development registry')
    jobs = []; generators = []
    definitions = [('A', 'imu_cmd_rpm', 24, False), ('B', 'imu_rpm', 24, False),
                   ('C', 'imu_rpm', 12, False), ('D', 'imu_rpm', 6, False), ('E', 'imu_rpm', 24, True)]
    for tag, feature, seconds, equal in definitions:
        train = pool(config, registry, 'train', feature, seconds, equal)
        validation = pool(config, registry, 'validation', feature, 24)
        for seed in config['training_seeds']:
            jobs.append(train_job(config, registry, train, validation, seed, tag))
            write_json(root(config) / 'development_progress.json', dict(jobs=jobs, generators=generators))
    for seconds in config['augmentation_prefixes']:
        train = pool(config, registry, 'train', 'imu_rpm', seconds)
        validation = pool(config, registry, 'validation', 'imu_rpm', 24)
        for seed in config['training_seeds']:
            gen = generator_job(config, train, seed); generators.append(gen)
            raw_job = next(j for j in jobs if j['tag'] == ('D' if seconds == 6 else 'B') and j['seed'] == seed)
            bank, labels = synthetic_bank(config, train, validation, gen, raw_job)
            for ratio in config['augmentation_ratios']:
                if ratio == 0: continue
                counts = augmentation_counts(train['y'], ratio)
                for method in ('ros', 'gan'):
                    if method == 'ros':
                        extra, extra_y, refs = ros_additions(train, counts, seed+10000)
                    else:
                        extra, extra_y = bank_additions(bank, labels, counts); refs = np.full(len(extra_y), -1)
                    if {i: int(np.sum(extra_y == i)) for i in range(19)} != counts:
                        raise AssertionError('Augmentation count differs from floor protocol')
                    tag = f'{method}_p{seconds}_r{ratio:g}'
                    job = train_job(config, registry, train, validation, seed, tag, method, ratio,
                                    np.concatenate([train['X'], extra]), np.r_[train['y'], extra_y])
                    write_json(Path(job['directory']) / 'augmentation_audit.json', dict(
                        counts={str(c): n for c, n in counts.items()}, original=job['class_original'], joined=job['class_joined'],
                        references_sha256=array_hash(refs), generator_signature=gen['signature'] if method == 'gan' else None,
                        training_pool=train['signature'], ratio=ratio, floor_rule=True))
                    jobs.append(job)
                    write_json(root(config) / 'development_progress.json', dict(jobs=jobs, generators=generators))
    choices = {str(s): {m: select_ratio(jobs, m, s) for m in ('ros', 'gan')} for s in config['augmentation_prefixes']}
    duration = []
    for s, tag in [(6, 'D'), (12, 'C'), (24, 'B')]:
        values = [j['validation']['prefix_24']['macro_f1'] for j in jobs if j['tag'] == tag]
        duration.append(dict(prefix_s=s, mean_macro_f1=float(np.mean(values)), scores=values))
    recommendation = min(duration, key=lambda v: (-round(v['mean_macro_f1'], 8), v['prefix_s']))
    selected = {tag: [j['signature'] for j in jobs if j['tag'] == tag] for tag, *_ in definitions}
    for s in config['augmentation_prefixes']:
        for display, method, ratio in [('ROS', 'ros', choices[str(s)]['ros']['selected_ratio']),
                                       ('GAN', 'gan', choices[str(s)]['gan']['selected_ratio']),
                                       ('ROS_matched_GAN', 'ros', choices[str(s)]['gan']['selected_ratio'])]:
            tag = ('D' if s == 6 else 'B') if ratio == 0 else f'{method}_p{s}_r{ratio:g}'
            selected[f'{display}_{s}'] = [j['signature'] for j in jobs if j['tag'] == tag]
    result = dict(config_signature=stable_hash(config), development_registry_signature=registry['signature'],
                  jobs=jobs, generators=generators, ratio_choices=choices, selected_models=selected,
                  duration_validation=duration, recommended_prefix_s=recommendation['prefix_s'])
    write_json(root(config) / 'development_complete.json', result)
    return result


def freeze(config, development, test_registry):
    complete_path = root(config) / 'development_complete.json'
    complete = read(complete_path)
    if len(complete['jobs']) != 51 or len(complete['generators']) != 6:
        raise RuntimeError('All 51 classifier and 6 generator jobs must complete before test unlock')
    locked = {str(ROOT / name): sha256_file(ROOT / name) for name in TRAIN_CODE + ['configs/simulink_gan_v3_prefix.json']}
    for name in ('development_complete.json', 'development_sources.json', 'new_test_sources.json', 'predeclared_protocol.json'):
        path = root(config) / name; locked[str(path)] = sha256_file(path)
    for spec in complete['jobs'] + complete['generators']:
        directory = Path(spec['directory']); is_gan = 'generator_updates' in spec
        checkpoint = directory / ('checkpoint_3000.pt' if is_gan else 'checkpoint_1000.pt')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['signature'] != spec['signature']: raise RuntimeError('Unmatched trained model')
        if is_gan:
            if saved['generator_updates'] != 3000 or saved['critic_updates'] != 15000 or saved['command_indices']:
                raise RuntimeError('GAN actual update/condition mismatch')
        elif saved['step'] != 1000 or saved['dual_head']:
            raise RuntimeError('Classifier budget/head changed')
        for path in [checkpoint, directory / 'job.json', directory / 'result.json', directory / 'training_curve.json',
                     Path(spec['scaler_directory']) / 'scalers.pkl', Path(spec['scaler_directory']) / 'fit_sources.json']:
            locked[str(path)] = sha256_file(path)
        if not is_gan:
            locked[str(directory / 'alarm_selection.json')] = sha256_file(directory / 'alarm_selection.json')
    path = root(config) / 'freeze.json'
    if path.exists():
        existing = read(path); authorize_test(config, test_registry, existing['signature']); return existing
    payload = dict(**runtime_info(), config_signature=stable_hash(config),
                   development_registry_signature=development['signature'], test_registry_signature=test_registry['signature'],
                   selected_models=complete['selected_models'], ratio_choices=complete['ratio_choices'],
                   recommended_prefix_s=complete['recommended_prefix_s'], classifier_jobs=51, generator_jobs=6,
                   locked_files=locked, test_features_read_before_freeze=False)
    payload['signature'] = stable_hash(payload); immutable_json(path, payload)
    print('V3_FROZEN', payload['signature'], flush=True)
    return payload


def evaluate_test(config, registry, frozen):
    authorization = frozen['signature']; authorize_test(config, registry, authorization)
    access_path = root(config) / 'test_access_receipt.json'
    if not access_path.exists():
        immutable_json(access_path, dict(**runtime_info(), freeze_signature=authorization,
                       test_registry_signature=registry['signature'],
                       purpose='First authorized unified formal test read; all development already frozen'))
    complete = read(root(config) / 'development_complete.json')
    selected_ids = {s for signatures in frozen['selected_models'].values() for s in signatures}
    specs = [j for j in complete['jobs'] if j['signature'] in selected_ids]
    path = root(config) / 'test_progress.json'
    if path.exists():
        previous = read(path)
        if previous['freeze_signature'] != authorization or previous['test_registry_signature'] != registry['signature']:
            raise PermissionError('Test progress belongs to another freeze/source registry')
        results = previous['results']
        if any(r['signature'] not in selected_ids for r in results): raise ValueError('Unexpected cached test model')
    else:
        results = []
    existing = {r['signature'] for r in results}
    for spec in specs:
        if spec['signature'] in existing: continue
        values = pool(config, registry, 'test', spec['feature'], 24, authorization=authorization)
        model = load_classifier(config, spec)
        with (Path(spec['scaler_directory']) / 'scalers.pkl').open('rb') as f: cls, _ = pickle.load(f)
        output = classify.predict_classifier(model, transform(values['X'], cls))
        masks = scope_masks(values)
        groups = {str(g): {name: metric(values['y'][mask & (values['groups'] == g)],
                           output['logits'][mask & (values['groups'] == g)].argmax(1))
                           for name, mask in masks.items()} for g in config['test_groups']}
        records = prediction_records(config, registry, 'test', spec, model, authorization)
        alarm = alarm_metrics(records, **spec['alarm_policy'])
        alarm_groups = {str(g): alarm_metrics([r for r in records if r['group_id'] == g], **spec['alarm_policy'])
                        for g in config['test_groups']}
        result = dict(signature=spec['signature'], tag=spec['tag'], seed=spec['seed'], feature=spec['feature'],
                      prefix_s=spec['prefix_s'], method=spec['method'], ratio=spec['ratio'], alarm_policy=spec['alarm_policy'],
                      test=scoped_metrics(values, output), groups=groups, alarm=alarm, alarm_groups=alarm_groups,
                      alarm_gated_window=scoped_metrics(values, output, threshold=spec['alarm_policy']['threshold']),
                      evaluation_pool_signature=values['signature'], evaluation_source_ids=sorted(np.unique(values['source_ids']).tolist()))
        np.savez_compressed(Path(spec['directory']) / 'formal_test_predictions.npz', logits=output['logits'],
                            fault_probability=output['fault_probability'], y=values['y'], source_ids=values['source_ids'],
                            starts=values['starts'], ends=values['ends'])
        results.append(result)
        write_json(path, dict(freeze_signature=authorization, test_registry_signature=registry['signature'], results=results))
        print('V3_TEST', spec['tag'], spec['seed'], result['test']['prefix_24']['macro_f1'], flush=True)
    if len(results) != len(specs): raise RuntimeError('Incomplete unified test')
    final = dict(freeze_signature=authorization, test_registry_signature=registry['signature'],
                 selected_models=frozen['selected_models'], unique_tested_models=len(specs), results=results,
                 test_sources=57, independent_test_groups=3, all_development_preceded_test=True)
    write_json(root(config) / 'final_test_results.json', final)
    return final
