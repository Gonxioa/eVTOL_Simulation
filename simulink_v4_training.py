"""Isolated V4 exact-step trainer with full-train validation and reproducible resume.

Uses classify.FaultClassifier unchanged; extends the V3 step-training protocol
with variable windows, one CNN control, per-class/group diagnostics and budgets.
"""
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import classify
from provenance import code_hashes, stable_hash, sha256_file, write_json, runtime_info
from simulink_data import ROOT, read
from simulink_v3_experiment import metric
from simulink_v3_data import immutable_json, array_hash
from simulink_v4_data import root, fit_scalers, masks
from simulink_v4_models import build, statistical_features

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CODE = ['classify.py', 'simulink_v3_experiment.py', 'simulink_v4_models.py', 'simulink_v4_training.py', 'simulink_v4_data.py']


def transform(x, scaler):
    return scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)


def predict(model, x, batch_size=128):
    model.eval(); output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            bx = torch.as_tensor(x[start:start+batch_size], dtype=torch.float32, device=DEVICE)
            output.append(model(bx).cpu().numpy())
    return np.concatenate(output)


def diagnostics(y, logits):
    if not len(y): return dict(window_count=0, accuracy=None, macro_f1=None, present_class_macro_f1=None)
    output = metric(y, logits.argmax(1)); cm = np.asarray(output['confusion_matrix'])
    probabilities = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1).numpy()
    output['cross_entropy'] = float(log_loss(y, probabilities, labels=np.arange(19)))
    output['prediction_frequencies'] = (cm.sum(0)/cm.sum()).tolist()
    output['healthy_recall'] = None if not cm[0].sum() else float(cm[0, 0]/cm[0].sum())
    output['joint_accuracy_by_effectiveness'] = {str(eta): float(np.diag(cm)[a:b].sum()/cm[a:b].sum()) if cm[a:b].sum() else None
        for eta, a, b in [(0., 1, 7), (.2, 7, 13), (.8, 13, 19)]}
    output['motor_accuracy_on_fault'] = float(sum(cm[i, j] for i in range(1, 19) for j in range(1, 19)
        if (i-1) % 6 == (j-1) % 6)/max(cm[1:].sum(), 1))
    output['severity_only_accuracy_on_fault'] = float(sum(cm[i, j] for i in range(1, 19) for j in range(1, 19)
        if (i-1)//6 == (j-1)//6)/max(cm[1:].sum(), 1))
    return output


def evaluate(p, logits):
    result = {name: diagnostics(p['y'][mask], logits[mask]) for name, mask in masks(p).items()}
    result['groups'] = {str(g): diagnostics(p['y'][p['groups'] == g], logits[p['groups'] == g]) for g in sorted(set(p['groups']))}
    result['worst_group_macro_f1'] = min(r['macro_f1'] for r in result['groups'].values())
    for low, high, name in [(0., .25, 'fault_fraction_0_25'), (.25, .5, 'fault_fraction_25_50'),
                             (.5, .75, 'fault_fraction_50_75'), (.75, 1.00001, 'fault_fraction_75_100')]:
        mask = (p['y'] != 0) & (p['physical_fault_fraction'] >= low) & (p['physical_fault_fraction'] < high)
        result[name] = diagnostics(p['y'][mask], logits[mask])
    return result


def model_latency(model, x):
    bx = torch.as_tensor(x[:1], dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.inference_mode():
        for _ in range(5): model(bx)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        before = time.perf_counter()
        for _ in range(30): model(bx)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
    return (time.perf_counter()-before)*1000/30


def _state_path(spec, step): return Path(spec['directory']) / f'checkpoint_{step}.pt'


def load_model(config, spec, step=None):
    state = torch.load(_state_path(spec, step or spec['step']), map_location=DEVICE, weights_only=False)
    if state['signature'] != spec['signature']: raise ValueError('Model checkpoint identity mismatch')
    model = build(config, spec['family'], spec['window'], spec['features']).to(DEVICE)
    model.load_state_dict(state['model']); model.eval()
    return model


def train(config, train_pool, validation, family, seed, max_steps, method='raw', ratio=0., raw=None, labels=None):
    torch.set_num_threads(4)
    if sorted(set(train_pool['y'])) != list(range(19)): raise ValueError('Every training pool must retain all 19 classes')
    if train_pool['identity']['role'] != 'train' or validation['identity']['role'] != 'validation':
        raise PermissionError('Training/validation role violation')
    cls, _, scale_directory = fit_scalers(config, train_pool)
    raw = train_pool['X'] if raw is None else raw; labels = train_pool['y'] if labels is None else labels
    x = transform(raw, cls); vx = transform(validation['X'], cls)
    identity = dict(config_signature=stable_hash(config), train_pool=train_pool['signature'], validation_pool=validation['signature'],
        family=family, features=x.shape[-1], window=x.shape[1], seed=seed, method=method, ratio=ratio,
        max_declared_steps=max(config['budget_candidates']), batch=config['batch_size'], learning_rate=config['learning_rate'],
        X_sha256=array_hash(x), y_sha256=array_hash(labels), scaler_sha256=sha256_file(Path(scale_directory)/'scalers.pkl'),
        code=code_hashes(ROOT, CODE))
    signature = stable_hash(identity)
    directory = root(config)/'classifiers'/f'{family}_w{x.shape[1]}_s{seed}_{signature[:16]}'
    directory.mkdir(parents=True, exist_ok=True)
    immutable_json(directory/'job.json', dict(**identity, signature=signature, scaler_directory=scale_directory))
    classify.set_seed(seed)
    model = build(config, family, x.shape[1], x.shape[-1]).to(DEVICE)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    rng = np.random.default_rng(seed+17); start = 0; history = []; losses = []; elapsed = 0.
    if (directory/'last.pt').exists():
        saved = torch.load(directory/'last.pt', map_location=DEVICE, weights_only=False)
        if saved['signature'] != signature: raise ValueError('Incompatible optimizer state')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        start=saved['step']; history=saved['history']; initial=saved['initial_model']; elapsed=saved['training_wall_s']
        rng.bit_generator.state=saved['sampler_rng']; torch.set_rng_state(saved['torch_rng'].cpu())
        if DEVICE.type == 'cuda': torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    if start >= max_steps:
        spec = read(directory/'result.json')
        if not _state_path(spec, max_steps).exists(): raise RuntimeError('Missing requested budget checkpoint')
        return spec
    started = time.perf_counter(); trained_since=0.
    for step in range(start+1, int(max_steps)+1):
        before = time.perf_counter(); model.train()
        indices = rng.integers(0, len(labels), size=config['batch_size'])
        bx = torch.as_tensor(x[indices], dtype=torch.float32, device=DEVICE)
        by = torch.as_tensor(labels[indices], dtype=torch.long, device=DEVICE)
        optimizer.zero_grad(set_to_none=True); logits=model(bx)
        loss=torch.nn.functional.cross_entropy(logits, by); loss.backward()
        grad=float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None)))
        if not np.isfinite(float(loss)) or not np.isfinite(grad): raise FloatingPointError('Nonfinite classifier optimization')
        optimizer.step(); losses.append(float(loss.detach())); trained_since += time.perf_counter()-before
        if step in config['checkpoints'] or step == max_steps:
            full_train = diagnostics(labels, predict(model, x)); val = evaluate(validation, predict(model, vx))
            record = dict(step=step, batch_loss=float(loss.detach()), recent_training_loss_mean=float(np.mean(losses[-250:])),
                gradient_norm=grad, full_train=full_train, validation=val, inference_ms_per_window=model_latency(model, vx),
                training_wall_s=elapsed+trained_since, wall_s_including_evaluation=time.perf_counter()-started,
                parameter_count=sum(p.numel() for p in model.parameters()), actual_updates=step,
                selected_training_windows=len(labels), expected_window_draws=step*config['batch_size']/len(labels))
            history.append(record)
            delta = max(float((v.detach().cpu()-initial[k].cpu()).abs().max()) for k, v in model.state_dict().items())
            state=dict(signature=signature, step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                initial_model=initial, parameter_max_delta=delta, sampler_rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else [],
                training_wall_s=elapsed+trained_since, history=history)
            temporary=directory/'checkpoint.tmp'; torch.save(state, temporary); temporary.replace(directory/f'checkpoint_{step}.pt')
            import shutil
            shutil.copy2(directory/f'checkpoint_{step}.pt', directory/'last.pt')
            write_json(directory/'training_curve.json', history)
            print('V4_CLASSIFIER', family, 'w', x.shape[1], 'seed', seed, 'step', step,
                  'train_f1',round(full_train['macro_f1'],5),'validation_f1',round(val['prefix_24']['macro_f1'],5),flush=True)
    spec = dict(**identity, signature=signature, directory=str(directory), scaler_directory=scale_directory,
                step=max_steps, prefix_s=train_pool['identity']['prefix_s'], grid=train_pool['identity']['grid'],
                sampling_seed=train_pool['identity']['sampling_seed'], history=history,
                training_windows=len(labels), original_windows=len(train_pool['y']),
                class_original=np.bincount(train_pool['y'], minlength=19).tolist(),
                class_joined=np.bincount(labels, minlength=19).tolist(), completed_utc=runtime_info()['created_utc'])
    write_json(directory/'result.json', spec)
    return spec


def record_at(spec, step):
    return next(row for row in spec['history'] if row['step'] == step)


def rank_candidates(rows):
    """Rows are aggregate three-seed candidate records, never individual lucky seeds."""
    return sorted(rows, key=lambda r: (-round(r['mean_macro_f1'], 8), r['window'], r['steps'], r['parameters'], r['family']))


def aggregate(specs, steps):
    rows = []
    for step in steps:
        selected = [s for s in specs if any(r['step'] == step for r in s['history'])]
        if not selected: continue
        if len(selected) != 3 or {s['seed'] for s in selected} != {42, 43, 44}: raise ValueError('Selection needs all three training seeds')
        results = [record_at(s, step) for s in selected]
        rows.append(dict(family=selected[0]['family'], window=selected[0]['window'], steps=step,
            mean_macro_f1=float(np.mean([r['validation']['prefix_24']['macro_f1'] for r in results])),
            std_macro_f1=float(np.std([r['validation']['prefix_24']['macro_f1'] for r in results])),
            train_macro_f1=float(np.mean([r['full_train']['macro_f1'] for r in results])),
            parameters=results[0]['parameter_count'], mean_inference_ms=float(np.mean([r['inference_ms_per_window'] for r in results])),
            signatures=[s['signature'] for s in selected], source_scope='same three validation groups'))
    return rows


def statistical_job(config, train_pool, validation, seed):
    cls, _, scale_directory = fit_scalers(config, train_pool)
    x = statistical_features(transform(train_pool['X'], cls)); vx = statistical_features(transform(validation['X'], cls))
    identity=dict(train_pool=train_pool['signature'], validation_pool=validation['signature'], seed=seed,
        model='random_forest', trees=config['statistical_trees'], min_leaf=config['statistical_min_leaf'],
        X_sha256=array_hash(x), config_signature=stable_hash(config), code=code_hashes(ROOT, CODE))
    key=stable_hash(identity); directory=root(config)/'statistics'/key
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists(): return read(directory/'result.json')
    before=time.perf_counter()
    model=RandomForestClassifier(n_estimators=config['statistical_trees'],min_samples_leaf=config['statistical_min_leaf'],
                                max_features='sqrt',n_jobs=4,random_state=seed)
    model.fit(x,train_pool['y'])
    elapsed=time.perf_counter()-before
    def logits(features): return np.log(np.maximum(model.predict_proba(features),1e-12))
    result=dict(**identity,signature=key,window=train_pool['identity']['window'],feature_count=x.shape[1],
        training_wall_s=elapsed,tree_node_count=sum(t.tree_.node_count for t in model.estimators_),
        full_train=diagnostics(train_pool['y'],logits(x)),validation=evaluate(validation,logits(vx)),scaler_directory=scale_directory)
    with (directory/'model.pkl').open('wb') as f: pickle.dump(model,f)
    write_json(directory/'result.json',result)
    print('V4_STATISTICAL',seed,train_pool['identity']['window'],result['validation']['prefix_24']['macro_f1'],flush=True)
    return result
