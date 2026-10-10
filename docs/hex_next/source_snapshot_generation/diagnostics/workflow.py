"""Small trained LSTM interface check; deliberately no formal performance claim.

One paired group per family trains, another checks replay. These groups are
development data, never frozen final test data. WGAN is not run in this batch.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from provenance import sha256_file, stable_hash, write_json
from .models import FaultLSTM
from .features import FEATURE_SETS, packets_from_frame
from .continuous import replay, causal_windows, offline_labels


class TorchPredictor:
    def __init__(self, checkpoint, settings):
        artifact = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if artifact['feature_names'] != FEATURE_SETS[settings['feature_set']] or artifact['window'] != settings['window'] or artifact['sample_hz'] != settings['sample_hz']:
            raise RuntimeError('Checkpoint feature/window/sample contract mismatch')
        self.artifact = artifact; self.mean = artifact['mean'].numpy(); self.scale = artifact['scale'].numpy()
        self.model = FaultLSTM(**artifact['model_config']); self.model.load_state_dict(artifact['state_dict']); self.model.eval()

    def __call__(self, window):
        if window.shape != (self.artifact['window'], len(self.artifact['feature_names'])) or not np.isfinite(window).all():
            raise ValueError('Invalid diagnostic inference window')
        scaled = (window-self.mean)/self.scale
        with torch.inference_mode():
            return torch.softmax(self.model(torch.as_tensor(scaled[None], dtype=torch.float32)), dim=1)[0].numpy()


def fit_interface(preprocessed, out, settings):
    preprocessed, out = Path(preprocessed), Path(out); out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((preprocessed/'preprocess_metadata.json').read_text())
    for name, digest in metadata['array_sha256'].items():
        if sha256_file(preprocessed/name) != digest:
            raise RuntimeError('Preprocessed arrays changed after completion')
    if sha256_file(preprocessed/'window_manifest.json') != metadata['window_manifest_sha256']:
        raise RuntimeError('Window provenance changed')
    entries = json.loads((preprocessed/'window_manifest.json').read_text()); eligible = np.load(preprocessed/'training_continuous_indices.npy')
    X = np.load(preprocessed/'X_raw.npy'); y = np.load(preprocessed/'y_all.npy'); groups = np.load(preprocessed/'groups.npy')
    fit = settings['interface_fit']; train_groups = sorted({e['group_id'] for e in entries if e['repeat'] == fit['training_repeat']})
    check_groups = sorted({e['group_id'] for e in entries if e['repeat'] == fit['replay_repeat']})
    if not train_groups or not check_groups or set(train_groups) & set(check_groups):
        raise RuntimeError('Need disjoint paired development train/replay groups')
    torch.set_num_threads(1); torch.manual_seed(fit['seed']); rng = np.random.default_rng(fit['seed'])
    indices = []
    for label in range(19):
        candidates = np.flatnonzero((y == label) & np.isin(groups, train_groups))
        if not len(candidates):
            raise RuntimeError('Development training is missing a class; retain limitation rather than synthesize inputs')
        if len(candidates) > fit['max_windows_per_class']:
            candidates = np.sort(rng.choice(candidates, fit['max_windows_per_class'], replace=False))
        indices.extend(candidates.tolist())
    indices = np.array(indices); train_X = X[indices]; train_y = y[indices]
    mean = train_X.reshape(-1, train_X.shape[-1]).mean(axis=0)
    scale = train_X.reshape(-1, train_X.shape[-1]).std(axis=0); scale = np.maximum(scale, 1e-6)
    normalized = ((train_X-mean)/scale).astype(np.float32)
    model_config = {'n_features': X.shape[-1], 'hidden_dim': fit['hidden_dim'], 'layers': fit['layers'], 'n_classes': 19, 'attention_pooling': True}
    model = FaultLSTM(**model_config); optimizer = torch.optim.Adam(model.parameters(), lr=fit['learning_rate'])
    loader = DataLoader(TensorDataset(torch.from_numpy(normalized), torch.from_numpy(train_y)), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(fit['seed']))
    losses = []
    for epoch in range(fit['epochs']):
        model.train(); total = 0.; count = 0
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True); loss = torch.nn.functional.cross_entropy(model(bx), by)
            loss.backward(); optimizer.step(); total += float(loss.detach())*len(by); count += len(by)
        losses.append(total/count); print('Interface LSTM epoch', epoch+1, 'loss', losses[-1], flush=True)
    artifact = {'schema': 'causal_lstm_checkpoint_v1', 'purpose': fit['purpose'], 'formal_protocol': False,
                'model_config': model_config, 'state_dict': model.state_dict(), 'mean': torch.from_numpy(mean), 'scale': torch.from_numpy(scale),
                'feature_names': metadata['feature_names'], 'feature_set': metadata['feature_set'], 'window': metadata['window'],
                'sample_hz': settings['diagnostics']['sample_hz'], 'train_groups': train_groups, 'replay_groups': check_groups,
                'training_window_ids': [entries[int(eligible[i])]['window_id'] for i in indices],
                'preprocess_identity': metadata['preprocess_identity']}
    temp = out/'interface_model.pt.tmp'; torch.save(artifact, temp); temp.replace(out/'interface_model.pt')
    report = {'purpose': fit['purpose'], 'formal_accuracy_claim': False, 'train_groups': train_groups, 'replay_groups': check_groups,
              'training_windows': len(indices), 'class_counts': {str(c): int(np.sum(train_y == c)) for c in range(19)},
              'training_window_ids': artifact['training_window_ids'], 'scaler_mean': mean.tolist(), 'scaler_std': scale.tolist(),
              'preprocess_identity': metadata['preprocess_identity'], 'epochs': fit['epochs'], 'losses': losses,
              'checkpoint_sha256': sha256_file(out/'interface_model.pt'), 'model_config': model_config,
              'alarm_settings': settings['diagnostics'], 'threshold_status': 'predeclared_design_not_selected_or_calibrated'}
    write_json(out/'interface_fit.json', report)
    return report


def evaluate_replay(folder, preprocessed, out, settings):
    from hex_research.continuous_data import validate
    folder, preprocessed, out = Path(folder), Path(preprocessed), Path(out); validate(folder)
    metadata = json.loads((preprocessed/'preprocess_metadata.json').read_text())
    fit = json.loads((out/'interface_fit.json').read_text())
    if fit['preprocess_identity'] != metadata['preprocess_identity'] or fit['checkpoint_sha256'] != sha256_file(out/'interface_model.pt'):
        raise RuntimeError('Checkpoint/source integrity mismatch')
    torch.set_num_threads(1); predictor = TorchPredictor(out/'interface_model.pt', settings['diagnostics'])
    manifest = json.loads((folder/'dataset_generation_manifest.json').read_text())
    if metadata['preprocess_identity']['dataset_signature'] != manifest['plan_signature']:
        raise RuntimeError('Replay dataset was not the checkpoint development source')
    results = []; true = []; predicted = []; equality = []; logs = out/'predictions'; logs.mkdir(exist_ok=True)
    for source in manifest['runs']:
        if source['group_id'] not in fit['replay_groups']:
            continue
        path = folder/source['file']; side = json.loads(path.with_suffix('.json').read_text()); frame = pd.read_csv(path)
        packets = packets_from_frame(frame); predictions = replay(packets, predictor, settings['diagnostics'])
        X, times = causal_windows(packets, settings['diagnostics'])
        # Same model/scaler, point-at-a-time replay vs independently reconstructed
        # causal offline windows. No batch-wide normalization or future fill.
        offline_prob = np.array([predictor(window) for window in X])
        replay_prob = np.array([p['probabilities'] for p in predictions if p['valid']])
        difference = float(np.max(np.abs(offline_prob-replay_prob))) if len(X) else 0.
        equality.append({'record_id': source['record_id'], 'max_probability_difference': difference})
        labels = offline_labels(times, source['label'], side['fault_start_s'], side['fault_end_s'])
        pure = labels >= 0; true.extend(labels[pure].tolist()); predicted.extend(np.argmax(offline_prob[pure], axis=1).tolist())
        events = [p for p in predictions if p['alarm']]
        start = side['fault_start_s']; post = [p for p in events if start is not None and p['time'] >= start]
        evaluable = source['fault_evaluable'] if source['label'] else True
        results.append({'record_id': source['record_id'], 'group_id': source['group_id'], 'family': source['family'], 'label': source['label'],
                        'termination': source['termination'], 'duration_s': source['duration_s'], 'fault_start_s': start,
                        'fault_to_end_s': source['post_fault_duration_s'], 'prediction_count': len(predictions),
                        'fault_evaluable': evaluable, 'alarm_times_s': [p['time'] for p in events],
                        'pre_fault_false_alarms': len(events) if start is None else sum(p['time'] < start for p in events),
                        'detected_before_stop': bool(post) if source['label'] else None,
                        'alarm_delay_s': float(post[0]['time']-start) if post else None,
                        'first_alarm_label': post[0]['alarm_motor_severity_label'] if post else None})
        write_json(logs/(source['record_id']+'.json'), predictions)
        print('Replay', source['record_id'], 'alarms', len(events), 'detected', bool(post) if source['label'] else 'normal', flush=True)
    normal = [r for r in results if r['label'] == 0]; faults = [r for r in results if r['label'] != 0]
    healthy_time = sum(r['duration_s'] for r in normal); false_alarms = sum(len(r['alarm_times_s']) for r in normal)
    evaluable = [r for r in faults if r['fault_evaluable']]; detected = [r for r in faults if r['detected_before_stop']]
    report = {'purpose': 'development_replay_interface_not_formal_performance', 'formal_accuracy_claim': False,
              'train_groups': fit['train_groups'], 'replay_groups': fit['replay_groups'], 'records': results,
              'healthy_normal_only_seconds': healthy_time, 'normal_only_false_alarm_events': false_alarms,
              'false_alarms_per_healthy_hour': false_alarms/healthy_time*3600 if healthy_time else None,
              'healthy_time_note': 'Only Normal records count; paired identical healthy prefixes are not multiplied into exposure',
              'fault_events': len(faults), 'evaluable_fault_events': len(evaluable), 'unevaluable_fault_events': len(faults)-len(evaluable),
              'detected_before_stop': len(detected), 'missed_evaluable_fault_events': sum(not r['detected_before_stop'] for r in evaluable),
              'alarm_delays_s': [r['alarm_delay_s'] for r in detected],
              'pure_window_macro_f1': float(f1_score(true, predicted, labels=list(range(19)), average='macro', zero_division=0)),
              'pure_window_accuracy': float(accuracy_score(true, predicted)), 'confusion_matrix': confusion_matrix(true, predicted, labels=list(range(19))).tolist(),
              'replay_offline_probability_checks': equality, 'all_replay_offline_equal': all(r['max_probability_difference'] == 0 for r in equality),
              'unpassed_or_unvalidated': ['Formal recognition performance, probability calibration and augmentation improvement are not established',
                                        'Short development healthy exposure cannot establish a reliable operational false-alarm rate']}
    write_json(out/'continuous_replay_report.json', report)
    plot_report(folder, out, results, settings)
    return report


def plot_report(folder, out, records, settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [next(r for r in records if r['family'] == family and r['label'] == 1) for family in ('hover', 'line', 'circle', 'vertical')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, source in zip(axes.flat, selected):
        predictions = json.loads((out/'predictions'/(source['record_id']+'.json')).read_text())
        axis.plot([p['time'] for p in predictions], [p['fault_probability'] for p in predictions], label='P(fault)')
        axis.axvline(source['fault_start_s'], color='black', linestyle='--', label='offline injection label')
        axis.axhline(settings['diagnostics']['fault_probability_threshold'], color='gray', linestyle=':', label='declared alarm threshold')
        axis.set(title=source['family']+' / M0 Full', xlabel='time [s]', ylabel='fault probability', ylim=(0, 1))
        axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'continuous_diagnosis.png', dpi=150); plt.close(fig)
