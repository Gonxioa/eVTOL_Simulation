"""Full causal windows; labeling and provenance stay outside feature extraction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from provenance import stable_hash, sha256_file, write_json
from .features import packets_from_frame, FEATURE_SETS
from .continuous import causal_windows, offline_labels


def prepare(folder, out, settings=None):
    from hex_research.continuous_data import validate, VERSION
    folder, out = Path(folder), Path(out); validate(folder)
    manifest = json.loads((folder/'dataset_generation_manifest.json').read_text())
    settings = dict(manifest['diagnostic_settings'] if settings is None else settings)
    if settings['feature_set'] not in FEATURE_SETS:
        raise ValueError('V5 requires an explicit observation-only feature set')
    out.mkdir(parents=True, exist_ok=True)
    identity = {'dataset_signature': manifest['plan_signature'], 'dataset_manifest_sha256': sha256_file(folder/'dataset_generation_manifest.json'), 'settings': settings}
    if (out/'preprocess_metadata.json').exists():
        prior = json.loads((out/'preprocess_metadata.json').read_text())
        if prior['preprocess_identity'] != identity:
            raise RuntimeError('Refuse to mix feature versions or changed data in existing output')
    all_X = []; all_y = []; all_groups = []; records = []; train_indices = []; seen_healthy = {}
    for source in manifest['runs']:
        path = folder/source['file']; frame = pd.read_csv(path); side = json.loads(path.with_suffix('.json').read_text())
        X, times = causal_windows(packets_from_frame(frame), settings)
        labels = offline_labels(times, source['label'], side['fault_start_s'], side['fault_end_s'])
        for window, time, label in zip(X, times, labels):
            index = len(all_y)
            window_id = stable_hash({'record': source['record_id'], 'start': float(time[0]), 'end': float(time[-1]), 'features': settings['feature_set'], 'dataset': manifest['plan_signature']})
            duplicate = None
            if label == 0:
                key = (source['group_id'], float(time[0]), float(time[-1]), window.tobytes())
                duplicate = seen_healthy.get(key)
                if duplicate is None:
                    seen_healthy[key] = window_id
            keep = bool(label >= 0 and duplicate is None)
            if keep:
                train_indices.append(index)
            records.append({'window_id': window_id, 'record_id': source['record_id'], 'group_id': source['group_id'],
                            'task_family': source['family'], 'repeat': source['repeat'], 'label': int(label),
                            'phase': 'transition' if label < 0 else 'healthy' if label == 0 else 'fault_injected',
                            'start_s': float(time[0]), 'end_s': float(time[-1]),
                            'eligible_for_training': keep, 'duplicate_healthy_of': duplicate})
            all_X.append(window); all_y.append(label); all_groups.append(source['group_id'])
    X = np.stack(all_X); y = np.array(all_y, dtype=np.int64); groups = np.array(all_groups, dtype=np.int64)
    selected = np.array(train_indices, dtype=np.int64)
    np.save(out/'X_continuous.npy', X); np.save(out/'y_continuous.npy', y); np.save(out/'groups_continuous.npy', groups)
    np.save(out/'X_raw.npy', X[selected]); np.save(out/'y_all.npy', y[selected]); np.save(out/'groups.npy', groups[selected])
    np.save(out/'training_continuous_indices.npy', selected)
    write_json(out/'window_manifest.json', records); write_json(out/'feature_names.json', FEATURE_SETS[settings['feature_set']])
    counts = {str(label): int(np.sum(y[selected] == label)) for label in range(19)}
    metadata = {'dataset_version': VERSION, 'model_version': manifest['model_version'], 'formal_protocol': False,
                'preprocess_identity': identity, 'feature_set': settings['feature_set'], 'feature_names': FEATURE_SETS[settings['feature_set']],
                'settings': settings, 'window': settings['window'], 'step': settings['step'], 'all_windows': len(X),
                'training_eligible_windows': len(selected), 'transition_windows': int(np.sum(y < 0)),
                'deduplicated_healthy_windows': sum(r['duplicate_healthy_of'] is not None for r in records),
                'class_counts': counts, 'normalization': 'none; fit only on later selected training groups',
                'window_manifest_sha256': sha256_file(out/'window_manifest.json'),
                'source_dataset': str(folder.resolve()), 'group_policy': 'paired group and all derived windows stay together',
                'array_sha256': {p.name: sha256_file(p) for p in out.glob('*.npy')}}
    write_json(out/'preprocess_metadata.json', metadata)
    print('V5 causal preprocessing', X.shape, 'eligible', len(selected), 'transition', metadata['transition_windows'],
          'deduplicated', metadata['deduplicated_healthy_windows'], flush=True)
    return metadata
