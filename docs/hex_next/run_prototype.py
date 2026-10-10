"""Reproduce the explicit development fitting-depth check, preserving epoch3."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from diagnostics.workflow import fit_interface, evaluate_replay
from provenance import sha256_file, stable_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'output_hex_next_20261009/batch2')
    parser.add_argument('--preset', type=Path, default=Path(__file__).with_name('PROTOTYPE_30_EPOCHS.json'))
    args = parser.parse_args()
    preset = json.loads(args.preset.read_text(encoding='utf-8'))
    settings = json.loads((ROOT/'hex_research/configs/continuous.json').read_text(encoding='utf-8'))
    settings['interface_fit'].update(preset['interface_fit']); settings['diagnostics']['feature_set'] = preset['feature_set']
    model_out = args.out/'diagnostics'/preset['name']
    if model_out.exists():
        raise SystemExit('Prototype output exists. Keep it read-only; choose a new preset name for another fit.')
    model_out.mkdir(parents=True)
    write_json(model_out/'predeclared_prototype.json', preset)
    started = time.perf_counter()
    result = fit_interface(args.out/'preprocessed'/preset['feature_set'], model_out, settings)
    fit_seconds = time.perf_counter()-started
    report = evaluate_replay(args.out/'dataset_v5', args.out/'preprocessed'/preset['feature_set'], model_out, settings)
    previous = json.loads((args.out/'diagnostics/imu_only/interface_fit.json').read_text(encoding='utf-8'))
    identity = {'training_window_ids': previous['training_window_ids'] == result['training_window_ids'],
                'train_groups': previous['train_groups'] == result['train_groups'],
                'replay_groups': previous['replay_groups'] == result['replay_groups'],
                'scaler_mean': previous['scaler_mean'] == result['scaler_mean'],
                'scaler_std': previous['scaler_std'] == result['scaler_std'],
                'model_config': previous['model_config'] == result['model_config'],
                'alarm_settings': previous['alarm_settings'] == result['alarm_settings']}
    if not all(identity.values()):
        raise RuntimeError('Fitting-depth check changed other intended fixed conditions')
    write_json(model_out/'prototype_receipt.json', {'preset': preset, 'fit_wall_seconds': fit_seconds,
               'fixed_condition_checks': identity, 'script_sha256': sha256_file(__file__),
               'formal_performance_claim': False})
    print(json.dumps({k: report[k] for k in ('normal_only_false_alarm_events', 'healthy_normal_only_seconds',
                     'detected_before_stop', 'fault_events', 'pure_window_macro_f1', 'all_replay_offline_equal')}, indent=2))


if __name__ == '__main__':
    main()
