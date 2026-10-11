"""Windows entry points for the first two next-step batches."""
import argparse
import json
from pathlib import Path
from provenance import write_json
from .configuration import load, ROOT
from .commands import task_library


def main():
    parser = argparse.ArgumentParser(description='Parameterized research tasks and causal diagnostic development')
    parser.add_argument('command', choices=['library', 'validate', 'generate', 'check', 'preprocess', 'interface-fit', 'replay'])
    parser.add_argument('--out', type=Path, default=ROOT/'output_hex_next_20261009'/'batch2')
    parser.add_argument('--settings', type=Path, default=Path(__file__).parent/'configs/continuous.json')
    parser.add_argument('--feature-set', choices=['imu_only', 'imu_attitude', 'imu_attitude_command'], default='imu_only')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    cfg = load(); settings = json.loads(args.settings.read_text(encoding='utf-8')); settings['diagnostics']['feature_set'] = args.feature_set
    if args.command == 'library':
        write_json(args.out/'task_library.json', {'settings': settings, 'groups': task_library(settings)})
    elif args.command == 'validate':
        from .continuous_data import validate_tasks
        report = validate_tasks(cfg, settings, args.out)
        if not report['all_normal_passed']:
            raise SystemExit('New-task normal validation failed; see saved actual results')
    elif args.command == 'generate':
        from .continuous_data import generate
        generate(cfg, settings, args.out)
    elif args.command == 'check':
        from .continuous_data import validate
        validate(args.out/'dataset_v5')
    elif args.command == 'preprocess':
        from diagnostics.preprocessing import prepare
        prepare(args.out/'dataset_v5', args.out/'preprocessed'/args.feature_set, settings['diagnostics'])
    elif args.command == 'interface-fit':
        from diagnostics.workflow import fit_interface
        fit_interface(args.out/'preprocessed'/args.feature_set, args.out/'diagnostics'/args.feature_set, settings)
    elif args.command == 'replay':
        from diagnostics.workflow import evaluate_replay
        evaluate_replay(args.out/'dataset_v5', args.out/'preprocessed'/args.feature_set,
                        args.out/'diagnostics'/args.feature_set, settings)


if __name__ == '__main__':
    main()
