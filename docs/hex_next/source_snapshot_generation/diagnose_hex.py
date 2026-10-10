"""Diagnose an observation CSV without labels, references or fault schedules."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from diagnostics.features import packets_from_frame, FEATURE_SETS
from diagnostics.continuous import replay
from diagnostics.workflow import TorchPredictor
from provenance import write_json, sha256_file


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Causal diagnosis from observation-only CSV')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feature-set', choices=list(FEATURE_SETS), default='imu_only')
    parser.add_argument('--settings', type=Path, default=root/'hex_research/configs/continuous.json')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; choose a new prediction log')
    settings = json.loads(args.settings.read_text())['diagnostics']; settings['feature_set'] = args.feature_set
    torch.set_num_threads(1); predictor = TorchPredictor(args.checkpoint, settings)
    # No sidecar, training labels, trajectory name or fault schedule is read.
    packets = packets_from_frame(pd.read_csv(args.input))
    predictions = replay(packets, predictor, settings)
    write_json(args.output, {'schema': 'continuous_predictions_v1', 'settings': settings,
               'model_purpose': predictor.artifact['purpose'], 'formal_validation': False,
               'input_sha256': sha256_file(args.input), 'checkpoint_sha256': sha256_file(args.checkpoint),
               'predictions': predictions})
    print('Saved', len(predictions), 'causal predictions to', args.output)


if __name__ == '__main__':
    main()
