"""Audit preserved artifacts and index the completed v2 release; no simulation."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DOCS = Path(__file__).resolve().parent
OUT = ROOT / 'output_hex_rate_tuning_20261010/release_v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    baseline = read(DOCS / 'baseline.json')
    changed = [name for name, digest in baseline['protected_files'].items()
               if not (ROOT / name).exists() or sha(ROOT / name) != digest]
    if changed:
        raise RuntimeError('Protected artifacts changed: ' + str(changed))
    fit = read(DOCS / 'v2_interface_fit.json')
    checkpoint = OUT / 'diagnostics/imu_only_epoch30/interface_model.pt'
    if sha(checkpoint) != fit['checkpoint_sha256']:
        raise RuntimeError('New checkpoint does not match its recorded hash')
    rows = []
    for path in sorted((OUT / 'dataset_v5').glob('g[0-9][0-9][0-9]_*.json')):
        side = read(path)
        metrics = side['flight_metrics']
        if sha(path.with_suffix('.csv')) != side['csv_sha256']:
            raise RuntimeError('New CSV changed: ' + str(path))
        rows.append(dict(record_id=side['record_id'], csv=path.with_suffix('.csv').name,
                         label=side['scenario']['label'],
                         **{key: metrics[key] for key in (
                             'termination', 'actual_duration_s', 'position_rmse_m',
                             'final_position_error_m', 'maximum_tilt_deg',
                             'maximum_rate_rad_s', 'nominal_bound_fraction')},
                         csv_sha256=side['csv_sha256']))
    if len(rows) != 152:
        raise RuntimeError('Expected 152 release records')
    with (DOCS / 'v2_record_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    receipt = dict(
        executed_utc=datetime.now(timezone.utc).isoformat(),
        branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        numerical_code_commit='9ba6fa0',
        final_unit_regression_tests_previously_observed_passed=48,
        final_healthy_passed=32, final_partial_passed=192, final_long_passed=64,
        new_data_records=len(rows), old_protected_files_checked=len(baseline['protected_files']),
        changed_old_files=changed, new_checkpoint_sha256_verified=sha(checkpoint),
        diagnosis_reliability_passed=False,
        replay_url='http://127.0.0.1:8766/?group=6&label=13',
        browser_observed=dict(partial_M0_group6_end_s=8.0,
                              partial_M0_group6_speed_m_s_display=0.017,
                              full_M0_group6_end_s=3.77, full_M0_group6_reason='z < -2 m',
                              model_version='hex_research_v2_integral_recovery',
                              config_hash='bcf8e537fd9d2b36853cc0a5dbc97f776222d68a165e947109cb7d161c7ec8c0'),
        git_push_performed=False, git_merge_performed=False,
        report_sha256={p.name: sha(p) for p in DOCS.glob('*.json') if p.name != 'FINAL_RECEIPT.json'},
        validation_document_sha256=sha(DOCS / 'VALIDATION_REPORT.md'),
        metrics_csv_sha256=sha(DOCS / 'v2_record_metrics.csv'),
        full_page_screenshot_sha256=sha(DOCS / 'figures/v2_replay.jpg'),
        sample_screenshot_sha256=sha(DOCS / 'figures/v2_current_sample.jpg'))
    (DOCS / 'FINAL_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(protected=len(baseline['protected_files']), changed=changed,
                          new_checkpoint_verified=True, records=len(rows),
                          completed=sum(row['termination'] == 'completed' for row in rows)), ensure_ascii=False))


if __name__ == '__main__':
    main()
