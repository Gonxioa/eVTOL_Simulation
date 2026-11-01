"""Refresh current source hashes and verify the completed naming migration."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SIM = ROOT / 'simulink'
ORIGINAL = (HERE / 'source_commit.txt').read_text().strip()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def git_bytes(path):
    return subprocess.run(['git', 'show', f'{ORIGINAL}:{path}'], cwd=ROOT,
                          check=True, capture_output=True).stdout


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


verification = read(HERE / 'verification.json')
assert verification['passed'] and verification['signal_max_errors'] == [0, 0]
paths = {SIM / 'hex_evtol.slx', SIM / 'internal/layout_template.slx',
         SIM / 'configs/research.json', SIM / 'validation/SIMULATION_TUNING.md'}
for folder in (SIM, SIM / 'blocks', SIM / 'functions', SIM / 'validation'):
    paths.update(folder.glob('*.m'))
paths.update((SIM / 'validation').glob('*.py'))
hashes = {p.relative_to(SIM).as_posix(): sha(p) for p in sorted(paths)}
write(SIM / 'validation/SIMULATION_SOURCE_HASHES.json', {
    'profile': 'control_v3_20261023', 'sha256': hashes,
    'naming_verification': '../../maintenance/naming_20261101/verification.json',
    'original_evidence_commit': ORIGINAL,
})

report_path = SIM / 'validation/MY_SIMULATION_REPORT.json'
report = read(report_path)
original_report = json.loads(git_bytes('simulink/validation/MY_SIMULATION_RECOVERY.json'))
report['source_sha256'] = {name: sha(SIM / name) for name in report['source_sha256']}
report['source_hash_scope'] = 'Current names after numeric equivalence verification.'
report['original_execution_evidence'] = {
    'commit': ORIGINAL,
    'report': 'simulink/validation/MY_SIMULATION_RECOVERY.json',
    'source_sha256': original_report['source_sha256'],
    'wall_time_scope': 'wall_time_s above is from the original recorded execution.',
}
report['naming_equivalence'] = verification
write(report_path, report)

active = list(SIM.glob('*.m'))
for folder in ('blocks', 'functions', 'validation'):
    active += list((SIM / folder).glob('*.m'))
active += list((SIM / 'validation').glob('*.py')) + [SIM / 'configs/research.json']
for path in active:
    source = path.read_text(encoding='utf-8-sig')
    assert 'recovery' not in source.lower(), f'Old active name: {path}'
    if path.suffix == '.m':
        match = re.search(r'^function\s+(?:(?:\[[^\]]+\]|\w+)\s*=\s*)?(\w+)\s*\(', source, re.M)
        # Ordinary scripts may contain local helpers after script statements.
        if source.lstrip().startswith('function'):
            assert match and match[1] == path.stem, f'Function/file mismatch: {path}'
with zipfile.ZipFile(SIM / 'hex_evtol.slx') as model:
    for name in model.namelist():
        if name.endswith('.xml'):
            assert b'recovery' not in model.read(name).lower(), f'Old model name: {name}'
assert [p.name for p in SIM.glob('*.slx')] == ['hex_evtol.slx']
assert not [p for p in SIM.rglob('*') if 'recovery' in p.name.lower()]

# Keep the failed/initial evidence byte-for-byte, including its historical names.
frozen = ('LOOP_INITIAL_RESULTS', 'SOURCE_SLX_PARITY_INITIAL',
          'SOURCE_SLX_PARITY_FAILED_1a71a5e4')
for stem in frozen:
    current = SIM / 'validation' / f'SIMULATION_{stem}.json'
    assert current.read_bytes() == git_bytes(f'simulink/validation/RECOVERY_{stem}.json')

folder = SIM / 'output/simulations/control_v3_20261023'
suites = {}
for suite in ('CORE', 'HEALTHY', 'IDEAL', 'SEEDS'):
    data = read(SIM / f'validation/SIMULATION_{suite}_RESULTS.json')
    assert data['passed'] and data['profile'] == 'control_v3_20261023'
    for case in data['cases']:
        assert (folder / case['name'] / 'result.mat').is_file()
    suites[suite] = len(data['cases'])
assert sum(suites.values()) == 106
input_audit = read(SIM / 'validation/SIMULATION_INPUT_AUDIT.json')
assert input_audit['passed'] and len(input_audit['cases']) == 106
assert len(list((folder / 'my_circle_run').glob('*.png'))) == 8
healthy = folder / 'healthy_four_tasks_20261027'
healthy_plots = list(healthy.glob('four_tasks*comparison.png'))
for task in ('00_hover', '01_line', '02_circle', '03_vertical'):
    healthy_plots += list((healthy / task).glob('*.png'))
assert len(healthy_plots) == 24
write(HERE / 'static_checks.json', {
    'passed': True, 'active_sources_checked': len(active), 'source_hash_count': len(hashes),
    'current_model_only': True, 'model_xml_normal_names': True,
    'historical_reports_byte_preserved': len(frozen), 'native_case_data_retained': suites,
    'saved_trace_input_audit_cases': len(input_audit['cases']),
    'standard_png_count': 8, 'healthy_png_count': 24,
})
print('NAMING_STATIC_CHECKS_PASSED', len(active), len(hashes), suites)
