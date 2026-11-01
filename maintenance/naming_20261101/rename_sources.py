"""Rename the active workflow; immutable archive records are intentionally frozen."""
from pathlib import Path
import json,re,subprocess,sys,hashlib

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')
# The common configuration helpers remain private building blocks.
renames=[
 ('simulink/hex_parameters.m','simulink/hex_base_parameters.m'),
 ('simulink/hex_block_parameters.m','simulink/hex_common_block_parameters.m'),
 ('simulink/hex_recovery_parameters.m','simulink/hex_parameters.m'),
 ('simulink/hex_recovery_block_parameters.m','simulink/hex_block_parameters.m'),
 ('simulink/hex_recovery.slx','simulink/hex_evtol.slx'),
 ('simulink/run_hex_recovery.m','simulink/run_hex_simulink.m'),
 ('simulink/configure_hex_recovery.m','simulink/configure_hex_simulink.m'),
 ('simulink/open_hex_recovery.m','simulink/open_hex_simulink.m'),
 ('simulink/build_hex_recovery_model.m','simulink/build_hex_model.m'),
 ('simulink/hex_recovery_metrics.m','simulink/hex_metrics.m'),
 ('simulink/blocks/recovery_controller_step.m','simulink/blocks/controller_step.m'),
 ('simulink/blocks/recovery_reference_step.m','simulink/blocks/reference_step.m'),
 ('simulink/internal/recovery_layout_template.slx','simulink/internal/layout_template.slx'),
 ('simulink/validation/validate_hex_recovery.m','simulink/validation/validate_hex_simulink.m'),
 ('simulink/validation/validate_hex_recovery_components.m','simulink/validation/validate_hex_components.m'),
 ('simulink/validation/validate_hex_recovery_loops.m','simulink/validation/validate_hex_loops.m'),
 ('simulink/validation/finalize_hex_recovery.m','simulink/validation/finalize_hex_delivery.m'),
 ('simulink/validation/screen_hex_recovery.m','simulink/validation/screen_hex_control.m'),
 ('simulink/validation/analyze_hex_recovery.py','simulink/validation/analyze_hex_simulation.py'),
 ('simulink/validation/recovery_overview.png','simulink/validation/flight_overview.png'),
 ('simulink/validation/MY_SIMULATION_RECOVERY.json','simulink/validation/MY_SIMULATION_REPORT.json'),
]
snapshot=Path(r'C:/Users/lenovo/AppData/Local/Temp/evtol_naming_20261101/before/validation')
for path in sorted(snapshot.glob('RECOVERY_*')):
    if not path.name.startswith('RECOVERY_'):continue
    old='simulink/validation/'+path.name
    renames.append((old,old.replace('RECOVERY_','SIMULATION_',1)))
ids={
 'hex_recovery_parameters':'hex_parameters','hex_parameters':'hex_base_parameters',
 'hex_recovery_block_parameters':'hex_block_parameters','hex_block_parameters':'hex_common_block_parameters',
 'hex_recovery_metrics':'hex_metrics','build_hex_recovery_model':'build_hex_model',
 'configure_hex_recovery':'configure_hex_simulink','open_hex_recovery':'open_hex_simulink',
 'run_hex_recovery':'run_hex_simulink','validate_hex_recovery_components':'validate_hex_components',
 'validate_hex_recovery_loops':'validate_hex_loops','validate_hex_recovery':'validate_hex_simulink',
 'finalize_hex_recovery':'finalize_hex_delivery','screen_hex_recovery':'screen_hex_control',
 'analyze_hex_recovery':'analyze_hex_simulation','recovery_controller_step':'controller_step',
 'recovery_reference_step':'reference_step','recovery_reference_audit':'reference_audit',
 'recovery_damage':'fault_severity','recovery_v3_20261023':'control_v3_20261023',
 'recovery_path_progress':'path_progress','recovery_layout_template':'layout_template',
 'recovery_validation':'validation','recovery_loops':'loop_tests','recovery_tuning':'tuning',
 'recovery_overview':'flight_overview','hex_recovery':'hex_evtol',
 'MY_SIMULATION_RECOVERY':'MY_SIMULATION_REPORT','RECOVERY_':'SIMULATION_',
 'Recovery Progress Log':'Path Progress Log','Recovery Progress':'Path Progress',
 'Recovery Feedback':'State Feedback',
}
ids.update({k.upper():v.upper() for k,v in list(ids.items()) if k.upper()!=k})
token=re.compile('|'.join(re.escape(k) for k in sorted(ids,key=len,reverse=True)))
def convert(s):
    # Process each old identifier once, avoiding cascading helper/factory renames.
    s=token.sub(lambda m:ids[m[0]],s)
    s=re.sub(r'(?<=\.)recovery\b','fault_response',s)
    s=re.sub(r'"recovery"(?=\s*:)','"fault_response"',s)
    s=s.replace("'recovery'","'simulations'").replace('"output/recovery"','"output/simulations"')
    s=s.replace('output/recovery/','output/simulations/').replace('output\\recovery\\','output\\simulations\\')
    s=s.replace('recovery_', 'return_')
    s=re.sub(r'\brecovery\b','control',s)
    s=s.replace('Recovery','Control')
    return s

staged=subprocess.check_output(['git','diff','--cached','--name-status','--find-renames=100%'],cwd=ROOT,text=True)
completed={tuple(line.split('\t')[1:]) for line in staged.splitlines() if line.startswith('R100\t')}
for old,new in renames:
    if (old,new) in completed:continue
    source=(ROOT/old).resolve();destination=(ROOT/new).resolve()
    assert source.is_relative_to(ROOT) and destination.is_relative_to(ROOT)
    if destination.is_file():
        previous=subprocess.check_output(['git','rev-parse',(HERE/'source_commit.txt').read_text().strip()+':'+old],cwd=ROOT,text=True).strip()
        payload=destination.read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
        if actual==previous:continue
    assert source.is_file() and not destination.exists(),(old,new)
    subprocess.run(['git','mv','--',old,new],cwd=ROOT,check=True)

changed=[];frozen=[]
for p in sorted((ROOT/'simulink').rglob('*')):
    if not p.is_file() or 'output' in p.relative_to(ROOT/'simulink').parts or p.suffix not in ['.m','.py','.md','.json','.txt']:continue
    # Failed/initial comparisons preserve original bytes and original provenance.
    if p.suffix=='.json' and ('_INITIAL' in p.name or '_FAILED_' in p.name):
        frozen.append(p.relative_to(ROOT).as_posix());continue
    content=p.read_bytes();bom=content.startswith(b'\xef\xbb\xbf')
    old=content.decode('utf-8-sig')
    new=convert(old)
    if p.name=='hex_metrics.m':new=re.sub(r'\bcontrol\b(?=\s*=|\s*<=|\s*;|\s*,|\s*\))','return_time',new)
    if new!=old:
        p.write_bytes((b'\xef\xbb\xbf' if bom else b'')+new.encode('utf-8'))
        changed.append(p.relative_to(ROOT).as_posix())

for name in ['README.md','CURRENT_DESIGN.md']:
    p=ROOT/name;old=p.read_text(encoding='utf-8')
    # Git branch names and archive URLs are provenance, not model identifiers.
    branch='codex/current-design-and-recovery';placeholder='FROZEN_CURRENT_BRANCH'
    new=convert(old.replace(branch,placeholder)).replace(placeholder,branch)
    p.write_text(new,encoding='utf-8')
p=ROOT/'PROJECT_HANDOFF.md';old=p.read_text(encoding='utf-8');split=old.index('## 2026-10-25')
p.write_text(convert(old[:split])+old[split:],encoding='utf-8')

report={'previous_commit':(HERE/'source_commit.txt').read_text().strip(),'files':renames,
        'identifiers':ids,'modified_text_files':changed,'frozen_historical_reports':frozen,
        'field_names':{'recovery':'fault_response','recovery_reference_audit':'reference_audit','recovery_damage':'fault_severity'},
        'new_profile':'control_v3_20261023','old_profile':'recovery_v3_20261023'}
(HERE/'mapping.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'renamed_files':len(renames),'updated_text_files':len(changed),'historical_reports_preserved':len(frozen)}))
