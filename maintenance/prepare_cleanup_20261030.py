"""Read-only preflight. Produce an exact deletion list after remote verification."""
from pathlib import Path
from collections import Counter
import hashlib,json,os,subprocess,sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'maintenance/archive_20261030'
T=Path(os.environ['LOCALAPPDATA'])/'Temp'
W=Path(r'C:\Users\lenovo\.codex\worktrees\hex-audit-control\eVTOL_Simulation')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def files(p):
    if p.is_file():yield p;return
    for base,dirs,names in os.walk(p,followlinks=False):
        for d in dirs:assert not (Path(base)/d).is_symlink() and not (Path(base)/d).is_junction()
        for name in names:yield Path(base)/name

def cached(p):return any(x in ['__pycache__','slprj','.pytest_cache'] for x in p.parts) or p.name.endswith(('.pyc','.slxc')) or p.name=='.git'

def main():
    verified=json.loads((A/'verified_assets.json').read_text(encoding='utf-8'))
    verified_ok=bool(verified.get('complete') and verified.get('server_requeried'))
    assert verified_ok or '--draft' in sys.argv,'Archive upload is incomplete.'
    validation=json.loads((A/'latest_layout_validation.json').read_text(encoding='utf-8'))
    assert validation['passed'] and validation['main_workspace_model_modified']==False
    manifest=json.loads((A/'manifest.json').read_text(encoding='utf-8'))
    plan=json.loads((A/'cleanup_plan.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'simulink/hex_recovery.slx')==plan['main_model_sha256']
    expected={str(Path(x['source']).resolve()).casefold():x for x in manifest['files']}
    local_resources_file=A/'retained_checkpoints.json'
    local_resources=json.loads(local_resources_file.read_text(encoding='utf-8'))['files'] if local_resources_file.exists() else []
    retained={str(Path(x['source']).resolve()).casefold():x for x in local_resources}
    local=subprocess.check_output(['git','for-each-ref','--format=%(objectname) %(refname)','refs/heads'],cwd=ROOT,text=True).splitlines()
    remote=subprocess.check_output(['git','-c','http.proxy=http://127.0.0.1:18081','ls-remote','--heads','origin'],cwd=ROOT,text=True).splitlines()
    remote={line.split()[1]:line.split()[0] for line in remote}
    branch_snapshot=[]
    for line in local:
        commit,ref=line.split();assert remote.get(ref)==commit,('Unpublished branch',ref)
        branch_snapshot.append({'branch':ref.removeprefix('refs/heads/'),'commit':commit})
    (A/'remote_refs_verified.json').write_text(json.dumps(branch_snapshot,indent=2),encoding='utf-8')
    targets=[]
    def add(p,kind):
        if not p.exists():return
        p=p.resolve()
        assert p!=ROOT and p!=ROOT/'.git'
        if p.is_relative_to(ROOT):
            first=p.relative_to(ROOT).parts[0]
            assert first not in plan['protected_roots'] and first!='.git',str(p)
        else:assert p in [W,T/'evtol_archive_20261030',T/'evtol_current_verify_20261030'],str(p)
        targets.append({'path':str(p),'kind':kind,'files':0,'bytes':0})
    for rel in plan['legacy_code_files']+plan['legacy_code_directories']:add(ROOT/rel,'historical_code')
    output=ROOT/'simulink/output'
    keep={'recovery','recovery_validation','recovery_loops','kx_evidence_20261026','normalization_comparison_20261027','document_format_20261028','document_review_20261028'}
    for p in output.iterdir():
        if p.name not in keep:add(p,'cache' if p.name=='slprj' else 'historical_results')
    for p in (output/'recovery').iterdir():
        if p.name!='recovery_v3_20261023':add(p,'historical_results')
    for p in (output/'recovery_validation').iterdir():
        if p.name!='recovery_v3_20261023':add(p,'historical_results')
    add(output/'recovery/recovery_v3_20261023/healthy_four_tasks_20261027/before_axis_split','historical_results')
    for rel in ['plots','runs','output','output_hex_audit_baseline','simulink.zip','simulink/hex_evtol.slx.bak']:add(ROOT/rel,'historical_results')
    for rel in ['slprj','simulink/slprj','__pycache__','tests/__pycache__','uavfd_f550/__pycache__','uavfd_f550/tests/__pycache__']:
        add(ROOT/rel,'cache')
    for parent in [ROOT,ROOT/'simulink']:
        for p in parent.glob('*.slxc'):add(p,'cache')
    add(W,'old_worktree')
    for target in targets:
        p=Path(target['path'])
        for f in files(p):
            target['files']+=1;target['bytes']+=f.stat().st_size
            if target['kind']=='cache' or cached(f):continue
            if target['kind']=='historical_code':
                content=f.read_bytes();blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
                archived=subprocess.check_output(['git','rev-parse','archive-20261030:'+f.relative_to(ROOT).as_posix()],cwd=ROOT,text=True).strip()
                assert blob==archived,('Changed historical code',str(f))
            else:
                entry=expected.get(str(f.resolve()).casefold())
                if entry is not None:
                    assert entry['size']==f.stat().st_size and sha(f)==entry['sha256'],('File changed after archive',str(f))
                else:
                    resource=retained.get(str(f.resolve()).casefold())
                    assert resource is not None,('File was not archived or retained',str(f))
                    assert sha(f)==resource['sha256']==sha(Path(resource['retained_path'])),('Retained model mismatch',str(f))
    # Temporary artifacts belong to this operation and are removed last.
    for p in [T/'evtol_archive_20261030',T/'evtol_current_verify_20261030']:
        add(p,'operation_temporary_files')
    report={'root':str(ROOT),'archive_url':verified['release_url'],'remote_archives_verified':verified_ok,
            'source_model_sha256':plan['main_model_sha256'],'targets':targets,
            'source_files_to_remove':sum(x['files'] for x in targets),
            'source_bytes_to_remove':sum(x['bytes'] for x in targets),
            'counts_by_kind':dict(Counter(x['kind'] for x in targets))}
    filename='deletion_preflight.json' if verified_ok else 'deletion_preflight_draft.json'
    (A/filename).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='targets'},ensure_ascii=False))

if __name__=='__main__':main()
