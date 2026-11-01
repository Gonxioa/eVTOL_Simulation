"""Archive project versions/results to GitHub Releases, verifying server SHA-256.

Credentials come from Git Credential Manager in memory and are never logged.
This script never removes source files. Local cleanup is a separate operation.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import urllib.request, urllib.error, urllib.parse

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'maintenance'/'archive_20261030'
STAGING=Path(os.environ['LOCALAPPDATA'])/'Temp'/'evtol_archive_20261030'
WORKTREE=Path(r'C:\Users\lenovo\.codex\worktrees\hex-audit-control\eVTOL_Simulation')
REPO='Gonxioa/eVTOL_Simulation'
TAG='archive-20261030'
EXCLUDED={'.git','__pycache__','slprj','.pytest_cache','document_format_20261028','document_review_20261028'}
MODEL_SUFFIXES={'.pt','.pth','.ckpt','.onnx','.h5','.hdf5','.joblib','.pkl','.pickle'}
sys.stdout.reconfigure(encoding='utf-8')

def emit(**v):print(json.dumps(v,ensure_ascii=False),flush=True)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(name,value):(AUDIT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def enumerate_files(source):
    if source.is_file():yield source;return
    for current,dirs,names in os.walk(source,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDED and not (Path(current)/d).is_symlink())
        for name in sorted(names):
            p=Path(current)/name
            if p.is_symlink() or name.endswith(('.pyc','.slxc')) or p.suffix.lower() in MODEL_SUFFIXES or name=='.git':continue
            yield p

def pack():
    AUDIT.mkdir(parents=True,exist_ok=True);STAGING.mkdir(parents=True,exist_ok=True)
    sources=[(ROOT/'simulink/output','project/simulink/output'),
             (ROOT/'plots','project/plots'),(ROOT/'runs','project/runs'),
             (ROOT/'output_hex_audit_baseline','project/output_hex_audit_baseline'),
             (ROOT/'output','project/output'),
             (ROOT/'simulink.zip','project/simulink.zip'),
             (ROOT/'simulink/hex_evtol.slx.bak','project/simulink/hex_evtol.slx.bak'),
             (WORKTREE,'old_worktree')]
    files=[]
    for source,prefix in sources:
        if not source.exists():continue
        for p in enumerate_files(source):
            relative=prefix if source.is_file() else prefix+'/'+p.relative_to(source).as_posix()
            files.append({'source':str(p),'path':relative,'size':p.stat().st_size})
    save('sources.json',{'date':'2026-10-30','repository':REPO,'tag':TAG,'files':files})
    assets=[];part=[];total=0;index=1;processed=0
    def flush(batch,number):
        nonlocal processed
        target=STAGING/f'evtol-versions-results-{number:02d}.zip'
        with ZipFile(target,'w',compression=ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
            for item in batch:
                p=Path(item['source']);before=p.stat()
                digest=hashlib.sha256()
                with p.open('rb') as src,z.open(item['path'],'w',force_zip64=True) as dst:
                    for block in iter(lambda:src.read(1024*1024),b''):
                        digest.update(block);dst.write(block)
                after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),str(p)
                item['sha256']=digest.hexdigest();item['asset']=target.name
                processed+=1
        with ZipFile(target) as z:assert z.testzip() is None
        assets.append({'name':target.name,'size':target.stat().st_size,'sha256':sha(target)})
        emit(stage='packed',asset=target.name,MiB=round(target.stat().st_size/1048576,2),files_done=processed,files_total=len(files))
    for item in files:
        if part and total+item['size']>256*1024*1024:
            flush(part,index);part=[];total=0;index+=1
        part.append(item);total+=item['size']
    if part:flush(part,index)
    bundle=STAGING/'all-code-history.bundle'
    subprocess.run(['git','bundle','create',str(bundle),'--all'],cwd=ROOT,check=True)
    subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    assets.append({'name':bundle.name,'size':bundle.stat().st_size,'sha256':sha(bundle)})
    manifest={'date':'2026-10-30','repository':REPO,'tag':TAG,
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'exclusions':sorted(EXCLUDED),'assets':assets,'files':files}
    save('manifest.json',manifest)
    (STAGING/'archive-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    emit(stage='pack_complete',assets=len(assets)+1,files=len(files),original_bytes=sum(f['size'] for f in files),upload_bytes=sum(a['size'] for a in assets))

def credentials():
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0')
    process=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,timeout=30,env=env)
    fields=dict(line.split('=',1) for line in process.stdout.splitlines() if '=' in line)
    secret=fields.get('password');assert secret,'GitHub credential unavailable.'
    return secret

def upload():
    token=credentials()
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'eVTOL-version-archive','X-GitHub-Api-Version':'2022-11-28'}
    def api(path,method='GET',body=None):
        data=None if body is None else json.dumps(body).encode()
        request=urllib.request.Request('https://api.github.com/repos/'+REPO+path,data=data,method=method,headers=dict(headers,**({'Content-Type':'application/json'} if data else {})))
        with urllib.request.urlopen(request,timeout=60) as response:
            raw=response.read()
            return json.loads(raw) if raw else None
    body='整理前所有本地代码分支已推送；附件保存历史与当前仿真输出、旧工作目录及完整 Git bundle。\n\narchive-manifest.json 记录逐文件 SHA-256、归档分卷和原路径。仿真结果按路径解压恢复。原始数据集、训练权重和个人文档不包含在公开归档中。\n\n仅在服务器返回的附件 SHA-256 与本地一致后，才执行本地历史版本清理。'
    try:release=api('/releases/tags/'+TAG)
    except urllib.error.HTTPError as exc:
        if exc.code!=404:raise
        release=api('/releases','POST',{'tag_name':TAG,'name':'2026-10-30 全版本代码与仿真结果归档','body':body,'draft':False,'prerelease':False,'make_latest':'false'})
    save('release.json',{'id':release['id'],'html_url':release['html_url'],'tag_name':TAG})
    manifest=json.loads((AUDIT/'manifest.json').read_text(encoding='utf-8'))
    assets=manifest['assets']+[{'name':'archive-manifest.json','size':(STAGING/'archive-manifest.json').stat().st_size,'sha256':sha(STAGING/'archive-manifest.json')}]
    existing={a['name']:a for a in api(f"/releases/{release['id']}/assets?per_page=100")}
    verified=[];verification_lock=Lock()
    def upload_one(item):
        saved=existing.get(item['name'])
        expected='sha256:'+item['sha256']
        if saved is not None and saved.get('state') in ['new','starter'] and not saved.get('digest'):
            api('/releases/assets/'+str(saved['id']),'DELETE')
            saved=None
        if saved is None:
            target=STAGING/item['name'];assert sha(target)==item['sha256']
            url=release['upload_url'].split('{')[0]+'?name='+urllib.parse.quote(item['name'])
            emit(stage='upload_start',asset=item['name'],MiB=round(item['size']/1048576,2))
            # Curl handles the proxy's streamed-upload connection reliably.
            # Feed authentication only over stdin; never put a token in argv/files.
            config=''.join('header = "'+key+': '+val+'"\n' for key,val in dict(headers,**{'Content-Type':'application/octet-stream'}).items())
            proxy=urllib.request.getproxies().get('https','')
            command=['curl.exe','--silent','--show-error','--fail-with-body','--http1.1',
                     '--connect-timeout','15','--max-time','1800','--request','POST',
                     '--upload-file',str(target),'--url',url,'--config','-']
            if proxy:command+=['--proxy',proxy]
            uploaded=subprocess.run(command,input=config.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1810)
            if uploaded.returncode:
                raise RuntimeError('Upload transport failed: '+uploaded.stderr.decode('utf-8',errors='replace')[:300]+' '+uploaded.stdout.decode('utf-8',errors='replace')[:300])
            saved=json.loads(uploaded.stdout)
        assert saved.get('size')==item['size'],('size mismatch',item['name'])
        assert saved.get('digest')==expected,('server SHA-256 mismatch or unavailable',item['name'],saved.get('digest'))
        with verification_lock:
            verified.append(dict(item,id=saved['id'],download_url=saved['browser_download_url'],server_digest=saved['digest']))
            verified.sort(key=lambda v:v['name'])
            save('verified_assets.json',{'release_url':release['html_url'],'verified':verified,'complete':False})
            emit(stage='asset_verified',asset=item['name'],verified=len(verified),total=len(assets))
    errors=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(upload_one,item):item['name'] for item in assets}
        for future in as_completed(futures):
            try:future.result()
            except Exception as exc:
                errors.append({'asset':futures[future],'error':str(exc)})
                emit(stage='upload_error',**errors[-1])
    if errors:
        save('upload_errors.json',errors)
        raise RuntimeError('Some assets did not verify; resume upload to retry. No source files were deleted.')
    # Re-query the server rather than trusting only the upload response.
    remote={a['name']:a for a in api(f"/releases/{release['id']}/assets?per_page=100")}
    for item in verified:assert remote[item['name']]['digest']==item['server_digest'] and remote[item['name']]['state']=='uploaded'
    save('verified_assets.json',{'release_url':release['html_url'],'verified':verified,'complete':True,'server_requeried':True})
    emit(stage='all_remote_assets_verified',release_url=release['html_url'],assets=len(verified))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pack','upload']);args=parser.parse_args()
    {'pack':pack,'upload':upload}[args.stage]()
