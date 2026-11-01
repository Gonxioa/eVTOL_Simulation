"""Preserve serialized training artifacts locally and omit them from public ZIPs."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json,os,shutil,subprocess,urllib.request
from archive_20261030 import ROOT,AUDIT,STAGING,MODEL_SUFFIXES,sha,save,emit

manifest=json.loads((AUDIT/'manifest.json').read_text(encoding='utf-8'))
omitted=[f for f in manifest['files'] if Path(f['path']).suffix.lower() in MODEL_SUFFIXES]
assert omitted
protected=ROOT/'模型'/'retained_training_artifacts_20261030'
retained=[]
for item in omitted:
    destination=protected/item['path'];destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(item['source'],destination)
    assert sha(destination)==item['sha256']
    retained.append(dict(item,retained_path=str(destination)))
save('retained_checkpoints.json',{'protected_directory':str(protected),'count':len(retained),'files':retained})
omitted_paths={f['path'] for f in omitted}
affected=sorted({f['asset'] for f in omitted})
for name in affected:
    source=STAGING/name;temporary=STAGING/(name+'.sanitized')
    with ZipFile(source) as old,ZipFile(temporary,'w',compression=ZIP_DEFLATED,compresslevel=3,allowZip64=True) as new:
        for entry in old.infolist():
            if entry.filename in omitted_paths:continue
            with old.open(entry) as f,new.open(entry,'w',force_zip64=True) as target:
                shutil.copyfileobj(f,target,1024*1024)
    with ZipFile(temporary) as z:assert z.testzip() is None
    os.replace(temporary,source)
    for item in manifest['assets']:
        if item['name']==name:item.update(size=source.stat().st_size,sha256=sha(source))
manifest['files']=[f for f in manifest['files'] if f['path'] not in omitted_paths]
manifest['excluded_serialized_training_suffixes']=sorted(MODEL_SUFFIXES)
manifest['retained_local_training_artifacts']=len(retained)
save('manifest.json',manifest)
(STAGING/'archive-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
# Remove only assets created by this archive operation that must be replaced.
p=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,check=True)
token=dict(x.split('=',1) for x in p.stdout.splitlines() if '=' in x)['password']
headers={'Authorization':'Bearer '+token,'User-Agent':'eVTOL-version-archive','Accept':'application/vnd.github+json'}
base='https://api.github.com/repos/Gonxioa/eVTOL_Simulation'
release=json.loads((AUDIT/'release.json').read_text(encoding='utf-8'))
with urllib.request.urlopen(urllib.request.Request(base+f"/releases/{release['id']}/assets?per_page=100",headers=headers),timeout=30) as response:remote=json.load(response)
for item in remote:
    if item['name'] in affected+['archive-manifest.json']:
        with urllib.request.urlopen(urllib.request.Request(base+'/releases/assets/'+str(item['id']),method='DELETE',headers=headers),timeout=30) as response:assert response.status==204
save('verified_assets.json',{'release_url':release['html_url'],'verified':[],'complete':False,'replacement_assets':affected})
emit(stage='training_artifacts_retained_locally',count=len(retained),affected_assets=affected,public_files=len(manifest['files']))
