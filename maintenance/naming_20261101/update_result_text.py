from pathlib import Path
from decimal import Decimal
import hashlib,json,re,sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
mapping=json.loads((HERE/'mapping.json').read_text(encoding='utf-8'))
pairs=mapping['identifiers']
pattern=re.compile('|'.join(re.escape(k) for k in sorted(pairs,key=len,reverse=True)))
def convert(s):
    s=pattern.sub(lambda m:pairs[m[0]],s)
    s=re.sub(r'"recovery"(?=\s*:)','"fault_response"',s)
    s=s.replace('output/recovery/','output/simulations/')
    s=s.replace('output\\\\recovery\\\\','output\\\\simulations\\\\')
    s=s.replace('output\\recovery\\','output\\simulations\\')
    s=s.replace('recovery_','return_')
    s=re.sub(r'\brecovery\b','control',s).replace('Recovery','Control')
    return s
def numbers(v):
    if isinstance(v,dict):return tuple(x for item in v.values() for x in numbers(item))
    if isinstance(v,list):return tuple(x for item in v for x in numbers(item))
    if isinstance(v,(int,float,Decimal)) or v is None:return (v,)
    return ()
counts={'csv':0,'json':0,'text':0};payloads=[]
for top in ['simulations','validation','loop_tests']:
    for p in (ROOT/'simulink/output'/top).rglob('*'):
        if not p.is_file() or p.suffix.lower() not in ['.csv','.json','.md','.txt']:continue
        raw=p.read_bytes();bom=raw.startswith(b'\xef\xbb\xbf');body=raw[3:] if bom else raw
        if p.suffix=='.csv':
            header,sep,data=body.partition(b'\n')
            changed=convert(header.decode('utf-8')).encode('utf-8')+sep+data
            assert changed.partition(b'\n')[2]==data
            payloads.append({'path':str(p),'data_sha256':hashlib.sha256(data).hexdigest()})
            counts['csv']+=1
        else:
            old=body.decode('utf-8');new=convert(old);changed=new.encode('utf-8')
            if p.suffix=='.json':
                assert numbers(json.loads(old,parse_float=Decimal))==numbers(json.loads(new,parse_float=Decimal))
                counts['json']+=1
            else:counts['text']+=1
        if changed!=body:p.write_bytes((b'\xef\xbb\xbf' if bom else b'')+changed)
(HERE/'text_data_migration.json').write_text(json.dumps({'counts':counts,'numeric_csv_payloads':payloads,'numeric_json_values_preserved':True},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(counts))
