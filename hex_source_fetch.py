from pathlib import Path
import urllib.request, hashlib, json
from concurrent.futures import ThreadPoolExecutor

root = Path(__file__).resolve().parent
out = root/'output_hex_audit/source_snapshots'; out.mkdir(parents=True,exist_ok=True)
urls = {
 'rflymad_v2.html':'https://arxiv.org/html/2311.11340v2',
 'rflymad_dataset.html':'https://rfly-openha.github.io/documents/4_resources/dataset.html',
 'px4_112_control.html':'https://docs.px4.io/v1.12/en/flight_stack/controller_diagrams',
 'px4_modern_allocation.html':'https://docs.px4.io/main/en/concept/control_allocation.html',
 'PositionControl.cpp':'https://raw.githubusercontent.com/PX4/PX4-Autopilot/v1.12.3/src/modules/mc_pos_control/PositionControl/PositionControl.cpp',
 'AttitudeControl.cpp':'https://raw.githubusercontent.com/PX4/PX4-Autopilot/v1.12.3/src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp',
 'RateControl.cpp':'https://raw.githubusercontent.com/PX4/PX4-Autopilot/v1.12.3/src/modules/mc_rate_control/RateControl/RateControl.cpp',
 'px4_1123_ref.json':'https://api.github.com/repos/PX4/PX4-Autopilot/git/ref/tags/v1.12.3',
 'hummingbird_params.py':'https://raw.githubusercontent.com/spencerfolk/rotorpy/v2.1.2/rotorpy/vehicles/hummingbird_params.py',
 'rotorpy_simulate.py':'https://raw.githubusercontent.com/spencerfolk/rotorpy/v2.1.2/rotorpy/simulate.py',
 'rotorpy_environments.py':'https://raw.githubusercontent.com/spencerfolk/rotorpy/v2.1.2/rotorpy/environments.py',
 'rotorpy_LICENSE.md':'https://raw.githubusercontent.com/spencerfolk/rotorpy/v2.1.2/LICENSE.md',
 'official_multirotor.py':'https://raw.githubusercontent.com/spencerfolk/rotorpy/v2.1.2/rotorpy/vehicles/multirotor.py',
 'official_quadrotor_control.py':'https://raw.githubusercontent.com/spencerfolk/rotorpy/v2.1.2/rotorpy/controllers/quadrotor_control.py',
 'giribet_2018.pdf':'https://www.ijeetc.com/uploadfile/2018/0612/20180612065846780.pdf',
 'px4_tag_object.json':'https://api.github.com/repos/PX4/PX4-Autopilot/git/tags/3b967f543d11c8ae8fff9687c3de8d37744799fd',
 'rotorpy_paper.html':'https://arxiv.org/html/2306.04485',
 'giribet_2018.html':'https://doi.org/10.18178/ijeetc.7.2.58-65',
}
def fetch(pair):
 name,url=pair
 row={'name':name,'url':url,'access_date':'2026-10-06'}
 try:
  request=urllib.request.Request(url,headers={'User-Agent':'research-model-audit'})
  with urllib.request.urlopen(request,timeout=25) as r:
   data=r.read(); row.update(status=r.status,bytes=len(data),final_url=r.url,sha256=hashlib.sha256(data).hexdigest())
  (out/name).write_bytes(data)
 except Exception as e: row['error']=str(e)
 return row
with ThreadPoolExecutor(max_workers=5) as pool: rows=list(pool.map(fetch,urls.items()))
(root/'docs/hex_audit/source_fetch.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
for r in rows: print(r['name'],r.get('status',r.get('error')),r.get('bytes'))
