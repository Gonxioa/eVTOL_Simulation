"""Version-4 research dataset, explicit early termination and integrity gates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from provenance import write_json,stable_hash,sha256_file
from .configuration import ROOT,rotorpy_fields
from .simulation import simulate,FaultSchedule,export,fingerprint

VERSION='hex_v4_research_cascade'

def generate(cfg,out,groups=2,duration=6.):
    from scenario_config import SCENARIOS,serializable_scenarios,effectiveness_table
    gate=json.loads((out/'normal_validation.json').read_text(encoding='utf-8'))
    if not gate['base_research_normal_gate'] or gate['config_hash']!=stable_hash(cfg): raise RuntimeError('Matching normal research gate has not passed')
    if groups<2 or duration<4: raise ValueError('Interface dataset needs at least 2 groups and duration >=4s')
    folder=out/'dataset_v4'; folder.mkdir(exist_ok=True)
    plan={'dataset_version':VERSION,'config':cfg,'groups':groups,'duration_s':duration,'fault_time_s':3.,
          'window_policy':{'start_s':3.05,'end_s':3.65,'window':50,'step':50,'status':'predeclared_common_early_transient_interface_window'},
          'feedback_mode':'observed_feedback','noise':True,'wind':True,
          'fingerprint':fingerprint(cfg)['code_hashes'],'scenarios':serializable_scenarios()}
    plan['plan_signature']=stable_hash(plan)
    previous=folder/'generation_plan.json'
    if previous.exists() and json.loads(previous.read_text(encoding='utf-8'))!=plan: raise RuntimeError('Incompatible generation plan; use a new output directory')
    write_json(previous,plan)
    summaries=[]; hashes={}
    for group in range(1,groups+1):
        for scenario in SCENARIOS:
            path=folder/f'{scenario.name}__run{group:03d}.csv'
            # Resume only exact plan AND hashed CSV/sidecar, never partial writes.
            seed=cfg['seed']+group*10000+scenario.label*31
            if path.exists():
                side_path=path.with_suffix('.json')
                if not side_path.exists(): raise RuntimeError('CSV has no completion sidecar')
                side=json.loads(side_path.read_text(encoding='utf-8'))
                if side.get('plan_signature')!=plan['plan_signature'] or side['csv_sha256']!=sha256_file(path): raise RuntimeError('Resume signature or integrity mismatch')
            else:
                schedule=FaultSchedule(scenario.fault_motor,scenario.simulated_factor,3.) if scenario.label else FaultSchedule()
                run=simulate(cfg,duration,schedule=schedule,seed=seed,feedback='observed_feedback',noise=True,wind=True)
                run.update(schedule=schedule,matching_fault_time=3.)
                df,side=export(run,path,scenario,group); side['plan_signature']=plan['plan_signature']; write_json(path.with_suffix('.json'),side)
            hashes[path.name]=sha256_file(path); summaries.append({'file':path.name,'label':scenario.label,'group':group,'termination':side['termination'],'end_s':side['actual_duration_s']})
            print('dataset',path.name,side['termination'],side['actual_duration_s'],flush=True)
    fields=rotorpy_fields(cfg)
    manifest={'dataset_version':VERSION,'model_version':cfg['model_version'],'plan_signature':plan['plan_signature'],'fault_time_s':3.,
              'config_hash':stable_hash(cfg),'physics_parameters':{'mass':fields['mass'],'k_eta':fields['k_eta'],'num_rotors':6,'gravity':cfg['airframe']['gravity_m_s2'],'parameter_status':cfg['parameter_status']},
              'label_effectiveness':effectiveness_table(),'scenarios':serializable_scenarios(),'csv_sha256':hashes,
              'sidecar_sha256':{Path(n).with_suffix('.json').name:sha256_file(folder/Path(n).with_suffix('.json')) for n in hashes},
              'analysis_interval':plan['window_policy'],'formal_protocol':False,'purpose':'interface_smoke_not_accuracy_claim',
              'runs':summaries,'normal_gate':gate['base_research_normal_gate']}
    write_json(folder/'dataset_generation_manifest.json',manifest)
    validate_dataset(folder)
    return manifest

def validate_dataset(folder):
    folder=Path(folder); manifest=json.loads((folder/'dataset_generation_manifest.json').read_text(encoding='utf-8'))
    if manifest['dataset_version']=='hex_v5_continuous_tasks':
        from .continuous_data import validate
        return validate(folder)
    if manifest['dataset_version']!=VERSION: raise RuntimeError('Unsupported dataset version')
    if set(manifest['csv_sha256'])!={p.name for p in folder.glob('*.csv')}: raise RuntimeError('CSV inventory mismatch')
    statuses=[]
    for name,digest in manifest['csv_sha256'].items():
        path=folder/name; side_name=path.with_suffix('.json').name
        if sha256_file(path)!=digest or sha256_file(folder/side_name)!=manifest['sidecar_sha256'][side_name]: raise RuntimeError('Dataset hash mismatch')
        side=json.loads(path.with_suffix('.json').read_text(encoding='utf-8')); df=pd.read_csv(path)
        if len(df)!=side['rows'] or not np.isfinite(df.select_dtypes(include=np.number).drop(columns=['fault_time'],errors='ignore').to_numpy()).all(): raise RuntimeError('Data rows/nonfinite values')
        t=df['time'].to_numpy(); rate=side['config']['timing']['export_hz']
        if np.any(np.diff(t)<=0) or not np.allclose(np.diff(t),1/rate,atol=1e-9): raise RuntimeError('Export clock mismatch')
        if side['termination']=='completed' and abs(t[-1]-side['requested_duration_s'])>1e-9: raise RuntimeError('False completion')
        if side['provenance']['config_hash']!=manifest['config_hash'] or df['config_hash'].nunique()!=1 or df['config_hash'].iloc[0]!=manifest['config_hash']: raise RuntimeError('Mixed physical configuration')
        cmd=df[[f'cmd_motor{i}' for i in range(6)]].to_numpy(); omega=df[[f'truth_rotor{i}' for i in range(6)]].to_numpy()
        maximum=side['config']['propulsion']['omega_max_rad_s']
        if np.any(cmd<0) or np.any(cmd>maximum+1e-8) or np.any(omega<0) or np.any(omega>maximum+1e-8): raise RuntimeError('Physical speed bounds violated')
        kT=side['config']['propulsion']['kT']; actual=kT*omega**2
        if not np.allclose(actual.sum(axis=1),df['dynamic_propulsive_thrust'],atol=1e-8,rtol=1e-10): raise RuntimeError('Actual dynamic thrust is not from actual rotor states')
        eta=df[[f'effectiveness{i}' for i in range(6)]].to_numpy()
        if not np.allclose((eta*kT*cmd**2).sum(axis=1),df['predicted_plant_thrust'],atol=1e-8,rtol=1e-10): raise RuntimeError('Steady command prediction mismatch')
        statuses.append({'file':name,'status':'PASS' if side['termination']=='completed' else 'EXPLICIT_EARLY_STOP','termination':side['termination'],'rows':len(df),'end_s':float(t[-1])})
    write_json(folder/'integrity_report.json',{'all_integrity_passed':True,'runs':statuses,'early_stop_count':sum(r['status']=='EXPLICIT_EARLY_STOP' for r in statuses),'formal_training_approved':False})
    print('dataset integrity:',len(statuses),'sources checked; early stops retained.',flush=True)
    return statuses
