"""Predeclared, paired offline integral tuning. Fault truth goes only to plant.

Separate candidate files and actual traces preserve every trial and failure.
This runner does not modify active gains, old data or diagnostic checkpoints.
"""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'1')
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from provenance import stable_hash, sha256_file, write_json
from .configuration import ROOT, physical, matrix
from .commands import task_library, ParameterizedTask
from .continuous_data import initial_state
from .simulation import simulate, FaultSchedule
from .validation import metric

OUT=ROOT/'output_hex_rate_tuning_20261010'
BASE=ROOT/'docs/hex_rate_tuning/baseline_source/hex_research/configs/research.json'
SETTINGS=ROOT/'docs/hex_rate_tuning/baseline_source/hex_research/configs/continuous.json'
CONDITIONS=('ideal','wind','noise_wind','delay_noise_wind')


def candidates(base):
    trials=[('baseline',[1,1,.6],[3,3,2]),
            ('gain_only_rp',[10,10,.6],[3,3,2]),
            ('limit_only_rp',[1,1,.6],[20,20,2]),
            ('combined_rp10',[10,10,.6],[20,20,2]),
            ('combined_rp20',[20,20,.6],[20,20,2]),
            ('combined_rp30',[30,30,.6],[20,20,2]),
            ('user_probe',[10,10,6],[20,20,20]),
            ('yaw_gain6',[10,10,6],[20,20,2]),
            ('rp_limit12',[10,10,.6],[12,12,2])]
    result={}
    for name,gain,limit in trials:
        cfg=deepcopy(base);cfg['control']['rate_i']=gain;cfg['control']['rate_integral_limit_rad_s2']=limit
        result[name]=cfg
    return result


def plan(out):
    base=json.loads(BASE.read_text(encoding='utf-8'));settings=json.loads(SETTINGS.read_text(encoding='utf-8'))
    p=physical(base);A=matrix(p);hover=np.r_[p['mass']*p['gravity'],0,0,0]
    authority=[]
    for motor in range(6):
        eta=np.ones(6);eta[motor]=.8
        # Offline steady equilibrium for the nominal minimum-norm allocator.
        demand=np.linalg.solve(A@np.diag(eta)@np.linalg.pinv(A),hover)
        alpha=np.linalg.solve(p['inertia'],demand[1:])
        authority.append({'motor':motor,'nominal_compensation_wrench':demand.tolist(),
                          'rate_integral_equilibrium_rad_s2':alpha.tolist()})
    value={'schema':'rate_integral_design_v1','baseline_config':base,'settings':settings,
           'candidates':candidates(base),'criteria':settings['normal_acceptance'],
           'promotion_position_reserve_fraction':.10,
           'long_hold':{'hover_duration_s':30.,'task_duration_s':20.,'tail_s':5.,
                        'tail_position_rmse_m':.10,'tail_position_peak_m':.15,'tail_mean_speed_m_s':.05},
           'conditions':list(CONDITIONS),'pilot':'g001 M0 Partial, ideal and delay_noise_wind; all 9 candidates',
           'screen':'g001/g003/g005/g007, M0/M1, ideal/delay_noise_wind; shortlisted candidates',
           'full_validation':'8 groups x 6 motors x 4 conditions (192 Partial) plus 32 healthy',
           'long_validation':'2 hover groups x 6 motors x 4 conditions at 30s, plus 4 moving groups x 4 conditions at 20s (M0)',
           'controller_inputs':'state observations and task reference only; no fault truth or predictions',
           'authority_estimates':authority,
           'numerical_source_sha256':{name:sha256_file(ROOT/name) for name in
              ('hex_research/control.py','hex_research/simulation.py','hex_research/dynamics.py',
               'hex_research/math3d.py','hex_research/observation.py','hex_research/commands.py','control_allocation.py')},
           'status':'predeclared_research_design_not_real_aircraft_confirmation'}
    value['signature']=stable_hash(value)
    target=out/'predeclared_plan.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=value:
        raise RuntimeError('Plan/source changed; use a new output directory, preserve prior trials')
    if not target.exists():write_json(target,value)
    for name,cfg in value['candidates'].items():
        target=out/'configs'/f'{name}.json'
        if not target.exists():write_json(target,cfg)
    return value


def trace(path,run):
    keys=('rate_integral','rate_integral_applied','velocity_integral','velocity_integral_applied',
          'position_error','velocity_error','rate_error','rate_reference','observed_rate','observed_velocity',
          'rate_integral_limit_active','velocity_integral_limit_active','rate_integral_blocked_nominal',
          'rate_integral_clipped_step','velocity_integral_clipped_step','rate_integral_actual_step',
          'velocity_integral_actual_step','rate_integral_requested_step','rate_integral_allowed_step',
          'velocity_integral_feedback_step','velocity_integral_antiwindup_step')
    rows=[]
    for r in run['rows']:
        row={'time':r['time'],'feedback_timestamp':r['feedback_timestamp'],
             'attitude_error_angle_rad':r['control']['attitude_error_angle_rad']}
        for j,a in enumerate('xyz'):
            for prefix,values in [('position',r['state'][:3]),('velocity',r['state'][3:6]),
                                  ('body_rate',r['state'][10:13]),('reference',r['reference']['x'])]:
                row[f'{prefix}_{a}']=float(values[j])
            for key in keys:row[f'{key}_{a}']=float(r['control'][key][j])
        for m in range(6):
            row[f'command{m}']=float(r['control']['cmd_motor_speeds'][m]);row[f'rotor{m}']=float(r['state'][13+m])
            row[f'effectiveness{m}']=float(r['effectiveness'][m])
            row[f'lower_bound{m}']=int(r['control']['allocator_lower_active'][m])
            row[f'upper_bound{m}']=int(r['control']['allocator_upper_active'][m])
        for j,a in enumerate(('thrust','roll','pitch','yaw')):row[f'nominal_residual_{a}']=float(r['control']['nominal_residual_wrench'][j])
        rows.append(row)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def evaluate(job):
    cfg=deepcopy(job['config']);g=deepcopy(job['group']);condition=job['condition'];duration=job['duration']
    if condition=='delay_noise_wind':cfg['sensor']['feedback_delay_s']=job['settings']['normal_validation_delay_s']
    g['task']['duration_s']=duration
    start=g['fault_time_s'] if job['motor']>=0 else None
    run=simulate(cfg,duration,schedule=FaultSchedule(job['motor'],.8 if start is not None else 1.,start),
                 initial=initial_state(cfg,g),task_provider=ParameterizedTask(g['task']),seed_streams=g['seed_streams'],
                 feedback='ideal_feedback' if condition in ('ideal','wind') else 'observed_feedback',
                 noise=condition in ('noise_wind','delay_noise_wind'),wind=condition!='ideal')
    m=metric(run);rows=run['rows'];times=np.array([r['time'] for r in rows])
    errors=np.array([r['state'][:3]-r['reference']['x'] for r in rows]);error=np.linalg.norm(errors,axis=1)
    speed=np.linalg.norm([r['state'][3:6] for r in rows],axis=1)
    post=times>=(start or 0);tail=times>=duration-job['long']['tail_s']
    if not np.any(tail):tail=np.arange(len(rows))==len(rows)-1
    criteria=job['criteria'];checks={'completed':m['termination']=='completed' and abs(m['actual_duration_s']-duration)<1e-9,
        'rmse':m['position_rmse_m']<=criteria['position_rmse_m'],
        'final_error':m['final_position_error_m']<=criteria['final_position_error_m'],
        'tilt':m['maximum_tilt_deg']<=criteria['maximum_tilt_deg'],
        'rate':m['maximum_rate_rad_s']<=criteria['maximum_body_rate_rad_s'],
        'saturation':m['nominal_bound_fraction']<=criteria['maximum_nominal_bound_fraction']}
    extra={'post_fault_position_rmse_m':float(np.sqrt(np.mean(error[post]**2))) if np.any(post) else None,
           'maximum_position_error_m':float(max(error)),'final_speed_m_s':float(speed[-1]),
           'tail_position_rmse_m':float(np.sqrt(np.mean(error[tail]**2))),
           'tail_position_peak_m':float(max(error[tail])),'tail_mean_speed_m_s':float(np.mean(speed[tail])),
           'maximum_attitude_error_deg':float(max(r['control']['attitude_error_angle_rad'] for r in rows)*180/np.pi)}
    for name in ('rate_integral','velocity_integral'):
        values=np.array([r['control'][name] for r in rows]);flags=np.array([r['control'][name+'_limit_active'] for r in rows])
        extra['maximum_abs_'+name]=np.max(abs(values),axis=0).tolist()
        extra[name+'_limit_fraction']=np.mean(flags,axis=0).tolist()
        extra[name+'_first_limit_time_s']=[float(times[np.flatnonzero(flags[:,j])[0]]) if np.any(flags[:,j]) else None for j in range(3)]
    extra['rate_integral_blocked_fraction']=np.mean([r['control']['rate_integral_blocked_nominal'] for r in rows],axis=0).tolist()
    if duration>8:
        for field in ('tail_position_rmse_m','tail_position_peak_m','tail_mean_speed_m_s'):
            checks[field]=extra[field]<=job['long'][field]
    path=Path(job['path']);trace(path.with_suffix('.csv'),run)
    result={'case_id':job['case_id'],'candidate':job['candidate'],'group_id':g['group_id'],
            'family':g['task']['family'],'motor':job['motor'],'condition':condition,'duration_s':duration,
            'config_hash':stable_hash(cfg),'plan_signature':job['signature'],'checks':checks,'passed':all(checks.values()),
            'failed_checks':[k for k,v in checks.items() if not v],**m,**extra,'trace_sha256':sha256_file(path.with_suffix('.csv'))}
    write_json(path,result)
    return result


def execute(plan,stage,names,out,workers):
    groups=task_library(plan['settings']);jobs=[];cached=[]
    if stage=='pilot':cases=[(groups[0],0,c,8.) for c in ('ideal','delay_noise_wind')]
    elif stage=='screen':cases=[(g,m,c,8.) for g in groups if g['group_id'] in (1,3,5,7) for m in (0,1) for c in ('ideal','delay_noise_wind')]
    elif stage=='validation':cases=[(g,m,c,8.) for g in groups for m in range(-1,6) for c in CONDITIONS]
    elif stage=='long':
        cases=[(g,m,c,30.) for g in groups if g['task']['family']=='hover' for m in range(6) for c in CONDITIONS]
        cases += [(g,0,c,20.) for g in groups if g['group_id'] in (3,4,5,6) for c in CONDITIONS]
    else:raise ValueError(stage)
    for name in names:
        cfg=plan['candidates'].get(name)
        if cfg is None:
            cfg=json.loads((out/'configs'/f'{name}.json').read_text(encoding='utf-8'))
        for group,motor,condition,duration in cases:
            case_id=f'g{group["group_id"]:03d}_m{motor}_{condition}_{duration:g}s'
            path=out/stage/name/(case_id+'.json')
            job={'config':cfg,'group':group,'condition':condition,'duration':duration,'motor':motor,
                 'criteria':plan['criteria'],'settings':plan['settings'],'long':plan['long_hold'],
                 'candidate':name,'case_id':case_id,'path':str(path),'signature':plan['signature']}
            if path.is_file():
                value=json.loads(path.read_text(encoding='utf-8'))
                expected=deepcopy(cfg)
                if condition=='delay_noise_wind':expected['sensor']['feedback_delay_s']=plan['settings']['normal_validation_delay_s']
                if value['plan_signature']!=plan['signature'] or value['config_hash']!=stable_hash(expected) or sha256_file(path.with_suffix('.csv'))!=value['trace_sha256']:
                    raise RuntimeError('Prior trial changed; do not overwrite '+str(path))
                cached.append(value)
            else:jobs.append(job)
    results=cached[:];total=len(jobs)+len(cached)
    print(f'{stage}: {len(cached)} cached, {len(jobs)} to run, workers={workers}',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(evaluate,job):job for job in jobs}
        for future in as_completed(futures):
            value=future.result();results.append(value)
            print(f'{stage} {len(results)}/{total} {value["candidate"]} {value["case_id"]} '
                  f'{"PASS" if value["passed"] else "FAIL"} rmse={value["position_rmse_m"]:.4f} final={value["final_position_error_m"]:.4f}',flush=True)
    results.sort(key=lambda r:(r['candidate'],r['group_id'],r['motor'],r['condition']))
    summary={}
    for name in names:
        subset=[r for r in results if r['candidate']==name]
        summary[name]={'cases':len(subset),'passed':sum(r['passed'] for r in subset),
                       'worst_rmse_m':max(r['position_rmse_m'] for r in subset),
                       'worst_final_error_m':max(r['final_position_error_m'] for r in subset),
                       'position_reserve_fraction':1-max(max(r['position_rmse_m']/plan['criteria']['position_rmse_m'],r['final_position_error_m']/plan['criteria']['final_position_error_m']) for r in subset),
                       'failed_cases':[r['case_id'] for r in subset if not r['passed']]}
    report={'executed_utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'plan_signature':plan['signature'],
            'runner_sha256':sha256_file(Path(__file__)),'summary':summary,'results':results}
    write_json(out/(stage+'_summary.json'),report)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('plan','pilot','screen','validation','long'))
    parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--candidates',nargs='+')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if not 1<=args.workers<=8:parser.error('workers must be 1…8')
    declared=plan(args.out)
    if args.stage=='plan':print(declared['signature']);return
    names=args.candidates or list(declared['candidates'])
    execute(declared,args.stage,names,args.out,args.workers)


if __name__=='__main__':main()
