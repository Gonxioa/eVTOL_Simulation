"""Generate and diagnose a controller version only after actual paired gates."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(variable,'1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from provenance import sha256_file, stable_hash, write_json
from hex_research.configuration import load
from hex_research.continuous_data import plan_for, initial_state, export_v5, generate
from hex_research.commands import ParameterizedTask
from hex_research.simulation import simulate, FaultSchedule
from hex_research.validation import metric


def one_record(job):
    from scenario_config import SCENARIOS
    cfg,g,label,signature,folder=job;scenario=SCENARIOS[label]
    path=Path(folder)/f'g{g["group_id"]:03d}_{g["task"]["family"]}_c{label:02d}.csv'
    if path.exists():
        side=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if side['plan_signature']!=signature or side['csv_sha256']!=sha256_file(path):
            raise RuntimeError('Prior record changed; preserve it: '+str(path))
        return {'record':side['record_id'],'cached':True,'termination':side['termination']}
    schedule=FaultSchedule(scenario.fault_motor,scenario.simulated_factor,g['fault_time_s']) if label else FaultSchedule()
    run=simulate(cfg,g['task']['duration_s'],schedule=schedule,seed=g['seed_streams']['sensor'],
                 seed_streams=g['seed_streams'],task_provider=ParameterizedTask(g['task']),
                 initial=initial_state(cfg,g),feedback='observed_feedback',noise=True,wind=True)
    run.update(schedule=schedule,matching_fault_time=g['fault_time_s'])
    frame,side=export_v5(run,path,scenario,g,signature)
    side['flight_metrics']=metric(run);write_json(path.with_suffix('.json'),side)
    return {'record':side['record_id'],'cached':False,'termination':side['termination']}


def check_gates(tuning,candidate):
    reports=[json.loads((tuning/(stage+'_summary.json')).read_text(encoding='utf-8')) for stage in ('validation','long')]
    selected=json.loads((tuning/'configs'/f'{candidate}.json').read_text(encoding='utf-8'))
    if load()!=selected:raise RuntimeError('Active configuration is not the validated selected version')
    counts=(224,64)
    for report,count in zip(reports,counts):
        summary=report['summary'][candidate]
        if summary['cases']!=count or summary['passed']!=count:raise RuntimeError('Actual validation/long gate failed')
    plan=json.loads((tuning/'predeclared_plan.json').read_text(encoding='utf-8'))
    for name,digest in plan['numerical_source_sha256'].items():
        if sha256_file(ROOT/name)!=digest:raise RuntimeError('Validated numerical source changed: '+name)
    for report in reports:
        if report['plan_signature']!=plan['signature']:raise RuntimeError('Gate used different predeclared criteria')
        for r in report['results']:
            if r['candidate']!=candidate:continue
            expected=deepcopy(selected)
            if r['condition']=='delay_noise_wind':expected['sensor']['feedback_delay_s']=plan['settings']['normal_validation_delay_s']
            if r['config_hash']!=stable_hash(expected):raise RuntimeError('Gate configuration mismatch: '+r['case_id'])
    if reports[0]['summary'][candidate]['position_reserve_fraction']<plan['promotion_position_reserve_fraction']:
        raise RuntimeError('Predeclared positional reserve was not met')
    normals=[r for r in reports[0]['results'] if r['candidate']==candidate and r['motor']==-1]
    if len(normals)!=32 or not all(r['passed'] for r in normals):raise RuntimeError('Need 32 actual healthy checks')
    return selected,plan['settings'],normals


def dataset(tuning,out,candidate,workers):
    cfg,settings,normals=check_gates(tuning,candidate)
    plan=plan_for(cfg,settings);out.mkdir(parents=True,exist_ok=True)
    folder=out/'dataset_v5';folder.mkdir(exist_ok=True);target=folder/'generation_plan.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=plan:
        raise RuntimeError('Use a fresh release output; preserve the previous source')
    write_json(target,plan)
    gate={'all_normal_passed':True,'cases':normals,'plan_signature':plan['plan_signature'],
          'config_hash':stable_hash(cfg),'acceptance':settings['normal_acceptance'],
          'real_aircraft_validation':False,'source_validation_summary_sha256':sha256_file(tuning/'validation_summary.json')}
    write_json(out/'task_normal_validation.json',gate)
    jobs=[(cfg,g,label,plan['plan_signature'],str(folder)) for g in plan['groups'] for label in range(19)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one_record,job) for job in jobs]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result();print(f'generated {i}/152 {result["record"]} {result["termination"]}',flush=True)
    # Use the existing writer/validator to inspect already generated sources,
    # construct its manifest and check paired prefixes. No second simulation.
    manifest=generate(cfg,settings,out)
    write_json(out/'release_generation_receipt.json',{'executed_utc':datetime.now(timezone.utc).isoformat(),
        'runner_sha256':sha256_file(__file__),'candidate':candidate,'workers':workers,
        'config_hash':stable_hash(cfg),'plan_signature':plan['plan_signature'],'records':len(manifest['runs']),
        'validation_summary_sha256':sha256_file(tuning/'validation_summary.json'),
        'long_summary_sha256':sha256_file(tuning/'long_summary.json'),'old_data_modified':False})


def diagnose(tuning,out,candidate):
    cfg,settings,_=check_gates(tuning,candidate)
    from diagnostics.preprocessing import prepare
    from diagnostics.workflow import fit_interface,evaluate_replay
    feature='imu_only';model=out/'diagnostics/imu_only_epoch30'
    if model.exists():raise RuntimeError('Keep existing model read-only; use a fresh model output')
    settings=deepcopy(settings);settings['interface_fit']['epochs']=30
    preprocessed=out/'preprocessed'/feature
    prepare(out/'dataset_v5',preprocessed,settings['diagnostics'])
    model.mkdir(parents=True);write_json(model/'predeclared_training.json',settings)
    fit=fit_interface(preprocessed,model,settings)
    report=evaluate_replay(out/'dataset_v5',preprocessed,model,settings)
    write_json(model/'training_receipt.json',{'executed_utc':datetime.now(timezone.utc).isoformat(),
              'runner_sha256':sha256_file(__file__),'config_hash':stable_hash(cfg),
              'feature_set':feature,'epochs':30,'formal_performance_claim':False,
              'train_groups':fit['train_groups'],'replay_groups':fit['replay_groups']})
    print(json.dumps({k:report[k] for k in ('fault_events','detected_before_stop','missed_evaluable_fault_events',
                   'normal_only_false_alarm_events','pure_window_macro_f1','all_replay_offline_equal')},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('dataset','diagnose'))
    parser.add_argument('--tuning',type=Path,default=ROOT/'output_hex_rate_tuning_20261010')
    parser.add_argument('--out',type=Path,default=ROOT/'output_hex_rate_tuning_20261010/release_v2')
    parser.add_argument('--candidate',default='velocity_z09');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('workers must be 1…8')
    if args.stage=='dataset':dataset(args.tuning,args.out,args.candidate,args.workers)
    else:diagnose(args.tuning,args.out,args.candidate)


if __name__=='__main__':main()
