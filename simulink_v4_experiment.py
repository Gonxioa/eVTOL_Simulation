"""Registered batch-A selection; every selection uses real validation groups only."""
import pickle
from pathlib import Path
import numpy as np

from provenance import stable_hash, sha256_file, write_json, runtime_info
from simulink_data import read, ROOT
from simulink_v3_data import immutable_json, array_hash
from simulink_v4_data import root, pool, snapshot, registry, actual_source_audit
from simulink_v4_training import train, aggregate, rank_candidates, statistical_job, record_at


def decision(config, name, payload):
    result=dict(decision_id=name,config_signature=stable_hash(config),**payload)
    result['signature']=stable_hash(result)
    immutable_json(root(config)/'decisions'/f'{name}.json',result)
    print('V4_DECISION',name,result.get('selected'),flush=True)
    return result


def memorization_pool(config, reg):
    p=pool(config,reg,'train',100,24,'native'); rng=np.random.default_rng(config['memorization_seed'])
    chosen=np.sort(np.concatenate([rng.choice(np.flatnonzero(p['y']==c),config['memorization_windows_per_class'],replace=False) for c in range(19)]))
    result={k:(v[chosen] if isinstance(v,np.ndarray) and len(v)==len(p['y']) else v) for k,v in p.items()}
    result['identity']=dict(p['identity'],memorization_subset=True,selection_seed=config['memorization_seed'],parent_pool=p['signature'])
    result['signature']=stable_hash(dict(identity=result['identity'],indices=chosen.tolist()))
    result['X_sha256']=array_hash(result['X']); result['y_sha256']=array_hash(result['y'])
    selection=[]
    for source in sorted(set(result['source_ids'])):
        mask=result['source_ids']==source
        selection.append(dict(source_id=str(source),group=int(result['groups'][mask][0]),label=int(result['y'][mask][0]),
                              starts=result['starts'][mask].tolist(),ends=result['ends'][mask].tolist(),count=int(mask.sum())))
    result['selection']=selection
    return result


def run_batch_A(config):
    snapshot(config); reg=registry(config); audit=actual_source_audit(config,reg)
    from simulink_v4_analysis import analyze
    if not (root(config)/'signal_analysis/analysis_receipt.json').exists(): analyze(config,reg)
    jobs={}; statistics=[]
    def add(spec):
        jobs[spec['signature']]=spec
        write_json(root(config)/'batch_A_progress.json',dict(jobs=list(jobs.values()),statistical=statistics))
    memory=memorization_pool(config,reg); validation=pool(config,reg,'validation',100,24,'native')
    memory_job=train(config,memory,validation,'lstm',config['memorization_seed'],config['memorization_max_steps'],method='memorization')
    add(memory_job)
    screen=[]; common_refs={}
    for window in config['window_candidates']:
        tp=pool(config,reg,'train',window,24,'common'); vp=pool(config,reg,'validation',window,24,'common')
        specs=[]
        for seed in config['training_seeds']:
            s=train(config,tp,vp,'lstm',seed,config['screen_steps']); specs.append(s); add(s)
        screen+=aggregate(specs,[config['screen_steps']]); common_refs[('lstm',window)]=specs
    chosen=rank_candidates(screen)[:config['max_selected_windows']]
    windows=[r['window'] for r in chosen]
    decision(config,'A_window_screen',dict(question='Which of four lengths is competitive at common endpoints and 1000 LSTM updates?',
        candidates=screen,selected=windows,validation_rule=config['ranking'],
        scope='Common endpoints 299+50j; training counts4794; not a proof of globally optimal history length'))
    budget_rows=[]
    for window in windows:
        tp=pool(config,reg,'train',window,24,'common'); vp=pool(config,reg,'validation',window,24,'common')
        for family in ('lstm','cnn'):
            specs=[]
            for seed in config['training_seeds']:
                s=train(config,tp,vp,family,seed,max(config['budget_candidates'])); specs.append(s); add(s)
            common_refs[(family,window)]=specs; budget_rows+=aggregate(specs,config['budget_candidates'])
        for seed in config['training_seeds']:
            statistics.append(statistical_job(config,tp,vp,seed))
    budget_choice=rank_candidates(budget_rows)[0]; steps=budget_choice['steps']
    decision(config,'A_budget_representation',dict(question='Do more optimization and one fixed CNN representation improve held-out recognition?',
        candidates=budget_rows,selected=budget_choice,validation_rule=config['ranking'],
        scope='Finite 1000/2000/4000 budgets; continuation is one trajectory, not independent repeats'))
    native_rows=[]; native_refs={}
    for window in windows:
        tp=pool(config,reg,'train',window,24,'native'); vp=pool(config,reg,'validation',window,24,'native')
        for family in ('lstm','cnn'):
            specs=[]
            for seed in config['training_seeds']:
                s=train(config,tp,vp,family,seed,steps); specs.append(s); add(s)
            native_refs[(family,window)]=specs; native_rows+=aggregate(specs,[steps])
    native_choice=rank_candidates(native_rows)[0]; window=native_choice['window']; family=native_choice['family']
    decision(config,'A_native_joint_review',dict(question='Does the chosen representation/window remain competitive when healthy data start at zero?',
        candidates=native_rows,selected=native_choice,validation_rule=config['ranking'],
        scope='All native pools rebuilt and trained from scratch; common-grid weights not silently reused'))
    prefix_rows=[]; prefixes={}
    vp=pool(config,reg,'validation',window,24,'native')
    for seconds in config['prefix_candidates_s']:
        tp=pool(config,reg,'train',window,seconds,'native'); specs=[]
        for seed in config['training_seeds']:
            s=train(config,tp,vp,family,seed,steps); specs.append(s); add(s)
        candidate=aggregate(specs,[steps])[0]; candidate['prefix_s']=seconds
        candidate['original_windows']=len(tp['y']); prefix_rows.append(candidate); prefixes[seconds]=specs
    prefix_choice=min(prefix_rows,key=lambda r:(-round(r['mean_macro_f1'],8),r['prefix_s']))
    prefix=prefix_choice['prefix_s']
    decision(config,'A_prefix_coverage',dict(question='Which 0-based prefix is the raw-data research reference under the fixed model budget?',
        candidates=prefix_rows,selected=prefix_choice,
        validation_rule='Mean three-seed common complete-24s validation argmax macro-F1, tie shorter prefix',
        scope='A prefix choice does not establish a GAN-specific optimal coverage'))
    short=min([r for r in prefix_rows if r['prefix_s'] in (6,12)],key=lambda r:(-round(r['mean_macro_f1'],8),r['prefix_s']))
    equal_rows=[]
    for sampling_seed in config['equal_sampling_seeds']:
        tp=pool(config,reg,'train',window,24,'native',sampling_seed,short['prefix_s']); specs=[]
        target=pool(config,reg,'train',window,short['prefix_s'],'native')
        if [(r['source_id'],r['count']) for r in tp['selection']] != [(r['source_id'],r['count']) for r in target['selection']]:
            raise AssertionError('Equal control does not match each source/class count')
        for seed in config['training_seeds']:
            s=train(config,tp,vp,family,seed,steps); specs.append(s); add(s)
        r=aggregate(specs,[steps])[0]; r.update(sampling_seed=sampling_seed,matched_prefix=short['prefix_s'],original_windows=len(tp['y']))
        equal_rows.append(r)
    decision(config,'A_equal_count_controls',dict(question='Does wider time coverage help at the same per-source sample count?',
        candidates=equal_rows,selected=None,validation_rule='Diagnostic crossed 3 draw seeds x 3 classifier seeds; never pick the best draw seed',
        scope='No new independent flight sources; equal controls do not replace the raw-prefix selection'))
    raw=prefixes[prefix]; teacher=[]; failure=[]
    for s in raw:
        m=record_at(s,steps)['validation']['prefix_24']; recalls=np.asarray(m['class_recall'])
        bad=np.flatnonzero(recalls<=0).tolist()
        h=m['healthy_recall']; partial=m['joint_accuracy_by_effectiveness']['0.8']
        if bad or h<=1/19 or partial<=1/19: failure.append(dict(seed=s['seed'],zero_recall_classes=bad,healthy_recall=h,partial_joint_accuracy=partial))
        teacher.append(dict(seed=s['seed'],class_recall=m['class_recall'],healthy_recall=h,partial_joint_accuracy=partial))
    gate=dict(passed=not failure,failures=failure,teachers=teacher,rule=config['teacher_gate'],
              meaning='Minimum non-collapse research gate, not generated-label certification')
    decision(config,'A_teacher_gate',dict(question='Is the raw auxiliary teacher non-collapsed enough to attempt exploratory G1?',
        selected=gate['passed'],evidence=gate,validation_rule=config['teacher_gate'],scope='If false, G1 is not executed; G0/ROS remain valid research controls'))
    if len(jobs)>config['max_batch_A_neural_trajectories'] or len(statistics)>config['max_batch_A_statistical_fits']:
        raise AssertionError('Registered batch-A task cap exceeded')
    result=dict(**runtime_info(),config_signature=stable_hash(config),registry_signature=reg['signature'],source_audit=audit,
        selected_window=window,selected_family=family,selected_steps=steps,selected_prefix_s=prefix,
        selected_pool=pool(config,reg,'train',window,prefix,'native')['signature'],raw_models=raw,teacher_gate=gate,
        screen=screen,budget_candidates=budget_rows,native_joint=native_rows,prefix_candidates=prefix_rows,equal_controls=equal_rows,
        neural_jobs=list(jobs.values()),statistical_jobs=statistics,memorization_job=memory_job,
        actual_neural_trajectories=len(jobs),historical_test_features_read=False,formal_confirmation_run=False)
    write_json(root(config)/'batch_A_complete.json',result)
    locked={}
    for s in raw:
        for p in [Path(s['directory'])/f'checkpoint_{steps}.pt',Path(s['scaler_directory'])/'scalers.pkl',Path(s['scaler_directory'])/'fit_sources.json']:
            locked[str(p)]=sha256_file(p)
    payload=dict(config_signature=stable_hash(config),registry_signature=reg['signature'],pool_signature=result['selected_pool'],
        window=window,family=family,steps=steps,prefix_s=prefix,teacher_gate=gate,locked_files=locked,
        basis='Three-seed raw-data validation ranking and identical real-information budget for raw/ROS/GAN')
    payload['signature']=stable_hash(payload); immutable_json(root(config)/'raw_baseline_freeze.json',payload)
    print('V4_BATCH_A_COMPLETE',window,family,steps,prefix,'teacher_gate',gate['passed'],flush=True)
    return result
