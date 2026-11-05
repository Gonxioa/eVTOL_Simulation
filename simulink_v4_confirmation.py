"""Budget-planned unseen paired confirmation, immutable freeze and one test unlock."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import torch

from provenance import stable_hash,sha256_file,write_json,runtime_info
from simulink_data import ROOT,read,prepare_plan,checked_plan
from simulink_v3_data import immutable_json
from simulink_v4_data import root,registry,pool,authorize,signals,verified_hash,snapshot
from simulink_v4_training import load_model,predict,transform,evaluate,record_at
from simulink_v4_augmentation import prediction_records,alarm_metrics


def optimizer_steps(state):
    return sorted({int(v['step'].item() if hasattr(v['step'],'item') else v['step']) for v in state['state'].values()})


def planned_groups(differences,halfwidth,minimum,maximum):
    values=np.asarray(differences,float)
    if len(values)<2 or not np.isfinite(values).all() or halfwidth<=0 or minimum<1 or maximum<minimum:
        raise ValueError('Invalid group-level planning inputs')
    sd=float(np.std(values,ddof=1)); estimated=int(np.ceil((1.96*sd/halfwidth)**2))
    return sd,estimated,int(np.clip(estimated,minimum,maximum))


def decision_register(config):
    """Record the already registered choices and evidence before test access."""
    base=root(config); a,b,_=all_models(config); decisions=[]
    def row(name,question,candidates,selected,rule,evidence,limits,status='executed'):
        decisions.append(dict(name=name,question=question,candidates=candidates,selected=selected,validation_rule=rule,
            fixed_controls='12 measured channels; unchanged simulation, controller, split and physical fault; fixed seeds/budgets',
            evidence_sha256={str(base/p):sha256_file(base/p) for p in evidence},
            scope='Development training/validation only; no confirmation scores',limitations=limits,status=status))
    row('window','Which finite window gives the reference neural result?',config['window_candidates'],a['selected_window'],
        'Common endpoints, LSTM1000; retain2. Joint LSTM/CNN budget comparison; native-grid joint refit.',
        ['decisions/A_window_screen.json','decisions/A_native_joint_review.json'],
        'CNN only tested at retained windows; not a global window optimum. Different windows have different warm-up.')
    row('representation_budget','What can the fixed finite neural budget learn?',a['budget_candidates'],
        dict(family=a['selected_family'],steps=a['selected_steps']),
        'Mean pooled full24 validation19-class argmax macroF1; registered ties.',
        ['decisions/A_budget_representation.json'],'RF is a separate strong diagnostic;4000 steps is not a convergence certificate.')
    row('coverage','Which prefix is the common real-information budget?',a['prefix_candidates'],a['selected_prefix_s'],
        'Full24 validation19-class argmax macroF1, average seeds; ties prefer shorter prefix.',
        ['decisions/A_prefix_coverage.json','decisions/A_equal_count_controls.json'],
        'Equal-count controls are diagnostics, not selection candidates; coverage chosen before GAN, not jointly optimal.')
    row('teacher','May the exploratory external-teacher constraint run?',config['class_constraint_weights'],a['teacher_gate'],
        'All19 class recalls positive and healthy/80%-joint accuracy above1/19 in every raw seed.',
        ['decisions/A_teacher_gate.json'],'Minimum noncollapse prerequisite only; cannot certify synthetic labels.')
    row('GAN_checkpoint','Which common saved generation checkpoint?',config['generator_checkpoints'],b['selected_G_step'],
        'G0 independent TSTR, mean full24 validation macroF1; ties fewer Gupdates. G1 shares step.',
        ['decisions/B_G_checkpoint.json'],'Validation adapted; independent new test is the confirmation. Loss minimum is not the selector.')
    row('augmentation','Which per-class ratio and class weight?',dict(ratios=config['augmentation_ratios'],weights=config['class_constraint_weights']),
        dict(GAN=b['ratio_choices'],ROS=b['ROS_choice'],G1_weight=b['selected_G1_weight']),
        'Full24 validation macroF1 across3 paired joint repeats; ties lower ratio then lower weight.',
        ['batch_B_complete.json'],'No guarantee of GAN improvement; preserve exact-count ROS and raw, including rejected candidates.')
    row('alarm','Which fixed causal alarm working points?',dict(thresholds=config['alarm_thresholds'],holds=config['alarm_holds']),
        {s:dict(threshold=p['threshold'],hold=p['hold']) for s,p in b['alarm_policies'].items()},
        'Unchanged validation objective: miss +0.05*falseevents/min +0.05*mean_delay/24; registered ties.',
        ['batch_B_complete.json'],'Separate from19-class argmax. Fresh rising edges only; zero false alarms alone is insufficient.')
    row('deferred','Which optional variants remain unexecuted?',['G2','filter','stage conditioning','cross-task'],None,
        'Disabled before development, not enabled by confirmation outcomes.',['registered_config.json'],
        'No claims for these variants.',status='not_executed_registered_out_of_scope')
    for d in decisions:
        d['decision_id']='V4_'+d['name']
        d['basis']='Approved execution plan and registered bounded engineering budget; the metric is not a reliability acceptance criterion.'
        d['controlled_variables']=d['fixed_controls']
        d['evidence']=d['evidence_sha256']
    receipt=dict(**runtime_info(),config_signature=stable_hash(config),decisions=decisions,new_test_features_read=False)
    path=base/'decision_register.json'
    if path.exists():
        old=read(path)
        # Schema-only additions before confirmation unlock preserve the choices.
        normalized=[]
        for d in old['decisions']:
            d=dict(d); d['decision_id']='V4_'+d['name']
            d['basis']='Approved execution plan and registered bounded engineering budget; the metric is not a reliability acceptance criterion.'
            d['controlled_variables']=d['fixed_controls']; d['evidence']=d['evidence_sha256']; normalized.append(d)
        if normalized!=decisions or old['config_signature']!=receipt['config_signature']:
            raise RuntimeError('Development decision register changed')
        if normalized!=old['decisions']:
            if (base/'freeze.json').exists(): raise RuntimeError('Cannot revise a frozen decision register')
            old['decisions']=normalized; old['schema_completed_utc']=runtime_info()['created_utc']
            old['signature']=stable_hash({k:v for k,v in old.items() if k!='signature'}); write_json(path,old)
        return old
    receipt['signature']=stable_hash(receipt); immutable_json(path,receipt); return receipt


def confirmation_plan(config):
    decision_register(config)
    b=read(root(config)/'batch_B_complete.json')
    if b['config_signature']!=stable_hash(config): raise ValueError('Development config mismatch')
    choices=[('G0',0.,b['ratio_choices']['0.0']['selected']['mean_macro_f1'])]
    if b['selected_G1_weight'] is not None:
        w=b['selected_G1_weight']; choices.append(('G1',w,b['ratio_choices'][str(w)]['selected']['mean_macro_f1']))
    primary=min(choices,key=lambda r:(-round(r[2],8),r[1]))[0]
    comparator='ROS_matched_'+primary
    models={m['signature']:m for m in b['models']}
    differences=[]
    for group in (6,7,8):
        a=[record_at(models[s],models[s]['step'])['validation']['groups'][str(group)]['macro_f1'] for s in b['selected'][primary]]
        c=[record_at(models[s],models[s]['step'])['validation']['groups'][str(group)]['macro_f1'] for s in b['selected'][comparator]]
        differences.append(float(np.mean(a)-np.mean(c)))
    sd,estimated,groups=planned_groups(differences,config['confirmation_planning_halfwidth'],
                                     config['confirmation_min_groups'],config['confirmation_max_groups'])
    folder=root(config)/'raw_confirmation'; used=set(); old_plans=[]
    for p in sorted((ROOT/'output_simulink_gan').rglob('collection_plan.json')):
        if p.parent.resolve()==folder.resolve(): continue
        plan=read(p); used.update(int(r['seed']) for r in plan['runs']); old_plans.append(dict(path=str(p),sha256=sha256_file(p)))
    seeds=list(range(max(used)+1,max(used)+1+groups))
    design=dict(config_signature=stable_hash(config),B_complete_sha256=sha256_file(root(config)/'batch_B_complete.json'),
        primary_comparison=[primary,comparator],validation_group_differences=differences,planning_sample_sd=sd,
        requested_normal_approximate_halfwidth=config['confirmation_planning_halfwidth'],unclipped_groups=estimated,
        selected_groups=groups,records=19*groups,seeds=seeds,previously_used_seeds=sorted(used),previous_plan_hashes=old_plans,
        limitations='Only 3 development groups: unstable normal-approximation planning; min/max are budget caps, not statistical sufficiency or guaranteed precision',
        stop_policy='One predeclared collection; no extra groups based on confirmation scores')
    design['signature']=stable_hash(design)
    immutable_json(root(config)/'confirmation_design.json',design)
    path=prepare_plan(folder,seeds=tuple(seeds),task=2,dataset_version=config['profile']+'_confirmation')
    print('V4_CONFIRMATION_PLAN',groups,'groups',19*groups,'records',seeds,'primary',primary,flush=True)
    return path


def confirmation_registry(config):
    folder=root(config)/'raw_confirmation'; plan=checked_plan(folder); design=read(root(config)/'confirmation_design.json')
    records=[]; failures=[]
    for run in plan['runs']:
        path=folder/run['run_key']; meta_path=path/'run_metadata.json'
        if not meta_path.exists(): failures.append(dict(run=run,error='missing metadata')); continue
        meta=read(meta_path)
        if meta['status']!='completed' or meta.get('termination')!='completed':
            failures.append(dict(run=run,error=meta['status'])); continue
        if meta['run']!=run or meta['plan_signature']!=plan['plan_signature'] or meta['sources']!=plan['sources']:
            raise ValueError('Confirmation provenance mismatch')
        if meta['last_time_s']!=24 or meta['fault_start_s']!=3: raise ValueError('Confirmation duration/injection changed')
        for file,key in [('signals.csv','csv_sha256'),('result.mat','mat_sha256')]:
            if sha256_file(path/file)!=meta[key]: raise ValueError('Collected file identity mismatch')
        records.append(dict(source_id='v4_confirm_'+run['run_key'],source_path=str(path),plan_path=str(folder/'collection_plan.json'),
            group_id=12+run['group_id'],role='test',run=run,csv_sha256=meta['csv_sha256'],mat_sha256=meta['mat_sha256'],
            metadata_sha256=sha256_file(meta_path),flight_acceptance_passed=bool(meta['flight_metrics']['passed'])))
    write_json(root(config)/'confirmation_collection_status.json',dict(complete=len(records),expected=design['records'],failures=failures))
    if failures or len(records)!=design['records']:
        raise RuntimeError('Incomplete confirmation retained; no hard classes silently removed')
    if set(r['run']['seed'] for r in records)&set(design['previously_used_seeds']): raise ValueError('Previously exposed confirmation seed')
    result=dict(profile=config['profile'],records=records,design_signature=design['signature'],plan_signature=plan['plan_signature'])
    result['signature']=stable_hash(result)
    immutable_json(root(config)/'confirmation_sources.json',result)
    return result


def all_models(config):
    a=read(root(config)/'batch_A_complete.json'); b=read(root(config)/'batch_B_complete.json')
    neural={s['signature']:s for s in a['neural_jobs']}
    for row in b['TSTR']: neural[row['model']['signature']]=row['model']
    for s in b['all_augmented_models']: neural[s['signature']]=s
    return a,b,list(neural.values())


def freeze(config):
    decision_register(config)
    reg=confirmation_registry(config); a,b,neural=all_models(config)
    frozen_path=root(config)/'freeze.json'
    if frozen_path.exists():
        old=read(frozen_path); authorize(config,reg,'test',old['signature']); return old
    locked={}
    code=['classify.py','WGAN.py','simulink_data.py','replay_diagnostics.py','simulink_v3_data.py','simulink_v3_experiment.py',
          'simulink_v4_data.py','simulink_v4_models.py','simulink_v4_training.py','simulink_v4_gan.py',
          'simulink_v4_experiment.py','simulink_v4_augmentation.py','simulink_v4_confirmation.py','run_simulink_v4.py',
          'configs/simulink_gan_v4_evidence.json','configs/simulink_gan_v4_augmentation.json']
    for name in code: locked[str(ROOT/name)]=sha256_file(ROOT/name)
    for name in ['registered_config.json','registered_batch_B.json','development_sources.json','confirmation_sources.json',
                 'confirmation_design.json','raw_baseline_freeze.json','batch_A_complete.json','batch_B_complete.json','decision_register.json']:
        locked[str(root(config)/name)]=sha256_file(root(config)/name)
    if (root(config)/'hardware.json').exists(): locked[str(root(config)/'hardware.json')]=sha256_file(root(config)/'hardware.json')
    for s in a['statistical_jobs']:
        for p in [root(config)/'statistics'/s['signature']/'model.pkl',root(config)/'statistics'/s['signature']/'result.json']:
            locked[str(p)]=sha256_file(p)
    for s in neural:
        checkpoint=Path(s['directory'])/f'checkpoint_{s["step"]}.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['signature']!=s['signature'] or state['step']!=s['step'] or optimizer_steps(state['optimizer'])!=[s['step']]:
            raise RuntimeError('Actual classifier optimizer/update mismatch')
        if not state['parameter_max_delta']>0: raise RuntimeError('No classifier parameter updates')
        for p in [checkpoint,Path(s['directory'])/'job.json',Path(s['directory'])/'result.json',
                  Path(s['directory'])/'training_curve.json',Path(s['scaler_directory'])/'scalers.pkl',Path(s['scaler_directory'])/'fit_sources.json']:
            locked[str(p)]=sha256_file(p)
    for g in b['generators']:
        checkpoint=Path(g['directory'])/f'checkpoint_{g["generator_updates"]}.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if optimizer_steps(state['optimizer_g'])!=[3000] or optimizer_steps(state['optimizer_d'])!=[15000]:
            raise RuntimeError('Actual GAN optimizer budget mismatch')
        if state['sampling_indices'].shape!=(15000,config['batch_size']): raise RuntimeError('Missing actual critic sample indices')
        for p in [checkpoint,Path(g['directory'])/'job.json',Path(g['directory'])/'training_curve.json',Path(g['directory'])/'result.json',
                  Path(g['directory'])/f'bank_{b["selected_G_step"]}.npz',Path(g['directory'])/f'quality_{b["selected_G_step"]}.json']:
            locked[str(p)]=sha256_file(p)
        for step in config['generator_checkpoints']:
            for filename in (f'checkpoint_{step}.pt',f'bank_{step}.npz',f'quality_{step}.json'):
                p=Path(g['directory'])/filename; locked[str(p)]=sha256_file(p)
        locked[str(Path(g['directory'])/'sampling_indices.npz')]=sha256_file(Path(g['directory'])/'sampling_indices.npz')
    for s in b['models']:
        locked[str(Path(s['directory'])/'v4_alarm_selection.json')]=sha256_file(Path(s['directory'])/'v4_alarm_selection.json')
        locked[str(Path(s['directory'])/'v4_alarm_grid.json')]=sha256_file(Path(s['directory'])/'v4_alarm_grid.json')
    payload=dict(**runtime_info(),config_signature=stable_hash(config),test_registry_signature=reg['signature'],
        raw_baseline_signature=b['raw_baseline_signature'],selected=b['selected'],neural_trajectories=len(neural),
        generator_jobs=len(b['generators']),locked_files=locked,all_development_complete_before_test=True,
        historical_tests_excluded=True,new_test_features_read=False)
    payload['signature']=stable_hash(payload); immutable_json(frozen_path,payload)
    print('V4_FROZEN',payload['signature'],flush=True)
    return payload


def test(config):
    frozen=read(root(config)/'freeze.json'); reg=read(root(config)/'confirmation_sources.json')
    authorize(config,reg,'test',frozen['signature'])
    access=root(config)/'test_access_receipt.json'
    if not access.exists(): immutable_json(access,dict(**runtime_info(),freeze_signature=frozen['signature'],test_registry_signature=reg['signature']))
    b=read(root(config)/'batch_B_complete.json'); window=b['models'][0]['window']
    p=pool(config,reg,'test',window,24,'native',token=frozen['signature'])
    progress_path=root(config)/'confirmation_progress.json'
    if progress_path.exists():
        previous=read(progress_path)
        if previous['freeze_signature']!=frozen['signature']: raise PermissionError('Confirmation cache belongs to another freeze')
        results=previous['results']
    else: results=[]
    for spec in b['models']:
        if any(r['signature']==spec['signature'] for r in results): continue
        model=load_model(config,spec)
        with (Path(spec['scaler_directory'])/'scalers.pkl').open('rb') as f: cls,_=pickle.load(f)
        logits=predict(model,transform(p['X'],cls)); metrics=evaluate(p,logits)
        records=prediction_records(config,reg,'test',spec,frozen['signature'])
        policy=b['alarm_policies'][spec['signature']]; threshold=policy['threshold']; hold=policy['hold']
        scores=alarm_metrics(records,threshold,hold)
        by_group={str(g):alarm_metrics([r for r in records if r['group_id']==g],threshold,hold) for g in sorted(set(p['groups']))}
        z=logits-logits.max(1,keepdims=True); probs=np.exp(z); probs/=probs.sum(1,keepdims=True)
        gated=np.where(1-probs[:,0]>=threshold,logits[:,1:].argmax(1)+1,0)
        from simulink_v3_experiment import metric
        from simulink_v4_data import masks
        gated_metrics={name:metric(p['y'][mask],gated[mask]) if mask.any() else {'window_count':0} for name,mask in masks(p).items()}
        result=dict(signature=spec['signature'],seed=spec['seed'],method=spec['method'],ratio=spec['ratio'],
            test=metrics,alarm=scores,alarm_by_group=by_group,alarm_policy=dict(threshold=threshold,hold=hold),
            alarm_gated_window=gated_metrics,pool_signature=p['signature'])
        np.savez_compressed(Path(spec['directory'])/'V4_confirmation_predictions.npz',logits=logits,y=p['y'],
                            source_ids=p['source_ids'],starts=p['starts'],ends=p['ends'])
        results.append(result)
        write_json(progress_path,dict(freeze_signature=frozen['signature'],results=results))
        print('V4_CONFIRMATION',spec['method'],spec['seed'],metrics['prefix_24']['macro_f1'],'miss',scores['missed'],flush=True)
    final=dict(freeze_signature=frozen['signature'],test_registry_signature=reg['signature'],selected=b['selected'],
               results=results,groups=len(set(p['groups'])),records=len(reg['records']),same_evaluation_pool=p['signature'])
    write_json(root(config)/'final_confirmation_results.json',final)
    return final


def audit(config):
    frozen=read(root(config)/'freeze.json'); reg=read(root(config)/'confirmation_sources.json')
    authorize(config,reg,'test',frozen['signature']); a,b,neural=all_models(config)
    rows=[]; checks=[]
    def check(name,ok):
        checks.append(dict(check=name,passed=bool(ok)))
        if not ok: raise AssertionError(name)
    legacy=snapshot(config)
    check('all legacy frozen artifacts unchanged',len(legacy['files'])==1972)
    for s in neural:
        state=torch.load(Path(s['directory'])/f'checkpoint_{s["step"]}.pt',map_location='cpu',weights_only=False)
        check('actual Adam budget '+s['signature'],optimizer_steps(state['optimizer'])==[s['step']])
        check('actual measured feature width '+s['signature'],s['features']==12)
        first_key='lstm.weight_ih_l0' if s['family']=='lstm' else 'encoder.0.weight'
        check('actual learned input tensor width '+s['signature'],state['model'][first_key].shape[1]==12)
        if s['method']=='ros' or s['method'].startswith('G'):
            original=np.asarray(s['class_original']); joined=np.asarray(s['class_joined'])
            check('actual per-class floor additions '+s['signature'],np.array_equal(joined-original,np.floor(original*s['ratio']).astype(int)))
        rows.append(dict(signature=s['signature'],family=s['family'],window=s['window'],prefix_s=s['prefix_s'],seed=s['seed'],
            method=s['method'],ratio=s['ratio'],updates=s['step'],training_windows=s['training_windows'],original_windows=s['original_windows']))
    pd.DataFrame(rows).to_csv(root(config)/'audit/actual_neural_budgets.csv',index=False,encoding='utf-8-sig')
    for file in (root(config)/'scalers').glob('*/fit_sources.json'):
        receipt=read(file); pool_file=root(config)/'pools'/f'{receipt["pool"]}.pkl'
        if not pool_file.exists():
            from simulink_v4_experiment import memorization_pool
            p=memorization_pool(config,registry(config))
            with (file.parent/'scalers.pkl').open('rb') as f: cls,gan=pickle.load(f)
            check('actual memorization selected-only scaler '+file.parent.name,receipt['window_count']==152 and
                  receipt['X_sha256']==p['X_sha256'] and np.allclose(cls.mean_,p['X'].reshape(-1,12).mean(0,dtype=np.float64)))
            continue
        with pool_file.open('rb') as f: p=pickle.load(f)
        with (file.parent/'scalers.pkl').open('rb') as f: cls,gan=pickle.load(f)
        check('selected-only scaler '+file.parent.name,
              p['identity']['role']=='train' and receipt['X_sha256']==p['X_sha256'] and
              int(cls.n_samples_seen_)==len(p['y'])*p['X'].shape[1] and
              np.allclose(cls.mean_,p['X'].reshape(-1,12).mean(0,dtype=np.float64)))
    raw=read(root(config)/'development_sources.json'); formal=raw['records']+reg['records']
    roles={role:{r['run']['seed'] for r in formal if r['role']==role} for role in ('train','validation','test')}
    check('role seeds disjoint',not(roles['train']&roles['validation'] or roles['train']&roles['test'] or roles['validation']&roles['test']))
    old={r['run']['seed'] for r in raw['historical_test_records']}
    check('new test excludes historical exposed seeds',not(old&roles['test']))
    source_rows=[]
    for role,registry_ in [('train',raw),('validation',raw),('test',reg)]:
        for sample in signals(config,registry_,role,token=frozen['signature'] if role=='test' else None):
            r=sample['record']; source_rows.append(dict(role=role,group=r['group_id'],seed=r['run']['seed'],source_id=r['source_id'],
                label=r['run']['label'],motor=r['run']['motor_simulink'],factor=r['run']['factor'],rows=len(sample['X']),
                finite=bool(np.isfinite(sample['X']).all()),flight_acceptance_passed=r['flight_acceptance_passed'],
                csv_sha256=r['csv_sha256'],mat_sha256=r['mat_sha256'],metadata_sha256=r['metadata_sha256']))
    source_df=pd.DataFrame(source_rows); source_df.to_csv(root(config)/'audit/formal_source_manifest.csv',index=False,encoding='utf-8-sig')
    final=read(root(config)/'final_confirmation_results.json'); identities=[]
    for result in final['results']:
        spec=next(s for s in b['models'] if s['signature']==result['signature'])
        with np.load(Path(spec['directory'])/'V4_confirmation_predictions.npz') as p:
            identities.append(stable_hash(dict(y=p['y'].tolist(),source=p['source_ids'].tolist(),ends=p['ends'].tolist())))
            check('main classification uses argmax '+spec['signature'],
                  abs(np.mean(p['logits'].argmax(1)==p['y'])-result['test']['prefix_24']['accuracy'])<1e-12)
        expected_exposure=final['groups']*(23.99-(spec['window']-1)/100)
        check('deduplicated healthy exposure '+spec['signature'],abs(result['alarm']['normal_observation_s']-expected_exposure)<1e-8)
    check('all final models share same windows/labels/sources',len(set(identities))==1)
    access=read(root(config)/'test_access_receipt.json')
    check('first test read follows freeze',access['created_utc']>=frozen['created_utc'])
    check('full finite records kept',source_df.rows.eq(2401).all() and source_df.finite.all())
    result=dict(**runtime_info(),passed=True,checks=checks,count=len(checks),neural_trajectories=len(neural),
        classifier_updates=sum(s['step'] for s in neural),generator_updates=3000*len(b['generators']),critic_updates=15000*len(b['generators']),
        formal_records=len(source_df),independent_condition_groups=9+final['groups'],
        retained_flight_acceptance_failures=int((~source_df.flight_acceptance_passed).sum()),freeze_signature=frozen['signature'])
    write_json(root(config)/'audit/final_audit.json',result)
    print('V4_FINAL_AUDIT',len(checks),'checks passed',flush=True)
    return result
