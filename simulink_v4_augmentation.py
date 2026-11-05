"""Batch B: exact matched-information GAN/ROS comparisons and causal validation."""
import pickle
from pathlib import Path
import numpy as np

from provenance import stable_hash,sha256_file,write_json,runtime_info
from simulink_data import ROOT,read
from simulink_v3_data import immutable_json
from replay_diagnostics import select_alarm_policy,score_alarm_records
from simulink_v4_data import root,pool,registry,signals,authorize
from simulink_v4_training import train,load_model,record_at,predict,transform
from simulink_v4_gan import train_generator

B_CONFIG=ROOT/'configs/simulink_gan_v4_augmentation.json'


def counts(y,ratio): return {c:int(np.floor(np.sum(y==c)*ratio)) for c in range(19)}


def additions(x,y,desired):
    idx=np.concatenate([np.flatnonzero(y==c)[:n] for c,n in desired.items()]).astype(np.int64)
    return x[idx],y[idx]


def ros_additions(p,desired,seed):
    rng=np.random.default_rng(seed)
    idx=np.concatenate([rng.choice(np.flatnonzero(p['y']==c),n,replace=True) for c,n in desired.items()]).astype(np.int64)
    return p['X'][idx],p['y'][idx]


def prediction_records(config,reg,role,spec,token=None):
    authorize(config,reg,role,token)
    identity=dict(registry=reg['signature'],role=role,signature=spec['signature'],step=spec['step'],
        checkpoint=sha256_file(Path(spec['directory'])/f'checkpoint_{spec["step"]}.pt'),
        scaler=sha256_file(Path(spec['scaler_directory'])/'scalers.pkl'),stride=config['replay_stride'],
        code_sha256=sha256_file(__file__))
    key=stable_hash(identity); path=root(config)/'replay_cache'/f'{role}_{key}.pkl'
    if path.exists():
        if read(path.with_suffix('.json'))!=identity: raise ValueError('Causal replay cache mismatch')
        with path.open('rb') as f: return pickle.load(f)
    model=load_model(config,spec)
    with (Path(spec['scaler_directory'])/'scalers.pkl').open('rb') as f: scaler,_=pickle.load(f)
    output=[]; window=spec['window']
    for sample in signals(config,reg,role,token=token):
        values=sample['X']; ends=np.arange(window-1,len(values),config['replay_stride'])
        x=np.asarray([values[e-window+1:e+1] for e in ends],np.float32)
        logits=predict(model,transform(x,scaler))
        z=logits-logits.max(1,keepdims=True); probs=np.exp(z); probs/=probs.sum(1,keepdims=True)
        r=sample['record']
        output.append(dict(source_id=r['source_id'],group_id=r['group_id'],label=r['run']['label'],
            time_s=sample['time'][ends],fault_start_s=sample['meta']['fault_start_s'],
            fault_probability=1-probs[:,0],fault_class=logits[:,1:].argmax(1)+1,
            first_available_time_s=float(sample['time'][ends[0]]),predictions=len(ends)))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f: pickle.dump(output,f)
    write_json(path.with_suffix('.json'),identity)
    return output


def alarm_metrics(records,threshold,hold):
    result=score_alarm_records(records,threshold,hold)
    delays=np.asarray(result['detection_delays_s'])
    for t in (.2,.5,1.,3.): result[f'detected_within_{t:g}s_rate']=float(np.sum(delays<=t+1e-9)/max(result['fault_records'],1))
    result['delay_quantiles_s']=None if not len(delays) else dict(zip(('p25','median','p75','p95'),np.quantile(delays,[.25,.5,.75,.95]).tolist()))
    result['first_available_time_s']=records[0]['first_available_time_s']
    result['predictions_per_record']=records[0]['predictions']
    result['prefault_judgments_per_fault_record']=int(np.sum(np.asarray(records[0]['time_s'])<3))
    return result


def choose_policy(config,reg,spec):
    path=Path(spec['directory'])/'v4_alarm_selection.json'
    if path.exists(): return read(path)
    records=prediction_records(config,reg,'validation',spec)
    p=select_alarm_policy(records,config['alarm_thresholds'],config['alarm_holds'])
    result=dict(threshold=p['threshold'],hold=p['hold'],validation=alarm_metrics(records,p['threshold'],p['hold']))
    write_json(path,result); write_json(path.with_name('v4_alarm_grid.json'),p['grid'])
    return result


def run_batch_B(config):
    a=read(root(config)/'batch_A_complete.json'); frozen=read(root(config)/'raw_baseline_freeze.json')
    for file,digest in frozen['locked_files'].items():
        if sha256_file(file)!=digest: raise ValueError('The common raw-data baseline changed')
    bconfig=read(B_CONFIG)
    immutable_json(root(config)/'registered_batch_B.json',dict(protocol=bconfig,signature=stable_hash(bconfig),raw_baseline_signature=frozen['signature']))
    reg=registry(config); window=a['selected_window']; prefix=a['selected_prefix_s']; steps=a['selected_steps']; family=a['selected_family']
    tp=pool(config,reg,'train',window,prefix,'native'); vp=pool(config,reg,'validation',window,24,'native')
    matching=pool(config,reg,'validation',window,prefix,'native')
    raw=a['raw_models']; generators=[]; tstr=[]; checkpoint_rows=[]; all_jobs=[]
    def status(): write_json(root(config)/'batch_B_progress.json',dict(generators=generators,TSTR=tstr,classifiers=all_jobs))
    for index,seed in enumerate(config['generator_seeds']):
        g=train_generator(config,bconfig,tp,matching,seed,0.); generators.append(g)
        for checkpoint in config['generator_checkpoints']:
            with np.load(Path(g['directory'])/f'bank_{checkpoint}.npz') as bank: x,y=bank['X'],bank['y']
            s=train(config,tp,vp,family,bconfig['TSTR_seeds'][index],steps,method=f'TSTR_{g["signature"]}_{checkpoint}',raw=x,labels=y)
            tstr.append(dict(generator=g['signature'],G_step=checkpoint,model=s)); status()
    for checkpoint in config['generator_checkpoints']:
        results=[record_at(r['model'],steps)['validation']['prefix_24']['macro_f1'] for r in tstr if r['G_step']==checkpoint]
        checkpoint_rows.append(dict(step=checkpoint,mean_macro_f1=float(np.mean(results)),scores=results))
    selected_G=min(checkpoint_rows,key=lambda r:(-round(r['mean_macro_f1'],8),r['step']))['step']
    immutable_json(root(config)/'decisions/B_G_checkpoint.json',dict(selected=selected_G,candidates=checkpoint_rows,rule=bconfig['checkpoint_selection']))
    allowed=[0.]
    if a['teacher_gate']['passed']:
        allowed=config['class_constraint_weights']
        for weight in allowed:
            if not weight: continue
            for index,seed in enumerate(config['generator_seeds']):
                teacher=dict(raw[index],step=steps)
                g=train_generator(config,bconfig,tp,matching,seed,weight,teacher); generators.append(g)
                with np.load(Path(g['directory'])/f'bank_{selected_G}.npz') as bank: x,y=bank['X'],bank['y']
                s=train(config,tp,vp,family,bconfig['TSTR_seeds'][index],steps,method=f'TSTR_{g["signature"]}_{selected_G}',raw=x,labels=y)
                tstr.append(dict(generator=g['signature'],G_step=selected_G,model=s)); status()
    else:
        write_json(root(config)/'G1_not_executed.json',dict(reason='Registered non-collapse teacher prerequisite failed',evidence=a['teacher_gate'],
            status='not_executed_unmet_prerequisite',G0_continues=True))
    all_jobs.extend(dict(s,step=steps) for s in raw)
    ros={}; gan={weight:{} for weight in allowed}
    for ratio in config['augmentation_ratios']:
        if ratio==0:
            ros[ratio]=[dict(s,step=steps) for s in raw]
            for weight in allowed: gan[weight][ratio]=ros[ratio]
            continue
        desired=counts(tp['y'],ratio); ros[ratio]=[]
        for index,seed in enumerate(config['training_seeds']):
            extra,labels=ros_additions(tp,desired,config['ros_seeds'][index])
            s=train(config,tp,vp,family,seed,steps,'ros',ratio,np.concatenate([tp['X'],extra]),np.r_[tp['y'],labels])
            ros[ratio].append(s); all_jobs.append(s); status()
        for weight in allowed:
            gan[weight][ratio]=[]
            for index,seed in enumerate(config['training_seeds']):
                g=next(g for g in generators if g['weight']==weight and g['seed']==config['generator_seeds'][index])
                with np.load(Path(g['directory'])/f'bank_{selected_G}.npz') as bank: extra,labels=additions(bank['X'],bank['y'],desired)
                if {c:int(np.sum(labels==c)) for c in range(19)}!=desired: raise AssertionError('GAN class count mismatch')
                s=train(config,tp,vp,family,seed,steps,f'G{weight:g}_{g["signature"]}_{selected_G}',ratio,np.concatenate([tp['X'],extra]),np.r_[tp['y'],labels])
                gan[weight][ratio].append(s); all_jobs.append(s); status()
    def select(mapping):
        rows=[]
        for ratio,specs in mapping.items():
            values=[record_at(s,steps)['validation']['prefix_24']['macro_f1'] for s in specs]
            rows.append(dict(ratio=ratio,mean_macro_f1=float(np.mean(values)),scores=values,signatures=[s['signature'] for s in specs]))
        return dict(selected=min(rows,key=lambda r:(-round(r['mean_macro_f1'],8),r['ratio'])),candidates=rows)
    ros_choice=select(ros); choices={str(w):select(mapping) for w,mapping in gan.items()}
    selected={'raw':[dict(s,step=steps) for s in raw], 'ROS_best':ros[ros_choice['selected']['ratio']],
              'G0':gan[0.][choices['0.0']['selected']['ratio']], 'ROS_matched_G0':ros[choices['0.0']['selected']['ratio']]}
    if len(allowed)>1:
        best=min([w for w in allowed if w],key=lambda w:(-round(choices[str(w)]['selected']['mean_macro_f1'],8),w))
        selected['G1']=gan[best][choices[str(best)]['selected']['ratio']]
        selected['ROS_matched_G1']=ros[choices[str(best)]['selected']['ratio']]
        selected_G1_weight=best
    else: selected_G1_weight=None
    unique={s['signature']:dict(s,step=steps) for specs in selected.values() for s in specs}
    policies={signature:choose_policy(config,reg,s) for signature,s in unique.items()}
    result=dict(**runtime_info(),config_signature=stable_hash(config),B_protocol_signature=stable_hash(bconfig),
        selected_G_step=selected_G,selected_G1_weight=selected_G1_weight,teacher_gate_passed=a['teacher_gate']['passed'],
        ratio_choices=choices,ROS_choice=ros_choice,selected={name:[s['signature'] for s in specs] for name,specs in selected.items()},
        models=list(unique.values()),all_augmented_models=all_jobs,generators=generators,TSTR=tstr,alarm_policies=policies,
        raw_baseline_signature=frozen['signature'],historical_test_features_read=False)
    if (len(generators)>bconfig['max_generator_jobs'] or len(tstr)>bconfig['max_TSTR_classifier_jobs'] or
            len(all_jobs)-len(raw)>bconfig['max_main_augmented_classifier_jobs']):
        raise AssertionError('Registered batch-B task cap exceeded')
    write_json(root(config)/'batch_B_complete.json',result)
    print('V4_BATCH_B_COMPLETE','Gstep',selected_G,'G1',selected_G1_weight,'unique final models',len(unique),flush=True)
    return result
