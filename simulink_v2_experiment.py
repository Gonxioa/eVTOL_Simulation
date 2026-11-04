"""Fixed-source, validation-selected V2 protocol using existing network classes."""
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import classify
import WGAN
from provenance import code_hashes, stable_hash, write_json, sha256_file
from simulink_data import (ROOT,FEATURES,read,v2_registry,v2_pool,v2_feature,v2_load_record,
                           observable_features,prepare_plan)
from experiment_protocol import augmentation_counts,stratified_generation_quality
from replay_diagnostics import motor_response_scores,select_alarm_policy,score_alarm_records


def configure(config,feature):
    torch.set_num_threads(4)
    classify.N_CLASSES=19; classify.N_FEATURES=len(FEATURES[feature]); classify.SEQ_LEN=config['window']
    classify.HIDDEN_DIM=config['classifier_hidden_dim']; classify.N_LAYERS=config['classifier_layers']
    classify.ATTENTION_POOLING=True; classify.FEATURE_NAMES=FEATURES[feature]
    classify.BATCH_SIZE=config['batch_size']; classify.LR=config['classifier_lr']
    WGAN.HIDDEN_DIM=config['generator_hidden_dim']; WGAN.N_LAYERS=config['generator_layers']
    WGAN.NOISE_DIM=config['generator_noise_dim']; WGAN.EMBED_DIM=config['generator_embedding_dim']
    WGAN.N_CRITIC=config['n_critic']; WGAN.LAMBDA_GP=config['gradient_penalty']
    WGAN.LR_G=config['generator_lr']; WGAN.LR_D=config['critic_lr']; WGAN.TEMPORAL_NOISE=True
    WGAN.configure_dimensions(classify.N_FEATURES,19,config['window'])


def root(config): return ROOT/config['results_root']


def metrics(y,prediction,threshold=.5):
    probability=prediction['fault_probability']; logits=prediction['logits']
    pred=np.where(probability>=threshold,logits[:,1:].argmax(1)+1,0)
    normal=y==0; fault=~normal
    false=float(np.mean(pred[normal]!=0)); miss=float(np.mean(pred[fault]==0))
    return dict(accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,labels=range(19),average='macro',zero_division=0)),
                healthy_false_positive=false,fault_miss=miss,balanced_detection_error=.5*(false+miss),
                detection_threshold=threshold,window_count=len(y))


def scalers(config,registry,train,feature):
    values=v2_feature(train,feature)
    signature=stable_hash(dict(pool=train['signature'],feature=feature))
    directory=root(config)/'scalers'/signature[:16]; directory.mkdir(parents=True,exist_ok=True)
    path=directory/'scalers.pkl'
    if path.exists():
        with path.open('rb') as f: cls,gan=pickle.load(f)
    else:
        cls=StandardScaler().fit(values.reshape(-1,values.shape[-1]))
        gan=MinMaxScaler(feature_range=(-1,1)).fit(values.reshape(-1,values.shape[-1]))
        with path.open('wb') as f: pickle.dump((cls,gan),f)
        write_json(directory/'fit_sources.json',dict(signature=signature,training_groups=sorted(np.unique(train['groups']).tolist()),
                   training_source_indices=sorted(np.unique(train['source_indices']).tolist()),feature_names=FEATURES[feature],
                   registry_signature=registry['signature']))
    return cls,gan,str(directory)


def transform(values,scaler):
    return scaler.transform(values.reshape(-1,values.shape[-1])).reshape(values.shape).astype(np.float32)


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def load_classifier(config,spec,step=None):
    configure(config,spec['feature'])
    path=Path(spec['directory'])/f'checkpoint_{step or spec["steps"]}.pt'
    saved=torch.load(path,map_location=classify.DEVICE,weights_only=False)
    if saved['signature']!=spec['signature']: raise ValueError('Classifier checkpoint signature mismatch')
    model=classify.FaultClassifier(dual_head=spec['dual_head']).to(classify.DEVICE)
    model.load_state_dict(saved['model']); model.eval()
    return model


def train_job(config,registry,train,validation,feature,dual_head,seed,tag='raw',X_raw=None,y=None,max_steps=None):
    configure(config,feature); cls,_,scale_dir=scalers(config,registry,train,feature)
    raw=v2_feature(train,feature) if X_raw is None else X_raw
    y=train['y'] if y is None else y
    x=transform(raw,cls); valx=transform(v2_feature(validation,feature),cls)
    steps=max_steps or config['classifier_max_steps']
    payload=dict(feature=feature,dual_head=dual_head,seed=seed,steps=steps,lr=config['classifier_lr'],
                 batch_size=config['batch_size'],hidden_dim=config['classifier_hidden_dim'],layers=config['classifier_layers'],
                 detector_weight=config['detector_loss_weight'],X_sha256=array_hash(x),y_sha256=array_hash(y),
                 train_pool=train['signature'],validation_pool=validation['signature'],
                 code_sha256=code_hashes(ROOT,['classify.py','simulink_data.py','experiment_protocol.py']))
    signature=stable_hash(payload); directory=root(config)/'classifiers'/f'{tag}_{feature}_s{seed}_{signature[:12]}'
    directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'job.json',dict(**payload,signature=signature,scaler_directory=scale_dir))
    def callback(model,step):
        result=metrics(validation['y'],classify.predict_classifier(model,valx))
        print('VALIDATION',tag,feature,'dual',dual_head,'step',step,json.dumps(result),flush=True)
        return result
    model=classify.train_classifier_steps(x,y,directory,max_steps=steps,checkpoints=config['classifier_checkpoints'],
               seed=seed,dual_head=dual_head,detector_weight=config['detector_loss_weight'],validation_callback=callback,
               signature=signature,batch_size=config['batch_size'],lr=config['classifier_lr'])
    result=metrics(validation['y'],classify.predict_classifier(model,valx))
    spec=dict(feature=feature,dual_head=dual_head,seed=seed,steps=steps,directory=str(directory),
              signature=signature,scaler_directory=scale_dir,validation=result,training_receipt=model.training_receipt)
    write_json(directory/'result.json',spec)
    return spec


def baseline_stage(config,registry,train,validation):
    destination=root(config)/'development'; destination.mkdir(parents=True,exist_ok=True)
    results={}
    for feature in config['feature_sets']:
        results[feature]=train_job(config,registry,train,validation,feature,False,config['development_seed'],'single')
    write_json(destination/'baselines.json',results)
    return results


def head_stage(config,registry,train,validation):
    path=root(config)/'development/baselines.json'
    if not path.exists(): baseline_stage(config,registry,train,validation)
    baselines=read(path); feature=config['primary_feature']; single=baselines[feature]
    dual=train_job(config,registry,train,validation,feature,True,config['development_seed'],'dual')
    a,b=single['validation'],dual['validation']
    improved=(b['balanced_detection_error']<=a['balanced_detection_error']-config['head_improvement_margin'] and
              b['macro_f1']>=a['macro_f1']-.05)
    choice=dict(dual_head=improved,single=single,dual=dual,
                selection_rule='BER at 0.5 improves by >=0.005; macro F1 drop <=0.05; validation only')
    choice['primary']=dual if improved else single
    choice['ablations']={feature:choice['primary']}
    for feature in config['feature_sets']:
        if feature==config['primary_feature']: continue
        choice['ablations'][feature]=train_job(config,registry,train,validation,feature,improved,config['development_seed'],'fixed_head') if improved else baselines[feature]
    write_json(root(config)/'development/head_selection.json',choice)
    print('HEAD_SELECTION',json.dumps(dict(dual_head=improved,single=a,dual=b)),flush=True)
    return choice


def generator_job(config,registry,train,mode,seed,max_steps=None):
    feature=config['primary_feature']; configure(config,feature)
    _,gan,scaler_dir=scalers(config,registry,train,feature)
    x=transform(v2_feature(train,feature),gan)
    indices=[FEATURES[feature].index(f'cmd_motor{i}') for i in range(6)] if mode=='command' else None
    steps=max_steps or config['generator_max_steps']
    payload=dict(mode=mode,seed=seed,steps=steps,train_pool=train['signature'],batch=config['batch_size'],
                 hidden=config['generator_hidden_dim'],layers=config['generator_layers'],noise=config['generator_noise_dim'],
                 embedding=config['generator_embedding_dim'],n_critic=config['n_critic'],gp=config['gradient_penalty'],
                 lr_g=config['generator_lr'],lr_d=config['critic_lr'],command_indices=indices,
                 X_sha256=array_hash(x),y_sha256=array_hash(train['y']),code_sha256=code_hashes(ROOT,['WGAN.py']))
    signature=stable_hash(payload); directory=root(config)/'generators'/f'{mode}_s{seed}_{signature[:12]}'
    directory.mkdir(parents=True,exist_ok=True); write_json(directory/'job.json',dict(**payload,signature=signature))
    WGAN.train_generator_steps(x,train['y'],directory,generator_max_steps=steps,checkpoints=config['generator_checkpoints'],
                               command_indices=indices,seed=seed,batch_size=config['batch_size'],signature=signature)
    spec=dict(mode=mode,seed=seed,steps=steps,signature=signature,directory=str(directory),scaler_directory=scaler_dir,
              command_indices=indices,feature=feature)
    write_json(directory/'result.json',spec)
    return spec


def generator_train_stage(config,registry,train):
    result={mode:generator_job(config,registry,train,mode,config['development_seed']) for mode in ('plain','command')}
    write_json(root(config)/'development/generator_training.json',result)
    return result


def load_generator(config,spec,step=None):
    configure(config,spec['feature'])
    saved=torch.load(Path(spec['directory'])/f'checkpoint_{step or spec["steps"]}.pt',map_location=WGAN.DEVICE,weights_only=False)
    if saved['signature']!=spec['signature']: raise ValueError('GAN checkpoint signature mismatch')
    model=WGAN.Generator(spec['command_indices']).to(WGAN.DEVICE)
    model.load_state_dict(saved['generator']); model.eval()
    return model


def synthetic(config,registry,train,spec,counts,seed,step=None):
    feature=spec['feature']; _,gan,_=scalers(config,registry,train,feature)
    raw=v2_feature(train,feature); scaled=transform(raw,gan); model=load_generator(config,spec,step)
    values,labels,indices=WGAN.generate_from_pool(model,counts,scaled,train['y'],seed,config['batch_size'])
    result=gan.inverse_transform(values.reshape(-1,values.shape[-1])).reshape(values.shape).astype(np.float32) if len(values) else values
    if spec['command_indices']:
        result[:,:,spec['command_indices']]=raw[indices][:,:,spec['command_indices']]
        if not np.array_equal(result[:,:,spec['command_indices']],raw[indices][:,:,spec['command_indices']]):
            raise AssertionError('Conditional commands changed')
        if not np.array_equal(train['y'][indices],labels): raise AssertionError('Command condition label mismatch')
    return result,labels,indices


def prediction_records(config,registry,role,spec=None,rule=False,authorization=None):
    identity=None if rule else dict(feature=spec['feature'],dual_head=spec['dual_head'],steps=spec['steps'],
                checkpoint_sha256=sha256_file(Path(spec['directory'])/f'checkpoint_{spec["steps"]}.pt'),
                scaler_sha256=sha256_file(Path(spec['scaler_directory'])/'scalers.pkl'))
    source=dict(registry=registry['signature'],role=role,rule=rule,model=identity,
                code_sha256=code_hashes(ROOT,['replay_diagnostics.py','classify.py','simulink_data.py']))
    key=stable_hash(source); path=root(config)/'prediction_cache'/f'{role}_{key[:16]}.pkl'
    if role=='test':
        from simulink_data import v2_authorize
        v2_authorize(config,role,authorization)
    if path.exists():
        with path.open('rb') as f: return pickle.load(f)
    records=[]; feature=config['primary_feature'] if rule else spec['feature']
    if not rule:
        model=load_classifier(config,spec)
        with (Path(spec['scaler_directory'])/'scalers.pkl').open('rb') as f: cls,_=pickle.load(f)
    for record in registry['records']:
        if record['role']!=role: continue
        meta,df=v2_load_record(config,record,authorization)
        values=observable_features(df,meta,feature); times=df.time_s.to_numpy()
        ends=np.arange(config['window']-1,len(times),config['replay_stride'])
        if rule:
            commands=df[[f'rotor_command_rad_s_{i}' for i in range(1,7)]].to_numpy()
            rpm=df[[f'feedback_state_{i}' for i in range(14,20)]].to_numpy()
            output=motor_response_scores(commands,rpm,meta['sample_dt_s'],meta['parameters']['tau'],meta['feedback_delay_s'])
            probability=output['fault_probability'][ends]; fault_class=output['fault_class'][ends]
        else:
            windows=np.asarray([values[end-config['window']+1:end+1] for end in ends])
            output=classify.predict_classifier(model,transform(windows,cls))
            probability=output['fault_probability']; fault_class=output['logits'][:,1:].argmax(1)+1
        records.append(dict(source_id=record['source_id'],group_id=record['group_id'],label=record['run']['label'],
                            fault_start_s=meta['fault_start_s'],time_s=times[ends],fault_probability=probability,fault_class=fault_class))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f: pickle.dump(records,f)
    write_json(path.with_suffix('.json'),source)
    return records


def generator_quality_stage(config,registry,train,validation,head,generators):
    feature=config['primary_feature']; cls,_,_=scalers(config,registry,train,feature)
    rows=[]; counts={i:config['quality_samples_per_class'] for i in range(19)}
    for mode,spec in generators.items():
        for step in config['generator_checkpoints']:
            key=stable_hash(dict(generator=spec['signature'],step=step,counts=counts))
            folder=root(config)/'quality'/f'{mode}_{step}_{key[:12]}'; folder.mkdir(parents=True,exist_ok=True)
            samples=folder/'quality_samples.npz'
            if samples.exists():
                saved=dict(np.load(samples)); x,y,indices=saved['X'],saved['y'],saved['command_source_indices']
            else:
                x,y,indices=synthetic(config,registry,train,spec,counts,config['development_seed']+10000+step,step)
                np.savez_compressed(samples,X=x,y=y,command_source_indices=indices)
            quality=stratified_generation_quality(v2_feature(train,feature),train['y'],v2_feature(validation,feature),
                      validation['y'],x,y,FEATURES[feature],train['groups'],mode=='command')
            tstr=train_job(config,registry,train,validation,feature,head['dual_head'],config['development_seed'],
                           f'quality_{mode}_{step}',x,y,config['quality_classifier_max_steps'])
            real_model=load_classifier(config,head['primary'])
            trts=metrics(y,classify.predict_classifier(real_model,transform(x,cls)))
            row=dict(mode=mode,generator_updates=step,tstr_validation=tstr['validation'],trts=trts,
                     quality=quality,samples_path=str(samples),quality_classifier=tstr,
                     quality_only_count=len(y),augmentation_joined_count=0)
            write_json(folder/'quality.json',row); rows.append(row)
            print('GAN_VALIDATION_QUALITY',mode,step,'TSTR',tstr['validation'],'quality',quality['summary'],flush=True)
    # A common budget is chosen for both GAN structures; never choose by generator loss.
    scores={step:float(np.mean([r['tstr_validation']['macro_f1'] for r in rows if r['generator_updates']==step]))
            for step in config['generator_checkpoints']}
    selected=max(scores,key=lambda step:(scores[step],-step))
    report=dict(rows=rows,shared_generator_steps=selected,budget_scores=scores,
                selection_rule='Highest mean validation TSTR macro F1 across both GANs; ties choose smaller shared budget')
    write_json(root(config)/'development/gan_quality_selection.json',report)
    return report


def augment_job(config,registry,train,validation,head,seed,method,ratio,generator=None,generator_step=None):
    feature=config['primary_feature']; raw=v2_feature(train,feature)
    counts=augmentation_counts(train['y'],19,'ratio',ratio)
    if ratio==0:
        spec=train_job(config,registry,train,validation,feature,head,seed,'raw_fixed')
        return dict(method=method,ratio=ratio,classifier=spec,generated_joined=0,counts=counts)
    if method=='ros':
        rng=np.random.default_rng(seed+123); refs=[]; generated_labels=[]
        for label,count in counts.items():
            refs.extend(rng.choice(np.flatnonzero(train['y']==label),size=count,replace=True).tolist())
            generated_labels.extend([label]*count)
        x=raw[refs]; y=np.asarray(generated_labels,np.int64); indices=np.asarray(refs)
    else:
        x,y,indices=synthetic(config,registry,train,generator,counts,seed+123,generator_step)
    actual={i:int(np.sum(y==i)) for i in range(19)}
    if actual!=counts or not np.isfinite(x).all(): raise ValueError('Augmentation count/value audit failed')
    tag=f'{method}_r{ratio:g}_g{generator_step or 0}'
    spec=train_job(config,registry,train,validation,feature,head,seed,tag,
                   np.concatenate([raw,x]),np.concatenate([train['y'],y]))
    directory=Path(spec['directory'])
    np.savez_compressed(directory/'augmentation.npz',X=x,y=y,command_source_indices=indices)
    receipt={i:dict(original=int(np.sum(train['y']==i)),planned=counts[i],actual_generated=actual[i],
                    joined_training=actual[i],quality_only=0) for i in range(19)}
    write_json(directory/'augmentation_counts.json',receipt)
    return dict(method=method,ratio=ratio,classifier=spec,generated_joined=len(y),counts=receipt,
                generator=generator,generator_step=generator_step)


def select_workpoint(config,registry,spec=None,rule=False):
    records=prediction_records(config,registry,'validation',spec,rule)
    return select_alarm_policy(records,config['alarm_thresholds'],config['alarm_holds'])


def gans_stage(config,registry,train,validation):
    head=read(root(config)/'development/head_selection.json')
    path=root(config)/'development/generator_training.json'
    generators=read(path) if path.exists() else generator_train_stage(config,registry,train)
    quality=generator_quality_stage(config,registry,train,validation,head,generators)
    rows=[]; selections={}; seed=config['development_seed']; shared=quality['shared_generator_steps']
    for method in ('ros','plain','command'):
        for ratio in config['augmentation_ratios']:
            result=augment_job(config,registry,train,validation,head['dual_head'],seed,method,ratio,
                               generators.get(method),shared if method!='ros' else None)
            result['alarm_policy']=select_workpoint(config,registry,result['classifier'])
            rows.append(result)
            print('AUGMENTATION_VALIDATION',method,ratio,result['classifier']['validation'],
                  'alarm objective',result['alarm_policy']['validation']['objective'],flush=True)
        choices=[r for r in rows if r['method']==method]
        # Validation macro F1 is primary; alarm objective and smaller augmentation break ties.
        selections[method]=max(choices,key=lambda r:(r['classifier']['validation']['macro_f1'],
                                  -r['alarm_policy']['validation']['objective'],-r['ratio']))
    baselines=read(root(config)/'development/baselines.json')
    primary=baselines[config['primary_feature']]
    short=dict(primary); short['steps']=config['classifier_short_steps']
    curve=read(Path(primary['directory'])/'training_curve.json')
    short['validation']=next(r['validation'] for r in curve if r['step']==short['steps'])
    raw=train_job(config,registry,train,validation,config['primary_feature'],head['dual_head'],seed,'raw_fixed')
    controls=dict(raw_short=short,raw_single=primary,raw_fixed=raw,dual_candidate=head['dual'])
    control_policies={name:select_workpoint(config,registry,spec) for name,spec in controls.items()}
    ablations={}
    for feature in config['feature_sets']:
        if feature==config['primary_feature']: continue
        spec=train_job(config,registry,train,validation,feature,head['dual_head'],seed,'fixed_ablation')
        ablations[feature]=dict(classifier=spec,alarm_policy=select_workpoint(config,registry,spec))
    report=dict(shared_generator_steps=shared,dual_head=head['dual_head'],selected=selections,all_ratios=rows,
                generators=generators,controls=controls,control_policies=control_policies,ablations=ablations,
                rule_policy=select_workpoint(config,registry,rule=True),selection_scope='train and validation only')
    write_json(root(config)/'development/selection.json',report)
    return report


CORE_FILES=['classify.py','WGAN.py','experiment_protocol.py','simulink_data.py',
            'replay_diagnostics.py','simulink_v2_experiment.py','run_simulink_gan.py']


def read_frozen(config,registry):
    payload=read(root(config)/'frozen_configuration.json')
    unsigned={k:v for k,v in payload.items() if k!='signature'}
    if stable_hash(unsigned)!=payload['signature']: raise ValueError('Frozen configuration signature is invalid')
    if payload['config_signature']!=stable_hash(config) or payload['registry_signature']!=registry['signature']:
        raise ValueError('Data/configuration changed after freeze; use a new results_root')
    if payload['code_sha256']!=code_hashes(ROOT,CORE_FILES):
        raise ValueError('Protocol code changed after freeze; do not reuse final test')
    for path,digest in payload['scaler_sha256'].items():
        if sha256_file(path)!=digest: raise ValueError('Scaler changed after configuration freeze')
    return payload


def freeze_stage(config,registry):
    path=root(config)/'frozen_configuration.json'
    if path.exists(): return read_frozen(config,registry)
    selection=read(root(config)/'development/selection.json')
    payload=dict(config=config,config_signature=stable_hash(config),registry_signature=registry['signature'],
                 code_sha256=code_hashes(ROOT,CORE_FILES),selection=selection,
                 classifier_steps=config['classifier_max_steps'],training_seeds=config['final_seeds'],
                 selection_uses_only='six train and three validation paired groups',
                 final_test_groups=config['roles']['test'],test_features_read_before_freeze=0)
    def scaler_paths(value):
        if isinstance(value,dict):
            found=[str(Path(value['scaler_directory'])/'scalers.pkl')] if 'scaler_directory' in value else []
            return found+[p for child in value.values() for p in scaler_paths(child)]
        if isinstance(value,list): return [p for child in value for p in scaler_paths(child)]
        return []
    payload['scaler_sha256']={path:sha256_file(path) for path in sorted(set(scaler_paths(selection)))}
    access=root(config)/'feature_access.jsonl'
    if access.exists():
        if any(json.loads(line)['role']=='test' for line in access.read_text(encoding='utf-8').splitlines()):
            raise RuntimeError('Test features were accessed before configuration freeze')
    payload['signature']=stable_hash(payload); write_json(path,payload)
    snapshots=root(config)/'frozen_source'; snapshots.mkdir(parents=True,exist_ok=True)
    for name in CORE_FILES: shutil.copy2(ROOT/name,snapshots/name)
    print('CONFIGURATION_FROZEN',payload['signature'],flush=True)
    return payload


def final_train_stage(config,registry,train,validation):
    frozen=read_frozen(config,registry); selected=frozen['selection']; results={}
    for seed in frozen['training_seeds']:
        if seed==config['development_seed']:
            models=dict(selected['controls'])
            gens=selected['generators']
            for method,choice in selected['selected'].items(): models[method]=choice['classifier']
            for mode in ('plain','command'): models[f'ros_matched_{mode}']=matched_ros(selected,mode)['classifier']
            for feature,row in selected['ablations'].items(): models[f'ablation_{feature}']=row['classifier']
        else:
            primary=config['primary_feature']; dual=selected['dual_head']
            single=train_job(config,registry,train,validation,primary,False,seed,'final_single')
            candidate=train_job(config,registry,train,validation,primary,True,seed,'final_dual')
            fixed=candidate if dual else single
            short=dict(single); short['steps']=config['classifier_short_steps']
            short['validation']=next(row['validation'] for row in read(Path(single['directory'])/'training_curve.json') if row['step']==short['steps'])
            models=dict(raw_single=single,dual_candidate=candidate,raw_fixed=fixed,raw_short=short)
            gens={mode:generator_job(config,registry,train,mode,seed,selected['shared_generator_steps']) for mode in ('plain','command')}
            for method,choice in selected['selected'].items():
                if choice['ratio']==0: models[method]=fixed
                else:
                    row=augment_job(config,registry,train,validation,dual,seed,method,choice['ratio'],
                                    gens.get(method),selected['shared_generator_steps'] if method!='ros' else None)
                    models[method]=row['classifier']
            for mode in ('plain','command'):
                ratio=selected['selected'][mode]['ratio']
                models[f'ros_matched_{mode}']=fixed if ratio==0 else augment_job(
                    config,registry,train,validation,dual,seed,'ros',ratio)['classifier']
            for feature in selected['ablations']:
                models[f'ablation_{feature}']=train_job(config,registry,train,validation,feature,dual,seed,'final_ablation')
        hashes={name:sha256_file(Path(spec['directory'])/f'checkpoint_{spec["steps"]}.pt') for name,spec in models.items()}
        gen_hashes={mode:sha256_file(Path(spec['directory'])/f'checkpoint_{selected["shared_generator_steps"]}.pt') for mode,spec in gens.items()}
        results[str(seed)]=dict(models=models,checkpoint_sha256=hashes,generators=gens,
                               selected_generator_updates=selected['shared_generator_steps'],generator_checkpoint_sha256=gen_hashes)
        write_json(root(config)/f'final_training_seed{seed}.json',dict(frozen_signature=frozen['signature'],**results[str(seed)]))
    receipt=dict(frozen_signature=frozen['signature'],seeds=results,all_models_complete=True)
    write_json(root(config)/'final_training_complete.json',receipt)
    print('ALL_FROZEN_MODELS_READY',list(results),flush=True)
    return receipt


def matched_ros(selected,mode):
    ratio=selected['selected'][mode]['ratio']
    return next(row for row in selected['all_ratios'] if row['method']=='ros' and row['ratio']==ratio)


def policy_for(selected,name):
    if name in selected['control_policies']: return selected['control_policies'][name]
    if name.startswith('ros_matched_'): return matched_ros(selected,name[len('ros_matched_'):])['alarm_policy']
    if name.startswith('ablation_'): return selected['ablations'][name[len('ablation_'):]]['alarm_policy']
    return selected['selected'][name]['alarm_policy']


def window_rule_prediction(pool,registry,records):
    lookup={r['source_id']:r for r in records}; probability=[]; labels=[]
    for source_index,end in zip(pool['source_indices'],pool['end_s']):
        record=lookup[registry['records'][int(source_index)]['source_id']]
        idx=int(np.argmin(np.abs(np.asarray(record['time_s'])-end)))
        if abs(record['time_s'][idx]-end)>1e-8: raise ValueError('Replay/window timestamps differ')
        probability.append(record['fault_probability'][idx]); labels.append(record['fault_class'][idx])
    logits=np.zeros((len(labels),19)); logits[np.arange(len(labels)),labels]=1
    return dict(fault_probability=np.asarray(probability),logits=logits)


def test_stage(config,registry):
    frozen=read_frozen(config,registry)
    complete=read(root(config)/'final_training_complete.json')
    if complete['frozen_signature']!=frozen['signature'] or not complete['all_models_complete']:
        raise PermissionError('All frozen models must be complete before unified testing')
    final_path=root(config)/'final_test_results.json'
    if final_path.exists():
        saved=read(final_path)
        current={seed:entry['checkpoint_sha256'] for seed,entry in complete['seeds'].items()}
        if saved['frozen_signature']!=frozen['signature'] or saved['model_checkpoints']!=current:
            raise RuntimeError('Incompatible final test receipt or changed model weights')
        print('FINAL_TEST_REUSE',frozen['signature'],flush=True); return saved
    for seed,entry in complete['seeds'].items():
        for mode,spec in entry['generators'].items():
            checkpoint=Path(spec['directory'])/f'checkpoint_{entry["selected_generator_updates"]}.pt'
            if sha256_file(checkpoint)!=entry['generator_checkpoint_sha256'][mode]:
                raise ValueError('Frozen generator checkpoint changed')
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if state['generator_updates']!=entry['selected_generator_updates'] or state['critic_updates']!=config['n_critic']*state['generator_updates']:
                raise ValueError('Generator/critic optimizer update count mismatch')
    # Only now does the protocol read numeric test features.
    test=v2_pool(config,registry,'test',frozen['signature']); results=[]
    for seed,entry in complete['seeds'].items():
        for name,spec in entry['models'].items():
            checkpoint=Path(spec['directory'])/f'checkpoint_{spec["steps"]}.pt'
            if sha256_file(checkpoint)!=entry['checkpoint_sha256'][name]: raise ValueError('Frozen classifier was modified')
            policy=policy_for(frozen['selection'],name); model=load_classifier(config,spec)
            with (Path(spec['scaler_directory'])/'scalers.pkl').open('rb') as f: cls,_=pickle.load(f)
            prediction=classify.predict_classifier(model,transform(v2_feature(test,spec['feature']),cls))
            records=prediction_records(config,registry,'test',spec,authorization=frozen['signature'])
            alarm=score_alarm_records(records,policy['threshold'],policy['hold'])
            per_group={int(group):metrics(test['y'][test['groups']==group],
                         {k:v[test['groups']==group] for k,v in prediction.items()},policy['threshold'])
                       for group in config['roles']['test']}
            result=dict(seed=int(seed),variant=name,feature=spec['feature'],dual_head=spec['dual_head'],
                        model_checkpoint_sha256=entry['checkpoint_sha256'][name],
                        effective_classifier_updates=spec['steps'],window_metrics=metrics(test['y'],prediction,policy['threshold']),
                        per_group=per_group,alarm=alarm,validation_selected_threshold=policy['threshold'],hold=policy['hold'])
            results.append(result)
            target=root(config)/'test_predictions'/f'seed{seed}_{name}.npz'; target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,y=test['y'],groups=test['groups'],end_s=test['end_s'],**prediction)
    rule_records=prediction_records(config,registry,'test',rule=True,authorization=frozen['signature'])
    policy=frozen['selection']['rule_policy']; rule_predictions=window_rule_prediction(test,registry,rule_records)
    rule=dict(window_metrics=metrics(test['y'],rule_predictions,policy['threshold']),
              alarm=score_alarm_records(rule_records,policy['threshold'],policy['hold']),
              policy=dict(threshold=policy['threshold'],hold=policy['hold']),
              scope='Known nominal motor dynamics/delay, observed commands/RPM only; not a real-flight validation')
    report=dict(frozen_signature=frozen['signature'],test_groups=config['roles']['test'],test_records=57,
                model_checkpoints={seed:entry['checkpoint_sha256'] for seed,entry in complete['seeds'].items()},
                test_independent_paired_groups=3,training_seeds=config['final_seeds'],results=results,rule=rule,
                independence_note='Training-seed repeats share the same three held-out source groups; windows and 9 seed/group cells are not 9 independent flights.')
    write_json(final_path,report)
    print('UNIFIED_FINAL_TEST_COMPLETE',len(results),'model/seed evaluations',flush=True)
    return report


def main_action(action,config):
    if len(config['new_seeds'])!=8 or len(set(config['final_seeds']))<3:
        raise ValueError('V2 requires eight new paired conditions and at least three distinct training seeds')
    if config['classifier_short_steps'] not in config['classifier_checkpoints']:
        raise ValueError('Short-budget checkpoint must be retained')
    if action in ('plan','collect'):
        path=prepare_plan(ROOT/config['new_raw'],config['new_seeds'],config['task_family'],config['dataset_version']+'_new8')
        if action=='collect':
            import subprocess
            subprocess.run(['C:/Program Files/MATLAB/R2025b/bin/matlab.exe','-batch',
                            f"addpath('{(ROOT/'simulink').as_posix()}'); collect_hex_dataset('{path.as_posix()}')"],check=True,cwd=ROOT)
        print(path); return
    registry=v2_registry(config)
    if action=='registry': print(registry['signature']); return
    if (root(config)/'frozen_configuration.json').exists() and action in ('baselines','heads','gan_train','gans'):
        raise RuntimeError('Development is frozen. Use a new results_root for further selection.')
    if action=='test': return test_stage(config,registry)
    if action=='freeze': return freeze_stage(config,registry)
    train=v2_pool(config,registry,'train'); validation=v2_pool(config,registry,'validation')
    print('V2_POOLS',len(train['y']),len(validation['y']),'TEST_LOCKED',not (root(config)/'frozen_configuration.json').exists(),flush=True)
    if action=='preprocess': return
    if action=='baselines': baseline_stage(config,registry,train,validation); return
    if action=='heads': head_stage(config,registry,train,validation); return
    if action=='gan_train': generator_train_stage(config,registry,train); return
    if action=='gans': gans_stage(config,registry,train,validation); return
    if action=='final_train': return final_train_stage(config,registry,train,validation)
    if action=='all':
        if not (root(config)/'frozen_configuration.json').exists():
            baseline_stage(config,registry,train,validation); head_stage(config,registry,train,validation)
            gans_stage(config,registry,train,validation); freeze_stage(config,registry)
        final_train_stage(config,registry,train,validation); return test_stage(config,registry)
    raise ValueError(f'V2 action not implemented yet: {action}')
