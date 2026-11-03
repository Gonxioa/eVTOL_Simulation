"""Reproducible entrypoint that orchestrates the EXISTING preprocessing/WGAN/classifier."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import shutil

from provenance import write_json, code_hashes, stable_hash
from simulink_data import ROOT, PROFILE, read, prepare_plan, preprocess, finalize_dataset


def environment(config,feature,stage):
    base=ROOT/config['results_root']; processed=base/'processed'/f'{config["experiment_revision"]}_{feature}'
    output=base/'experiments'/f'{config["experiment_revision"]}_{stage}_{feature}_ratio{config["augmentation_ratio"]:g}_seed{config["seed"]}'
    env=os.environ.copy()
    values=dict(DATA_PROFILE=PROFILE,DATA_DIR_OVERRIDE=processed,OUTPUT_DIR_OVERRIDE=output,
                CLASSIFIER_SAVE_DIR_OVERRIDE=output,EXPERIMENT_SEED=config['seed'],N_FOLDS=config['folds'],
                CLASSIFIER_EPOCHS=1 if stage=='smoke' else config['classifier_epochs'],
                GAN_EPOCHS=1 if stage=='smoke' else config['gan_epochs'],
                GAN_QUALITY_CLASSIFIER_EPOCHS=1 if stage=='smoke' else config['quality_classifier_epochs'],
                GAN_QUALITY_SAMPLES=config['quality_samples_per_class'],GAN_QUALITY_EVAL=1,
                CLASSIFIER_HIDDEN_DIM=config['classifier_hidden_dim'],GAN_HIDDEN_DIM=config['gan_hidden_dim'],
                CLASSIFIER_LAYERS=config['layers'],GAN_LAYERS=config['layers'],
                CLASSIFIER_BATCH_SIZE=config['batch_size'],GAN_BATCH_SIZE=config['batch_size'],
                GAN_GENERATE_BATCH_SIZE=config['batch_size'],GAN_N_CRITIC=config['gan_n_critic'],
                TRAIN_MAX_GROUPS=config['train_max_groups'],TRAIN_WINDOWS_PER_CLASS=config['train_windows_per_class'],
                GAN_AUGMENT_MODE=config['augmentation_mode'],GAN_AUGMENT_RATIO=config['augmentation_ratio'],
                GAN_PHYSICS_LAMBDAS=0,GAN_EARLY_STOPPING=0,GAN_RESTORE_BEST=0,
                GAN_TARGET_POLICY='max',GAN_MAX_GENERATED_PER_CLASS=0,CLASSIFIER_RESUME=1,
                CLASSIFIER_ATTENTION_POOLING=1,FOLD_GAN_SAVE_MODELS=1,GAN_TEMPORAL_NOISE=1,
                MPLBACKEND='Agg',PYTHONUTF8=1)
    env.update({k:str(v) for k,v in values.items()})
    return env,processed,output,values


def train(config,feature,stage):
    env,processed,output,values=environment(config,feature,stage)
    output.mkdir(parents=True,exist_ok=True)
    snapshot=dict(config=config,feature_set=feature,stage=stage,environment={k:str(v) for k,v in values.items()},
                  code_sha256=code_hashes(ROOT,['classify.py','WGAN.py','simulink_data.py','experiment_protocol.py','run_simulink_gan.py']))
    snapshot['signature']=stable_hash(snapshot)
    path=output/'resume_config.json'
    if path.exists() and read(path)['signature']!=snapshot['signature']:
        raise RuntimeError('Incompatible experiment resume directory; select a new results_root')
    write_json(path,snapshot)
    for name in snapshot['code_sha256']:
        shutil.copy2(ROOT/name,output/('source_'+name))
    with (output/'training.log').open('a',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-u',str(ROOT/'classify.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Training failed; see {output / "training.log"}')
    print(stage,feature,read(output/'summary.json'),flush=True)
    return env,output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['plan','collect','preprocess','smoke','prototype','replay','all'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/simulink_gan_prototype.json')
    parser.add_argument('--feature-set',choices=['imu','imu_cmd','imu_rpm','imu_cmd_rpm'])
    args=parser.parse_args(); config=read(args.config)
    if config['profile']!=PROFILE or config['physical_regularization']!=0:
        raise ValueError('Unsupported profile or unvalidated physical target')
    feature=args.feature_set or config['feature_set']; raw=ROOT/config['data_dir']
    if args.action=='plan': print(prepare_plan(raw)); return
    if args.action=='collect':
        plan=prepare_plan(raw)
        matlab=Path('C:/Program Files/MATLAB/R2025b/bin/matlab.exe')
        subprocess.run([str(matlab),'-batch',f"addpath('{(ROOT/'simulink').as_posix()}'); collect_hex_dataset('{plan.as_posix()}')"],check=True,cwd=ROOT)
        finalize_dataset(raw); return
    if args.action in ('preprocess','all'):
        _,destination,_,_=environment(config,feature,'prototype')
        preprocess(raw,destination,feature,config['window'],config['step'],config['fault_buffer_s'])
    if args.action in ('smoke','all'): train(config,feature,'smoke')
    if args.action in ('prototype','all'): train(config,feature,'prototype')
    if args.action in ('replay','all'):
        env,_,output,_=environment(config,feature,'prototype')
        for fold in range(1,config['folds']+1):
            subprocess.run([sys.executable,str(ROOT/'replay_diagnostics.py'),'--fold',str(output/'folds'/f'fold_{fold}'),
                            '--data',str(raw),'--stride',str(config['replay_stride']),
                            '--probability',str(config['replay_probability']),'--hold',str(config['replay_hold'])],
                           check=True,cwd=ROOT,env=env)


if __name__=='__main__': main()
