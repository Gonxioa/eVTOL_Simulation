"""Independent audit of V2 fitted arrays, optimizer counts and sealed-test artifacts."""
import argparse
import json
import pickle
from pathlib import Path
import subprocess
import numpy as np
import torch

from provenance import sha256_file,write_json,stable_hash
from simulink_data import ROOT,read,v2_registry,v2_pool,v2_feature,FEATURES
from simulink_v2_experiment import read_frozen,array_hash,transform


def audit(config):
    base=ROOT/config['results_root']; registry=v2_registry(config); frozen=read_frozen(config,registry)
    final=read(base/'final_test_results.json'); complete=read(base/'final_training_complete.json')
    assert final['frozen_signature']==complete['frozen_signature']==frozen['signature']
    assert set(complete['seeds'])==set(map(str,config['final_seeds']))
    assert frozen['test_features_read_before_freeze']==0
    access=[json.loads(line) for line in (base/'feature_access.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(row['authorization']==frozen['signature'] for row in access if row['role']=='test')
    pools={role:v2_pool(config,registry,role,frozen['signature'] if role=='test' else None)
           for role in ('train','validation','test')}
    for role,pool in pools.items():
        assert set(pool['groups'])==set(config['roles'][role])
        assert set(pool['y'])==set(range(19))
        healthy=pool['y']==0
        assert np.all(pool['regime'][healthy]=='normal')
        for source,end,label in zip(pool['source_indices'],pool['end_s'],pool['y']):
            record=registry['records'][int(source)]
            assert record['role']==role and record['run']['label']==label
            if label: assert end>=3
        expected_healthy=47*len(config['roles'][role])
        assert int(np.sum(healthy))==expected_healthy
    role_sets=[set(pool['groups']) for pool in pools.values()]
    assert not role_sets[0]&role_sets[1] and not role_sets[0]&role_sets[2] and not role_sets[1]&role_sets[2]
    # All model/physical settings agree; only nuisance seed and injected class vary.
    common=[]
    for record in registry['records']:
        p=read(Path(record['source_path'])/'run_metadata.json')['parameters']
        common.append(stable_hash({k:v for k,v in p.items() if k not in ('seed','fault_motor','fault_factor')}))
    assert len(set(common))==1
    assert sum(r['source_pool']=='v1' for r in registry['records'])==76
    assert all(sha256_file(Path(r['source_path'])/'signals.csv')==r['csv_sha256'] for r in registry['records'])
    training=pools['train']; checked={}; classifier_receipts=[]; generator_receipts=[]
    for seed,entry in complete['seeds'].items():
        for mode,spec in entry['generators'].items():
            path=Path(spec['directory'])/f'checkpoint_{entry["selected_generator_updates"]}.pt'
            assert sha256_file(path)==entry['generator_checkpoint_sha256'][mode]
            state=torch.load(path,map_location='cpu',weights_only=False); job=read(Path(spec['directory'])/'job.json')
            assert state['generator_updates']==entry['selected_generator_updates']==frozen['selection']['shared_generator_steps']
            assert state['critic_updates']==config['n_critic']*state['generator_updates']
            assert all(int(value['step'])==state['generator_updates'] for value in state['optimizer_g']['state'].values())
            assert all(int(value['step'])==state['critic_updates'] for value in state['optimizer_d']['state'].values())
            assert state['generator_parameter_max_delta']>0
            assert job['train_pool']==training['signature']
            with (Path(spec['scaler_directory'])/'scalers.pkl').open('rb') as handle: _,gan=pickle.load(handle)
            raw=v2_feature(training,config['primary_feature'])
            assert job['X_sha256']==array_hash(transform(raw,gan)) and job['y_sha256']==array_hash(training['y'])
            generator_receipts.append(dict(seed=int(seed),mode=mode,generator_updates=state['generator_updates'],
                                            critic_updates=state['critic_updates'],parameter_max_delta=state['generator_parameter_max_delta']))
            del state
        for name,spec in entry['models'].items():
            path=Path(spec['directory'])/f'checkpoint_{spec["steps"]}.pt'
            assert sha256_file(path)==entry['checkpoint_sha256'][name]
            if str(path) in checked:
                classifier_receipts.append(dict(seed=int(seed),variant=name,**checked[str(path)])); continue
            state=torch.load(path,map_location='cpu',weights_only=False); job=read(Path(spec['directory'])/'job.json')
            expected=config['classifier_short_steps'] if name=='raw_short' else config['classifier_max_steps']
            assert state['step']==spec['steps']==expected
            assert all(int(value['step'])==expected for value in state['optimizer']['state'].values())
            assert job['train_pool']==training['signature']
            raw=v2_feature(training,spec['feature'])
            with (Path(spec['scaler_directory'])/'scalers.pkl').open('rb') as handle: cls,gan=pickle.load(handle)
            np.testing.assert_allclose(cls.mean_,raw.mean((0,1),dtype=np.float64),rtol=0,atol=1e-7)
            np.testing.assert_allclose(cls.var_,raw.var((0,1),dtype=np.float64),rtol=0,atol=1e-5)
            np.testing.assert_array_equal(gan.data_min_,raw.min((0,1))); np.testing.assert_array_equal(gan.data_max_,raw.max((0,1)))
            assert int(cls.n_samples_seen_)==len(raw)*config['window']
            source_info=read(Path(spec['scaler_directory'])/'fit_sources.json')
            assert source_info['training_groups']==config['roles']['train']
            arrays=Path(spec['directory'])/'augmentation.npz'; y=training['y']; joined=0
            if arrays.exists():
                added=dict(np.load(arrays)); extra=added['X']; ey=added['y']; refs=added['command_source_indices']; joined=len(ey)
                counts=read(Path(spec['directory'])/'augmentation_counts.json')
                for label in range(19):
                    c=counts[str(label)]
                    assert c['original']==np.sum(y==label)
                    assert c['planned']==c['actual_generated']==c['joined_training']==np.sum(ey==label)
                    assert c['quality_only']==0
                if np.all(refs>=0):
                    assert np.array_equal(y[refs],ey)
                    assert set(training['groups'][refs])<=set(config['roles']['train'])
                    if Path(spec['directory']).name.startswith('command_'):
                        idx=[FEATURES[spec['feature']].index(f'cmd_motor{i}') for i in range(6)]
                        np.testing.assert_array_equal(extra[:,:,idx],raw[refs][:,:,idx])
                    else: np.testing.assert_array_equal(extra,raw[refs])
                raw=np.concatenate([raw,extra]); y=np.concatenate([y,ey])
            assert job['X_sha256']==array_hash(transform(raw,cls)) and job['y_sha256']==array_hash(y)
            weights=state['binary_weights']; n0=np.sum(y==0); n1=np.sum(y!=0)
            np.testing.assert_allclose(weights[0]*n0,weights[1]*n1,rtol=1e-12)
            receipt=dict(optimizer_updates=expected,training_examples=len(y),generated_joined=joined,dual_head=spec['dual_head'],
                         checkpoint_sha256=entry['checkpoint_sha256'][name])
            checked[str(path)]=receipt; classifier_receipts.append(dict(seed=int(seed),variant=name,**receipt)); del state
    for row in final['results']:
        assert set(map(int,row['per_group']))==set(config['roles']['test'])
        assert row['alarm']['fault_records']==54
        assert abs(row['alarm']['normal_observation_s']-69)<1e-8
    protected=['simulink/hex_evtol.slx','simulink/configs','simulink/hex_parameters.m','simulink/hex_base_parameters.m',
               'simulink/hex_prepare_inputs.m','simulink/hex_metrics.m','simulink/blocks','simulink/functions',
               'scenario_config.py','uavfd_f550']
    diff=subprocess.check_output(['git','diff','6f946123638f8629bb4c79126e4fe8bd83638b57','--name-only','--',*protected],cwd=ROOT,text=True)
    assert not diff.strip(),diff
    report=dict(passed=True,frozen_signature=frozen['signature'],record_count=228,independent_paired_groups=12,
                roles=config['roles'],old_source_files_referenced=76,old_hashes_preserved=True,model_parameters_identical=True,
                statistical_unit='paired condition group, not simulation windows or training-seed repeats',
                window_counts={r:len(pool['y']) for r,pool in pools.items()},
                transition_windows={r:int(np.sum(pool['regime']=='transition')) for r,pool in pools.items()},
                generator_receipts=generator_receipts,classifier_receipts=classifier_receipts,
                test_access_only_with_frozen_authorization=True,physical_model_and_uavfd_unchanged=True,
                optimizer_counts_verified=True,scalers_training_only_verified=True)
    write_json(base/'behavioral_audit.json',report); print(json.dumps({k:v for k,v in report.items() if k not in ('generator_receipts','classifier_receipts')},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,default=ROOT/'configs/simulink_gan_v2.json')
    args=parser.parse_args(); audit(read(args.config))
