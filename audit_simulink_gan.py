"""Audit actual collected records and fitted artifacts, not just source syntax."""
import argparse
import json
import pickle
from pathlib import Path
import subprocess
import numpy as np

from provenance import sha256_file, write_json
from run_simulink_gan import environment
from simulink_data import ROOT, read, finalize_dataset
from experiment_protocol import generation_checks


def audit(config,feature):
    _,processed,output,_=environment(config,feature,'prototype')
    manifest=finalize_dataset(ROOT/config['data_dir'])
    x=np.load(processed/'X_raw.npy'); y=np.load(processed/'y_all.npy'); groups=np.load(processed/'groups.npy')
    folds=[]
    for fold in range(1,config['folds']+1):
        path=output/'folds'/f'fold_{fold}'; spec=read(path/'fold_signature.json')
        tr=np.asarray(spec['train_idx']); te=np.asarray(spec['test_idx'])
        assert not set(groups[tr]) & set(groups[te])
        assert len(set(groups[te]))==1 and set(y[te])==set(range(19))
        assert set(np.flatnonzero(groups==groups[te][0]))==set(te)
        assert len(set(groups[tr]))==config['train_max_groups']
        assert all(np.sum(y[tr]==i)==config['train_windows_per_class'] for i in range(19))
        for filename in ('classifier_scaler.pkl','gan_minmax_scaler.pkl'):
            with (path/filename).open('rb') as f: scaler=pickle.load(f)
            if 'minmax' in filename:
                np.testing.assert_array_equal(scaler.data_min_,x[tr].min((0,1)))
                np.testing.assert_array_equal(scaler.data_max_,x[tr].max((0,1)))
            else:
                np.testing.assert_allclose(scaler.mean_,x[tr].mean((0,1),dtype=np.float64),rtol=0,atol=1e-9)
        counts=read(path/'wgan_plain/augmentation_counts.json'); ros=read(path/'random_oversampling_counts.json')
        gen=np.load(path/'wgan_plain/y_gen.npy'); quality=np.load(path/'wgan_plain/y_quality.npy')
        for i in range(19):
            item=counts[str(i)]; target=int(np.floor(np.sum(y[tr]==i)*config['augmentation_ratio']))
            assert item['original']==np.sum(y[tr]==i)
            assert item['planned']==item['actual_generated']==item['joined_training']==np.sum(gen==i)==target
            assert item['quality_only']==np.sum(quality==i)==config['quality_samples_per_class']
            assert ros[str(i)]['joined_training']==target and ros[str(i)]['quality_only']==0
        losses=np.load(path/'wgan_plain/models/gan_losses.npz')
        assert losses['generator_updates'][0]>0 and losses['generator_parameter_max_delta'][0]>0
        raw=np.load(path/'wgan_plain/X_gen_raw.npy'); assert np.isfinite(raw).all()
        checks=generation_checks(x[tr],y[tr],raw,gen,read(processed/'feature_names.json'),groups[tr])
        for variant in ('no_gan','class_weighted','random_oversampling','wgan_plain'):
            saved=np.load(path/variant/'test_predictions.npz')
            np.testing.assert_array_equal(saved['test_indices'],te)
            np.testing.assert_array_equal(saved['test_groups'],groups[te])
        folds.append(dict(fold=fold,training_groups=spec['train_groups'],test_groups=spec['test_groups'],
                          original_training_windows=len(tr),generated_joined=len(gen),quality_only=len(quality),
                          generator_updates=int(losses['generator_updates'][0]),
                          generator_parameter_max_delta=float(losses['generator_parameter_max_delta'][0]),
                          joined_sample_quality_checks=checks))
    # Protected math/model files must remain byte-identical to the starting version.
    protected=['simulink/hex_evtol.slx','simulink/configs/research.json','simulink/hex_parameters.m',
               'simulink/hex_base_parameters.m','simulink/hex_prepare_inputs.m','simulink/hex_metrics.m',
               'simulink/blocks','simulink/functions','scenario_config.py','uavfd_f550']
    changed=subprocess.check_output(['git','diff','65583a23fcea6a6887d1e77e591c49c827ef4d03','--name-only','--',*protected],cwd=ROOT,text=True)
    assert not changed.strip(),f'Protected simulation/legacy/UAV-FD source changed: {changed}'
    failed=[r['run_key'] for r in manifest['records'] if not r['flight_acceptance_passed']]
    result=dict(passed=True,feature_set=feature,source_runs=len(manifest['records']),paired_groups=len(set(groups)),
                windows=len(y),flight_acceptance_failures_retained=failed,folds=folds,
                source_plan_signature=manifest['plan_signature'],protected_math_unchanged=True,
                source_manifest_sha256=sha256_file(processed/'source_manifest.json'))
    write_json(output/'behavioral_audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,default=ROOT/'configs/simulink_gan_prototype.json')
    parser.add_argument('--feature-set',default='imu_cmd_rpm')
    args=parser.parse_args(); audit(read(args.config),args.feature_set)
