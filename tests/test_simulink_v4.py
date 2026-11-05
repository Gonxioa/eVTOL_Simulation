"""V4 endpoint fairness, native prefix geometry, cache integrity and true resume."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from simulink_data import read
from simulink_v3_data import array_hash
from simulink_v4_data import CONFIG,indices,pool,fit_scalers,root
from simulink_v4_models import FixedCNN,statistical_features
from simulink_v4_training import train,diagnostics,evaluate,rank_candidates


def fixture_pool(role):
    rng=np.random.default_rng(19); y=np.tile(np.arange(19),2)
    x=rng.normal(size=(len(y),8,12)).astype(np.float32)
    identity=dict(role=role,feature='imu_rpm',prefix_s=24,grid='native',sampling_seed=None)
    return dict(X=x,y=y,groups=np.full(len(y),0 if role=='train' else 6),source_ids=np.array([f'{role}_{v}' for v in y]),
        starts=np.full(len(y),400),ends=np.full(len(y),407),physical_fault_fraction=(y!=0).astype(float),
        crosses_fault=np.zeros(len(y),bool),signature='fixture_'+role,identity=identity,
        X_sha256=array_hash(x),y_sha256=array_hash(y),selection=[])


class V4Tests(unittest.TestCase):
    def test_common_windows_have_same_ends_and_counts(self):
        reference=None
        for w in (50,100,200,300):
            s,e,f,t=indices(w,24,'common',0)
            self.assertEqual((len(s),e[0],e[-1]),(43,299,2399))
            if reference is not None: np.testing.assert_array_equal(reference,e)
            reference=e
            fs,fe,_,_=indices(w,24,'common',1)
            self.assertEqual((len(fe),fe[0]),(42,349))
            self.assertEqual(6*(len(e)+18*len(fe)),4794)

    def test_native_prefix_counts_from_zero(self):
        counts={50:[720,2088,4824],100:[714,2082,4818],200:[702,2070,4806],300:[690,2058,4794]}
        for w,expected in counts.items():
            actual=[]
            for prefix in (6,12,24):
                s,e,_,_=indices(w,prefix,'native',0); _,fe,_,_=indices(w,prefix,'native',1)
                self.assertEqual(s[0],0); self.assertTrue(np.all(e<prefix*100))
                actual.append(6*(len(e)+18*len(fe)))
            self.assertEqual(actual,expected)

    def test_fault_fraction_and_transport_metadata(self):
        for w,ratio in zip((50,100,200,300),(1.,.5,.25,1/6)):
            s,e,f,t=indices(w,24,'common',1)
            self.assertAlmostEqual(f[0],ratio)
            self.assertAlmostEqual(t[0],47/w)
            self.assertEqual(e[0],349)
        with self.assertRaises(ValueError): indices(400,24,'common',1)

    def test_no_future_or_cross_record_padding(self):
        for w in (50,300):
            s,e,_,_=indices(w,6,'native',1)
            self.assertTrue(np.all(e-s==w-1)); self.assertTrue(np.all(s>=0)); self.assertTrue(np.all(e<600))
            self.assertTrue(np.all(e>=300))

    def test_test_arrays_locked_before_cache_access(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=dict(read(CONFIG),results_root=folder)
            with self.assertRaises(PermissionError): pool(cfg,{'signature':'fixture'},'test',100)
            self.assertFalse(list(Path(folder).rglob('*.pkl')))

    def test_statistical_features_are_finite_and_fixed_size(self):
        for w in (50,100,200,300):
            out=statistical_features(np.ones((3,w,12),np.float32))
            self.assertEqual(out.shape,(3,198)); self.assertTrue(np.isfinite(out).all())

    def test_fixed_CNN_handles_registered_lengths(self):
        cfg=read(CONFIG)
        for w in (50,100,200,300):
            model=FixedCNN(cfg); out=model(torch.zeros(2,w,12))
            self.assertEqual(tuple(out.shape),(2,19)); self.assertTrue(torch.isfinite(out).all())

    def test_phase_macro_f1_support_and_joint_severity(self):
        y=np.arange(1,19); logits=np.eye(19)[y]*8
        m=diagnostics(y,logits)
        self.assertAlmostEqual(m['macro_f1'],18/19); self.assertEqual(m['present_class_macro_f1'],1.)
        self.assertIsNone(m['healthy_recall']); self.assertEqual(m['joint_accuracy_by_effectiveness']['0.8'],1.)
        # Predict opposite motors of the correct severity: severity is correct, joint localization is wrong.
        opposite=1+(y-1)//6*6+((y-1)%6+3)%6
        m=diagnostics(y,np.eye(19)[opposite]*8)
        self.assertEqual(m['severity_only_accuracy_on_fault'],1.); self.assertEqual(m['motor_accuracy_on_fault'],0.)

    def test_empty_mixed_scope_is_recorded_not_invented(self):
        p=fixture_pool('validation'); out=evaluate(p,np.eye(19)[p['y']]*8)
        self.assertEqual(out['mixed']['window_count'],0); self.assertIsNone(out['mixed']['macro_f1'])

    def test_equal_source_counts_scaler_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=dict(read(CONFIG),results_root=folder)
            records=[]
            for label in range(19):
                x=np.repeat(np.arange(2401,dtype=np.float32)[:,None],12,axis=1)+label
                records.append(dict(record=dict(source_id=f's{label}',group_id=0,run=dict(label=label)),
                    meta={'fault_start_s':3.},time=np.arange(2401)/100,X=x,phase=np.arange(2401)/100,progress=np.ones(2401)))
            reg={'signature':'fixture'}
            with patch('simulink_v4_data.signals',return_value=records):
                p=pool(cfg,reg,'train',100,24,'native',731905,6)
                d=pool(cfg,reg,'train',100,6,'native')
            self.assertEqual(len(p['y']),119)
            self.assertEqual([(r['source_id'],r['count']) for r in p['selection']],[(r['source_id'],r['count']) for r in d['selection']])
            cls,_,_=fit_scalers(cfg,p)
            np.testing.assert_allclose(cls.mean_,p['X'].reshape(-1,12).mean(0,dtype=np.float64))
            self.assertTrue(np.any(p['starts']>=1200)); self.assertEqual(int(cls.n_samples_seen_),11900)
            import pickle
            path=root(cfg)/'pools'/f'{p["signature"]}.pkl'
            p['X'][0,0,0]+=999
            with path.open('wb') as f: pickle.dump(p,f)
            with self.assertRaises(ValueError): pool(cfg,reg,'train',100,24,'native',731905,6)

    def test_budget_resume_restores_actual_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=dict(read(CONFIG),results_root=str(Path(directory)/'full'),lstm_hidden=4,lstm_layers=1,
                     budget_candidates=[1,2],checkpoints=[1,2])
            p,v=fixture_pool('train'),fixture_pool('validation')
            with patch('simulink_v4_training.DEVICE',torch.device('cpu')):
                a=train(cfg,p,v,'lstm',42,2)
                c2=dict(cfg,results_root=str(Path(directory)/'resume'))
                train(c2,p,v,'lstm',42,1); b=train(c2,p,v,'lstm',42,2)
            sa=torch.load(Path(a['directory'])/'last.pt',map_location='cpu',weights_only=False)
            sb=torch.load(Path(b['directory'])/'last.pt',map_location='cpu',weights_only=False)
            self.assertEqual({int(s['step']) for s in sb['optimizer']['state'].values()},{2})
            for name,t in sa['model'].items(): torch.testing.assert_close(t,sb['model'][name],rtol=0,atol=0)
            self.assertEqual([h['step'] for h in sb['history']],[1,2])
            self.assertGreater(sb['parameter_max_delta'],0.)

    def test_ranking_ties_and_registered_seed_design(self):
        rows=[dict(mean_macro_f1=.5,window=w,steps=s,parameters=p,family=f) for w,s,p,f in
              [(200,1000,10,'cnn'),(100,4000,10,'cnn'),(100,1000,20,'lstm'),(100,1000,10,'cnn')]]
        self.assertEqual(rank_candidates(rows)[0]['family'],'cnn'); self.assertEqual(rank_candidates(rows)[0]['steps'],1000)
        cfg=read(CONFIG)
        self.assertFalse(set(cfg['training_seeds'])&set(cfg['generator_seeds']))
        self.assertFalse(cfg['G2_enabled']); self.assertFalse(cfg['synthetic_filter_enabled'])


if __name__=='__main__': unittest.main()
