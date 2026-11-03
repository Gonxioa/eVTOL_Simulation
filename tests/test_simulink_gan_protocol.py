"""Behavioral contract tests; synthetic fixtures here are never research data."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import classify
import WGAN
from experiment_protocol import augmentation_counts,budget_indices,assert_complete_pairs,count_receipt,generation_checks
from simulink_data import scenarios,observable_features,FEATURES,prepare_plan,checked_plan
from replay_diagnostics import causal_predictions


class ProtocolTests(unittest.TestCase):
    def test_complete_pair_holdout_and_budget(self):
        y=np.tile(np.repeat(np.arange(19),3),4); groups=np.repeat(np.arange(4),57)
        assert_complete_pairs(y,groups,19)
        for tr,te in StratifiedGroupKFold(4,shuffle=True,random_state=42).split(y,y,groups):
            self.assertFalse(set(groups[tr]) & set(groups[te]))
            self.assertEqual(set(y[te]),set(range(19)))
            chosen=budget_indices(tr,y,groups,7,2,4)
            self.assertTrue(set(chosen)<=set(tr)); self.assertEqual(len(set(groups[chosen])),2)
            self.assertEqual(set(np.bincount(y[chosen])),{4})
        with self.assertRaises(ValueError): assert_complete_pairs(y[1:],groups[1:],20)

    def test_balanced_ratio_counts_and_quality_separation(self):
        y=np.repeat(np.arange(19),60)
        self.assertEqual(sum(augmentation_counts(y,19).values()),0)
        for ratio,n in [(0,0),(.5,30),(1,60),(2,120)]:
            planned=augmentation_counts(y,19,'ratio',ratio)
            self.assertEqual(set(planned.values()),{n})
            gen=np.repeat(np.arange(19),n); q=np.repeat(np.arange(19),7)
            receipt=count_receipt(y,gen,planned,19,q)
            self.assertEqual(receipt[0]['joined_training'],n)
            self.assertEqual(receipt[0]['quality_only'],7)

    def test_random_oversampling_matches_gan_ratio(self):
        old=(classify.N_CLASSES,classify.AUGMENT_MODE,classify.AUGMENT_RATIO)
        try:
            classify.N_CLASSES=19; classify.AUGMENT_MODE='ratio'
            y=np.repeat(np.arange(19),6); x=np.arange(len(y))[:,None,None].astype(np.float32)
            for ratio,expected in [(0,6),(.5,9),(1,12),(2,18)]:
                classify.AUGMENT_RATIO=ratio
                _,labels=classify.random_oversample(x,y,42)
                self.assertEqual(set(np.bincount(labels)),{expected})
        finally:
            classify.N_CLASSES,classify.AUGMENT_MODE,classify.AUGMENT_RATIO=old

    def test_label_mapping_strict_full_and_legacy_unchanged(self):
        from scenario_config import FULL_FAULT_FACTOR_OVERRIDES
        table=scenarios(); self.assertEqual(len(table),19)
        for item in table[1:]: self.assertEqual(item['motor_python']+1,item['motor_simulink'])
        self.assertTrue(all(s['factor']==0 for s in table if s['severity']=='full'))
        self.assertEqual(FULL_FAULT_FACTOR_OVERRIDES,{4:.05,5:.05})

    def test_observable_units_delay_and_no_truth_fallback(self):
        n=8; data={}
        for prefix,number in [('accelerometer_sensor_m_s2',3),('gyro_sensor_rad_s',3),('rotor_command_rad_s',6)]:
            for i in range(1,number+1): data[f'{prefix}_{i}']=np.arange(n)+i*10.
        for i in range(14,20): data[f'feedback_state_{i}']=np.arange(n)+900.+i
        for i in range(1,7): data[f'rotor_actual_rad_s_{i}']=np.full(n,99999.)
        df=pd.DataFrame(data); meta={'feedback_delay_s':.03,'parameters':{'sensor_dt':.01}}
        x=observable_features(df,meta,'imu_cmd_rpm'); self.assertEqual(x.shape,(n,18))
        self.assertEqual(x[4,0],11.)  # delayed measured accelerometer; no gravity subtraction
        self.assertEqual(x[4,6],14.)  # current command, rad/s unchanged
        self.assertEqual(x[4,12],918.)  # feedback_state_14, not mechanical speed
        for feature,width in [('imu',6),('imu_cmd',12),('imu_rpm',12),('imu_cmd_rpm',18)]:
            self.assertEqual(observable_features(df,meta,feature).shape[1],width)
        with self.assertRaises(ValueError): observable_features(df.drop(columns='feedback_state_14'),meta,'imu_cmd_rpm')

    def test_scalers_only_fit_training(self):
        old=classify.N_FEATURES; classify.N_FEATURES=2
        train=np.arange(24,dtype=np.float32).reshape(3,4,2); test=np.full((1,4,2),1e6,np.float32)
        _,_,_,cls,gan=classify.fit_scalers(train,test)
        np.testing.assert_allclose(cls.mean_,train.reshape(-1,2).mean(0))
        np.testing.assert_allclose(gan.data_max_,train.reshape(-1,2).max(0))
        classify.N_FEATURES=old

    def test_data_config_signature_and_incompatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); prepare_plan(folder); plan=checked_plan(folder)
            self.assertEqual(len(plan['runs']),76)
            with self.assertRaises(RuntimeError): prepare_plan(folder,seeds=(7,8,9,10))
            classify.atomic_pickle(folder/'checkpoint.pkl',{'signature':'old'})
            self.assertIsNone(classify.load_checkpoint(folder/'checkpoint.pkl','new'))

    def test_quality_checks_duplicates_against_raw_similarity(self):
        rng=np.random.default_rng(12); x=rng.normal(size=(8,10,2)).astype(np.float32); y=np.repeat([0,1],4)
        report=generation_checks(x,y,x[:2],y[:2],['a','b'],np.tile([0,1],4))
        self.assertTrue(report['finite']); self.assertEqual(report['exact_training_duplicates'],2)
        self.assertIsNotNone(report['raw_cross_source_nearest_rms_p05'])

    def test_causal_prefix_invariance(self):
        class Dummy(torch.nn.Module):
            def forward(self,x):
                return torch.stack((x.mean((1,2)),-x.mean((1,2))),1)
        model=Dummy().to(classify.DEVICE); x=np.arange(30,dtype=np.float32).reshape(15,2)
        scaler=StandardScaler().fit(x[:6]); first=causal_predictions(model,scaler,x[:10],4,1,.0,2)
        x[10:]=1e7; after=causal_predictions(model,scaler,x,4,1,.0,2)
        self.assertEqual(first,after[:len(first)])

    def test_generator_updates_with_one_small_batch(self):
        WGAN.configure_dimensions(6,19,8)
        rng=np.random.default_rng(1); x=rng.uniform(-1,1,(19,8,6)).astype(np.float32); y=np.arange(19)
        with tempfile.TemporaryDirectory() as folder:
            g,*_=WGAN.train(x,y,n_epochs=1,save_dir=folder,save_models=False,verbose=False)
            self.assertGreater(g.training_history['generator_updates'],0)
            self.assertGreater(g.training_history['generator_parameter_max_delta'],0)

    def test_zero_ratio_generation_is_empty_without_forward(self):
        class MustNotRun(torch.nn.Module):
            def forward(self,*args): raise AssertionError('Zero augmentation must not sample a network')
        WGAN.configure_dimensions(6,19,100)
        x,y=WGAN.generate_synthetic(MustNotRun(),list(range(19)),{i:0 for i in range(19)},verbose=False)
        self.assertEqual(x.shape,(0,100,6)); self.assertEqual(len(y),0)


if __name__=='__main__': unittest.main()
