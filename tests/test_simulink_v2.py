"""V2 budgets, conditions, source lock and separate alarm clocks."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

import classify
import WGAN
from simulink_data import read,ROOT,v2_authorize,v2_pool
from replay_diagnostics import alarm_trace,motor_response_scores,score_alarm_records
from experiment_protocol import stratified_generation_quality


class V2Tests(unittest.TestCase):
    def setUp(self):
        self.cstate={k:getattr(classify,k) for k in ('N_FEATURES','N_CLASSES','SEQ_LEN','HIDDEN_DIM','N_LAYERS','DEVICE')}
        self.gstate={k:getattr(WGAN,k) for k in ('N_FEATURES','N_CLASSES','SEQ_LEN','HIDDEN_DIM','N_LAYERS','DEVICE','NOISE_DIM','EMBED_DIM','N_CRITIC')}
        classify.N_FEATURES=2; classify.N_CLASSES=19; classify.SEQ_LEN=8
        classify.HIDDEN_DIM=8; classify.N_LAYERS=1; classify.DEVICE=torch.device('cpu')
        WGAN.HIDDEN_DIM=8; WGAN.N_LAYERS=1; WGAN.NOISE_DIM=4; WGAN.EMBED_DIM=4
        WGAN.N_CRITIC=1; WGAN.DEVICE=torch.device('cpu'); WGAN.configure_dimensions(18,19,8)
        torch.set_num_threads(1)

    def tearDown(self):
        for k,v in self.cstate.items(): setattr(classify,k,v)
        for k,v in self.gstate.items(): setattr(WGAN,k,v)

    def test_fixed_source_roles_and_test_lock(self):
        cfg=read(ROOT/'configs/simulink_gan_v2.json'); roles=cfg['roles']
        self.assertEqual([len(roles[k]) for k in ('train','validation','test')],[6,3,3])
        self.assertEqual(len(set(sum(roles.values(),[]))),12)
        self.assertTrue(set(range(4))<=set(roles['train']))
        with tempfile.TemporaryDirectory() as directory:
            cfg=dict(cfg,results_root=directory)
            with self.assertRaises(PermissionError): v2_authorize(cfg,'test')
            with self.assertRaises(PermissionError): v2_pool(cfg,{'signature':'fixture'},'test')
            self.assertFalse(list(Path(directory).glob('cache/*')))

    def test_equal_classifier_updates_and_balanced_detection_loss(self):
        rng=np.random.default_rng(10); x=rng.normal(size=(12,8,2)).astype(np.float32)
        y=np.asarray([0]*9+[1]*3)
        with tempfile.TemporaryDirectory() as directory:
            for dual in (False,True):
                model=classify.train_classifier_steps(x,y,Path(directory)/str(dual),max_steps=3,checkpoints=(1,3),seed=7,
                                                       dual_head=dual,signature='same',batch_size=4)
                self.assertEqual(model.training_receipt['optimizer_updates'],3)
                state=torch.load(Path(directory)/str(dual)/'last.pt',weights_only=False)
                weights=state['binary_weights']
                self.assertAlmostEqual(weights[0]*9,weights[1]*3)
                self.assertTrue(all(int(v['step'])==3 for v in state['optimizer']['state'].values()))

    def test_optimizer_resume_and_signature_rejection(self):
        rng=np.random.default_rng(3); x=rng.normal(size=(12,8,2)).astype(np.float32); y=np.asarray([0]*6+[1]*6)
        with tempfile.TemporaryDirectory() as directory:
            full=classify.train_classifier_steps(x,y,Path(directory)/'full',max_steps=3,checkpoints=(1,3),seed=8,dual_head=True,signature='A',batch_size=4)
            classify.train_classifier_steps(x,y,Path(directory)/'resume',max_steps=1,checkpoints=(1,),seed=8,dual_head=True,signature='A',batch_size=4)
            resumed=classify.train_classifier_steps(x,y,Path(directory)/'resume',max_steps=3,checkpoints=(1,3),seed=8,dual_head=True,signature='A',batch_size=4)
            for key,value in full.state_dict().items(): torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
            self.assertEqual(resumed.training_receipt['updates_this_call'],2)
            with self.assertRaises(RuntimeError):
                classify.train_classifier_steps(x,y,Path(directory)/'resume',max_steps=3,seed=8,dual_head=True,signature='B')

    def test_dual_and_single_shared_initial_encoder(self):
        classify.set_seed(42); a=classify.FaultClassifier()
        classify.set_seed(42); b=classify.FaultClassifier(dual_head=True)
        for key,value in a.state_dict().items(): torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)

    def test_command_condition_in_both_networks_and_response_gp(self):
        idx=list(range(6,12)); g=WGAN.Generator(idx); d=WGAN.Discriminator(idx)
        z=WGAN.sample_noise(3); y=torch.tensor([0,1,2]); commands=torch.randn(3,8,6)
        out=g(z,y,commands)
        torch.testing.assert_close(out[:,:,idx],commands,rtol=0,atol=0)
        x=out.detach().requires_grad_(True); score=d(x,y,commands)
        grad=torch.autograd.grad(score.sum(),x,create_graph=True)[0]
        self.assertEqual(float(grad[:,:,idx].abs().max()),0.)
        self.assertGreater(float(grad.abs().sum()),0.)
        gp=WGAN.gradient_penalty(d,torch.randn_like(out),out.detach(),y,commands)
        self.assertTrue(torch.isfinite(gp))
        with self.assertRaises(ValueError): g(z,y)
        with self.assertRaises(ValueError): d(out,y)

    def test_both_gans_exact_effective_update_budget(self):
        rng=np.random.default_rng(6); x=rng.uniform(-1,1,(19,8,18)).astype(np.float32); y=np.arange(19)
        with tempfile.TemporaryDirectory() as directory:
            for mode,idx in [('plain',None),('command',list(range(6,12)))]:
                g,_=WGAN.train_generator_steps(x,y,Path(directory)/mode,generator_max_steps=2,checkpoints=(1,2),
                                               command_indices=idx,batch_size=2,signature=mode)
                self.assertEqual(g.step_receipt['generator_updates'],2); self.assertEqual(g.step_receipt['critic_updates'],2)
                generated,labels,refs=WGAN.generate_from_pool(g,{i:1 for i in range(19)},x,y,seed=7,batch_size=2)
                self.assertEqual(len(generated),19)
                if idx:
                    np.testing.assert_array_equal(generated[:,:,idx],x[refs][:,:,idx])
                    np.testing.assert_array_equal(y[refs],labels)

    def test_generator_optimizer_resume(self):
        rng=np.random.default_rng(8); x=rng.uniform(-1,1,(19,8,18)).astype(np.float32); y=np.arange(19)
        with tempfile.TemporaryDirectory() as directory:
            a,_=WGAN.train_generator_steps(x,y,Path(directory)/'full',generator_max_steps=2,checkpoints=(1,2),
                                           command_indices=list(range(6,12)),seed=3,batch_size=2,signature='C')
            WGAN.train_generator_steps(x,y,Path(directory)/'resume',generator_max_steps=1,checkpoints=(1,),
                                       command_indices=list(range(6,12)),seed=3,batch_size=2,signature='C')
            b,_=WGAN.train_generator_steps(x,y,Path(directory)/'resume',generator_max_steps=2,checkpoints=(1,2),
                                           command_indices=list(range(6,12)),seed=3,batch_size=2,signature='C')
            for key,value in a.state_dict().items(): torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
            self.assertEqual(b.step_receipt['updates_this_call'],1)

    def test_frozen_configuration_cannot_be_reselected_or_tampered(self):
        from simulink_v2_experiment import freeze_stage,read_frozen
        from provenance import write_json
        cfg=read(ROOT/'configs/simulink_gan_v2.json'); registry={'signature':'fixture'}
        with tempfile.TemporaryDirectory() as directory:
            cfg=dict(cfg,results_root=directory)
            write_json(Path(directory)/'development/selection.json',{'selection_scope':'validation fixture'})
            frozen=freeze_stage(cfg,registry)
            self.assertEqual(read_frozen(cfg,registry)['signature'],frozen['signature'])
            frozen['classifier_steps']=999
            write_json(Path(directory)/'frozen_configuration.json',frozen)
            with self.assertRaises(ValueError): read_frozen(cfg,registry)

    def test_matched_ros_is_selected_at_same_gan_ratio(self):
        from simulink_v2_experiment import matched_ros
        selected={'selected':{'plain':{'ratio':.25},'command':{'ratio':.5}},
                  'all_ratios':[{'method':'ros','ratio':ratio,'count':int(100*ratio)} for ratio in [0,.25,.5,1]]}
        self.assertEqual(matched_ros(selected,'plain')['count'],25)
        self.assertEqual(matched_ros(selected,'command')['count'],50)

    def test_detection_independent_of_class_and_localization_clock(self):
        alarm,motor=alarm_trace([.9,.9,.9,.9],[1,2,3,3],.5,2)
        np.testing.assert_array_equal(alarm,[False,True,True,True])
        np.testing.assert_array_equal(motor,[0,0,0,3])
        records=[dict(source_id='healthy',group_id=0,label=0,time_s=np.array([0.,1.,2.,3.]),
                      fault_probability=np.zeros(4),fault_class=np.ones(4,int)),
                 dict(source_id='fault',group_id=0,label=1,fault_start_s=1.,time_s=np.arange(5.),
                      fault_probability=np.asarray([0,.9,.9,.9,.9]),fault_class=np.asarray([2,2,2,1,1]))]
        result=score_alarm_records(records,.5,2)
        self.assertEqual(result['detection_delays_s'],[1.]); self.assertEqual(result['localization_delays_s'],[3.])
        self.assertEqual(result['normal_observation_s'],3.)

    def test_rule_uses_delayed_response_and_squares_speed_ratio(self):
        dt=.01; tau=np.full(6,.05); commands=np.full((200,6),500.); omega=commands.copy(); a=np.exp(-dt/tau)
        for k in range(1,len(omega)):
            target=commands[k-1].copy()
            if k>=31: target[4]*=np.sqrt(.2)
            omega[k]=a*omega[k-1]+(1-a)*target
        rpm=omega[np.maximum(np.arange(len(omega))-3,0)]
        result=motor_response_scores(commands,rpm,dt,tau,.03)
        self.assertAlmostEqual(result['estimated_effectiveness'][-1,4],.2,places=6)
        self.assertEqual(result['fault_class'][-1],11)
        self.assertTrue(np.all(result['fault_probability'][:31]==0))

    def test_constant_channels_and_lagged_quality(self):
        from simulink_data import FEATURES
        rng=np.random.default_rng(2); values=np.zeros((4,40,18),np.float32)
        u=rng.normal(500,100,(4,40)); values[:,:,6]=u; values[:,:,12]=np.c_[u[:,:3],u[:,:-3]]
        y=np.array([0,0,1,1]); groups=np.array([0,1,0,1])
        q=stratified_generation_quality(values,y,values,y,values,y,FEATURES['imu_cmd_rpm'],groups,True,max_lag=8)
        lag=next(row for row in q['per_class_motor_lag'] if row['label']==0 and row['motor_python']==0)
        self.assertAlmostEqual(lag['validation_peak_lag_s'],.03)
        const=next(row for row in q['per_class_motor_lag'] if row['motor_python']==1)
        self.assertTrue(const['constant_validation']); self.assertIsNone(const['lag_curve_mae'])


if __name__=='__main__': unittest.main()
