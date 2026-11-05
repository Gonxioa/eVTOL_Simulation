"""V4 GAN instrumentation preserves gradients, probe RNG and exact resume."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import WGAN
from simulink_data import read
from simulink_v4_data import CONFIG,fit_scalers
from simulink_v4_gan import differentiable_standardize,fixed_bank,configure,train_generator
from test_simulink_v4 import fixture_pool


class V4GANTests(unittest.TestCase):
    def setUp(self):
        self.keys=('DEVICE','N_FEATURES','N_CLASSES','SEQ_LEN','HIDDEN_DIM','N_LAYERS','NOISE_DIM','EMBED_DIM','TEMPORAL_NOISE')
        self.saved={k:getattr(WGAN,k) for k in self.keys}; self.threads=torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        for k,v in self.saved.items(): setattr(WGAN,k,v)
        torch.set_num_threads(self.threads)

    def test_scaler_bridge_is_exact_and_differentiable(self):
        rng=np.random.default_rng(3); raw=rng.normal(size=(30,12))*3+4
        cls=StandardScaler().fit(raw); gan=MinMaxScaler((-1,1)).fit(raw)
        x=torch.tensor(rng.uniform(-.5,.5,(2,50,12)),dtype=torch.float32,requires_grad=True)
        z=differentiable_standardize(x,gan,cls)
        expected=cls.transform(gan.inverse_transform(x.detach().numpy().reshape(-1,12))).reshape(2,50,12)
        np.testing.assert_allclose(z.detach().numpy(),expected,atol=1e-6)
        z.square().mean().backward(); self.assertTrue(torch.isfinite(x.grad).all()); self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_fixed_probes_restore_training_RNG(self):
        with tempfile.TemporaryDirectory() as directory:
            c=dict(read(CONFIG),results_root=directory,generator_hidden=4,generator_layers=1,generator_noise_dim=4,generator_embedding_dim=4)
            p=fixture_pool('train'); p['identity']['window']=50; p['X']=np.repeat(p['X'][:,:1],50,axis=1)
            from simulink_v3_data import array_hash
            p['X_sha256']=array_hash(p['X']); _,scaler,_=fit_scalers(c,p)
            with patch('simulink_v4_gan.DEVICE',torch.device('cpu')):
                configure(c,50); G=WGAN.Generator(); G.train()
                rng=torch.get_rng_state().clone(); a,ay=fixed_bank(c,p,G,scaler,142)
                self.assertTrue(torch.equal(rng,torch.get_rng_state())); self.assertTrue(G.training)
                b,by=fixed_bank(c,p,G,scaler,142)
            np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(ay,by)

    def test_resume_and_missing_probe_recovery_match_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            c=dict(read(CONFIG),results_root=str(Path(directory)/'full'),generator_hidden=4,generator_layers=1,
                   generator_noise_dim=4,generator_embedding_dim=4,generator_max_steps=2,generator_checkpoints=[1,2],n_critic=1,batch_size=4)
            p,v=fixture_pool('train'),fixture_pool('validation')
            from simulink_v3_data import array_hash
            for q in (p,v):
                q['X']=np.repeat(q['X'][:,:1],50,axis=1); q['identity']['window']=50; q['X_sha256']=array_hash(q['X'])
            with patch('simulink_v4_gan.DEVICE',torch.device('cpu')),patch('simulink_v4_gan.quality',return_value={'unit_probe':True}):
                full=train_generator(c,{},p,v,142,0.)
                rc=dict(c,results_root=str(Path(directory)/'resume'))
                with patch('simulink_v4_gan.fixed_bank',side_effect=RuntimeError('unit interruption after optimizer save')):
                    with self.assertRaises(RuntimeError): train_generator(rc,{},p,v,142,0.)
                resumed=train_generator(rc,{},p,v,142,0.)
            a=torch.load(Path(full['directory'])/'last.pt',map_location='cpu',weights_only=False)
            b=torch.load(Path(resumed['directory'])/'last.pt',map_location='cpu',weights_only=False)
            for name,value in a['generator'].items(): torch.testing.assert_close(value,b['generator'][name],rtol=0,atol=0)
            np.testing.assert_array_equal(a['sampling_indices'],b['sampling_indices'])
            self.assertEqual({int(s['step']) for s in b['optimizer_d']['state'].values()},{2})
            self.assertTrue((Path(resumed['directory'])/'bank_1.npz').exists())
            self.assertTrue((Path(resumed['directory'])/'quality_1.json').exists())


if __name__=='__main__': unittest.main()
