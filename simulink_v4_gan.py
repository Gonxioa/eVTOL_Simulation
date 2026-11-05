"""Original WGAN.Generator/Discriminator with isolated V4 instrumentation.

Training math follows WGAN.train_generator_steps (baseline 2e03333). Added:
loss decomposition, gradient norms, RNG-isolated fixed probes, and an optional
frozen real-data teacher term. The network backbone is not duplicated.
"""
from contextlib import nullcontext
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance,ks_2samp

import WGAN
import classify
from provenance import stable_hash,sha256_file,code_hashes,write_json,runtime_info
from simulink_data import ROOT,read,FEATURES
from simulink_v3_data import immutable_json,array_hash
from simulink_v4_data import root,fit_scalers
from simulink_v4_training import transform,load_model,DEVICE


def configure(config,window):
    torch.set_num_threads(4)
    WGAN.HIDDEN_DIM=config['generator_hidden']; WGAN.N_LAYERS=config['generator_layers']
    WGAN.NOISE_DIM=config['generator_noise_dim']; WGAN.EMBED_DIM=config['generator_embedding_dim']
    WGAN.TEMPORAL_NOISE=True; WGAN.DEVICE=DEVICE; WGAN.configure_dimensions(12,19,window)


def gradient_norm(model):
    return float(torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None)))


def penalty(discriminator,real,fake,labels):
    alpha=torch.rand(len(real),1,1,device=real.device)
    interpolated=(alpha*real+(1-alpha)*fake).requires_grad_(True)
    output=discriminator(interpolated,labels)
    gradients=torch.autograd.grad(output,interpolated,torch.ones_like(output),create_graph=True,retain_graph=True)[0]
    norm=gradients.reshape(len(real),-1).norm(2,dim=1)
    return ((norm-1)**2).mean(),norm.mean()


def differentiable_standardize(fake,gan_scaler,classifier_scaler):
    def tensor(x): return torch.as_tensor(x,dtype=fake.dtype,device=fake.device)
    raw=(fake-tensor(gan_scaler.min_))/tensor(gan_scaler.scale_)
    return (raw-tensor(classifier_scaler.mean_))/tensor(classifier_scaler.scale_)


def fixed_bank(config,p,G,gan_scaler,seed):
    counts={c:int(np.sum(p['y']==c)) for c in range(19)}
    devices=[torch.cuda.current_device()] if DEVICE.type=='cuda' else []
    previous=G.training
    with torch.random.fork_rng(devices=devices):
        values,labels,refs=WGAN.generate_from_pool(G,counts,transform(p['X'],gan_scaler),p['y'],seed+15000,config['batch_size'])
    G.train(previous)
    if np.any(refs!=-1) or values.shape[-1]!=12: raise AssertionError('Unexpected command conditioning')
    raw=gan_scaler.inverse_transform(values.reshape(-1,12)).reshape(values.shape).astype(np.float32)
    if not np.isfinite(raw).all(): raise FloatingPointError('Nonfinite synthetic bank')
    return raw,labels


def quality(config,train,validation,fake,labels):
    scale=np.maximum(train['X'].std((0,1)),1e-6); rows=[]; relations=[]; nearest=[]
    exact_training=0; exact_generated=0; distances=[]
    for c in range(19):
        real=train['X'][train['y']==c]; generated=fake[labels==c]; val=validation['X'][validation['y']==c]
        real_hash={__import__('hashlib').sha256(x.tobytes()).digest() for x in real}
        gen_hash=[__import__('hashlib').sha256(x.tobytes()).digest() for x in generated]
        exact_training+=sum(h in real_hash for h in gen_hash); exact_generated+=len(gen_hash)-len(set(gen_hash))
        probes=generated[:config['quality_probes_per_class']]
        for ch,name in enumerate(FEATURES['imu_rpm']):
            a=val[:,:,ch]; b=probes[:,:,ch]
            ac=a-a.mean(1,keepdims=True); bc=b-b.mean(1,keepdims=True)
            pa=(np.abs(np.fft.rfft(ac,axis=1))**2).mean(0); pb=(np.abs(np.fft.rfft(bc,axis=1))**2).mean(0)
            pa/=max(float(pa.sum()),1e-12); pb/=max(float(pb.sum()),1e-12)
            row=dict(label=c,channel=name,validation_std=float(a.std()),generated_std=float(b.std()),
                nearly_constant=bool(a.std()<(3. if ch>=6 else 1e-5)),
                normalized_wasserstein=float(wasserstein_distance(a.ravel(),b.ravel())/scale[ch]),
                ks=float(ks_2samp(a.ravel(),b.ravel()).statistic),demeaned_psd_l1=float(np.abs(pa-pb).sum()))
            for lag in (1,5,10,20):
                def correlation(x):
                    return float(np.mean((x[:,lag:]-x[:,lag:].mean())*(x[:,:-lag]-x[:,:-lag].mean()))/max(float(x[:,lag:].std()*x[:,:-lag].std()),1e-12))
                row[f'autocorrelation_error_lag{lag}']=abs(correlation(a)-correlation(b))
            rows.append(row)
        def corr(x): return np.nan_to_num(np.corrcoef(x.reshape(-1,12).T))
        relations.append(dict(label=c,cross_channel_correlation_mae=float(np.abs(corr(val)-corr(probes)).mean())))
        aa=(real/scale).reshape(len(real),-1); bb=(probes/scale).reshape(len(probes),-1)
        ds=cdist(bb,aa).min(1)/np.sqrt(aa.shape[1]); distances.extend(ds.tolist())
        raw_distance=cdist(aa[:min(64,len(aa))],aa)/np.sqrt(aa.shape[1])
        groups=train['groups'][train['y']==c]
        raw_distance[groups[:len(raw_distance),None]==groups[None,:]]=np.inf
        baseline=raw_distance.min(1); baseline=baseline[np.isfinite(baseline)]
        threshold=float(np.quantile(baseline,.05)) if len(baseline) else None
        nearest.append(dict(label=c,probe_count=len(probes),nearest_training_rms_median=float(np.median(ds)),
            real_cross_group_p05=threshold,near_fraction=None if threshold is None else float(np.mean(ds<threshold))))
    return dict(all_generated_finite=True,all_generated_count=len(labels),class_counts=np.bincount(labels,minlength=19).tolist(),
        exact_training_duplicates_all_admitted=exact_training,exact_generated_duplicates_all_admitted=exact_generated,
        per_class_channel=rows,per_class_relations=relations,per_class_nearest=nearest,
        summary=dict(normalized_wasserstein=float(np.mean([r['normalized_wasserstein'] for r in rows])),
                     ks=float(np.mean([r['ks'] for r in rows])),demeaned_psd_l1=float(np.mean([r['demeaned_psd_l1'] for r in rows]))),
        validation_reference='Same selected training prefix and window, independent validation groups',
        exhaustive_checks='All admitted windows: finite, counts, exact duplicates. Expensive distances and spectra: first32/class.',
        label_scope='Condition labels are requested classes, not independent physical truth; no generated absolute-stage label')


def train_generator(config,bconfig,train_pool,validation_prefix,seed,weight,teacher_spec=None):
    window=train_pool['identity']['window']; configure(config,window)
    cls,gan,scale_dir=fit_scalers(config,train_pool); x=transform(train_pool['X'],gan)
    if weight and teacher_spec is None: raise ValueError('G1 needs an eligible, frozen real-data teacher')
    teacher=None; teacher_hash=None
    if weight:
        teacher=load_model(config,teacher_spec)
        for p in teacher.parameters(): p.requires_grad_(False)
        teacher.eval(); teacher_hash=sha256_file(Path(teacher_spec['directory'])/f'checkpoint_{teacher_spec["step"]}.pt')
    identity=dict(config_signature=stable_hash(config),B_protocol=stable_hash(bconfig),train_pool=train_pool['signature'],
        X_sha256=array_hash(x),y_sha256=array_hash(train_pool['y']),seed=seed,weight=weight,teacher_sha256=teacher_hash,
        code=code_hashes(ROOT,['WGAN.py','simulink_v4_gan.py','simulink_v4_training.py','simulink_v4_data.py','simulink_v4_models.py']))
    signature=stable_hash(identity); directory=root(config)/'generators'/f'weight{weight:g}_s{seed}_{signature[:16]}'
    directory.mkdir(parents=True,exist_ok=True)
    immutable_json(directory/'job.json',dict(**identity,signature=signature,scaler_directory=scale_dir))
    classify.set_seed(seed)
    G=WGAN.Generator().to(DEVICE); D=WGAN.Discriminator().to(DEVICE)
    initial={k:v.detach().cpu().clone() for k,v in G.state_dict().items()}
    og=torch.optim.Adam(G.parameters(),lr=config['generator_lr'],betas=(0.,.9))
    od=torch.optim.Adam(D.parameters(),lr=config['critic_lr'],betas=(0.,.9))
    rng=np.random.default_rng(seed+23); history=[]; draws=[]; start=0; elapsed=0.
    if (directory/'last.pt').exists():
        saved=torch.load(directory/'last.pt',map_location=DEVICE,weights_only=False)
        if saved['signature']!=signature: raise ValueError('GAN resume signature mismatch')
        G.load_state_dict(saved['generator']); D.load_state_dict(saved['critic']); og.load_state_dict(saved['optimizer_g']); od.load_state_dict(saved['optimizer_d'])
        start=saved['generator_updates']; initial=saved['initial_generator']; history=saved['history']; elapsed=saved['training_wall_s']
        draws=saved['sampling_indices'].tolist()
        rng.bit_generator.state=saved['sampler_rng']; torch.set_rng_state(saved['torch_rng'].cpu())
        if DEVICE.type=='cuda': torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
    started=time.perf_counter()
    for step in range(start+1,config['generator_max_steps']+1):
        G.train(); D.train(); critic=[]
        for _ in range(config['n_critic']):
            indices=rng.integers(0,len(x),config['batch_size'])
            draws.append(indices.tolist())
            real=torch.as_tensor(x[indices],dtype=torch.float32,device=DEVICE)
            y=torch.as_tensor(train_pool['y'][indices],dtype=torch.long,device=DEVICE)
            for p in D.parameters(): p.requires_grad_(True)
            od.zero_grad(set_to_none=True)
            fake=G(WGAN.sample_noise(len(y)),y).detach()
            real_score=D(real,y).mean(); fake_score=D(fake,y).mean(); gp,gpnorm=penalty(D,real,fake,y)
            loss_d=fake_score-real_score+config['gradient_penalty']*gp
            loss_d.backward(); dnorm=gradient_norm(D); od.step()
            critic.append([float(real_score.detach()),float(fake_score.detach()),float(gp.detach()),float(gpnorm.detach()),float(loss_d.detach()),dnorm])
        for p in D.parameters(): p.requires_grad_(False)
        og.zero_grad(set_to_none=True); fake=G(WGAN.sample_noise(len(y)),y)
        adversarial=-D(fake,y).mean(); classification=fake.new_zeros(())
        if weight:
            # cuDNN eval-mode LSTM does not expose input backward; native autograd does.
            context=torch.backends.cudnn.flags(enabled=False) if teacher_spec['family']=='lstm' else nullcontext()
            with context: classification=torch.nn.functional.cross_entropy(teacher(differentiable_standardize(fake,gan,cls)),y)
        loss=adversarial+weight*classification; loss.backward(); gnorm=gradient_norm(G); og.step()
        means=np.mean(critic,axis=0)
        row=dict(generator_updates=step,critic_updates=step*config['n_critic'],loss_g=float(loss.detach()),
            loss_g_adversarial=float(adversarial.detach()),loss_g_classification=float(classification.detach()),
            gradient_norm_g=gnorm,real_critic_score=float(means[0]),fake_critic_score=float(means[1]),
            critic_gap=float(means[0]-means[1]),gradient_penalty=float(means[2]),interpolation_gradient_norm=float(means[3]),
            loss_d=float(means[4]),gradient_norm_d=float(means[5]))
        if not np.isfinite(list(row.values())).all(): raise FloatingPointError('Nonfinite GAN training')
        history.append(row)
        if step%100==0: print('V4_GENERATOR',weight,seed,step,'G',round(row['loss_g'],4),'D',round(row['loss_d'],4),'GP',round(row['gradient_penalty'],4),flush=True)
        if step in config['generator_checkpoints']:
            delta=max(float((v.detach().cpu()-initial[k].cpu()).abs().max()) for k,v in G.state_dict().items())
            if not delta>0: raise RuntimeError('Generator parameters did not change')
            state=dict(signature=signature,generator=G.state_dict(),critic=D.state_dict(),optimizer_g=og.state_dict(),optimizer_d=od.state_dict(),
                generator_updates=step,critic_updates=step*config['n_critic'],initial_generator=initial,parameter_max_delta=delta,
                history=history,sampler_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                sampling_indices=np.asarray(draws,np.int32),
                cuda_rng=torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else [],training_wall_s=elapsed+time.perf_counter()-started)
            temporary=directory/'checkpoint.tmp'; torch.save(state,temporary); temporary.replace(directory/f'checkpoint_{step}.pt')
            import shutil
            shutil.copy2(directory/f'checkpoint_{step}.pt',directory/'last.pt'); write_json(directory/'training_curve.json',history)
            # Probe generation restores the training RNG; numerical evidence only.
            bank,labels=fixed_bank(config,train_pool,G,gan,seed)
            np.savez_compressed(directory/f'bank_{step}.npz',X=bank,y=labels)
            q=quality(config,train_pool,validation_prefix,bank,labels)
            write_json(directory/f'quality_{step}.json',q)
    # Recover probe artifacts if a process stopped after an atomic optimizer save.
    for checkpoint in config['generator_checkpoints']:
        bank_path=directory/f'bank_{checkpoint}.npz'; quality_path=directory/f'quality_{checkpoint}.json'
        if bank_path.exists() and quality_path.exists(): continue
        saved=torch.load(directory/f'checkpoint_{checkpoint}.pt',map_location=DEVICE,weights_only=False)
        probe_model=WGAN.Generator().to(DEVICE); probe_model.load_state_dict(saved['generator'])
        bank,labels=fixed_bank(config,train_pool,probe_model,gan,seed)
        np.savez_compressed(bank_path,X=bank,y=labels)
        write_json(quality_path,quality(config,train_pool,validation_prefix,bank,labels))
    spec=dict(**identity,signature=signature,directory=str(directory),window=window,prefix_s=train_pool['identity']['prefix_s'],
        scaler_directory=scale_dir,generator_updates=config['generator_max_steps'],critic_updates=config['generator_max_steps']*config['n_critic'],
        completed_utc=runtime_info()['created_utc'],checkpoint_sha256=sha256_file(directory/f'checkpoint_{config["generator_max_steps"]}.pt'))
    if (directory/'result.json').exists():
        old=read(directory/'result.json'); spec['completed_utc']=old['completed_utc']
    immutable_json(directory/'result.json',spec)
    np.savez_compressed(directory/'sampling_indices.npz',indices=np.asarray(draws,np.int32))
    return spec
