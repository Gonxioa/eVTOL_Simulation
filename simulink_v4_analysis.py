"""Real measured-signal diagnostics; paired subtraction is not an online feature."""
import itertools
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from provenance import write_json
from simulink_data import FEATURES, ROOT, read
from simulink_v4_data import root, signals, pool, fit_scalers


def analyze(config, registry):
    destination=root(config)/'signal_analysis'; destination.mkdir(exist_ok=True)
    reference=pool(config,registry,'train',100,24,'native')
    scaler,_,_=fit_scalers(config,reference)
    samples=signals(config,registry,'train')+signals(config,registry,'validation')
    phases=[('before',0,300),('injection_transition',300,400),('early',400,600),('middle',600,1200),('late',1200,2400)]
    rows=[]; correlations=[]
    for sample in samples:
        r=sample['record']
        for name,start,end in phases:
            x=sample['X'][start:end].astype(float); centered=x-x.mean(0)
            psd=np.abs(np.fft.rfft(centered,axis=0))**2/(100*len(x))
            if len(x)%2==0: psd[1:-1]*=2
            else: psd[1:]*=2
            frequency=np.fft.rfftfreq(len(x),.01); variance=np.mean(centered**2,axis=0)
            corr=np.corrcoef(x.T)
            for ch,feature in enumerate(FEATURES['imu_rpm']):
                row=dict(role=r['role'],group=r['group_id'],seed=r['run']['seed'],source_id=r['source_id'],label=r['run']['label'],
                    phase=name,channel=feature,mean=float(x[:,ch].mean()),std=float(x[:,ch].std()),minimum=float(x[:,ch].min()),
                    maximum=float(x[:,ch].max()),mean_absolute_derivative=float(np.abs(np.diff(x[:,ch])).mean()*100),
                    frequency_resolution_hz=100/len(x))
                for lag in (1,5,10,20): row[f'autocorrelation_lag{lag}']=float(np.mean(centered[lag:,ch]*centered[:-lag,ch])/max(variance[ch],1e-12))
                for low,high in ((0,1),(1,5),(5,15),(15,50.000001)):
                    row[f'psd_band_{low:g}_{high:g}']=float(psd[(frequency>=low)&(frequency<high),ch].sum()*100/len(x))
                rows.append(row)
            for a,b in itertools.combinations(range(12),2):
                correlations.append(dict(source_id=r['source_id'],role=r['role'],group=r['group_id'],label=r['run']['label'],phase=name,
                    channel_a=FEATURES['imu_rpm'][a],channel_b=FEATURES['imu_rpm'][b],correlation=float(corr[a,b])))
    pd.DataFrame(rows).to_csv(destination/'per_source_channel_statistics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(correlations).to_csv(destination/'cross_channel_correlations.csv',index=False,encoding='utf-8-sig')
    distances=[]
    def distance(a,b,scope):
        ar,br=a['record'],b['record']
        for name,start,end in phases:
            delta=(a['X'][start:end]-b['X'][start:end])/scaler.scale_
            distances.append(dict(scope=scope,role=ar['role'],source_a=ar['source_id'],source_b=br['source_id'],
                group_a=ar['group_id'],group_b=br['group_id'],label_a=ar['run']['label'],label_b=br['run']['label'],phase=name,
                standardized_rms=float(np.sqrt(np.mean(delta**2)))))
    for role in ('train','validation'):
        selected=[s for s in samples if s['record']['role']==role]
        for group in sorted({s['record']['group_id'] for s in selected}):
            by_label={s['record']['run']['label']:s for s in selected if s['record']['group_id']==group}
            for a,b in [(0,c) for c in range(13,19)]+[(7,10),(8,11),(9,12)]: distance(by_label[a],by_label[b],'same_condition_different_class')
        for label in range(19):
            same=[s for s in selected if s['record']['run']['label']==label]
            for a,b in itertools.combinations(same,2): distance(a,b,'same_class_different_condition')
    pd.DataFrame(distances).to_csv(destination/'paired_and_cross_condition_distances.csv',index=False,encoding='utf-8-sig')
    hashes={}; collisions=[]
    import hashlib
    for x,label,source,start in zip(reference['X'],reference['y'],reference['source_ids'],reference['starts']):
        h=hashlib.sha256(x.tobytes()).hexdigest()
        if h in hashes and hashes[h]['label']!=int(label): collisions.append(dict(original=hashes[h],other=dict(label=int(label),source=str(source),start=int(start))))
        else: hashes[h]=dict(label=int(label),source=str(source),start=int(start))
    write_json(destination/'cross_label_exact_collisions.json',dict(count=len(collisions),cases=collisions,
        scope='Exact float32 100-sample measured windows; zero collisions does not prove separability'))
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
    first={s['record']['run']['label']:s for s in samples if s['record']['role']=='train' and s['record']['group_id']==0}
    for channel,feature in enumerate(FEATURES['imu_rpm']):
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        for ax,pairs,title in [(axes[0],[(0,'健康'),(13,'1号电机剩余80%效能')],'健康与轻度故障'),
                               (axes[1],[(7,'1号电机剩余20%效能'),(10,'4号电机剩余20%效能')],'对侧电机定位比较')]:
            for label,text in pairs: ax.plot(first[label]['time'],first[label]['X'][:,channel],label=text,linewidth=.8)
            ax.axvline(3,color='k',linestyle=':'); ax.legend(fontsize=8); ax.grid(alpha=.25); ax.set_title(title)
            ax.set_ylabel(('比力（m/s²）' if channel<3 else '角速度（rad/s）' if channel<6 else '转速观测（rad/s）'))
        axes[1].set_xlabel('Time (s)'); fig.suptitle(feature+'：预登记训练条件组0，未挑选最好案例')
        fig.tight_layout(); fig.savefig(destination/f'waveform_{feature}.png',dpi=150); plt.close(fig)
    legacy=read(ROOT/config['v3_root']/'development_complete.json')
    errors=[]
    for job in legacy['jobs']:
        for label,recall in enumerate(job['validation']['prefix_24']['class_recall']):
            errors.append(dict(version='historical_V3_validation',tag=job['tag'],seed=job['seed'],label=label,recall=recall))
    pd.DataFrame(errors).to_csv(destination/'historical_v3_real_ros_gan_recall.csv',index=False,encoding='utf-8-sig')
    receipt=dict(records=len(samples),channels=12,statistics_rows=len(rows),distance_rows=len(distances),
        paired_noise_note='Same-condition subtraction can cancel shared wind/noise; cross-condition distances are reported separately. This is not an online counterfactual feature.',
        interpretation='Weak differences or low classifier scores alone do not establish strict non-identifiability; no PCA/t-SNE conclusion is used.',
        phase_note='Nominal time bins only; late does not mean physically recovered. Actual task phase is in the source audit.')
    write_json(destination/'analysis_receipt.json',receipt)
    print('V4_SIGNAL_ANALYSIS',receipt,flush=True)
    return receipt
