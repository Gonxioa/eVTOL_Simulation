"""Descriptive V4 figures and evidence exports; never select or modify models."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from provenance import sha256_file,write_json
from simulink_data import read,FEATURES
from simulink_v4_data import root,pool,registry
from simulink_v4_training import record_at
from simulink_v4_augmentation import prediction_records
from replay_diagnostics import alarm_trace


def exports(config,a,b,final,target,frozen,confirmation_registry):
    base=root(config); figures=target/'figures'
    # Development candidates are reported even when they lost the selection.
    rows=[]; tstr=[]; channel=[]; relations=[]; nearest=[]; budget=[]
    for s in b['all_augmented_models']:
        h=record_at(s,s['step']); m=h['validation']['prefix_24']
        rows.append(dict(signature=s['signature'],method=s['method'],ratio=s['ratio'],seed=s['seed'],
            validation_macro_f1=m['macro_f1'],validation_accuracy=m['accuracy'],
            original_windows=s['original_windows'],training_windows=s['training_windows'],
            actual_updates=s['step'],class_original=str(s['class_original']),class_joined=str(s['class_joined'])))
    for r in b['TSTR']:
        s=r['model']; m=record_at(s,s['step'])['validation']['prefix_24']
        tstr.append(dict(generator=r['generator'],G_step=r['G_step'],classifier_seed=s['seed'],
            classifier_updates=s['step'],validation_macro_f1=m['macro_f1'],validation_accuracy=m['accuracy']))
    for g in b['generators']:
        for step in config['generator_checkpoints']:
            q=read(Path(g['directory'])/f'quality_{step}.json'); keys=dict(weight=g['weight'],seed=g['seed'],G_step=step)
            channel.extend(dict(**keys,**r) for r in q['per_class_channel'])
            relations.extend(dict(**keys,**r) for r in q['per_class_relations'])
            nearest.extend(dict(**keys,**r) for r in q['per_class_nearest'])
        import torch
        checkpoint=torch.load(Path(g['directory'])/f'checkpoint_{g["generator_updates"]}.pt',map_location='cpu',weights_only=False)
        budget.append(dict(signature=g['signature'],weight=g['weight'],seed=g['seed'],G_updates=g['generator_updates'],
            critic_updates=g['critic_updates'],parameter_max_delta=checkpoint['parameter_max_delta'],training_wall_s=checkpoint['training_wall_s']))
    for name,data in [('B_all_validation_candidates',rows),('B_independent_TSTR',tstr),
                      ('GAN_per_class_channel_quality',channel),('GAN_per_class_relationships',relations),
                      ('GAN_per_class_nearest_neighbors',nearest),('GAN_actual_budgets',budget)]:
        pd.DataFrame(data).to_csv(target/(name+'.csv'),index=False,encoding='utf-8-sig')
    write_json(target/'B_selection_evidence.json',dict(G_checkpoint=read(base/'decisions/B_G_checkpoint.json'),
        G1_weight=b['selected_G1_weight'],GAN_ratios=b['ratio_choices'],ROS_ratio=b['ROS_choice']))
    fig,ax=plt.subplots(figsize=(9,4.5))
    for name,mapping in [('ROS',b['ROS_choice'])]+[(f'G权重{w}',v) for w,v in b['ratio_choices'].items()]:
        rr=mapping['candidates']
        ax.errorbar([r['ratio'] for r in rr],[100*r['mean_macro_f1'] for r in rr],
                    yerr=[100*np.std(r['scores']) for r in rr],marker='o',capsize=3,label=name)
    ax.set_xlabel('每类新增比例'); ax.set_ylabel('共同完整24秒验证宏F1（%）'); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(figures/'B_ratio_validation.png',bbox_inches='tight'); plt.close(fig)
    # Aggregate confusion counts, not average normalized rows with unequal support.
    index={r['signature']:r for r in final['results']}; specs={s['signature']:s for s in b['models']}
    labels=['正常']+[f'{eta}% M{i}' for eta in (0,20,80) for i in range(1,7)]
    phase=[]
    for name,ids in b['selected'].items():
        for identity in ids:
            r=index[identity]
            for scope,m in r['test'].items():
                if not isinstance(m,dict) or not m.get('window_count'): continue
                phase.append(dict(model=name,seed=r['seed'],scope=scope,window_count=m['window_count'],
                    healthy_recall=m['healthy_recall'],motor_accuracy_on_fault=m['motor_accuracy_on_fault'],
                    severity_only_accuracy=m['severity_only_accuracy_on_fault'],
                    joint_0=m['joint_accuracy_by_effectiveness']['0.0'],joint_20=m['joint_accuracy_by_effectiveness']['0.2'],
                    joint_80=m['joint_accuracy_by_effectiveness']['0.8']))
        for scope in ('early_fault_3_6','late_fault_12_24'):
            cm=sum(np.asarray(index[i]['test'][scope]['confusion_matrix']) for i in ids)
            norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
            fig,ax=plt.subplots(figsize=(10,8)); image=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
            ax.set_xticks(range(19),labels,rotation=90); ax.set_yticks(range(19),labels)
            ax.set_xlabel('预测类别'); ax.set_ylabel('真实类别'); fig.colorbar(image,ax=ax,label='逐真实类别归一化比例')
            ax.set_title(f'{name}：'+('故障初期3～6秒' if scope.startswith('early') else '故障后期12～24秒')+
                         '，三个联合重复合计\n纯故障阶段没有健康真值支持，健康行留零')
            fig.tight_layout(); fig.savefig(figures/f'CM_{name}_{scope}.png',bbox_inches='tight'); plt.close(fig)
            pd.DataFrame(cm,index=labels,columns=labels).to_csv(target/f'CM_{name}_{scope}.csv',encoding='utf-8-sig')
    pd.DataFrame(phase).to_csv(target/'final_phase_joint_metrics.csv',index=False,encoding='utf-8-sig')
    # Deterministic illustrative cases: first source in sorted order, seed42 only.
    traces=[]
    for name,ids in b['selected'].items():
        spec=next(specs[s] for s in ids if specs[s]['seed']==config['training_seeds'][0]); r=index[spec['signature']]
        records=prediction_records(config,confirmation_registry,'test',spec,frozen['signature'])
        by_source={s['source_id']:s for s in records}; threshold=r['alarm_policy']['threshold']; hold=r['alarm_policy']['hold']
        for kind in ('false_alarm','miss'):
            candidates=([c for c in r['alarm']['cases'] if c['label']==0 and c['false_alarm_events']>0]
                        if kind=='false_alarm' else [c for c in r['alarm']['cases'] if c['label']!=0 and not c['detected']])
            if not candidates: continue
            case=sorted(candidates,key=lambda c:c['source_id'])[0]; source=by_source[case['source_id']]
            alarm,motor=alarm_trace(source['fault_probability'],source['fault_class'],threshold,hold); time=np.asarray(source['time_s'])
            fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
            axes[0].plot(time,source['fault_probability'],label='预测故障概率'); axes[0].axhline(threshold,color='r',ls='--',label='冻结阈值')
            axes[0].step(time,alarm.astype(int),where='post',alpha=.6,label='持续判断后的报警状态'); axes[0].legend(); axes[0].set_ylabel('概率／报警状态')
            axes[1].step(time,motor,where='post',label='定位电机，0为未定位')
            if case['label']:
                for ax in axes: ax.axvline(3,color='gray',ls=':')
                axes[1].axhline((case['label']-1)%6+1,color='r',ls='--',label='真实故障电机，仅评分参照')
            axes[1].legend(); axes[1].set_ylabel('电机编号'); axes[1].set_xlabel('Time (s)')
            for ax in axes: ax.grid(alpha=.2)
            fig.suptitle(f'{name}，种子{spec["seed"]}，'+('健康误报' if kind=='false_alarm' else '故障漏检')+'\n'+source['source_id'])
            fig.tight_layout(); fig.savefig(figures/f'alarm_{name}_{kind}.png',bbox_inches='tight'); plt.close(fig)
            traces.extend(dict(model=name,case_kind=kind,source_id=source['source_id'],time_s=float(t),
                fault_probability=float(p),threshold=threshold,hold=hold,alarm=int(al),motor=int(m))
                for t,p,al,m in zip(time,source['fault_probability'],alarm,motor))
    pd.DataFrame(traces).to_csv(target/'representative_alarm_traces.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(10,5))
    for name,ids in b['selected'].items():
        cases=[c for i in ids for c in index[i]['alarm']['cases'] if c['label']!=0]
        delays=sorted(c['detection_delay_s'] for c in cases if c['detected'])
        if delays: ax.step(delays,np.arange(1,len(delays)+1)/len(cases),where='post',label=name)
    ax.set_xlabel('故障注入后的检出延迟（秒）'); ax.set_ylabel('全部故障记录中累计检出比例'); ax.set_ylim(0,1); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(figures/'detection_delay_with_misses.png',bbox_inches='tight'); plt.close(fig)
    generated_examples(config,a,b,target)


def generated_examples(config,a,b,target):
    """Fixed first sample, every signal separately; no synthetic absolute phase."""
    from scipy.signal import periodogram
    train=pool(config,registry(config),'train',a['selected_window'],a['selected_prefix_s'],'native')
    validation=pool(config,registry(config),'validation',a['selected_window'],a['selected_prefix_s'],'native')
    choices=[('G0',0.)]+([] if b['selected_G1_weight'] is None else [('G1',b['selected_G1_weight'])])
    figdir=target/'generated_examples'; figdir.mkdir(exist_ok=True); acf_rows=[]
    for name,weight in choices:
        g=next(g for g in b['generators'] if g['weight']==weight and g['seed']==config['generator_seeds'][0])
        with np.load(Path(g['directory'])/f'bank_{b["selected_G_step"]}.npz') as bank: fake,y=bank['X'],bank['y']
        for label in range(19):
            real=validation['X'][validation['y']==label]; generated=fake[y==label][:config['quality_probes_per_class']]
            for ch,channel in enumerate(FEATURES['imu_rpm']):
                def acf(x):
                    v=x-x.mean(1,keepdims=True); denominator=np.mean(v*v)
                    return np.array([1.]+[float(np.mean(v[:,lag:]*v[:,:-lag])/max(float(denominator),1e-12)) for lag in range(1,min(31,v.shape[1]))])
                ra=acf(real[:,:,ch]); ga=acf(generated[:,:,ch])
                acf_rows.append(dict(variant=name,seed=g['seed'],label=label,channel=channel,
                    per_window_centered_acf_mae=float(np.abs(ra-ga).mean()),probes=len(generated)))
                if label!=7: continue  # Registered example: M1 with 20% remaining, first windows.
                fig,axes=plt.subplots(2,2,figsize=(11,6)); t=np.arange(real.shape[1])/100
                axes[0,0].plot(t,real[0,:,ch],label='真实验证首窗'); axes[0,0].plot(t,generated[0,:,ch],label='固定生成首窗')
                axes[0,0].set_xlabel('窗内相对时间（秒）'); axes[0,0].set_ylabel(channel); axes[0,0].legend()
                for x,title in [(real,'真实验证'),(generated,'生成')]:
                    f,p=periodogram(x[:,:,ch],fs=100,axis=1,detrend='constant')
                    axes[0,1].semilogy(f,np.maximum(p.mean(0),1e-16),label=title)
                axes[0,1].set_xlabel('频率（Hz）'); axes[0,1].set_ylabel('去均值功率谱密度'); axes[0,1].legend()
                axes[1,0].plot(np.arange(len(ra))*.01,ra,label='真实验证'); axes[1,0].plot(np.arange(len(ga))*.01,ga,label='生成')
                axes[1,0].set_xlabel('自相关滞后（秒）'); axes[1,0].set_ylabel('逐窗去均值自相关'); axes[1,0].legend()
                axes[1,1].hist(real[:,:,ch].ravel(),bins=40,density=True,alpha=.5,label='真实验证')
                axes[1,1].hist(generated[:,:,ch].ravel(),bins=40,density=True,alpha=.5,label='生成'); axes[1,1].set_xlabel(channel); axes[1,1].set_ylabel('概率密度'); axes[1,1].legend()
                for ax in axes.ravel(): ax.grid(alpha=.2)
                fig.suptitle(f'{name}，M1剩余20%效能，{channel}：固定示例，不赋予生成窗绝对阶段')
                fig.tight_layout(); fig.savefig(figdir/f'{name}_{channel}.png',bbox_inches='tight'); plt.close(fig)
        def corr(x): return np.nan_to_num(np.corrcoef(x.reshape(-1,12).T))
        real=validation['X'][validation['y']==7]; generated=fake[y==7][:config['quality_probes_per_class']]
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,x,title in zip(axes,(real,generated),('真实验证','生成')):
            image=ax.imshow(corr(x),vmin=-1,vmax=1,cmap='coolwarm'); ax.set_title(title)
            ax.set_xticks(range(12),FEATURES['imu_rpm'],rotation=90); ax.set_yticks(range(12),FEATURES['imu_rpm'])
        fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.8); fig.suptitle(f'{name}，M1剩余20%：通道关联，不是动力学闭合证明')
        fig.subplots_adjust(bottom=.27,left=.16,right=.88,wspace=.45); fig.savefig(figdir/f'{name}_correlation.png',bbox_inches='tight'); plt.close(fig)
    pd.DataFrame(acf_rows).to_csv(target/'GAN_per_window_centered_acf.csv',index=False,encoding='utf-8-sig')
    write_json(figdir/'example_selection.json',dict(label=7,window='First in source/pool and fixed latent order',
        generator_seed=config['generator_seeds'][0],G_step=b['selected_G_step'],no_synthetic_absolute_phase=True,
        original_quality_acf='Pooled mean-centering; supplementary CSV uses per-window centering. Neither is a label certificate.'))
