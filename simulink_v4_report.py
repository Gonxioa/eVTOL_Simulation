"""V4 evidence reports; development and unseen confirmation are kept separate."""
from pathlib import Path
import shutil
import pickle
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from provenance import write_json,sha256_file
from simulink_data import ROOT,read
from simulink_v4_data import root,authorize
from simulink_v4_training import record_at


def destination():
    p=ROOT/'docs/validation/simulink_gan_v4_20261105'; p.mkdir(parents=True,exist_ok=True); return p


def plot_setup():
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
                         'font.size':10,'savefig.dpi':150})


def manifest(folder):
    write_json(folder/'artifact_manifest.json',{p.relative_to(folder).as_posix():sha256_file(p)
        for p in folder.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'})


def report_A(config):
    base=root(config); a=read(base/'batch_A_complete.json'); target=destination(); figures=target/'figures'; figures.mkdir(exist_ok=True)
    plot_setup(); rows=[]; classes=[]; budgets=[]
    for s in a['neural_jobs']:
        last=s['history'][-1]
        budgets.append(dict(signature=s['signature'],family=s['family'],window=s['window'],prefix_s=s['prefix_s'],grid=s['grid'],
            sampling_seed=s['sampling_seed'],method=s['method'],seed=s['seed'],updates=s['step'],training_windows=s['training_windows'],
            parameters=last['parameter_count'],training_wall_s=last['training_wall_s'],inference_ms=last['inference_ms_per_window']))
        for h in s['history']:
            m=h['validation']['prefix_24']; tr=h['full_train']
            rows.append(dict(signature=s['signature'],family=s['family'],window=s['window'],prefix_s=s['prefix_s'],grid=s['grid'],
                sampling_seed=s['sampling_seed'],method=s['method'],seed=s['seed'],updates=h['step'],
                train_macro_f1=tr['macro_f1'],train_accuracy=tr['accuracy'],train_CE=tr['cross_entropy'],validation_macro_f1=m['macro_f1'],
                validation_accuracy=m['accuracy'],validation_CE=m['cross_entropy'],healthy_recall=m['healthy_recall'],
                joint_0=m['joint_accuracy_by_effectiveness']['0.0'],joint_20=m['joint_accuracy_by_effectiveness']['0.2'],
                joint_80=m['joint_accuracy_by_effectiveness']['0.8'],worst_group_f1=h['validation']['worst_group_macro_f1'],
                parameters=h['parameter_count'],training_wall_s=h['training_wall_s'],inference_ms=h['inference_ms_per_window']))
            for label,recall in enumerate(m['class_recall']):
                classes.append(dict(signature=s['signature'],family=s['family'],window=s['window'],prefix_s=s['prefix_s'],grid=s['grid'],
                    sampling_seed=s['sampling_seed'],seed=s['seed'],updates=h['step'],label=label,recall=recall,support=m['class_support'][label]))
    pd.DataFrame(rows).to_csv(target/'A_all_train_validation_curves.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(classes).to_csv(target/'A_per_class_validation_recall.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(budgets).to_csv(target/'A_actual_training_budgets.csv',index=False,encoding='utf-8-sig')
    stats=[]
    for s in a['statistical_jobs']:
        stats.append(dict(window=s['window'],seed=s['seed'],train_f1=s['full_train']['macro_f1'],validation_f1=s['validation']['prefix_24']['macro_f1'],
                          validation_accuracy=s['validation']['prefix_24']['accuracy'],feature_count=s['feature_count'],training_wall_s=s['training_wall_s']))
    pd.DataFrame(stats).to_csv(target/'A_statistical_baselines.csv',index=False,encoding='utf-8-sig')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,key,title in [(axes[0],'mean_macro_f1','共同完整24秒验证宏F1（%）'),(axes[1],'mean_inference_ms','单窗网络推理耗时（毫秒）')]:
        x=[r['window']/100 for r in a['screen']]; y=[r[key]*(100 if key=='mean_macro_f1' else 1) for r in a['screen']]
        ax.plot(x,y,'o-'); ax.set_xlabel('窗口长度（秒）'); ax.set_ylabel(title); ax.set_xticks([.5,1,2,3]); ax.grid(alpha=.25)
    fig.suptitle('共同末点、相同4794个训练窗口、LSTM1000次更新'); fig.tight_layout(); fig.savefig(figures/'A_window_screen.png',bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for family in ('lstm','cnn'):
        for w in sorted({r['window'] for r in a['budget_candidates']}):
            rr=[r for r in a['budget_candidates'] if r['family']==family and r['window']==w]
            ax.errorbar([r['steps'] for r in rr],[100*r['mean_macro_f1'] for r in rr],
                        yerr=[100*r['std_macro_f1'] for r in rr],marker='o',capsize=3,label=f'{family.upper()}，{w/100:g}秒')
    ax.set_xlabel('实际优化更新次数'); ax.set_ylabel('共同完整24秒验证宏F1（%）'); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(figures/'A_budget_representation.png',bbox_inches='tight'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    names=['prefix_6','prefix_12','prefix_24','early_fault_3_6','late_fault_12_24']
    labels=['前6秒','前12秒','完整24秒','故障初期','故障后期']
    by_id={s['signature']:s for s in a['neural_jobs']}
    for r in a['prefix_candidates']:
        specs=[by_id[k] for k in r['signatures']]
        for ax,key,title in [(axes[0],'macro_f1','19类宏F1（%）'),(axes[1],'accuracy','准确率（%）')]:
            values=np.array([[record_at(s,a['selected_steps'])['validation'][n][key] for n in names] for s in specs])
            ax.errorbar(range(len(names)),values.mean(0)*100,yerr=values.std(0)*100,marker='o',capsize=3,label=f'前{r["prefix_s"]}秒训练')
            ax.set_xticks(range(len(names)),labels); ax.set_ylabel(title); ax.legend(); ax.grid(alpha=.25)
    fig.suptitle('固定窗长、模型和预算后的原生前缀比较：全部是开发验证结果'); fig.tight_layout(); fig.savefig(figures/'A_prefix_validation.png',bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5))
    selected=[r for r in a['prefix_candidates'] if r['prefix_s'] in (6,12,24)]
    mean_equal=np.mean([r['mean_macro_f1'] for r in a['equal_controls']])
    ax.bar(['前6秒','前12秒','前24秒','全时段等量抽样\n三个抽样种子均值'],[100*r['mean_macro_f1'] for r in selected]+[100*mean_equal])
    ax.set_ylabel('共同完整24秒验证宏F1（%）'); ax.grid(axis='y',alpha=.25)
    fig.tight_layout(); fig.savefig(figures/'A_equal_count.png',bbox_inches='tight'); plt.close(fig)
    for p in (base/'decisions').glob('A_*.json'):
        shutil.copy2(p,target/p.name)
    for original,name in [('audit/source_manifest.csv','A_source_manifest.csv'),('audit/source_audit.json','A_source_audit.json'),
                           ('preflight.json','preflight.json'),('raw_baseline_freeze.json','raw_baseline_freeze.json'),
                           ('logs/unit_tests.txt','unit_tests.txt'),('logs/batch_a.log','A_training_log.txt')]:
        shutil.copy2(base/original,target/name)
    signal_fig=target/'real_signal_analysis'; signal_fig.mkdir(exist_ok=True)
    for p in (base/'signal_analysis').glob('*'):
        if p.is_file(): shutil.copy2(p,signal_fig/p.name)
    text=['# V4批次A：实际开发结果','',
        '**本报告仅含训练／验证结果，未将已曝光V2/V3测试当作新确认。**','',
        f'基线 `{config["baseline_commit"]}`。原始训练6组114条、验证3组57条，原始记录完整24秒。实际完成{a["actual_neural_trajectories"]}条神经训练轨迹、6个统计特征随机森林拟合。',
        '',f'按预登记规则，神经网络研究参考为：**{a["selected_window"]/100:g}秒窗口、{a["selected_family"].upper()}、{a["selected_steps"]}次更新、前{a["selected_prefix_s"]}秒真实训练数据**。',
        '', '## 窗口与预算证据','', '|窗口／秒|共同末点LSTM1000步验证宏F1|单窗网络推理毫秒|','|---|---:|---:|']
    for r in a['screen']: text.append(f'|{r["window"]/100:g}|{100*r["mean_macro_f1"]:.2f}%|{r["mean_inference_ms"]:.3f}|')
    text+=['','四个长度共用4794训练窗、2397验证窗。此表用于控制末点比较，不直接等同V3原生网格。选定窗口后，在从0秒开始的原生网格上重新训练、重新复核；没有静默复用优化器。',
        '', '|原生训练前缀|原始训练窗口|完整24秒验证宏F1|','|---|---:|---:|']
    for r in a['prefix_candidates']: text.append(f'|前{r["prefix_s"]}秒|{r["original_windows"]}|{100*r["mean_macro_f1"]:.2f}% ± {100*r["std_macro_f1"]:.2f}%|')
    text+=['','## 学习能力与关键限制','']
    mem=a['memorization_job']['history'][-1]['full_train']
    text.append(f'固定152个真实窗口的小样本检查，1000步训练宏F1为{100*mem["macro_f1"]:.2f}%。该检查没有充分记住全部样本；不能据此宣称优化充分或严格不可辨识。')
    text.append('')
    for window in sorted({r['window'] for r in stats}):
        rr=[r for r in stats if r['window']==window]
        text.append(f'{window/100:g}秒共同窗口的随机森林验证宏F1均值为{100*np.mean([r["validation_f1"] for r in rr]):.2f}%。它只使用相同12维信号的窗口统计描述，属于单独表示对照；不得忽略其高于神经网络的结果，也不据此证明所有19类都可靠可辨。')
        text.append('')
    text+=['4000步是有限预算上限，若曲线仍改善，不称为已充分收敛。等量全时段抽样的三个种子与三个训练种子完整交叉，共九次诊断；不选最好抽样种子，也不增加独立扰动来源。',
        '',f'G1最低非塌缩进入检查：**{"通过" if a["teacher_gate"]["passed"] else "未通过，G1跳过"}**。该条件只排除零召回或极弱教师，不认证生成标签。后续G1是探索性外部教师约束，必须以独立TSTR、真实验证和直接质量指标另行检查。',
        '', '## 真实信号分析','',
        '来源统计包含171条记录、12通道、五个名义时间阶段；保存均值、波动、变化率、去均值PSD、自相关及通道关联。配对差值与跨条件类内距离分开列出：相同条件的共享风噪会在减法中抵消，因此配对波形有差异不等于在线已有反事实参照。',
        '', '名义第7秒不保证实际运动完成。source_manifest记录7秒、12秒的实际任务phase；阶段只用于离线分析，不进入网络。完整有限但飞行验收失败的原记录全部保留。',
        '', '## 状态','',
        '批次A已实际完成并冻结共同真实训练池。批次B的实际状态以独立日志为准；新确认测试仍需全部开发选择完成后冻结再解锁。此处不声称GAN收益或新测试成绩。','']
    for p in sorted(figures.glob('A_*.png')): text.append(f'![{p.stem}](figures/{p.name})\n')
    (target/'批次A结果报告.md').write_text('\n'.join(text).rstrip()+'\n',encoding='utf-8')
    manifest(target); print('V4_REPORT_A',target,flush=True); return target


def report_final(config):
    target=report_A(config); base=root(config)
    a=read(base/'batch_A_complete.json'); b=read(base/'batch_B_complete.json'); final=read(base/'final_confirmation_results.json')
    frozen=read(base/'freeze.json'); audit=read(base/'audit/final_audit.json'); reg=read(base/'confirmation_sources.json')
    authorize(config,reg,'test',frozen['signature'])
    if final['freeze_signature']!=frozen['signature'] or not audit['passed']: raise ValueError('Report source/freeze mismatch')
    index={r['signature']:r for r in final['results']}; models={s['signature']:s for s in b['models']}
    rows=[]; group_rows=[]; recalls=[]; alarm_rows=[]; cases=[]; paired=[]
    for name,ids in b['selected'].items():
        for identity in ids:
            r=index[identity]; spec=models[identity]
            for scope,m in r['test'].items():
                if not isinstance(m,dict) or 'macro_f1' not in m: continue
                rows.append(dict(model=name,seed=r['seed'],scope=scope,window_count=m['window_count'],accuracy=m['accuracy'],macro_f1=m['macro_f1'],
                    present_class_macro_f1=m['present_class_macro_f1'],prediction_rule='19-class argmax'))
                if m['window_count']:
                    for label,recall in enumerate(m['class_recall']): recalls.append(dict(model=name,seed=r['seed'],scope=scope,label=label,recall=recall,support=m['class_support'][label]))
            for group,m in r['test']['groups'].items(): group_rows.append(dict(model=name,seed=r['seed'],group=int(group),accuracy=m['accuracy'],macro_f1=m['macro_f1']))
            alarm=r['alarm']
            alarm_rows.append(dict(model=name,seed=r['seed'],**{k:v for k,v in alarm.items() if not isinstance(v,(dict,list))},
                                  **{f'delay_{k}_s':v for k,v in (alarm['delay_quantiles_s'] or {}).items()}))
            for case in alarm['cases']: cases.append(dict(model=name,seed=r['seed'],**case))
    for variant in ('G0','G1'):
        if variant not in b['selected']: continue
        for comparator in ('raw','ROS_matched_'+variant):
            for seed in config['training_seeds']:
                ra=next(index[s] for s in b['selected'][variant] if index[s]['seed']==seed)
                rb=next(index[s] for s in b['selected'][comparator] if index[s]['seed']==seed)
                for group,m in ra['test']['groups'].items():
                    n=rb['test']['groups'][group]
                    paired.append(dict(variant=variant,comparator=comparator,seed=seed,group=int(group),
                        accuracy_difference_pp=100*(m['accuracy']-n['accuracy']),macro_f1_difference_pp=100*(m['macro_f1']-n['macro_f1'])))
    for file,frame in [('final_per_seed_metrics.csv',rows),('final_per_condition_metrics.csv',group_rows),
                       ('final_per_class_recall.csv',recalls),('final_alarm_per_seed.csv',alarm_rows),
                       ('final_alarm_per_record.csv',cases),('final_paired_differences.csv',paired)]:
        pd.DataFrame(frame).to_csv(target/file,index=False,encoding='utf-8-sig')
    paired_frame=pd.DataFrame(paired); uncertainty=[]
    rng=np.random.default_rng(20261105)
    for (variant,comparator),data in paired_frame.groupby(['variant','comparator']):
        values=data.groupby('group').macro_f1_difference_pp.mean().to_numpy()
        bootstrap=values[rng.integers(0,len(values),(4000,len(values)))].mean(1)
        uncertainty.append(dict(variant=variant,comparator=comparator,independent_groups=len(values),
            mean_difference_pp=float(values.mean()),minimum_group_pp=float(values.min()),maximum_group_pp=float(values.max()),
            cluster_bootstrap_lower_pp=float(np.quantile(bootstrap,.025)),cluster_bootstrap_upper_pp=float(np.quantile(bootstrap,.975)),
            scope='Average training repeats within group first; resample paired condition groups only. Small group-count interval is descriptive and unstable.'))
    pd.DataFrame(uncertainty).to_csv(target/'group_level_uncertainty.csv',index=False,encoding='utf-8-sig')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    names=[n for n in ('raw','ROS_best','G0','ROS_matched_G0','G1','ROS_matched_G1') if n in b['selected']]
    display={'raw':'原始数据','ROS_best':'验证最优ROS','G0':'原GAN（G0）','G1':'类别约束GAN（G1）',
             'ROS_matched_G0':'与G0同量ROS','ROS_matched_G1':'与G1同量ROS'}
    for ax,key,title in [(axes[0],'accuracy','准确率（%）'),(axes[1],'macro_f1','19类宏F1（%）')]:
        values=[[index[s]['test']['prefix_24'][key]*100 for s in b['selected'][n]] for n in names]
        ax.bar([display[n] for n in names],[np.mean(v) for v in values],yerr=[np.std(v) for v in values],capsize=3)
        ax.set_ylabel(title); ax.tick_params(axis='x',rotation=20); ax.grid(axis='y',alpha=.25)
    fig.suptitle('全新独立风噪条件组：全部选择冻结后的一次统一确认'); fig.tight_layout(); fig.savefig(target/'figures/final_comparison.png',bbox_inches='tight'); plt.close(fig)
    quality=[]; curves=[]
    for g in b['generators']:
        q=read(Path(g['directory'])/f'quality_{b["selected_G_step"]}.json')
        quality.append(dict(weight=g['weight'],seed=g['seed'],G_checkpoint=b['selected_G_step'],
            **q['summary'],all_generated_count=q['all_generated_count'],exact_training_duplicates=q['exact_training_duplicates_all_admitted'],
            exact_generated_duplicates=q['exact_generated_duplicates_all_admitted']))
        for row in read(Path(g['directory'])/'training_curve.json'): curves.append(dict(weight=g['weight'],seed=g['seed'],**row))
    pd.DataFrame(quality).to_csv(target/'GAN_quality_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(curves).to_csv(target/'GAN_decomposed_training_curves.csv',index=False,encoding='utf-8-sig')
    variants=sorted({g['weight'] for g in b['generators']}); fig,axes=plt.subplots(len(variants),3,figsize=(14,3.5*len(variants)),squeeze=False)
    for row,weight in enumerate(variants):
        for g in [g for g in b['generators'] if g['weight']==weight]:
            h=read(Path(g['directory'])/'training_curve.json'); x=[r['generator_updates'] for r in h]
            for ax,key in zip(axes[row],('loss_g','loss_d','gradient_penalty')):
                ax.plot(x,[r[key] for r in h],linewidth=.55,label=f'生成种子{g["seed"]}'); ax.set_xlabel('实际G更新次数')
                ax.set_ylabel(f'权重{weight:g}：{key}'); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.suptitle('G为单批损失，D为5次实际更新的均值；平滑程度不同不表示收敛优劣'); fig.tight_layout(); fig.savefig(target/'figures/GAN_loss_decomposition.png',bbox_inches='tight'); plt.close(fig)
    for original,target_name in [('freeze.json','freeze_receipt.json'),('confirmation_design.json','confirmation_design.json'),
        ('decision_register.json','decision_register.json'),
        ('hardware.json','hardware.json'),('registered_config.json','registered_config.json'),('registered_batch_B.json','registered_batch_B.json'),
        ('confirmation_matlab.log','confirmation_collection_log.txt'),('confirmation_collection_status.json','confirmation_collection_status.json'),
        ('test_access_receipt.json','test_access_receipt.json'),('audit/final_audit.json','final_audit.json'),
        ('audit/formal_source_manifest.csv','formal_source_manifest.csv'),('audit/actual_neural_budgets.csv','actual_neural_budgets.csv'),
        ('logs/batch_b.log','B_training_log.txt'),('logs/confirm_test.log','confirmation_test_log.txt')]:
        if (base/original).exists(): shutil.copy2(base/original,target/target_name)
    from simulink_v4_reporting_extra import exports
    exports(config,a,b,final,target,frozen,reg)
    def summary(name,key='macro_f1'):
        values=[index[s]['test']['prefix_24'][key]*100 for s in b['selected'][name]]
        return f'{np.mean(values):.2f} ± {np.std(values):.2f}'
    text=['# V4证据链：实际训练、增广与独立确认结果','',
        f'基线 `{config["baseline_commit"]}`。原始六训练组114条、三验证组57条；全新确认{final["groups"]}组、{final["records"]}条。旧V2/V3测试保持历史身份，未用于本轮正式确认。',
        '',f'实际完成{audit["neural_trajectories"]}条神经训练轨迹，{audit["classifier_updates"]}次分类器更新；'+
        f'{len(b["generators"])}个GAN，{audit["generator_updates"]}次G、{audit["critic_updates"]}次critic更新。统计基线另有6个随机森林拟合，不伪记为Adam训练步数。',
        '', '实际硬件为RTX 4060 Laptop GPU，PyTorch2.5.1+CUDA12.1、float32神经优化。参数量、训练墙钟及单窗推理耗时分别保存在A_actual_training_budgets.csv和GAN_actual_budgets.csv；训练曲线覆盖实际检查点。不同模型相同更新数不代表相同算力。',
        '',f'共同真实信息预算为前{a["selected_prefix_s"]}秒，{a["selected_window"]/100:g}秒窗口，{a["selected_family"].upper()}、{a["selected_steps"]}次更新。此前的窗口、预算、表示、覆盖和等量诊断见[批次A报告](批次A结果报告.md)。',
        '', '## 新确认主指标','',
        '以下为三个联合训练重复的均值±总体标准差，均是共同完整24秒、19类argmax结果；独立测试来源按条件组计，不按窗口数或训练种子数计。',
        '', '|方法|准确率／%|宏F1／%|','|---|---:|---:|']
    for name in names: text.append(f'|{name}|{summary(name,"accuracy")}|{summary(name)}|')
    alarm_values=[r['alarm']['false_alarm_events'] for r in final['results']]
    text+=['',f'**主要结果：** 原始数据宏F1为{summary("raw")}%，G0为{summary("G0")}%。'+
        (f'类别约束G1为{summary("G1")}%，其逐类同量ROS为{summary("ROS_matched_G1")}%。' if 'G1' in names else '')+
        '这些有限重复没有支持GAN稳定优于同量ROS。',
        '',f'**报警仍不可靠：** 每个模型的唯一健康观测总时长为{final["results"][0]["alarm"]["normal_observation_s"]:.2f}秒；'+
        f'各模型误报{min(alarm_values)}～{max(alarm_values)}次。故障在完整24秒内最终检出，不代表早期识别、正确定位或低误报达到要求。'+
        '因此当前只作为研究对照，优先复核真实信号表征、训练充分程度及误报控制，不据增广后略高的均值认定可用。']
    text+=['','raw为原始数据；G0为原类别条件WGAN-GP，G1为增加冻结真实数据教师类别约束的变体；ROS为有放回重复抽取已有真实训练窗。ROS_best由验证选择比例，ROS_matched按相应GAN逐类新增数量匹配。ROS_best与ROS_matched_G1本次是同一组权重，不能当成两次独立实验。',
        '', '|方法|健康召回|完全失效联合正确率|剩余20%联合正确率|剩余80%联合正确率|','|---|---:|---:|---:|---:|']
    for name in names:
        mm=[index[s]['test']['prefix_24'] for s in b['selected'][name]]
        columns=[np.mean([m['healthy_recall'] for m in mm])]+[np.mean([m['joint_accuracy_by_effectiveness'][str(eta)] for m in mm]) for eta in (0.,.2,.8)]
        text.append('|'+name+'|'+ '|'.join(f'{100*v:.2f}%' for v in columns)+'|')
    text+=['','联合正确率要求电机编号与失效程度同时正确。剩余80%类别整体仅约20%，是当前明显短板；完全失效的高正确率不能替代轻度故障定位能力。',
        '', '|方法|真实训练窗|新增生成／重复窗|分类训练总窗|','|---|---:|---:|---:|']
    for name in names:
        s=models[b['selected'][name][0]]
        text.append(f'|{name}|{s["original_windows"]}|{s["training_windows"]-s["original_windows"]}|{s["training_windows"]}|')
    text+=['','每种方法均使用同一114条原始训练来源，不因增加窗口副本或生成样本而增加独立仿真次数；最终确认是另外六个新条件组。']
    text+=['','## 开发选择与可用结论','',
        '开发验证的前6／12／24秒比较、等量抽样与窗口筛选均在独立新确认之前完成；新确认只评价已经选定的共同训练覆盖，不用新测试再次挑选前缀。',
        '', '|开发候选|所选新增比例|共同完整24秒验证宏F1／%|','|---|---:|---:|',
        f'|原始数据|0|{100*next(r["mean_macro_f1"] for r in a["prefix_candidates"] if r["prefix_s"]==a["selected_prefix_s"]):.2f}|',
        f'|ROS|{b["ROS_choice"]["selected"]["ratio"]:g}|{100*b["ROS_choice"]["selected"]["mean_macro_f1"]:.2f}|']
    for weight,choice in b['ratio_choices'].items():
        text.append(f'|GAN类别约束权重{weight}|{choice["selected"]["ratio"]:g}|{100*choice["selected"]["mean_macro_f1"]:.2f}|')
    text+=['','G1的两个权重和全部增广比例都保留，没有因表现较差从报告中删除。B_all_validation_candidates.csv记录每个联合重复和逐类新增数量，B_independent_TSTR.csv记录独立转移结果。',
        '', '统计特征随机森林在开发验证中明显高于对应神经参考结果，但其池为完整24秒共同网格，对照对象是同池神经网络，不与前12秒主池直接作等信息预算结论。它未被登记为本轮新确认主候选，不能把其开发成绩当成已经取得同等独立确认。后续优先复核表示与优化，当前结果不足以把12维信号判为严格不可辨识，也不足以部署为可靠的19类诊断器。',
        '', '本轮为独立12维实验，不重复V3的18维输入消融；输入维度影响应查看V3的同协议对照，不能把V3和V4不同模型预算的差值只归因于删除命令。']
    text+=['','## 主要配对结果与不确定性','']
    for row in uncertainty:
        text.append(f'{row["variant"]} 相对 {row["comparator"]}：先对每个条件组汇总训练重复，再跨组平均，宏F1差值{row["mean_difference_pp"]:+.2f}个百分点；'+
            f'组间范围[{row["minimum_group_pp"]:+.2f},{row["maximum_group_pp"]:+.2f}]，描述性组级bootstrap区间[{row["cluster_bootstrap_lower_pp"]:+.2f},{row["cluster_bootstrap_upper_pp"]:+.2f}]。')
        text.append('')
    text+=['小规模组级区间估计不稳；没有把重叠窗口或三个联合重复当作额外独立飞行。新组数在采集前按登记公式固定，并在6～12组预算范围内截断，不能把规划公式当作达到目标精度的保证。',
        '', '## GAN条件、约束和质量','',
        f'三个生成器种子142/143/144分别与分类器42/43/44配对，ROS抽样种子为242/243/244。这是三个联合重复，不是生成器／分类器方差的完全分解。G0和允许的G1都实际训练到3000次G、15000次critic；共同使用验证选择的第{b["selected_G_step"]}次G检查点。',
        '',f'G1最低非塌缩条件{"通过，已执行权重0.1和1.0的探索对照" if b["teacher_gate_passed"] else "未通过，G1按协议跳过"}。它不认证教师或生成标签。G2和样本筛选关闭，失败或无收益候选保留。',
        '', '生成质量的主要真实参照是相同窗长、相同训练覆盖范围的独立验证窗口；同时保留全24秒下游评价。记录去均值频谱、自相关、通道关系、分布距离和训练最近邻。全部入选窗口检查有限性、数量、精确重复；昂贵分布与距离诊断每类固定32个探针，不能将探针结果当全部样本的穷尽证明。',
        '', '条件标签只代表请求生成的类别。外部冻结教师提供探索性损失，TSTR分类器从随机初始化重新训练；不据教师自评分或漂亮波形认定物理故障标签正确。缺少命令和完整状态，不报告伪造的动力学闭合残差。带噪延迟转速观测保留停转附近可能的小负值。',
        '', '原GAN质量JSON的自相关按汇总数据中心化；GAN_per_window_centered_acf.csv额外报告逐窗去均值自相关，避免窗间均值差异掩盖窗内动态。generated_examples目录逐通道保存固定M1剩余20%示例、去均值PSD、逐窗自相关、分布及关联图；示例不按好坏挑选，生成样本没有真实绝对阶段。',
        '', '|类别约束权重|验证归一化Wasserstein均值|去均值PSD差异L1|精确训练重复数|独立TSTR完整验证宏F1|','|---|---:|---:|---:|---:|']
    for weight in sorted({g['weight'] for g in b['generators']}):
        q=[r for r in quality if r['weight']==weight]; ids={g['signature'] for g in b['generators'] if g['weight']==weight}
        t=[record_at(r['model'],r['model']['step'])['validation']['prefix_24']['macro_f1'] for r in b['TSTR'] if r['generator'] in ids and r['G_step']==b['selected_G_step']]
        text.append(f'|{weight:g}|{np.mean([r["normalized_wasserstein"] for r in q]):.3f}|{np.mean([r["demeaned_psd_l1"] for r in q]):.3f}|{sum(r["exact_training_duplicates"] for r in q)}|{100*np.mean(t):.2f}%|')
    text+=['','分布距离和PSD差异越小越接近相应真实验证参照；不把它们当成故障标签真值。生成频谱与逐窗自相关仍有明显偏差；没有精确复制也不等于足够多样或物理正确。',
        '', '## 连续因果诊断与异常','',
        '报警概率取19类softmax的1−正常概率；主窗口类别取19类argmax。两者不是同一个判定条件。每个窗口独立计算网络特征，不传递上一窗口隐藏状态；持续次数是报警状态机的记忆。',
        '阶段分类表中，早期按窗口末时刻落在[3,6)秒选取；后期要求窗口起点不早于12秒且末时刻早于24秒。阶段不是网络输入。纯故障阶段没有健康真值，固定19类宏F1上限为18/19；另列实际出现类别宏F1和支持数，不能直接与含健康的全时段分数作同口径比较。',
        '',
        '每条记录从首个完整窗口初始化，按0.1秒判断周期只读取过去与当前测量；无3秒真值重置。标准化只transform，不在新测试上fit。正常曝光只统计每个条件组唯一健康来源。检测、首次定位、最终正确定位分别评分；所有漏检保留在覆盖率分母。',
        '', '|方法|训练种子|阈值／持续|正常误报／观测秒|漏检／故障记录|0.5秒内检出|首次正确定位覆盖|最终正确定位覆盖|',
        '|---|---:|---|---|---|---:|---:|---:|']
    for name,ids in b['selected'].items():
        for identity in ids:
            r=index[identity]; al=r['alarm']; p=r['alarm_policy']
            text.append(f'|{name}|{r["seed"]}|{p["threshold"]:g}／{p["hold"]}|{al["false_alarm_events"]}／{al["normal_observation_s"]:.2f}|'+
                f'{al["missed"]}／{al["fault_records"]}|{100*al["detected_within_0.5s_rate"]:.1f}%|{100*al["first_correct_localization_fraction_all_faults"]:.1f}%|{100*al["localization_rate"]:.1f}%|')
    text+=['','单窗概率阈值成绩另存在原始JSON的alarm_gated_window字段，未混入argmax主表。0.2／0.5／1／3秒检出率、延迟分位数、首次定位准确率和逐记录异常均保存在CSV。',
        '', '本次没有完整24秒漏检记录，因此不制造全程漏检示例。另按固定种子、来源排序选取首个晚于1秒才检出的记录绘图，标为早期漏检；与健康误报时序共同保存在figures和representative_alarm_traces.csv。',
        '',f'完整有限但未通过飞行恢复验收的正式来源共{audit["retained_flight_acceptance_failures"]}条，保留并标记；没有删除困难类别或替换测试种子。',
        '', '## 冻结和完成范围','',f'冻结签名 `{frozen["signature"]}`。全部选择、模型、scaler和工作点在新测试特征解锁前固定，首次读取见test_access_receipt.json。',
        '', '采集前已通过confirmation_design绑定批次B完整选择记录、组数及种子；采集完成后登记实际原始来源哈希，再对所有模型、scaler、工作点和代码形成最终freeze。首次新测试特征读取发生在该最终freeze之后；没有把采集日志中的飞行恢复指标用于训练选择。',
        '', '模型、物理参数、PID、故障强度和测量链未为提高识别分数而修改。本轮只在同一绕圈任务、固定故障时刻和故障模型下改变独立风噪来源；跨任务、可变注入时刻、参数失配和实机尚未验证。4000步不保证收敛，类别约束不保证生成标签真实，验证最优不代表全局最优。',
        '',f'未执行：G2、阶段条件生成、跨窗有状态模型、筛选、额外任务／参数失配试验；这些不属于本次登记的首轮。实际执行和数量审计通过{audit["count"]}项，测试数量不替代识别效果或飞行验收。',
        '', '批次B有一次Windows原子写进度文件的访问冲突；日志保留原异常及恢复入口。已完成模型从原优化器检查点复用，后续完成标记和实际Adam状态另行审计，没有把恢复启动次数算成新增训练重复。',
        '',f'完整数据、权重、检查点、采样索引、缓存和日志保存在 `{config["results_root"]}`，此目录提供便于版本审阅的报告、图表、CSV和收据。','']
    for p in sorted((target/'figures').glob('*.png')): text.append(f'![{p.stem}](figures/{p.name})\n')
    (target/'结果报告.md').write_text('\n'.join(text).rstrip()+'\n',encoding='utf-8')
    manifest(target); print('V4_REPORT_FINAL',target,flush=True); return target
