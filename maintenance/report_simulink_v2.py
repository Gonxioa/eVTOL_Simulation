"""Publish V2 results from completed, audited artifacts; never selects models."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import shutil
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulink_data import ROOT,read
from provenance import write_json,sha256_file

NAMES={'raw_short':'原始数据150步','raw_single':'单头1000步','raw_fixed':'固定分类头1000步',
       'dual_candidate':'双头候选1000步','ros':'验证最优ROS','plain':'原GAN','command':'指令条件GAN',
       'ros_matched_plain':'同量ROS（原GAN）','ros_matched_command':'同量ROS（条件GAN）',
       'ablation_imu':'仅IMU','ablation_imu_cmd':'IMU＋指令','ablation_imu_rpm':'IMU＋转速'}


def average(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def main():
    config=read(ROOT/'configs/simulink_gan_v2.json'); base=ROOT/config['results_root']
    final=read(base/'final_test_results.json'); frozen=read(base/'frozen_configuration.json')
    audit=read(base/'behavioral_audit.json')
    assert audit['passed'] and final['frozen_signature']==audit['frozen_signature']==frozen['signature']
    evidence=ROOT/'docs/validation/simulink_gan_v2_20261104'; evidence.mkdir(parents=True,exist_ok=True)
    plots=base/'report'; plots.mkdir(parents=True,exist_ok=True)
    for name in ('source_registry.json','final_test_results.json',
                 'final_training_complete.json','behavioral_audit.json','tests_before_freeze.log'):
        shutil.copy2(base/name,evidence/name)
    for name in ('baselines.json','head_selection.json','gan_quality_selection.json'):
        shutil.copy2(base/'development'/name,evidence/name)
    if (base/'test_reuse.log').exists():
        shutil.copy2(base/'test_reuse.log',evidence/'test_reuse.log')
    # Keep the full immutable selection grids locally; publish concise receipts
    # rather than duplicating their per-case alarm traces several times in Git.
    def policy_receipt(policy):
        return {key:value for key,value in policy.items() if key!='grid' and key!='validation'} | {
            'validation':{key:value for key,value in policy.get('validation',{}).items() if key!='cases'}}
    selected=frozen['selection']
    selection_receipt={key:selected[key] for key in ('dual_head','shared_generator_steps')}
    selection_receipt['selected']={name:{'ratio':choice['ratio'],'classifier':choice['classifier'],
                                        'alarm_policy':policy_receipt(choice['alarm_policy'])}
                                    for name,choice in selected['selected'].items()}
    selection_receipt['control_policies']={name:policy_receipt(policy) for name,policy in selected['control_policies'].items()}
    selection_receipt['rule_policy']=policy_receipt(selected['rule_policy'])
    selection_receipt['all_ratios']=[{'method':row['method'],'ratio':row['ratio'],
                                     'classifier':row['classifier'],'alarm_policy':policy_receipt(row['alarm_policy'])}
                                    for row in selected['all_ratios']]
    write_json(evidence/'validation_selection_receipt.json',selection_receipt)
    write_json(evidence/'frozen_configuration_receipt.json',
               {key:value for key,value in frozen.items() if key!='selection'} |
               {'full_artifact_sha256':sha256_file(base/'frozen_configuration.json'),
                'selection_receipt':'validation_selection_receipt.json',
                'receipt_scope':'Full immutable configuration and selection grids remain in the local results root.'})
    tests=(base/'tests_before_freeze.log').read_text(encoding='utf-8')
    count=int(re.search(r'Ran (\d+) tests',tests).group(1)); assert tests.rstrip().endswith('OK')
    rows=[]; group_rows=[]
    for result in final['results']:
        wm=result['window_metrics']; alarm=result['alarm']; delays=alarm['detection_delays_s']; loc=alarm['localization_delays_s']
        rows.append(dict(seed=result['seed'],variant=result['variant'],feature=result['feature'],
                         accuracy=wm['accuracy'],macro_f1=wm['macro_f1'],healthy_window_fpr=wm['healthy_false_positive'],
                         false_alarm_events=alarm['false_alarm_events'],normal_observation_s=alarm['normal_observation_s'],
                         false_alarm_events_per_minute=alarm['false_alarm_events_per_minute'],
                         detected=alarm['detected'],missed=alarm['missed'],fault_records=alarm['fault_records'],
                         detection_delay_median_s=None if not delays else float(np.median(delays)),
                         correct_localization_delay_median_s=None if not loc else float(np.median(loc)),
                         first_localization_accuracy=alarm['first_localization_accuracy'],
                         eventual_localization_rate=alarm['localization_rate'],threshold=result['validation_selected_threshold'],
                         hold=result['hold'],optimizer_updates=result['effective_classifier_updates']))
        for group,gm in result['per_group'].items():
            group_rows.append(dict(seed=result['seed'],variant=result['variant'],group=int(group),**gm))
    df=pd.DataFrame(rows); group_df=pd.DataFrame(group_rows)
    df.to_csv(plots/'per_seed_results.csv',index=False,encoding='utf-8-sig')
    group_df.to_csv(plots/'per_group_results.csv',index=False,encoding='utf-8-sig')
    comparisons=[('raw_single','raw_short'),('dual_candidate','raw_single'),('plain','ros_matched_plain'),
                 ('command','ros_matched_command'),('command','plain')]
    pairs=[]
    for a,b in comparisons:
        left=group_df[group_df.variant==a].set_index(['seed','group']); right=group_df[group_df.variant==b].set_index(['seed','group'])
        for key in left.index:
            pairs.append(dict(comparison=f'{a} minus {b}',seed=int(key[0]),test_group=int(key[1]),
                              accuracy_difference=float(left.loc[key,'accuracy']-right.loc[key,'accuracy']),
                              macro_f1_difference=float(left.loc[key,'macro_f1']-right.loc[key,'macro_f1']),
                              healthy_fpr_difference=float(left.loc[key,'healthy_false_positive']-right.loc[key,'healthy_false_positive'])))
    paired=pd.DataFrame(pairs); paired.to_csv(plots/'paired_group_differences.csv',index=False,encoding='utf-8-sig')
    summary={}
    for name,part in df.groupby('variant'):
        summary[name]={}
        for metric in ('accuracy','macro_f1','healthy_window_fpr','false_alarm_events_per_minute',
                       'detected','missed','detection_delay_median_s','first_localization_accuracy'):
            valid=[float(v) for v in part[metric] if pd.notna(v)]
            summary[name][metric]=dict(mean=average(valid),training_seed_std=float(np.std(valid)) if valid else None)
    write_json(plots/'summary.json',summary)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'figure.dpi':120})
    selection=frozen['selection']; baseline=selection['controls']['raw_single']; dual=selection['controls']['dual_candidate']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for spec,label in [(baseline,'单头'),(dual,'双头候选')]:
        curve=read(Path(spec['directory'])/'training_curve.json')
        for axis,key,title in zip(axes.flat,['accuracy','macro_f1','healthy_false_positive','fault_miss'],
                                 ['19类准确率','宏F1','正常窗口误报率','故障窗口漏检率']):
            axis.plot([r['step'] for r in curve],[r['validation'][key] for r in curve],'o-',label=label)
            axis.set(xlabel='分类器优化更新次数',ylabel=title,title=title); axis.grid(alpha=.3); axis.legend()
    fig.suptitle('开发种子42：固定训练／验证来源，训练预算与检测头对照')
    fig.savefig(plots/'classifier_validation.png',dpi=180); plt.close(fig)
    quality=read(base/'development/gan_quality_selection.json'); qrows=quality['rows']; fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    quality_compact=[]
    for mode,label in [('plain','原GAN'),('command','指令条件GAN')]:
        records=sorted([r for r in qrows if r['mode']==mode],key=lambda r:r['generator_updates'])
        steps=[r['generator_updates'] for r in records]; f1=[]; wd=[]; lag=[]
        for r in records:
            response=[v for v in r['quality']['per_class_channel'] if not v['channel'].startswith('cmd_motor')]
            lags=[v['lag_curve_mae'] for v in r['quality']['per_class_motor_lag'] if v['lag_curve_mae'] is not None]
            collapsed=sum(not v['constant_validation'] and v['constant_generated'] for v in r['quality']['per_class_motor_lag'])
            value=dict(mode=mode,generator_updates=r['generator_updates'],tstr_macro_f1=r['tstr_validation']['macro_f1'],
                       response_normalized_wasserstein=float(np.mean([v['normalized_wasserstein'] for v in response])),
                       lag_mae=None if not lags else float(np.mean(lags)),lag_valid_pairs=len(lags),
                       raw_dynamic_generated_constant_pairs=collapsed,
                       exact_training_duplicates=r['quality']['nearest_and_duplicates']['exact_training_duplicates'])
            quality_compact.append(value); f1.append(value['tstr_macro_f1']); wd.append(value['response_normalized_wasserstein']); lag.append(value['lag_mae'])
        for axis,y,title in zip(axes,[f1,wd,lag],['验证TSTR宏F1（1000步质量分类器）','响应通道标准化Wasserstein','非恒定通道滞后曲线误差']):
            axis.plot(steps,y,'o-',label=label); axis.set(xlabel='有效生成器更新次数',title=title); axis.grid(alpha=.3); axis.legend()
    fig.suptitle('共同预算对照；给定指令不计入响应分布改善，滞后误差同时报告有效通道数')
    fig.savefig(plots/'generator_validation_quality.png',dpi=180); plt.close(fig)
    display=['raw_short','raw_single','dual_candidate','ros','plain','ros_matched_command','command']
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for axis,metric,title in [(axes[0],'accuracy','最终测试准确率'),(axes[1],'macro_f1','最终测试宏F1')]:
        means=[summary[n][metric]['mean'] for n in display]; stds=[summary[n][metric]['training_seed_std'] for n in display]
        axis.bar(np.arange(len(display)),means,yerr=stds,capsize=3)
        axis.set_xticks(np.arange(len(display)),[NAMES[n] for n in display],rotation=25,ha='right')
        axis.set(title=title,ylim=(0,1.05)); axis.grid(axis='y',alpha=.3)
    fig.suptitle('相同三个测试条件；误差线为三个训练种子的标准差，不是独立飞行置信区间')
    fig.savefig(plots/'final_test_comparison.png',dpi=180); plt.close(fig)
    for name in ('classifier_validation.png','generator_validation_quality.png','final_test_comparison.png'):
        shutil.copy2(plots/name,evidence/name)
    write_json(evidence/'quality_compact.json',quality_compact); write_json(evidence/'summary.json',summary)
    for name in ('per_seed_results.csv','per_group_results.csv','paired_group_differences.csv'):
        shutil.copy2(plots/name,evidence/name)
    test_table=['|对照|准确率均值 ± 训练种子标准差|宏F1|正常窗口误报率|','|---|---:|---:|---:|']
    for name in ['raw_short','raw_single','dual_candidate','ros','plain','ros_matched_plain','command','ros_matched_command',
                 'ablation_imu_cmd','ablation_imu_rpm','ablation_imu']:
        r=summary[name]
        test_table.append(f'|{NAMES[name]}|{100*r["accuracy"]["mean"]:.2f}% ± {100*r["accuracy"]["training_seed_std"]:.2f}%|{r["macro_f1"]["mean"]:.4f}|{100*r["healthy_window_fpr"]["mean"]:.2f}%|')
    rule=final['rule']; test_table.append(f'|电机响应残差规则|{rule["window_metrics"]["accuracy"]*100:.2f}%|{rule["window_metrics"]["macro_f1"]:.4f}|{rule["window_metrics"]["healthy_false_positive"]*100:.2f}%|')
    alarm_table=['|对照|冻结阈值／持续次数|平均误报事件数／每种子正常观测秒数|平均漏检／54|检出条件中位延迟的均值／s|首次定位正确率|',
                 '|---|---:|---:|---:|---:|---:|']
    for name in display:
        p=df[df.variant==name]; delay=average(p.detection_delay_median_s.replace({np.nan:None}).tolist())
        loc=average(p.first_localization_accuracy.replace({np.nan:None}).tolist())
        alarm_table.append(f'|{NAMES[name]}|{p.threshold.iloc[0]:.1f}／{p.hold.iloc[0]}|{p.false_alarm_events.mean():.2f}／{p.normal_observation_s.iloc[0]:.2f}|{p.missed.mean():.2f}／54|{"—" if delay is None else f"{delay:.3f}"}|{"—" if loc is None else f"{100*loc:.2f}%"}|')
    r=rule['alarm']
    rule_delay='—' if not r['detection_delays_s'] else format(np.median(r['detection_delays_s']),'.3f')
    rule_loc='—' if r['first_localization_accuracy'] is None else format(100*r['first_localization_accuracy'],'.2f')+'%'
    alarm_table.append(f'|电机响应残差规则|{r["threshold"]:.1f}／{r["hold"]}|{r["false_alarm_events"]}／{r["normal_observation_s"]:.2f}|{r["missed"]}／54|{rule_delay}|{rule_loc}|')
    paired_summary=[]
    for comparison,p in paired.groupby('comparison'):
        paired_summary.append(f'- `{comparison}`：逐种子×逐测试组准确率差均值 {100*p.accuracy_difference.mean():+.2f} 个百分点，范围 [{100*p.accuracy_difference.min():+.2f}, {100*p.accuracy_difference.max():+.2f}]；宏F1差均值 {p.macro_f1_difference.mean():+.4f}。')
    registry=read(base/'source_registry.json'); failed=[r['source_id'] for r in registry['records'] if not r['flight_acceptance_passed']]
    plain_advantage=100*(summary['plain']['accuracy']['mean']-summary['ros_matched_plain']['accuracy']['mean'])
    problematic=max((r for r in final['results'] if r['variant']=='command'),key=lambda r:r['alarm']['missed'])
    pre_active=sum(case.get('alarm_already_active_before_fault',False) for case in problematic['alarm']['cases'])
    lines=['# Simulink GAN V2：实际实验结果','',
           '日期：2026-11-04。基线 `6f94612`；功能分支 `codex/simulink-gan-v2`。', '',
           f'配置冻结签名：`{frozen["signature"]}`。冻结前测试特征读取数为 0；全部冻结模型完成后才统一评价。', '',
           '228 条仿真记录来自 12 个配对条件组；训练／验证／测试为 6／3／3。旧四组 76 条记录原位引用，新增八组 152 条记录；任务、模型、故障强度和控制参数保持不变。',
           '最终三个训练种子 42、43、44 共享相同三个测试条件，因此不能把九个种子×条件组合当成九次独立飞行。窗口均值是描述性分类指标；统计独立的测试条件只有三个。', '',
           '## 本轮结论','',
           f'增加训练预算是本轮最明确的收益：原始数据分类器准确率由 {100*summary["raw_short"]["accuracy"]["mean"]:.2f}% 提高到 {100*summary["raw_single"]["accuracy"]["mean"]:.2f}%。原 GAN 与同量 ROS 的准确率均值只差 {plain_advantage:+.2f} 个百分点，配对差异有正有负，目前没有充分证据认定 GAN 增广优于简单过采样。',
           '指令条件 GAN 在验证生成质量上更好，但最终报警稳定性更差。不能用较高的分类准确率或 TSTR 代替正常误报和新故障检出表现；本轮不建议据此替换现有诊断配置。',
           f'IMU＋指令消融准确率为 {100*summary["ablation_imu_cmd"]["accuracy"]["mean"]:.2f}%，仅 IMU 为 {100*summary["ablation_imu"]["accuracy"]["mean"]:.2f}%，已知电机响应规则达到 {100*rule["window_metrics"]["accuracy"]:.2f}%。这些结果提示当前分类能力主要依赖闭环指令及其响应关系；不能作为仅凭 IMU 就能识别真实桨叶损伤的证据。消融仅用于本轮评价，没有据测试结果重选主特征。', '',
           '## 固定选择与阶段结果','',
           '分类器统一 1000 次优化更新，保留同一训练轨迹的第 150 步作为短预算对照；所有长预算控制共用相同初始种子、学习率、批量设置和更新次数。训练、验证、测试各 4818／2409／2409 个窗口，包含正常全时段与故障过渡段，重复健康前缀不作为十八份来源。',
           '检测头提高了开发种子的 19 类验证分数，但正常误报和二分类平衡错误率没有达到预设采用门槛。因此主要增广比较保留单头，双头候选完整保留为独立对照，不根据最终测试重新改变这个决定。',
           '该采用门槛也有局限：预设要求平衡错误率绝对下降 0.005，而开发单头的错误率已为 0.00421，低于门槛本身；双头约为 0.00443。本轮门槛过于保守，不能由未被采用推断双头无价值。需要在新实验的验证集上重新预设相对改进或事件指标，不能据本轮测试追改规则。',
           '两种 GAN 在开发阶段均保存 500、1500、3000 次有效生成器更新；质量分类器使用 1000 次更新。验证选择共同预算 3000，原 GAN 比例 0.25、指令条件 GAN 比例 0.5、最优 ROS 比例 0.25。另保留两种 GAN 各自同量的 ROS，消除新增数量差异。',
           '比例、头设计和报警阈值／持续次数由验证组选择，三个最终训练种子共用同一套冻结选择；没有根据测试重新调参。', '',
           '![分类器训练预算与检测头验证曲线](validation/simulink_gan_v2_20261104/classifier_validation.png)', '',
           '## 最终测试','',*test_table,'',
           '![三个训练种子的最终测试对照](validation/simulink_gan_v2_20261104/final_test_comparison.png)', '',
           '误差范围是三个训练随机种子的波动，不是实飞有效性或统计显著性的证明。规则基线有已知电机动力学／延迟与当前故障模型匹配的优势，不能直接外推到桨叶气动损伤或未知传感器链。', '',
           '最终窗口分类先按各自验证冻结的阈值判定正常／故障，再从十八个故障类别中定位；因此表中为该工作点下的 19 类结果。开发训练曲线统一使用 0.5 阈值，两者口径不同。', '',
           '## 检测与定位','',*alarm_table,'',
           '检测延迟仅在检出样本中定义，漏检同时列出。首次定位准确率按已经产生定位的样本计算；原始 JSON 另有对全部故障的首次正确定位覆盖率、首次正确定位延迟及最终正确定位率。不能用“后来终于定位正确”代替首次定位准确率。',
           '每个训练种子的正常观测为相同三个健康来源，分别只统计一次；每条 24 秒记录在首个 100 点窗口形成后提供约 23 秒可评价时长，共 69 秒。故障记录中重复的健康前缀不增加观测分母。有限时长内零误报不代表长期误报率为零。', '',
           f'指令条件 GAN 的训练种子 {problematic["seed"]} 在 69 秒正常观测中产生 {problematic["alarm"]["false_alarm_events"]} 次误报；{pre_active}/54 条故障记录在故障发生前报警已经激活，按预设“新上升沿”规则漏检 {problematic["alarm"]["missed"]}/54。持续误报警不能算成成功检测新故障，条件检测延迟也不能掩盖这些漏检。冻结阈值在三个训练种子间迁移不稳定，后续需用新验证试验研究校准或报警状态机。', '',
           '## 逐组配对差异','',*paired_summary,'',
           '详细九个相关单元的差值保存在 `paired_group_differences.csv`，用于查看同一来源和同一训练种子的差异，不据此计算伪独立显著性。', '',
           '## 生成质量与限制','',
           '验证质量同时记录逐类逐通道分布、频谱、带常值标记的滞后曲线，以及训练最近邻和重复情况。响应通道的 Wasserstein 与给定指令列分开报告；条件生成复制指令是定义的一部分，不能算成学会了控制输入生成。',
           'TSTR 与滞后结果反映当前仿真分布上的改善，不等于完整动力学约束；物理正则保持关闭。所有 GAN 样本仍只来自六个训练条件，不能增加独立飞行来源。',
           '![生成器预算及响应质量验证](validation/simulink_gan_v2_20261104/generator_validation_quality.png)', '',
           f'数据完整且数值有限、但未通过飞行恢复验收的记录共 {len(failed)} 条，仍全部保留，清单见来源 registry。没有降低故障强度、放宽恢复阈值或删除困难类别。', '',
           '## 验证、产物与续跑','',
           f'{count} 项测试通过。实际产物审计核对了原始来源／角色、旧文件哈希、模型参数一致性、训练组内 scaler、真实优化器计数、GAN／ROS 数量、指令条件来源和冻结测试授权。',
           '主代码、配置和运行步骤见 [V2 使用说明](SIMULINK_GAN_V2.md)。完整原始记录、权重、逐时刻预测及日志在 `output_simulink_gan/v2_12groups/`；紧凑证据在 `docs/validation/simulink_gan_v2_20261104/`；统计 CSV 和图在结果目录的 `report/`。',
           '可在同一代码与配置下重新运行 `final_train` 恢复已保存优化器状态，并用 `test` 复用同一冻结测试收据。修改代码、数据或选择需要新的实验目录和新的研究协议，不能在本测试上反复选配置。', '',
           '## 完成状态与后续边界','',
           '代码已实现、规定训练预算已执行、固定划分流程已跑通；是否有稳定增益要结合上述同量配对结果和三种子波动判断，不能只看一个均值。训练收敛、跨任务／参数失配、真实桨叶损伤和实飞泛化均未验证。UAV-FD 最终测试集未解锁。',
           '下一步应在新来源上预先设定跨任务或参数失配试验，并用独立验证数据决定是否需要置信校准；不能根据本轮测试再修改头设计、比例或报警规则。', '']
    (ROOT/'docs/SIMULINK_GAN_V2_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(evidence/'completion.json',dict(code_implemented=True,planned_budgets_executed=True,pipeline_run=True,
               frozen_signature=frozen['signature'],statistical_independence='12 total paired groups, 3 final test groups',
               training_convergence_proven=False,real_flight_validated=False,tests_passed=count,
               source_registry_sha256=sha256_file(base/'source_registry.json')))
    print('V2_REPORT_WRITTEN',plots,evidence)


if __name__=='__main__': main()
