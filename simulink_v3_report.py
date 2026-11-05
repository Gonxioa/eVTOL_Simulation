"""Chinese V3 report and scientific figures from actual frozen-test results."""
import pickle
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from provenance import write_json, sha256_file, stable_hash
from replay_diagnostics import alarm_trace
from simulink_data import ROOT, read
from simulink_v3_data import root, authorize_test


NAMES = {'A': 'A：18维，前24秒', 'B': 'B：12维，前24秒', 'C': 'C：12维，前12秒',
         'D': 'D：12维，前6秒', 'E': 'E：12维，全时段等量抽样'}


def report(config):
    base = root(config); destination = ROOT / 'docs/validation/simulink_gan_v3_prefix_20261105'
    destination.mkdir(parents=True, exist_ok=True); plots = destination / 'figures'; plots.mkdir(exist_ok=True)
    complete = read(base / 'development_complete.json'); final = read(base / 'final_test_results.json')
    frozen = read(base / 'freeze.json'); audit = read(base / 'audit/audit.json')
    authorize_test(config, read(base / 'new_test_sources.json'), frozen['signature'])
    if (not audit['passed'] or final['freeze_signature'] != frozen['signature'] or
            audit['freeze_signature'] != frozen['signature'] or complete['config_signature'] != stable_hash(config)):
        raise ValueError('Report results, audit, config and freeze must identify the same experiment')
    by_id = {r['signature']: r for r in final['results']}
    selected = final['selected_models']
    def members(name): return [by_id[s] for s in selected[name]]
    def stats(name, scope='prefix_24', key='macro_f1'):
        a = np.asarray([r['test'][scope][key] for r in members(name)])
        return float(a.mean()), float(a.std())
    def cell(name, scope='prefix_24', key='macro_f1'):
        a, b = stats(name, scope, key); return f'{100*a:.2f} ± {100*b:.2f}'
    def delta(a, b, scope='prefix_24', key='macro_f1'):
        aa = {r['seed']: r['test'][scope][key] for r in members(a)}
        bb = {r['seed']: r['test'][scope][key] for r in members(b)}
        return np.asarray([100*(aa[s]-bb[s]) for s in config['training_seeds']])
    def mean_delta(a, b, scope='prefix_24', key='macro_f1'):
        return float(np.mean(delta(a, b, scope, key)))
    def diagnostic_classes(m):
        cm = np.asarray(m['confusion_matrix']); total = int(cm[1:, :].sum())
        motor = sum(cm[i, j] for i in range(1, 19) for j in range(1, 19) if (i-1) % 6 == (j-1) % 6)
        severity = sum(cm[i, j] for i in range(1, 19) for j in range(1, 19) if (i-1) // 6 == (j-1) // 6)
        return dict(healthy_recall=None if cm[0].sum() == 0 else float(cm[0, 0]/cm[0].sum()),
                    motor_accuracy_on_fault=float(motor/max(total, 1)), severity_accuracy_on_fault=float(severity/max(total, 1)))
    metric_rows = []; group_rows = []; class_rows = []; alarm_rows = []; alarm_group_rows = []; case_rows = []; cm_rows = []; gated_rows = []
    for name, signatures in selected.items():
        for signature in signatures:
            result = by_id[signature]
            for scope, m in result['test'].items():
                row = dict(model=name, job_tag=result['tag'], seed=result['seed'], feature=result['feature'],
                           training_prefix_s=result['prefix_s'], augmentation_ratio=result['ratio'], scope=scope,
                           accuracy=m['accuracy'], macro_f1=m['macro_f1'], present_class_macro_f1=m['present_class_macro_f1'],
                           windows=m['window_count'], prediction_rule='19-class argmax', **diagnostic_classes(m))
                metric_rows.append(row)
                for label in range(19):
                    class_rows.append(dict(model=name, seed=result['seed'], scope=scope, label=label,
                                           recall=m['class_recall'][label], support=m['class_support'][label]))
            for group, scopes in result['groups'].items():
                for scope, m in scopes.items():
                    group_rows.append(dict(model=name, seed=result['seed'], group_id=int(group), scope=scope,
                                           accuracy=m['accuracy'], macro_f1=m['macro_f1'],
                                           present_class_macro_f1=m['present_class_macro_f1'], windows=m['window_count']))
            alarm = result['alarm']
            alarm_rows.append(dict(model=name, job_tag=result['tag'], seed=result['seed'],
                       **{k: v for k, v in alarm.items() if k not in ('cases', 'policy', 'localization_rate_scope') and
                          not isinstance(v, (dict, list))},
                       **{f'detection_delay_{k}_s': v for k, v in (alarm['delay_quantiles_s'] or {}).items()}))
            for scope, m in result['alarm_gated_window'].items():
                gated_rows.append(dict(model=name, seed=result['seed'], scope=scope, threshold=result['alarm_policy']['threshold'],
                    accuracy=m['accuracy'], macro_f1=m['macro_f1'], windows=m['window_count'],
                    prediction_rule='single-window fault probability threshold; no persistence; separate from main argmax'))
            for case in alarm['cases']:
                case_rows.append(dict(model=name, job_tag=result['tag'], seed=result['seed'], **case))
            for group, group_alarm in result['alarm_groups'].items():
                alarm_group_rows.append(dict(model=name, seed=result['seed'], group_id=int(group),
                    **{k: v for k, v in group_alarm.items() if not isinstance(v, (dict, list))},
                    **{f'detection_delay_{k}_s': v for k, v in (group_alarm['delay_quantiles_s'] or {}).items()}))
    for result in final['results']:
        for scope, m in result['test'].items():
            for truth, prediction in np.argwhere(np.asarray(m['confusion_matrix']) != 0):
                cm_rows.append(dict(job_tag=result['tag'], seed=result['seed'], scope=scope, truth=int(truth), prediction=int(prediction),
                                    count=m['confusion_matrix'][truth][prediction]))
    frames = {'per_seed_metrics.csv': pd.DataFrame(metric_rows), 'per_condition_metrics.csv': pd.DataFrame(group_rows),
              'per_class_recall.csv': pd.DataFrame(class_rows), 'alarm_per_seed.csv': pd.DataFrame(alarm_rows),
              'alarm_per_record.csv': pd.DataFrame(case_rows), 'confusion_nonzero.csv': pd.DataFrame(cm_rows)}
    frames['alarm_gated_per_seed_metrics.csv'] = pd.DataFrame(gated_rows)
    frames['alarm_per_condition.csv'] = pd.DataFrame(alarm_group_rows)
    fault_cases = pd.DataFrame(case_rows)
    fault_cases = fault_cases.loc[fault_cases.label != 0].copy()
    fault_cases['severity'] = (fault_cases.label-1)//6
    severity_rows = []
    for (name, seed, severity), cases in fault_cases.groupby(['model', 'seed', 'severity']):
        delays = cases.detection_delay_s.to_numpy(dtype=float)
        finite = np.isfinite(delays)
        severity_rows.append(dict(model=name, seed=int(seed), severity=int(severity),
            remaining_effectiveness=(0., .2, .8)[int(severity)], fault_records=len(cases),
            detected=int(cases.detected.eq(True).sum()), missed=int(cases.detected.eq(False).sum()),
            first_correct_localization_coverage=float(cases.first_localization_correct.eq(True).sum()/len(cases)),
            conditional_median_delay_s=None if not finite.any() else float(np.median(delays[finite])),
            **{f'detected_within_{s:g}s_rate': float(np.sum(finite & (delays <= s+1e-9))/len(cases)) for s in (.2, .5, 1., 3.)}))
    frames['alarm_by_severity.csv'] = pd.DataFrame(severity_rows)
    val_rows = []
    for job in complete['jobs']:
        for scope, m in job['validation'].items():
            val_rows.append(dict(job_tag=job['tag'], seed=job['seed'], method=job['method'], feature=job['feature'],
                                 prefix_s=job['prefix_s'], ratio=job['ratio'], equal_count=job['equal_count'], scope=scope,
                                 accuracy=m['accuracy'], macro_f1=m['macro_f1'], windows=m['window_count']))
    frames['all_validation_candidates.csv'] = pd.DataFrame(val_rows)
    frames['all_validation_alarm_policies.csv'] = pd.DataFrame([
        dict(job_tag=j['tag'], seed=j['seed'], prefix_s=j['prefix_s'], ratio=j['ratio'],
             **j['alarm_policy'], false_alarm_events=j['validation_alarm']['false_alarm_events'],
             normal_observation_s=j['validation_alarm']['normal_observation_s'],
             missed=j['validation_alarm']['missed'], detected=j['validation_alarm']['detected'],
             conditional_mean_detection_delay_s=j['validation_alarm']['conditional_mean_detection_delay_s'])
        for j in complete['jobs']])
    differences = []
    contrasts = [('B', 'A', '移除指令'), ('D', 'B', '前6秒减前24秒'), ('C', 'B', '前12秒减前24秒'),
                 ('E', 'D', '等量全时段减前6秒')]
    for seconds in (6, 24):
        contrasts += [(f'GAN_{seconds}', f'ROS_matched_GAN_{seconds}', f'{seconds}秒GAN减同量ROS'),
                      (f'GAN_{seconds}', 'D' if seconds == 6 else 'B', f'{seconds}秒GAN减原始')]
    for a, b, description in contrasts:
        for seed in config['training_seeds']:
            ra = next(r for r in members(a) if r['seed'] == seed); rb = next(r for r in members(b) if r['seed'] == seed)
            for group in config['test_groups']:
                for scope in ('prefix_6', 'prefix_12', 'prefix_24', 'early_fault_3_6', 'late_fault_12_24'):
                    differences.append(dict(contrast=description, model_a=a, model_b=b, seed=seed, group_id=group, scope=scope,
                        accuracy_difference_pp=100*(ra['groups'][str(group)][scope]['accuracy']-rb['groups'][str(group)][scope]['accuracy']),
                        macro_f1_difference_pp=100*(ra['groups'][str(group)][scope]['macro_f1']-rb['groups'][str(group)][scope]['macro_f1'])))
    frames['paired_differences.csv'] = pd.DataFrame(differences)
    quality_rows = []; quality_channels = []
    for gen in complete['generators']:
        quality = read(Path(gen['directory']) / 'quality.json')
        quality_rows.append(dict(prefix_s=gen['prefix_s'], seed=gen['seed'], **quality['summary'],
                            **quality['nearest_and_duplicates'],
                            trained_raw_classifier_agreement=quality['trained_raw_classifier_label_agreement']['accuracy']))
        for row in quality['per_class_channel']:
            quality_channels.append(dict(prefix_s=gen['prefix_s'], seed=gen['seed'], **row))
    frames['generator_quality_summary.csv'] = pd.DataFrame(quality_rows)
    frames['generator_quality_class_channel.csv'] = pd.DataFrame(quality_channels)
    for filename, frame in frames.items(): frame.to_csv(destination / filename, index=False, encoding='utf-8-sig')
    for filename in ('source_manifest.csv', 'window_source_counts.csv', 'actual_training_budgets.csv', 'audit.json'):
        shutil.copy2(base / 'audit' / filename, destination / filename)
    for original, target in [('freeze.json', 'freeze_receipt.json'), ('test_access_receipt.json', 'test_access_receipt.json'),
                              ('preflight.json', 'preflight.json'), ('seed_selection_audit.json', 'seed_selection_audit.json')]:
        shutil.copy2(base / original, destination / target)
    shutil.copy2(ROOT / config['new_test_raw'] / 'collection_plan.json', destination / 'new_test_collection_plan.json')
    for original, target in [('logs/develop.log', 'training_log.txt'), ('collection_matlab.log', 'collection_log.txt'),
                              ('logs/unit_tests.txt', 'unit_tests.txt'), ('logs/test.log', 'formal_test_log.txt'),
                              ('logs/audit.log', 'audit_log.txt')]:
        if (base / original).exists(): shutil.copy2(base / original, destination / target)
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
                         'font.size': 10, 'figure.dpi': 125, 'savefig.dpi': 160})
    def save(fig, filename):
        fig.savefig(plots / filename, bbox_inches='tight', facecolor='white'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ('accuracy', 'macro_f1'), ('准确率', '19类宏F1')):
        means, deviations = zip(*(stats(n, key=key) for n in ('A', 'B')))
        ax.bar(['18维：含指令', '12维：无指令'], np.asarray(means)*100, yerr=np.asarray(deviations)*100, capsize=4,
               color=['#4c78a8', '#f58518']); ax.set_ylabel(title+'（%）'); ax.set_ylim(0, 105); ax.grid(axis='y', alpha=.25)
    fig.suptitle('相同前24秒训练、共同完整24秒测试：三个训练种子'); fig.tight_layout(); save(fig, '01_command_ablation.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, title in zip(axes, ('accuracy', 'macro_f1'), ('准确率', '19类宏F1')):
        for scope, label, color in [('prefix_6', '共同前6秒测试', '#4c78a8'), ('prefix_12', '共同前12秒测试', '#f58518'),
                                     ('prefix_24', '共同完整24秒测试', '#54a24b')]:
            values = [stats(n, scope, key) for n in ('D', 'C', 'B')]
            ax.errorbar([6, 12, 24], [a*100 for a, _ in values], yerr=[b*100 for _, b in values],
                        marker='o', capsize=4, label=label, color=color)
        ax.set_xlabel('可用训练前缀时长（秒）'); ax.set_xticks([6, 12, 24]); ax.set_ylabel(title+'（%）'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('12维输入：改变训练时长，评价范围保持一致'); fig.tight_layout(); save(fig, '02_prefix_training.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4)); x = np.arange(3)
    for ax, key, title in zip(axes, ('accuracy', 'macro_f1'), ('准确率', '19类宏F1')):
        for offset, name, label, color in [(-.18, 'D', 'D：前6秒，714窗', '#4c78a8'), (.18, 'E', 'E：全时段抽样，714窗', '#f58518')]:
            values = [stats(name, f'prefix_{s}', key) for s in (6, 12, 24)]
            ax.bar(x+offset, [a*100 for a, _ in values], .36, yerr=[b*100 for _, b in values], capsize=3, label=label, color=color)
        ax.set_xticks(x, ['前6秒测试', '前12秒测试', '完整24秒测试']); ax.set_ylabel(title+'（%）'); ax.grid(axis='y', alpha=.25); ax.legend(fontsize=8)
    fig.suptitle('D／E等量训练对照：逐来源、逐类别数量相同'); fig.tight_layout(); save(fig, '03_equal_count.png')
    for seconds in (6, 24):
        names = ['D' if seconds == 6 else 'B', f'ROS_{seconds}', f'GAN_{seconds}', f'ROS_matched_GAN_{seconds}']
        ratios = complete['ratio_choices'][str(seconds)]
        labels = ['原始数据', f'最优ROS\n比例{ratios["ros"]["selected_ratio"]:g}',
                  f'所选GAN\n比例{ratios["gan"]["selected_ratio"]:g}', f'同量ROS\n比例{ratios["gan"]["selected_ratio"]:g}']
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, scope, title in zip(axes, ('prefix_24', 'early_fault_3_6', 'late_fault_12_24'), ('完整24秒', '故障初期3～6秒', '故障后期12～24秒')):
            values = [stats(n, scope) for n in names]
            ax.bar(labels, [a*100 for a, _ in values], yerr=[b*100 for _, b in values], capsize=3,
                   color=['#4c78a8', '#54a24b', '#e45756', '#f58518']); ax.set_ylabel('固定19类宏F1（%）'); ax.set_title(title); ax.grid(axis='y', alpha=.25)
        fig.suptitle(f'12维、前{seconds}秒训练：增广比例由验证集预先选择'); fig.tight_layout(); save(fig, f'04_augmentation_{seconds}s.png')
    class_names = ['正常']+[f'M{i}-{eta}%' for eta in (0, 20, 80) for i in range(1, 7)]
    for name in ('B', 'C', 'D', 'E'):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, scope, title in zip(axes, ('early_fault_3_6', 'late_fault_12_24'), ('故障初期3～6秒', '故障后期12～24秒')):
            cm = np.sum([np.asarray(r['test'][scope]['confusion_matrix']) for r in members(name)], axis=0)
            cm = np.divide(cm, cm.sum(1, keepdims=True), out=np.zeros_like(cm, float), where=cm.sum(1, keepdims=True) != 0)
            im = ax.imshow(cm, vmin=0, vmax=1, cmap='Blues'); ax.set_xticks(range(19), class_names, rotation=75, fontsize=7)
            ax.set_yticks(range(19), class_names, fontsize=7); ax.set_xlabel('19类预测'); ax.set_ylabel('真实类别'); ax.set_title(title)
        fig.colorbar(im, ax=axes.tolist(), shrink=.75, label='按真实类别归一化比例')
        fig.suptitle(NAMES[name]+'：三个训练种子混淆矩阵合并后按行归一化'); save(fig, f'05_confusion_{name}.png')
    # Loss and validation curves, including every classifier seed; no smoothing hides anomalies.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = dict(zip('ABCDE', ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#b279a2']))
    for job in complete['jobs']:
        if job['tag'] not in colors: continue
        history = read(Path(job['directory']) / 'training_curve.json')
        axes[0].plot([h['step'] for h in history], [h['loss'] for h in history], color=colors[job['tag']], alpha=.7,
                     label=job['tag'] if job['seed'] == 42 else None)
        axes[1].plot([h['step'] for h in history], [h['validation']['prefix_24']['macro_f1']*100 for h in history],
                     color=colors[job['tag']], alpha=.7, label=job['tag'] if job['seed'] == 42 else None)
    axes[0].set_ylabel('所记录批次的训练交叉熵'); axes[1].set_ylabel('共同24秒验证宏F1（%）')
    for ax in axes: ax.set_xlabel('实际分类器优化更新次数'); ax.legend(); ax.grid(alpha=.25)
    fig.suptitle('A～E训练曲线：每条线对应一个训练种子'); fig.tight_layout(); save(fig, '06_classifier_curves.png')
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for gen in complete['generators']:
        history = read(Path(gen['directory']) / 'training_curve.json'); row = 0 if gen['prefix_s'] == 6 else 1
        for col, key in enumerate(('loss_g', 'loss_d')):
            axes[row, col].plot([h['generator_updates'] for h in history], [h[key] for h in history], linewidth=.55, alpha=.7, label=f'种子{gen["seed"]}')
            axes[row, col].set_ylabel(('生成器损失' if col == 0 else 'critic损失')+f'（训练前{gen["prefix_s"]}秒）')
    for ax in axes.flat: ax.set_xlabel('实际生成器更新次数'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('原类别条件12维WGAN-GP：每次生成器更新对应5次critic更新'); fig.tight_layout(); save(fig, '07_gan_curves.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, seconds in zip(axes, (6, 24)):
        for method, label, color in [('ros', 'ROS', '#54a24b'), ('gan', '原类别条件GAN', '#e45756')]:
            candidates = complete['ratio_choices'][str(seconds)][method]['rows']
            ax.errorbar([r['ratio'] for r in candidates], [100*r['mean_macro_f1'] for r in candidates],
                        yerr=[100*r['std_macro_f1'] for r in candidates], marker='o', capsize=4, label=label, color=color)
        ax.set_title(f'前{seconds}秒训练'); ax.set_xlabel('逐类新增比例'); ax.set_xticks([0, .25, .5, 1.])
        ax.set_ylabel('共同完整24秒验证宏F1（%）'); ax.legend(); ax.grid(alpha=.25)
    fig.suptitle('测试解锁前的增广比例选择：三种子均值与标准差'); fig.tight_layout(); save(fig, '09_validation_ratio_grid.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in 'ABCDE':
        alarms = [r['alarm'] for r in members(name)]
        delays = np.sort([v for a in alarms for v in a['detection_delays_s']])
        total = sum(a['fault_records'] for a in alarms)
        xx = np.r_[0., delays, 21.]; yy = np.r_[0., np.arange(1, len(delays)+1)/total, len(delays)/total]
        for ax in axes: ax.step(xx, yy*100, where='post', label=name, color=colors[name])
    for ax, maximum in zip(axes, (21, 3)):
        ax.set_xlim(0, maximum); ax.set_ylim(0, 102); ax.set_xlabel('故障后检出延迟（秒）')
        ax.set_ylabel('累计检出比例（%，漏检保留在分母）'); ax.legend(); ax.grid(alpha=.25)
    fig.suptitle('冻结的逐模型／逐种子验证工作点：完整延迟分布与初期细节'); fig.tight_layout(); save(fig, '10_detection_delay_cdf.png')
    # Fixed, reproducible representative-case selection: first by tag/seed/source,
    # not the most favorable or most dramatic trace.
    representatives = []
    ordered = sorted(final['results'], key=lambda r: (r['tag'], r['seed']))
    for kind in ('false_alarm', 'miss'):
        found = None
        for result in ordered:
            cases = sorted(result['alarm']['cases'], key=lambda c: c['source_id'])
            for case in cases:
                if (kind == 'false_alarm' and case.get('false_alarm_events', 0) > 0) or (kind == 'miss' and case.get('detected') is False):
                    found = (result, case); break
            if found: break
        if not found: continue
        result, case = found; trace = None
        for file in sorted((base / 'prediction_cache').glob('test_*.json')):
            identity = read(file)
            if identity['model_signature'] != result['signature']: continue
            with file.with_suffix('.pkl').open('rb') as f: records = pickle.load(f)
            trace = next(r for r in records if r['source_id'] == case['source_id']); break
        if trace is None: raise ValueError('Missing audited replay trace')
        active, motor = alarm_trace(trace['fault_probability'], trace['fault_class'], **result['alarm_policy'])
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(trace['time_s'], trace['fault_probability'], linewidth=1); axes[0].axhline(result['alarm_policy']['threshold'], color='r', linestyle='--', label='验证集所选阈值')
        axes[0].set_ylabel('预测故障概率'); axes[0].set_ylim(0, 1.02); axes[0].legend()
        axes[1].step(trace['time_s'], active.astype(int), where='post'); axes[1].set_ylabel('持续报警状态')
        axes[1].set_yticks([0, 1]); axes[1].set_ylim(-.05, 1.05)
        axes[2].step(trace['time_s'], motor, where='post', label='定位电机'); axes[2].set_ylabel('电机编号（0为未定位）'); axes[2].set_xlabel('Time (s)')
        axes[2].set_yticks(range(7)); axes[2].set_ylim(-.2, 6.2)
        if trace['label']:
            for ax in axes: ax.axvline(trace['fault_start_s'], color='black', linestyle=':', label='实际注入时刻')
            true_motor = np.where(trace['time_s'] >= trace['fault_start_s'], (trace['label']-1) % 6+1, 0)
            axes[2].step(trace['time_s'], true_motor, where='post', color='r', linestyle='--', label='真实故障电机')
        for ax in axes: ax.grid(alpha=.25)
        axes[2].legend(); title = '正常误报' if kind == 'false_alarm' else '故障漏检'
        fig.suptitle(f'{title}实例：{result["tag"]}，训练种子{result["seed"]}，{case["source_id"]}'); fig.tight_layout()
        save(fig, f'08_replay_{kind}.png')
        representatives.append(dict(kind=kind, tag=result['tag'], seed=result['seed'], source_id=case['source_id']))
        pd.DataFrame(dict(time_s=trace['time_s'], fault_probability=trace['fault_probability'], fault_class=trace['fault_class'],
                          alarm=active, localized_motor=motor)).to_csv(destination / f'representative_{kind}.csv', index=False)
    text = []
    def add(line=''): text.append(line)
    add('# V3：去除电机指令与训练时间覆盖独立实验')
    add('')
    add(f'基线 `{config["baseline_commit"]}`。本报告来自已完成的原生 Simulink 采集、实际训练与统一冻结测试；审计通过 {audit["count"]} 项。')
    add('')
    add('## 结论与范围')
    add('')
    add(f'1. 去掉电机指令：相同前24秒训练下，B 相对 A 的完整测试准确率变化 **{mean_delta("B", "A", key="accuracy"):+.2f} 个百分点**，宏F1变化 **{mean_delta("B", "A"):+.2f} 个百分点**。主实验仅使用12维传感器信号。')
    add(f'2. 只学前6秒：D 相对 B 的故障初期宏F1变化 **{mean_delta("D", "B", "early_fault_3_6"):+.2f} 个百分点**，故障后期变化 **{mean_delta("D", "B", "late_fault_12_24"):+.2f} 个百分点**。短时段模型仍完整评价到24秒。')
    add(f'3. 前12秒：C 的完整24秒测试宏F1为 **{cell("C")}%**，相对 B 变化 **{mean_delta("C", "B"):+.2f} 个百分点**。根据测试解锁前声明的验证指标，冻结推荐为 **前{complete["recommended_prefix_s"]}秒**；没有依据测试结果重新选择训练时长。')
    add(f'4. 数量与覆盖：D、E 都使用714个原始训练窗口；E 相对 D 的完整测试宏F1变化 **{mean_delta("E", "D"):+.2f} 个百分点**。这是固定抽样种子的一次等量对照；时间覆盖、抽到的具体运动阶段会同时变化，不能把全部差异唯一归因于某一项。')
    for seconds in (6, 24):
        changes = delta(f'GAN_{seconds}', f'ROS_matched_GAN_{seconds}')
        stable = bool(np.all(changes > 1e-9)); ratio = complete['ratio_choices'][str(seconds)]['gan']['selected_ratio']
        add(f'5. 前{seconds}秒增广：所选GAN比例为{ratio:g}，相对逐类同量ROS的完整测试宏F1差值为 **{changes.mean():+.2f} 个百分点**；三个种子分别为 '+
            '、'.join(f'{v:+.2f}' for v in changes)+f'。**{"三个种子均优于同量ROS" if stable else "未稳定超过同量ROS"}**。'+('比例为0表示验证选择回退原始数据，不能据此声称GAN增广有效。' if ratio == 0 else ''))
    add('')
    add('## 数据来源与训练口径')
    add('')
    add('正式使用12个独立配对条件组、228条原始仿真记录：原训练6组114条、原验证3组57条、新测试3组57条。每组为同一随机风与测量噪声条件下的1条健康记录与18条故障记录；独立扰动来源按12组计算，不能把228条记录当成228组独立扰动实验。旧测试3组57条原位保留归档，不参与本轮正式评价。')
    add('')
    add('按不同仿真记录计数，训练为6条健康＋108条故障，验证为3条健康＋54条故障，新测试为3条健康＋54条故障；合计12条健康、216条故障。窗口与增广样本不能加到这些原始记录数量中。')
    add('')
    add('任务保持原绕圈任务，原任务运动时段为第1～7秒，其余时段包含运动前悬停与运动后保持／恢复；因此前6秒训练未覆盖全部任务运动及后段稳定过程。每条仿真24秒，第3秒注入故障，失效效能仍为0、20%、80%。物理模型、PID、容错控制器、随机风、噪声及0.03秒反馈延迟均未修改。所有228条正式记录完整且有限，飞行恢复验收失败的记录数为'+str(audit['flight_acceptance_failures_retained'])+'；没有因识别困难而删类别或重选测试种子。')
    add('')
    source_table = pd.read_csv(base / 'audit/source_manifest.csv')
    for r in source_table.loc[~source_table.flight_acceptance_passed].itertuples():
        add(f'保留的未通过飞行恢复验收来源为 `{r.source_id}`（用途{r.role}、条件组{r.group_id}、种子{r.seed}、类别{r.label}）。该记录仍完整24秒；验收失败标记保留在来源清单，没有从窗口池剔除。')
        add('')
    add('训练种子42、43、44只控制网络训练与增广抽样，不增加独立飞行来源。新测试风噪种子固定为20261115、20261116、20261117。E组抽样种子固定为731905。')
    add('')
    add('|训练区间|每条健康记录|每条故障记录|每条件组|训练6组|验证／测试3组同时间子集|')
    add('|---|---:|---:|---:|---:|---:|')
    for s, h, f, n in [(6, 11, 6, 119), (12, 23, 18, 347), (24, 47, 42, 803)]: add(f'|[0,{s})秒|{h}|{f}|{n}|{6*n}|{3*n}|')
    add('')
    add('采样频率100 Hz，采样周期0.01秒；窗口长度100次采样，窗口移动周期0.5秒。窗口例如0.00～0.99秒、0.50～1.49秒；故障记录第一个保留窗口为2.50～3.49秒。窗口标签按末时刻确定，transition／post_fault仅作离线分段标记。原CSV和MAT均保留2401行完整24秒记录。')
    add('')
    add('12维严格来自加速度计、陀螺仪以及feedback_state_14～19。IMU按V2延迟一次，转速反馈已经带噪与延迟，未再次延迟。没有将电机命令、真实转速、效能、故障时刻、控制分配残差或来源信息送入主模型。改变命令与真值列不改变主特征的检查已在实际记录上通过。')
    add('')
    add('E逐来源、逐类别无放回抽取健康11窗及每条故障记录6窗。其StandardScaler与GAN用MinMaxScaler都只拟合真正选中的窗口；没有提前用完整24秒统计量拟合。')
    add('')
    add('所有分类器沿用单头19类LSTM、隐藏维度64、2层、原注意力池化、批量32、Adam学习率0.001，并实际更新1000次。原始D／C／B在采样有放回的训练下，每个训练窗口平均被抽取约44.82／15.37／6.64次；这是固定更新预算下不同重复使用程度，不能解释为相同epoch数。')
    add('')
    add('总计51个分类器任务、51000次分类器更新；6个原类别条件12维WGAN-GP，18000次生成器更新、90000次critic更新。各GAN均完成3000次G与15000次D更新，不使用电机指令条件。所有增广数量为逐类floor(原数量×比例)，具体数量见actual_training_budgets.csv。')
    add('')
    add('## 共同测试范围上的主指标')
    add('')
    add('下表为三个训练种子的均值±总体标准差，单位%。它描述训练随机性的波动；独立测试来源仍只有3个条件组。主指标统一采用19类logits.argmax，与报警阈值无关。')
    add('')
    add('|模型|前6秒准确率|前6秒宏F1|前12秒准确率|前12秒宏F1|完整24秒准确率|完整24秒宏F1|')
    add('|---|---:|---:|---:|---:|---:|---:|')
    for name in 'ABCDE': add('|'+NAMES[name]+'|'+'|'.join(cell(name, f'prefix_{s}', k) for s in (6, 12, 24) for k in ('accuracy', 'macro_f1'))+'|')
    add('')
    add('|模型|完整时段正常召回率|故障窗口电机编号准确率|故障窗口严重度准确率|')
    add('|---|---:|---:|---:|')
    for name in 'ABCDE':
        diagnostics = [diagnostic_classes(r['test']['prefix_24']) for r in members(name)]
        add('|'+name+'|'+'|'.join(f'{100*np.mean([d[k] for d in diagnostics]):.2f}' for k in
            ('healthy_recall', 'motor_accuracy_on_fault', 'severity_accuracy_on_fault'))+'|')
    add('')
    add('电机编号与严重度准确率均从同一个19类argmax结果推导，故障被预测为正常时记错；这两项是解释性分解，不替代19类主指标。')
    add('')
    add('一个与现有模型一致的解释是：电机动力学的目标为sqrt(eta)×command，而容错控制器会按1/sqrt(估计eta)补偿命令。补偿接近准确、且执行器未饱和时，不同剩余效能下的实际转速与IMU差异可能缩小，命令仍可能保留补偿幅度信息。这是依据[电机动力学](../../../simulink/functions/hex_derivative.m)和[控制器补偿](../../../simulink/blocks/controller_step.m)作出的机理解释；分类分数下降本身不能证明严格不可辨识，也不能推出实机规律。')
    add('')
    add('|模型|故障初期3～6秒宏F1|中期6～12秒宏F1|后期12～24秒宏F1|')
    add('|---|---:|---:|---:|')
    for name in 'ABCDE': add('|'+NAMES[name]+'|'+'|'.join(cell(name, s) for s in ('early_fault_3_6', 'middle_fault_6_12', 'late_fault_12_24'))+'|')
    add('')
    add('分阶段只包含故障记录：初期按窗口末时刻[3,6)秒，中期按末时刻[6,12)秒；后期要求整个窗口起点≥12秒且末时刻<24秒。上表保持固定19类宏F1，因此没有健康样本的阶段即使18种故障全对，固定19类宏F1上限也只有18/19。CSV另列实际出现类别宏F1，以便解释；不将这两种口径混用。逐类召回与非零混淆项均单独保存，未列出的混淆项视为0。混淆矩阵标签中的0%、20%、80%表示剩余推力效能。')
    add('')
    add('## 验证选择与增广比较')
    add('')
    add('比例候选0、0.25、0.5、1.0。每个训练时长、每种方法分别以三个种子的共同完整24秒验证宏F1均值选择；比较值保留8位小数，同分选较小比例。前12秒只做原始数据对照，未因测试结果临时增加GAN。')
    add('')
    add(f'正式测试包含{final["unique_tested_models"]}个不同权重的入选分类器。最佳ROS与同量ROS若比例相同，复用同一个已训练模型并在表中明确列出；其余比例保留验证结果，不根据测试重新选择。')
    add('')
    add('|训练时长|方法|验证所选比例|完整测试准确率|完整测试宏F1|')
    add('|---|---|---:|---:|---:|')
    for seconds in (6, 24):
        for label, name, ratio in [('原始', 'D' if seconds == 6 else 'B', 0.),
                                  ('验证最优ROS', f'ROS_{seconds}', complete['ratio_choices'][str(seconds)]['ros']['selected_ratio']),
                                  ('所选GAN', f'GAN_{seconds}', complete['ratio_choices'][str(seconds)]['gan']['selected_ratio']),
                                  ('与GAN逐类同量ROS', f'ROS_matched_GAN_{seconds}', complete['ratio_choices'][str(seconds)]['gan']['selected_ratio'])]:
            add(f'|前{seconds}秒|{label}|{ratio:g}|{cell(name, key="accuracy")}|{cell(name)}|')
    add('')
    add('生成质量按类别、按通道保存分布距离、PSD差异、恒定通道标记、训练样本最近邻和精确／近重复检查。诊断使用每类固定的前32个生成样本，共608个探针，分布参照为共同完整时段验证集；这不是对全部生成窗口的穷尽质量检查。质量评价的原始分类器已完成1000次更新。质量参考可读取训练与验证，但短时段GAN的梯度训练、scaler及最近邻训练参照仅使用前6秒。生成器损失下降不被作为类别正确或增广有效的证据。')
    add('')
    add('## 完整24秒因果报警')
    add('')
    add('报警窗口仍为100次采样，移动周期为0.1秒，共231次判断，覆盖0.99～23.99秒。推理只接受信号窗口，不接收故障时刻；每条记录从头初始化状态，不在第3秒重置。单头故障概率为1减19类softmax的正常概率，定位类别取18类故障输出的最大值。')
    add('')
    add('阈值0.1～0.9与持续次数2／3／5均逐模型、逐种子通过完整24秒验证记录选择，沿用V2目标函数及状态机。故障前已激活且持续到故障后的报警不作为新检出。正常曝光只统计每组唯一健康记录；每种子共69秒有效报警观测（3×23秒），没有重复累计18条故障记录的健康前缀。因果窗口形成前0～0.99秒不计报警观测时间。')
    add('')
    add('|模型|训练种子|阈值／持续次数|正常误报事件／观测秒数|漏检／54|0.2秒内检出|0.5秒内检出|1秒内检出|3秒内检出|首次定位准确率|首次正确定位覆盖率|最终正确定位覆盖率|')
    add('|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|')
    def percent(value): return '无定位' if value is None else f'{100*value:.1f}%'
    for name in selected:
        for r in sorted(members(name), key=lambda x: x['seed']):
            a = r['alarm']; p = r['alarm_policy']
            add(f'|{name}|{r["seed"]}|{p["threshold"]:g}／{p["hold"]}|{a["false_alarm_events"]}／{a["normal_observation_s"]:.0f}|{a["missed"]}／54|'+
                '|'.join(percent(a[f'detected_within_{s:g}s_rate']) for s in (.2, .5, 1., 3.))+'|'+
                '|'.join(percent(a[k]) for k in ('first_localization_accuracy', 'first_correct_localization_fraction_all_faults', 'localization_rate'))+'|')
    add('')
    add('检出延迟分位数与每条记录的原始延迟保存在alarm_per_seed.csv／alarm_per_record.csv；漏检不从检出率分母中排除。首次定位准确率的分母是产生首次定位的记录，首次正确定位覆盖率的分母是全部54条故障。报警阈值下的单窗口分类指标单独保存在alarm_gated_per_seed_metrics.csv；其判定只使用概率阈值，不叠加持续报警次数，未混入主表。')
    add('')
    add('代表误报／漏检图固定选择按任务标识、训练种子、来源编号排序后的首个符合条件案例；全部异常都保留在CSV中，没有用个别图替代总体指标。')
    add('')
    add('### 必须保留的逐种子异常')
    add('')
    for r in sorted(final['results'], key=lambda r: (-r['alarm']['false_alarm_events'], r['tag'], r['seed']))[:3]:
        a = r['alarm']
        add(f'- `{r["tag"]}`，种子{r["seed"]}：正常误报{a["false_alarm_events"]}次／69秒，漏检{a["missed"]}／54；该异常保留，不用均值掩盖。')
    zero_false = [r for r in final['results'] if r['alarm']['false_alarm_events'] == 0]
    if zero_false:
        r = min(zero_false, key=lambda r: (-r['alarm']['missed'], r['tag'], r['seed']))
        add(f'- 零误报并不表示报警可靠：`{r["tag"]}`，种子{r["seed"]}的正常误报为0，但仍漏检{r["alarm"]["missed"]}／54。')
    add('')
    add('alarm_by_severity.csv进一步按剩余效能0／20／80%拆分检出率、早期检出率和定位覆盖率。窗口19类得分与连续报警的用途不同，不能以较高窗口得分或零误报替代完整报警验收。当前12维主模型的分类与漏检表现仍是后续研究要解决的问题，本轮没有通过重新调参或挑选测试来源隐藏这些不足。')
    add('')
    add('## 冻结、审计与使用限制')
    add('')
    add(f'冻结收据：`{frozen["signature"]}`；冻结时间 `{frozen["created_utc"]}`。全部模型、scaler、比例、报警参数和源代码哈希在测试解锁前固定。首次授权读取时间见test_access_receipt.json；测试缓存的身份包含新来源签名、特征、模型权重和scaler哈希。')
    add('')
    add('结论仅适用于当前绕圈任务、仿真模型、固定故障时刻／强度及这3个新测试扰动条件组。训练与测试使用同一动力学和容错控制器，不能推出跨任务、参数失配或实机效果。等量抽样仅使用一个预声明抽样种子；三个训练种子不能替代更多独立飞行条件。')
    add('')
    add('完整原始记录、权重、训练曲线、报警网格、预测缓存及日志保存在项目下 `'+config['results_root']+'`；当前目录保留便于GitHub审阅的报告、CSV、主要图表与哈希收据。旧V2数据和权重原位保留。')
    add('')
    add('## 图表')
    add('')
    for path in sorted(plots.glob('*.png')): add(f'![{path.stem}](figures/{path.name})\n')
    (destination / '结果报告.md').write_text('\n'.join(text).rstrip()+'\n', encoding='utf-8')
    write_json(destination / 'summary.json', dict(freeze_signature=frozen['signature'], audit_passed=audit['passed'],
               recommended_prefix_s=complete['recommended_prefix_s'], selected_models=selected,
               results_root=config['results_root'], representatives=representatives,
               baseline_means={n: {s: {k: dict(mean=stats(n, s, k)[0], std=stats(n, s, k)[1])
                                       for k in ('accuracy', 'macro_f1')} for s in ('prefix_6', 'prefix_12', 'prefix_24',
                                       'early_fault_3_6', 'late_fault_12_24')} for n in 'ABCDE'}))
    artifacts = [p for p in destination.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json']
    write_json(destination / 'artifact_manifest.json', {p.relative_to(destination).as_posix(): sha256_file(p) for p in artifacts})
    print('V3_REPORT', destination, flush=True)
    return destination
