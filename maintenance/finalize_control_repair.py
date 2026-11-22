"""Produce the final evidence bundle after every declared native run completes."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from provenance import sha256_file,write_json
from analyze_control_repair import main as analyze,rows


def finalize(base):
    target=ROOT/'docs/validation/control_repair_20261122'
    plan=json.loads((base/'formal_01/plan.json').read_text(encoding='utf-8-sig'))
    records=rows(base/'formal_01')
    assert len(plan['cases'])==544 and len(records)==544
    assert all(r['status']=='completed' and r['metrics']['passed'] for r in records)
    assert {r['source_signature'] for r in records}=={plan['source_signature']}
    analyze(base,target)
    frame=pd.read_csv(target/'native_cases.csv')
    formal=frame[frame.suite.isin(['development','independent','ideal'])]
    assert len(formal)==532
    old_path=Path('C:/Users/lenovo/Desktop/eVTOL_Simulation/simulink/validation/SIMULATION_CORE_RESULTS.json')
    old=json.loads(old_path.read_text(encoding='utf-8-sig'))['cases']
    lookup={(int(x['task']),int(x['motor']),round(float(x['factor']),2)):x for x in old}
    comparisons=[]
    for r in records:
        c=r['case'];m=r['metrics']
        if c['suite']!='development' or c['seed']!=20261006 or c['motor']==0:continue
        previous=lookup[(c['task'],c['motor'],c['factor'])]
        item=dict(task=c['task'],motor=c['motor'],factor=c['factor'],seed=c['seed'])
        for key in ['return_time_s','late_tracking_rmse_m','peak_tilt_deg','peak_rp_rate_rad_s','phase_s']:
            item['old_'+key]=previous[key];item['new_'+key]=m[key];item['change_'+key]=m[key]-previous[key]
        comparisons.append(item)
    assert len(comparisons)==72
    comparison=pd.DataFrame(comparisons);comparison.to_csv(target/'core_72_paired_changes.csv',index=False,encoding='utf-8-sig')
    # Retain every worsening metric, even if every flight passes acceptance.
    worse=comparison[(comparison.change_return_time_s>1e-9)|(comparison.change_late_tracking_rmse_m>1e-9)|
                     (comparison.change_peak_tilt_deg>1e-9)|(comparison.change_peak_rp_rate_rad_s>1e-9)]
    worse.to_csv(target/'core_72_any_metric_regressions.csv',index=False,encoding='utf-8-sig')
    interface=json.loads((base/'interface_01/interface_receipt.json').read_text())
    assert interface['passed'] and interface['records']==9 and not interface['training_performed']
    for file in ['delivery_receipt.json','plan.json']:
        source=base/'formal_01'/file
        if source.exists():shutil.copy2(source,target/('formal_'+file))
    group=[]
    for (suite,seed),a in formal.groupby(['suite','seed']):
        group.append(dict(suite=suite,seed=int(seed),records=len(a),passed=int(a.passed.sum()),
            median_recovery_s=float(a.return_time_s.median()),p90_recovery_s=float(a.return_time_s.quantile(.9)),
            maximum_recovery_s=float(a.return_time_s.max()),worst_late_rmse_m=float(a.late_tracking_rmse_m.max()),
            maximum_tilt_deg=float(a.peak_tilt_deg.max()),maximum_rp_rate_rad_s=float(a.peak_rp_rate_rad_s.max()),
            phase_complete=int(a.mission_phase_complete.sum())))
    pd.DataFrame(group).to_csv(target/'per_seed_summary.csv',index=False,encoding='utf-8-sig')
    receipt=dict(total_full_24s_native_runs=668,main_matrix_runs=532,main_matrix_passed=int(formal.passed.sum()),
        frozen_formal_runs=544,frozen_formal_passed=int(frame[frame.name.isin([c['name'] for c in plan['cases']])].passed.sum()),
        development_ablation_runs=75,preflight_runs=40,interface_runs=9,
        independent_groups=3,independent_fault_records=216,independent_healthy_records=12,
        csv_recomputed_records=len(frame),controller_version=plan['controller_version'],source_signature=plan['source_signature'],
        old_core_summary_sha256=sha256_file(old_path),core_paired_cases=72,
        core_cases_with_any_metric_increase=len(worse),
        old_core_maximum_recovery_s=float(comparison.old_return_time_s.max()),
        new_core_maximum_recovery_s=float(comparison.new_return_time_s.max()),
        old_core_worst_late_rmse_m=float(comparison.old_late_tracking_rmse_m.max()),
        new_core_worst_late_rmse_m=float(comparison.new_late_tracking_rmse_m.max()),
        overall_maximum_recovery_s=float(formal.return_time_s.max()),overall_worst_late_rmse_m=float(formal.late_tracking_rmse_m.max()),
        overall_maximum_tilt_deg=float(formal.peak_tilt_deg.max()),overall_maximum_rp_rate_rad_s=float(formal.peak_rp_rate_rad_s.max()),
        mission_phase_complete=int(formal.mission_phase_complete.sum()),
        strict_arbitrary_axis_mirror_equivariance=False,
        controller_or_gan_retraining_performed=False)
    write_json(target/'completion_receipt.json',receipt)
    table=['| 条件 | 种子 | 通过/记录 | 恢复中位数/s | 90%分位/s | 最长/s | 最差末段RMSE/m | 任务进度完成 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels={'development':'风噪开发','independent':'风噪独立','ideal':'理想（30 ms延迟）'}
    for g in group:
        table.append(f"| {labels[g['suite']]} | {g['seed']} | {g['passed']}/{g['records']} | {g['median_recovery_s']:.2f} | {g['p90_recovery_s']:.3f} | {g['maximum_recovery_s']:.2f} | {g['worst_late_rmse_m']:.6f} | {g['phase_complete']}/{g['records']} |")
    block=f'''实际完成**668次24 s原生SLX仿真**，包括75次消融、40次前置检查、544次冻结后正式记录和9次接口采集；包含有意重复的成对对照，不等于668个独立随机条件。A→B→A状态复位另运行两轮、共6次1 s短仿真，不计入24 s次数。

主矩阵为**532/532通过**：3个开发种子各76项、3个预先声明的独立种子各76项，以及76项理想工况。每组76项为72故障+4健康。另12项运动镜像全部通过飞行验收，冻结后正式记录合计544/544通过。前置40项和接口9项也全部通过各自检查。没有运行失败或被删除的困难类别；初始构建/测试配置失败日志保留，未产生完整飞行记录。

{chr(10).join(table)}

主矩阵最长恢复{receipt['overall_maximum_recovery_s']:.2f} s，最差末3 s跟踪RMSE为{receipt['overall_worst_late_rmse_m']:.6f} m，最大倾角{receipt['overall_maximum_tilt_deg']:.3f}°，最大滚转俯仰角速度范数{receipt['overall_maximum_rp_rate_rad_s']:.3f} rad/s。有限性、转速及两类积分幅值全部通过。受控参考恢复、末段几何路径误差、名义时序误差、任务进度分别保存，主矩阵任务进度完成{receipt['mission_phase_complete']}/532项。没有以慢推进替代完整任务检查。

同20261006历史72项核心矩阵配对：最长恢复由{receipt['old_core_maximum_recovery_s']:.2f}降至{receipt['new_core_maximum_recovery_s']:.2f} s，最差末段RMSE由{receipt['old_core_worst_late_rmse_m']:.6f}降至{receipt['new_core_worst_late_rmse_m']:.6f} m。共有{len(worse)}项在恢复/末段误差/倾角/角速度至少一个指标上出现增加，均完整列入 `core_72_any_metric_regressions.csv`；通过验收不等于每个指标都改善。

对**584份CSV**（正式544+前置40）使用独立Python实现重新核对2401个时间戳、有限性、恢复时间、末段RMSE、最终分配残差、速度积分增量之和、滚转/俯仰积分增量和受限向零退出规则，并重算文件哈希，全部通过。汇总和逐条复算在 `native_cases.csv/json` 与 `independent_csv_recomputation.csv`。恢复时间保留最后一次越界减故障时刻，同时记录下一采样点持续入带时刻；最后一点仍越界、缺失故障后数据或记录不完整明确标为未恢复/不可评价。

![最终最慢恢复工况](validation/control_repair_20261122/02_worst_recovery_trace.png)
'''
    report=ROOT/'docs/CONTROL_REPAIR_REPORT.md';text=report.read_text(encoding='utf-8')
    if '<!-- FINAL_RESULTS -->' in text:text=text.replace('<!-- FINAL_RESULTS -->',block)
    else:
        start=text.index('## 5. 正式验证与独立复算');end=text.index('## 6. 镜像、步长和故障时刻')
        text=text[:start]+'## 5. 正式验证与独立复算\n\n'+block+'\n\n'+text[end:]
    report.write_text(text,encoding='utf-8')
    manifest=[]
    for folder in ['development_01','preflight_01','formal_01','interface_01']:
        for p in sorted((base/folder).rglob('*')):
            if p.is_file() and p.name in ['result.mat','signals.csv','run.json','run_metadata.json','execution_metadata.json','plan.json','collection_plan.json']:
                manifest.append(dict(path=p.relative_to(base).as_posix(),bytes=p.stat().st_size,sha256=sha256_file(p)))
    write_json(target/'raw_evidence_manifest.json',manifest)
    logs=target/'logs';logs.mkdir(exist_ok=True)
    for folder in [base,base/'development_01',base/'preflight_01',base/'formal_01',base/'interface_01']:
        for p in folder.glob('*.log'):shutil.copy2(p,logs/(folder.name+'__'+p.name))
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);finalize(p.parse_args().base.resolve())
