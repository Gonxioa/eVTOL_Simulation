"""Publish compact evidence from actual local prototype artifacts; no training."""
import json
import pickle
import shutil
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from provenance import write_json, sha256_file
from simulink_data import ROOT, read, VERSION


def main():
    base=ROOT/'output_simulink_gan'/VERSION
    evidence=ROOT/'docs/validation/simulink_gan_20261103'; evidence.mkdir(parents=True,exist_ok=True)
    config=read(ROOT/'configs/simulink_gan_prototype.json')
    test_log=(base.parent/'tests_final.log').read_text(encoding='utf-8')
    assert 'Ran 35 tests' in test_log and test_log.rstrip().endswith('OK')
    resume_log=(base/'experiments/v1_prototype_imu_cmd_rpm_ratio1_seed42/training.log').read_text(encoding='utf-8')
    assert resume_log.count('续跑')>=20
    lines=['# Simulink → 现有 WGAN-GP／故障分类：实际原型结果','',
           '日期：2026-11-03。起始代码 `65583a23fcea6a6887d1e77e591c49c827ef4d03`，功能分支 `codex/simulink-gan-profile`。',
           '', '代码与流程均已实际运行；训练尚不充分，未证明 GAN 有显著增益，未验证实飞有效性。', '',
           '## 数据和预算','',
           f'原生模型 `hex_evtol.slx`，控制版本 `control_v3_20261023`，数据版本 `{VERSION}`。四个配对扰动组 × 19 类 = 76 次 24 秒仿真。',
           '本次为预定绕圈任务、四个独立扰动种子、单电机剩余效能 80%/20%/严格 0%，启用容错控制、风、观测噪声和 30 ms 延迟。',
           '每条 20 个不重叠的 100 点窗口，全部位于 t>4 s，总计 1,520 个窗口。四折完整配对留出；每折两组预算内训练来源、每类 24 个窗口，共 456 个原始窗口。比例 1.0 新增 456 个 GAN 窗口；ROS 同量；另生成 152 个质量评估窗口，未加入分类训练。',
           '分类器与 GAN 各 10 epoch，质量分类器 3 epoch；隐藏维数均为 64、两层，架构类型沿用现有实现。训练随机种子 42，仅一次训练重复。所有对照分类器使用相同初始化种子、相同 epoch 和批量设置；扩充数据增加每 epoch 的优化步数，ROS 与 GAN 的训练样本数和步数一致。', '',
           '## 实际四折结果','',
           '|输入|对照|准确率均值 ± 折间标准差|宏 F1|正常类误报率（逐窗口 argmax）|',
           '|---|---|---:|---:|---:|']
    names={'no_gan':'原始数据','class_weighted':'类别加权','random_oversampling':'等量随机过采样','wgan_plain':'条件 WGAN-GP'}
    summaries={}; qualities={}; audits={}
    for feature in ('imu_cmd_rpm','imu'):
        exp=base/'experiments'/f'v1_prototype_{feature}_ratio1_seed42'
        summary=read(exp/'summary.json'); summaries[feature]=summary
        audits[feature]=read(exp/'behavioral_audit.json')
        assert audits[feature]['passed']
        for key,label in names.items():
            row=summary[key]
            lines.append(f'|{feature}|{label}|{row["acc_mean"]*100:.2f}% ± {row["acc_std"]*100:.2f}%|{row["f1_macro_mean"]:.4f}|{row["normal_false_alarm_mean"]*100:.2f}%|')
        for filename in ('summary.json','behavioral_audit.json','run_manifest.json','resume_config.json'):
            shutil.copy2(exp/filename,evidence/f'{feature}_{filename}')
        q=[]
        for fold in range(1,5):
            with (exp/f'folds/fold_{fold}/wgan_plain/synthetic_meta.pkl').open('rb') as f:
                quality=pickle.load(f)['quality']
            q.append(quality)
        qualities[feature]=q
        write_json(evidence/f'{feature}_gan_quality.json',q)
    cases=[c for p in (base/'experiments/v1_prototype_imu_cmd_rpm_ratio1_seed42').glob('folds/*/causal_replay/no_gan/report.json') for c in read(p)['cases']]
    assert len(cases)==76
    delays=[c['detection_delay_s'] for c in cases if c['detection_delay_s'] is not None]
    replay=dict(cases=cases,fault_records=sum(c['label']!=0 for c in cases),missed=sum(c['missed'] for c in cases),
                detected=len(delays),median_delay_detected_only_s=float(np.median(delays)),max_delay_detected_only_s=max(delays),
                pre_fault_alarm_fraction_mean=float(np.mean([c['pre_fault_false_alarm_fraction'] for c in cases])),
                first_alarm_motor_correct=sum(c['first_alarm_motor_correct'] is True for c in cases),
                policy={'probability':.5,'hold':3,'stride':10,'window':100},
                warning='Delay statistics condition on detection; misses are retained. Low false alarm with this threshold does not imply good sensitivity.')
    write_json(evidence/'causal_replay_summary.json',replay)
    for filename in ('collection_plan.json','dataset_manifest.json'):
        shutil.copy2(base/'raw'/filename,evidence/filename)
    shutil.copy2(base.parent/'tests_final.log',evidence/'tests_final.log')
    native=[read(p) for p in (base/'raw').glob('*/run_metadata.json')]
    failures=[dict(run_key=r['run']['run_key'],metrics=r['flight_metrics']) for r in native if not r['flight_metrics']['passed']]
    invalid=read(base/'failed_fast_restart_attempt/dataset_manifest.json')
    evidence_summary=dict(profile=config['profile'],dataset_version=VERSION,native_runs=len(native),paired_groups=4,
                          model_sha256=sha256_file(ROOT/'simulink/hex_evtol.slx'),source_plan_signature=read(base/'raw/collection_plan.json')['plan_signature'],
                          flight_acceptance_failures_retained=failures,failed_attempt_rejected_runs=len(invalid['failures']),
                          tests=35,tests_passed=True,smoke_features=['imu_cmd_rpm','imu'],prototype_features=['imu_cmd_rpm','imu'],
                          primary_checkpoint_reuse_verified=True,results=summaries,
                          versions={'python_runtime':str(Path(__import__('sys').executable)),'torch':torch.__version__,'numpy':np.__version__,
                                    'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
                                    'matlab':'R2025b','gpu':torch.cuda.get_device_name(0)},
                          completion={'code_implemented':True,'pipeline_run':True,'sufficient_training':False,'gan_benefit_established':False,'real_flight_validated':False})
    write_json(evidence/'evidence_summary.json',evidence_summary)
    lines += ['', '主方案 GAN 准确率低于直接训练和等量 ROS；IMU 的小幅变化也只是一次短训练的观察，不构成显著增益证据。类别均衡且初始化一致，所以类别加权与直接训练结果相同。正常类逐窗口误报仍高，当前分类器不能用作可靠在线诊断。', '',
              '## 真实失败与质量检查','',
              '原始控制验收仍使用 15 s 恢复阈值；`g000_label06`（第一组、6 号电机完全失效）恢复时间 15.41 s，未通过原控制验收。该记录完整保留并进入分组评估；没有调阈值、重抽种子或删除困难类别。76 条记录的数据完整性与注入/观测链审计通过，不等于 76 条控制验收全通过。',
              f'首次 Fast Restart 尝试因不可调参数未更新被拒绝。独立检查发现 {len(invalid["failures"])} 条与请求的故障或扰动不符，整个不完整配对批次未用于训练，文件保留在 `failed_fast_restart_attempt/`。最终采集关闭 Fast Restart。',
              '四折主方案各实际进行了 30 次生成器更新，参数最大变化约 0.005。质量样本有限、无精确训练重复，训练范围越界率为 0；这些检查不代表生成质量良好。TRTS 约 4.6%–5.3%、TSTR 约 5.3%–5.5%，接近 19 类随机水平。原始数据的同步指令—转速相关约 0.90，生成数据关系明显较弱；当前 GAN 还没有学好这些联合关系。', '',
              '## 因果回放','',
              f'直接训练基线回放四折留出的全部 76 条完整记录。预设策略为 100 点过去/当前窗口、10 点步长、置信度 0.5、连续三次同类报警。72 条故障中，{len(delays)} 条检出、{replay["missed"]} 条漏检；检出的条件中位延迟 {replay["median_delay_detected_only_s"]:.2f} s，最大 {max(delays):.2f} s；首次报警电机正确 {replay["first_alarm_motor_correct"]}/{len(delays)}。',
              '该报警策略在这批故障前/正常段未出现持续报警，但漏检很多。不能只报告条件检测延迟或零误报而隐藏漏检。逐窗口 argmax 误报与带置信度/持续条件的报警指标不是同一指标。', '',
              '## 验证与文件位置','',
              '35 项 Python 测试通过；包括完整来源/配对留出、预算内训练缩放、观测字段与单位、严格 0% 标签、零比例、均衡比例增广、质量数量分离、单小批次生成器更新、因果前缀不变性、配置变化拒绝复用。',
              '两个特征方案的实际产物审计均通过：每折 scaler 的统计量与预算内原始训练数据逐项一致；测试组完整留出；GAN/ROS 新增数量一致；生成器更新非零；四种对照使用相同测试索引。主方案再次执行原型命令，20 个已完成模型/生成缓存全部复用，汇总不变。',
              '受保护的 SLX、动力学/控制函数、基础与控制参数、旧标签定义及 UAV-FD 模块与起始提交一致。导出逻辑仅提取为共享函数。旧控制验收哈希描述当时版本；本次接口与采集版本以新的签名 collection_plan 和训练代码快照为准。',
              f'完整数据/日志/权重/配置位于 `output_simulink_gan/{VERSION}/`；详细路径与可复现命令见 [运行说明](SIMULINK_GAN.md)。紧凑证据保存在 `docs/validation/simulink_gan_20261103/`。原始 CSV/MAT、权重和大数组未加入 Git。', '',
              '## 剩余问题与最小下一步','',
              '先增加独立训练/验证来源，在训练域内为正常类误报和 GAN 联合关系建立验证检查，再使用预先固定的多训练种子与足够训练预算重复对照。现有外层测试不用于反复选超参数；需要新验证来源或明确嵌套划分。',
              '目前只有绕圈任务、四个配对扰动条件和单一模型/观测链；未验证跨任务、未见参数/控制器、真实桨叶缺损或实飞域泛化。现有基于转速的容错估计器继续独立运行，神经网络没有接入飞控。UAV-FD 最终测试集未解锁。', '']
    (ROOT/'docs/SIMULINK_GAN_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('EVIDENCE_WRITTEN',evidence)


if __name__=='__main__': main()
