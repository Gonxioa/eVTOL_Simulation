# Phys25 实时故障诊断开发验证

`audit`、`smoke` 和 `pilot` 已完成。种子 42 下，六种分段与早期权重设置分别用于
TCN-GRU（日志中的 `cnn`）和 LSTM，共 12 个候选。每个候选运行 12 轮，平均验证排名最优的
共同设置为 **100 点分段、故障后前 1 秒加权 3.0**。
[开发选型](development_selection.json) · [运行日志](logs/pilot.log)

## 代码与训练配置

| 文件 | 本轮改动 |
|---|---|
| [simulink_sequence_data.py](../../../simulink_sequence_data.py) | 从 12 维传感测量派生 25 维输入，新增相对转速、对侧电机差值和转向平衡项。用健康训练记录拟合 `healthy_asinh`。 |
| [simulink_sequence_model.py](../../../simulink_sequence_model.py) | 可选 0.3 秒与 2 秒因果 EMA，组合当前值、两条平滑序列和慢基线偏差。LSTM 与 TCN-GRU 保留跨包时序状态。 |
| [simulink_realtime_training.py](../../../simulink_realtime_training.py) | 保留分段训练，新增健康和轻故障记录权重、二分类检测损失、可保存的学习率调度状态及分严重度日志字段。 |
| [simulink_realtime_evaluation.py](../../../simulink_realtime_evaluation.py) | 分严重度统计在线指标，选型优先控制健康及故障前误报，再关注各严重度能否及时检出以及最弱严重度的定位表现。 |
| [simulink_realtime.json](../../../configs/simulink_realtime.json) | 本次 `pilot` 使用固定学习率 0.002、dropout 0.15、12 轮。设置健康权重 3.0、轻故障权重 1.5、二分类损失权重 0.5。 |
| [diagnose_step1_mild_fault.py](../../../diagnose_step1_mild_fault.py) | 在旧实时基线的开发集上分析配对物理残差和验证概率，并比较 `raw12` 与 `phys25` 的因果窗口特征。 |

数据沿用同一批 228 条原生绕圈记录。114 条训练、57 条验证，另留 57 条独立测试。每条 24 秒、
100 Hz、2,401 点。每轮训练遍历 273,714 点、更新 29 次。验证集有 3 条健康记录和 54 条故障记录。
[数据审计](data_audit.json) · [数据与源码签名](protocol.json)

## 从首轮试跑到当前结果

旧开发诊断的 0.5 秒随机森林探针中，80% 剩余效能类别的联合召回从 `raw12` 的 15.9%
提高到 `phys25` 的 56.9%。首轮 Phys25 `pilot` 中，第一组 CNN 第 10 轮状态宏 F1 为 0.4720，
按低误报工作点选出的 1 秒定位率为 0。较宽松工作点出现了部分定位，同时产生较多误报。
原先将 0.001 的学习率在 10 轮内按余弦退火，第 8 轮结束后已降至约 0.000186。
[旧开发集诊断](step1_diagnostics)

重跑同时调整了学习率、退火开关、dropout 和轮数。相同 CNN 候选到第 12 轮
1 秒定位为 28/54，轻故障 1 秒检出为 14/18，两类误报均为 0。
这个对照反映整组训练设置调整后的表现。单项参数的影响仍待独立对照。
[同配置候选第 12 轮明细](pilot/cnn_segment100_early1_span3/validation_epoch_012.json)

## 最佳 TCN-GRU

`cnn_segment100_early3_span1` 的第 12 轮取得状态宏 F1 **0.8015**。
其报警工作点使用 0.8 检出阈值、0.4 类别置信度阈值，检出与类别确认均持续 0.2 秒。
验证集健康曝光 71.70 秒和故障前曝光 157.14 秒中分别观测到 0 次误报。
[逐工作点验证记录](pilot/cnn_segment100_early3_span1/validation_epoch_012.json)

| 剩余效能 | 1 秒内检出 | 1 秒内正确联合定位 | 3 秒内正确联合定位 |
|---|---:|---:|---:|
| 0% | 18/18 | 18/18 | 18/18 |
| 20% | 18/18 | 12/18 | 12/18 |
| 80% | 18/18 | 6/18 | 7/18 |
| 合计 | 54/54 | 36/54 | 37/54 |

联合定位要求同时判断正确的电机和故障程度。LSTM 六组的保留检查点均为 0/54 的 1 秒联合定位。
共同配置的 LSTM 选择了第 8 轮，第 12 轮虽检出 15/18 条轻故障，但有 3 次健康误报，
超过开发选型的每分钟 1 次上限。
[LSTM 选中验证记录](pilot/lstm_segment100_early3_span1/validation_epoch_008.json) ·
[LSTM 第 12 轮](pilot/lstm_segment100_early3_span1/validation_epoch_012.json)

## 复现与产物

- [运行说明](../../REALTIME_USAGE.md)包含本地数据路径和流式回放入口。更换运行配置或源码后，
  为新实验指定独立结果目录。首次重跑时协议签名提示保留了这一约束。
- 每组的配置、训练曲线、选中结果与全部工作点验证快照见 [pilot](pilot)。
  两个模型的小规模流式一致性检查见 [smoke 收据](smoke_receipt.json)。
- [12 个可移植最佳检查点](../../../model_artifacts/realtime_phys25_20261124)包含模型权重、scaler、
  配置和报警工作点。其清单记录文件校验值与原训练检查点校验值。
- 发布前在同一 worktree 运行 `python -B -m unittest discover -s tests -q`，126 项测试通过。
  12 个可移植检查点均通过加载和校验。两个入选模型在 CPU 上复算 57 条验证飞行的事件指标，
  与本轮记录一致。两种分包方式也在一条完整轻故障记录上产生一致的在线输出。
- 当前进度为开发选型完成。正式阶段计划按 48 轮、42/43/44 三个种子执行，在正式结果冻结之后
  评价独立测试集。现阶段的主要验证问题是低误报下的轻故障联合定位。
