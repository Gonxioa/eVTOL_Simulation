# 角速度积分整定与轻微推力损失验证

本轮从本地提交 `f44cdbc` 开始，分支 `codex/rate-integral-tuning`。
开始日期2026-10-10，后续验证继续执行于2026-10-11；UTC执行时刻保存于每阶段机器报告。
既有串级结构、名义分配器、物理参数、随机过程和诊断观测白名单保持原设计。
故障效能只送入仿真动力学，控制器继续只读取反馈状态和任务参考。

## 固定基线与新增日志

`baseline.json`锁定修改前HEAD、43项基线测试结果及384个旧数据、预测和用户图片的SHA256。
`baseline_source/`保留修改前控制器、导出器和配置。新增日志的四项检查及原测试共47项通过，
其中同一含噪、风、故障输入下，新日志代码与旧控制器的命令、积分和完整物理轨迹逐位相同。
抗饱和逻辑未改写；测试确认名义饱和时阻止向外积累、反向误差允许积分退回。

## 固定的研究验证标准

沿用 `continuous.json` 的正常标准：位置RMSE≤0.25m、最终误差≤0.10m、倾角≤35°、
角速度模长≤3rad/s、名义电机触及边界的采样比例≤5%，并要求达到请求时长。
轻微故障也使用同一组标准；0.8是稳态推力效能，实际转速仍按一阶动力学响应。
作为本轮研究设计，正式启用候选前要求位置RMSE/最终误差相对原阈值至少留10%余量。
长期保持尾部5秒另要求位置RMSE≤0.10m、位置误差峰值≤0.15m、平均速度模长≤0.05m/s。
这些是预先固定的研究门槛，没有实机或法规认证含义。

- 单因素试验：9个候选，各跑g001/M0的理想及含噪风30ms延迟工况，共18项。
- 任务筛选：旧参数、横滚/俯仰积分增益20与30，四个任务配置×两个电机×两种条件，共48项。
- 完整验证：8组×6个电机×4种条件，192项0.8效能故障；另32项健康检查。
- 长时间保持：48项30秒故障悬停及16项20秒移动任务终点保持，共64项。

四种条件为理想反馈、风下理想反馈、10ms延迟含噪风反馈、30ms延迟含噪风反馈。
原任务、初始条件、五路随机种子和故障日程保留，用相同配对输入比较参数。
增益、限幅先分开试验，随后组合；偏航积分增益和上限单独比较，不按横滚/俯仰照搬。
预声明计划写在首条候选运行前；完整文件及签名见 `predeclared_plan.json`。

## 运行

项目根目录执行：

```powershell
python -m hex_research.control_tuning pilot --workers 4
python -m hex_research.control_tuning screen --candidates baseline combined_rp20 combined_rp30 --workers 4
python -m hex_research.control_tuning validation --candidates selected_rate_v2 --workers 4
python -m hex_research.control_tuning long --candidates selected_rate_v2 --workers 4
```

上述是保持原外环的内环候选对照。补充饱和恢复试验发现16秒时原竖直速度积分尚有0.135m/s²，
未达到提前设定的0.10要求；按 [velocity_recovery_plan.json](velocity_recovery_plan.json) 仅比较Ki_z=0.9/1.2。
两个值通过恢复检查，选择测试中较小的0.9，再对该完整配置重跑224项和64项验证。
最终启用参数为 `rate_i=[30,30,0.6]`、角速度积分上限 `[20,20,2]`、`velocity_i=[0.4,0.4,0.9]`。
位置、速度P/D、姿态P、角速度P/D、速度积分上限及偏航参数保持原值；最终48项单元/回归检查以实际日志为准。

最终候选复验与新版数据/诊断命令：

```powershell
python -m hex_research.control_tuning validation --candidates velocity_z09 --workers 8
python -m hex_research.control_tuning long --candidates velocity_z09 --workers 8
python docs/hex_rate_tuning/run_release.py dataset --workers 8
python docs/hex_rate_tuning/run_release.py diagnose
python view_hex_tasks.py --data-dir output_hex_rate_tuning_20261010/release_v2/dataset_v5 --predictions-dir output_hex_rate_tuning_20261010/release_v2/diagnostics/imu_only_epoch30/predictions --port 8766
```

新版输出 `output_hex_rate_tuning_20261010/release_v2`；旧152条记录及旧模型输出保持原样。
造数入口在实际32项健康、192项故障、64项长期检查及位置余量通过后才执行；
标准验证入口也增加签名检查，配置或源码改变时拒绝覆盖旧数据目录中的验证报告。

原始逐项控制CSV及失败JSON保存在 `output_hex_rate_tuning_20261010/`，不覆盖旧数据。
`configs/selected_rate_v2.json`是经过任务筛选后固定的候选；实际启用以最终验证报告为准。
并行进程独立运行每条轨迹，随机流来自原组种子，控制时钟按仿真时间推进。

参数及方程依据见 [PARAMETER_AND_LOG_LEDGER.md](PARAMETER_AND_LOG_LEDGER.md)。
最后的实际修改、验证、未通过项和新版数据入口见 [VALIDATION_REPORT.md](VALIDATION_REPORT.md)。
