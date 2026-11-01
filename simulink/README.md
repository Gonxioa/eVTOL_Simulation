# 当前六旋翼 Simulink 恢复模型

2026-11-01 整理后，日常运行模型只有 `hex_recovery.slx`，控制参数版本为 `recovery_v3_20261023`。
历史运行模型、接口迁移和候选版本已保存在 [GitHub 归档](https://github.com/Gonxioa/eVTOL_Simulation/releases/tag/archive-20261030) 与同名 Git 标签中。

## 日常运行

在 MATLAB R2025b 中进入本目录，打开 `my_simulation.m` 点击运行，或执行：

```matlab
my_simulation
```

当前示例：绕圈、24 秒、3 秒注入 1 号电机剩余 20% 效能故障；启用风、测量噪声、观测反馈和 30 ms 延迟。
场景设置集中在 `my_simulation.m`；增益和恢复阈值在 `hex_recovery_parameters.m`，基础参数在 `hex_parameters.m` 和 `configs/research.json`。

仅打开框图：

```matlab
open_hex_recovery
```

已保存框图的默认示例是 1 号电机完全失效，与可编辑脚本中的 20% 示例分开记录。

## 文件职责

|文件或目录|职责|
|---|---|
|`hex_recovery.slx`|唯一日常运行框图，内含数值代码与默认场景|
|`my_simulation.m`|设置场景，仿真、绘图并保存|
|`hex_recovery_parameters.m`|当前恢复控制参数|
|`hex_parameters.m`、`configs/research.json`|物理、执行器、风和传感器基础参数|
|`hex_prepare_inputs.m`|生成可重复的风和噪声序列|
|`configure_hex_recovery.m`|同步模型参数|
|`run_hex_recovery.m`|运行实际模型并保存 MAT/CSV|
|`plot_hex_results.m`|总览、六电机、陀螺仪与真值对比|
|`run_hex_healthy_assessment.m`|四健康任务仿真、指标与绘图|
|`plot_hex_healthy_assessment.m`|中文评估图，x/y/z 分别绘制|
|`diagnose_hex_healthy_tilt.m`|风、噪声和延迟对照|
|`blocks/`、`functions/`|当前控制器、任务、动力学和数学源码|
|`internal/recovery_layout_template.slx`|生成器所需内部布局模板，不能删除；不作为运行入口|
|`validation/`|恢复版验收脚本、最终报告和必要的失败记录|

模块源码已嵌入 SLX，改 `blocks/` 后需要同步或重新生成模型；`configure_hex_recovery` 只更新参数。
`build_hex_recovery_model` 使用内部布局模板重建模型，并先保存已有模型的备份。

## 本地结果

当前结果根目录：`output/recovery/recovery_v3_20261023/`。

- `my_circle_run/`：日常脚本输出，含 `result.mat`、`signals.csv`、总览、六台电机及陀螺仪对比，共八张图，各存 PNG/FIG。
- `healthy_four_tasks_20261027/`：四健康任务的最新评估，含四个任务子目录、24 张 PNG/FIG 图和指标表；图表在 10 月 28 日更新。
- `core_*`、`healthy_task*`、`ideal_*`、`seeds_*`：当前 v3 的 106 个验收工况数据。

运行 `my_simulation` 会覆盖同名结果；要保存新试验，应把脚本中两处 `my_circle_run` 改成同一个新名称。
四健康任务入口默认创建带时间戳的新目录。

MAT 保存完整数据和参数；CSV 用于查看时序；PNG 用于报告；FIG 用于 MATLAB 内继续编辑图。
旧基准、早期恢复候选和重复图备份从本地移除，完整归档附件含 SHA-256 清单，可从 GitHub 下载恢复。
原始数据集、训练权重、当前独立 F550 模块和文档材料单独保留。

## 验证和研究边界

四健康任务：

```matlab
report=run_hex_healthy_assessment;
```

恢复工况验证：

```matlab
addpath validation
validate_hex_recovery('core')
validate_hex_recovery('healthy')
validate_hex_recovery('ideal')
validate_hex_recovery('seeds')
validate_hex_recovery_components
validate_hex_recovery_loops
```

当前 v3 的 106 个实际模型工况验收记录见 `validation/RECOVERY_SUMMARY.md`，控制整定见 `validation/RECOVERY_TUNING.md`。
其中严格逐电机命令的源函数/SLX 数值对照未通过；状态及合力、力矩的行为对照通过，保留相关失败记录。
这不是实飞或多电机同时失效的保证。估计器依赖转速反馈、已知反馈延迟和当前电机目标缩放故障模型。

地面系 ENU，机体系 FLU，四元数 `[x y z w]`。任务口只输入期望位置和速度；姿态和角速度反馈保留。
控制与传感器周期 0.01 秒，动力学 RK4 内部最大步长 0.002 秒，积分按故障真实时刻分段。
不需要 Python、Optimization Toolbox 或 Aerospace Toolbox 来运行已保存框图。
