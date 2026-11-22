# 正式飞控模型运行与数据接口

本轮模型版本为 `control_recovery_v4_20261122`，研究对象为六旋翼。用于“基于AI可信度框架的动力提升航空器飞控与推进系统失效状态识别适航符合性验证关键技术研究”的仿真研究；验证范围和实际结果见同目录的修复报告。没有重新训练分类器或 GAN。

## 本机运行位置

独立修复分支：`codex/flight-control-recovery-20261122`。

本机检出目录：`C:/Users/lenovo/.codex/worktrees/evt-control-recovery-20261122`。原桌面项目及其中未提交的长序列研究修改保留原状。新模型是独立目录下的 `simulink/hex_evtol.slx`，不要把另一个目录中同名的旧模型混入当前 MATLAB 会话。

在 MATLAB 命令行执行：

```matlab
cd('C:/Users/lenovo/.codex/worktrees/evt-control-recovery-20261122/simulink')
my_simulation
```

也可以打开 `my_simulation.m`，修改任务、电机编号、剩余效能、故障时刻、随机种子后，点击编辑器“运行”。默认演示为绕圈任务、M1剩余20%效能、第3秒故障、运行24秒、风和噪声开启、30 ms反馈延迟。

单次运行及导出：

```matlab
P = hex_parameters();
P.task_family = 0;          % 0悬停，1直线，2绕圈，3升降
P.fault_motor = 1;          % 0健康；1～6故障电机
P.fault_factor = 0;         % 剩余效能：0.8、0.2、0
P.seed = 20261006;
P = hex_prepare_inputs(P);
r = run_hex_simulink(Parameters=P, OutputName="hover_m1_full");
m = hex_metrics(r);
```

单次结果默认保存在当前检出目录的 `simulink/output/simulations/control_recovery_v4_20261122/<OutputName>`。正式批量证据保存在桌面项目的 `output_control_repair_20261122`；仓库只提交汇总、审计、图和清单，大体积逐时刻文件保留在本机。

## 构建与版本检查

`build_hex_model` 在隔离的暂存目录内构建，核对全部7个数值模块的内嵌代码，编译成功后才替换正式SLX。旧SLX以内容哈希保存在 `simulink/build/history`。`hex_model_build.json` 记录源码及依赖哈希、内嵌代码哈希、MATLAB版本、SLX文件哈希。

`hex_ensure_model` 供单次运行、配置、打开模型、批量验收和数据采集调用。来源改变会自动重建；模型启动回调也检查来源与实际内嵌代码。发现正在运行、用户未保存的模型编辑或另一检出目录的同名模型时明确报错，并保留用户编辑。此时使用独立MATLAB会话即可，不需要放弃原模型修改。

独立记录使用普通仿真，运行入口强制关闭Fast Restart。回归测试实际执行A→B→A，并检查状态、反馈、控制审计完全一致，防止积分、滤波器、指令历史及效能估计跨记录串用。

## 原生验证

MATLAB回归：

```matlab
addpath(fullfile(pwd,'validation'))
test_hex_repair('C:/Users/lenovo/Desktop/eVTOL_Simulation/output_control_repair_20261122')
validate_hex_simulink('core')
validate_hex_simulink('ideal')
validate_hex_simulink('healthy')
validate_hex_simulink('seeds')
```

完整计划包含3个开发种子各76次、3个独立种子各76次和76次理想工况，共532次主矩阵仿真；另有12次运动镜像检查。开发前检查另有40次，包含6对悬停镜像、6对积分步长以及16个额外故障时刻。理想工况关闭风和测量噪声，仍保留30 ms反馈延迟和电机动态。

在项目根目录生成新的计划，避免覆盖已有证据：

```powershell
python maintenance/control_repair.py validation-plan --output C:/desired/new_native_run
```

然后在MATLAB中执行：

```matlab
run_hex_repair_plan('C:/desired/new_native_run/plan.json')
```

可增加第二、第三个参数指定首尾工况索引，在独立MATLAB进程中运行不重叠范围。每个进程使用独立缓存。续跑只复用计划、源码、CSV和MAT哈希均匹配的成功记录；失败记录保留，重新尝试需使用新目录。独立种子必须在参数冻结后统一运行；若据其结果调参，应更换版本并将这些种子归为开发条件。

全部完成后可执行 `finalize_hex_delivery('C:/desired/new_native_run/plan.json')`，重新核对所有MAT/CSV哈希和MAT内保存的验收结果；第二个参数设为 `true` 可额外在MATLAB中从MAT逐条重新计算全部指标。交付的 `independent_csv_recomputation.csv` 则是另一套Python实现从CSV重新计算恢复时间、末段误差、积分更新和分配残差的结果。历史v3验收收据不能给当前模型作证。

## 每条证据包括什么

`result.mat` 保存完整运行参数和预生成外生输入；`signals.csv` 保持原字段兼容并附加112个诊断字段；`execution_metadata.json` 描述字段名称、单位、坐标系、测量信号用途、源码与SLX构建身份、Git提交、完成状态、参数和文件哈希。批量运行还保存 `run.json` 和冻结的 `plan.json`。

100 Hz导出对应0.01 s采样周期，完整24 s记录包含起点和终点，共2401行。真实状态、反馈状态、参考值和控制内部量供离线核查；分类器仅接收下面12路测量。

## 后续长序列研究

输入为加速度计三轴（m/s²）、陀螺仪三轴（rad/s）、六路转速测量（rad/s）。IMU原始日志通过既有 `observable_features(...,'imu_rpm')` 进行一次因果30 ms对齐；转速直接读取已延迟的 `feedback_state_14..19`，不再次延迟。启动阶段保留首个观测值。没有命令、真值、效能、故障时刻或诊断量进入输入。

新版独立数据版本为 `simulink_control_v4_observable_20261122_v1`。用 `simulink_data.prepare_plan` 的 `controller_version`、`roles` 显式生成采集清单；一个配对条件组的19条记录属于同一用途。训练/验证/测试的种子分别确定，标准化只拟合训练来源。旧清单未声明版本时仍按旧控制器 `control_v3_20261023` 校验，不能把旧数据改标为新版。

例如在Python项目根目录：

```python
from simulink_data import prepare_plan
prepare_plan('C:/desired/new_dataset', seeds=[810001,810003,810007],
             dataset_version='simulink_control_v4_observable_20261122_v1',
             controller_version='control_recovery_v4_20261122',
             roles=['train','validation','test'])
```

在对应MATLAB检出目录执行 `collect_hex_dataset('C:/desired/new_dataset/collection_plan.json')`，每组19次，共57次。实际项目的规模划分应使用随交付保存的采集计划，不能把此三组示例当作充分训练规模。

已验证的兼容入口为 `maintenance/check_repaired_sequence.py`：通过现有 `load_flights`、`fit_scaler`、`collate` 和 `FlightClassifier`，读取整条2401×12飞行记录，对照整条与连续分块前向输出，保存当前状态、历史前缀和记录级标签。故障前逐时刻标签为健康；故障后为对应故障类。分块只用于计算，必须传递同一记录的模型记忆，每条新记录重新初始化。本轮只验证接口，随机初始化网络输出不代表识别率。
