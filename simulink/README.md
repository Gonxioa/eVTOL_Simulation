# 六旋翼 Simulink 模型

2026-10-23 单电机失效恢复实验：新增独立模型 `hex_recovery.slx`，按“稳定姿态/高度 → 返回既定几何轨迹 → 继续任务”的优先级运行。允许减慢或暂停任务进度，使用转速观测估计电机响应并重分配推力；控制块不接收故障真值。三类为单个电机剩余 80%、20%、严格 0% 效能，六个电机逐一测试。打开方式为 `open_hex_recovery`，运行示例为 `run_hex_recovery(Task="circle",FaultMotor=1,FaultFactor=0)`。原 `hex_evtol.slx` 与原健康基准保留。

恢复版的逐环整定、验收指标与限制见 [RECOVERY_TUNING.md](validation/RECOVERY_TUNING.md)，最终结果见 [RECOVERY_SUMMARY.md](validation/RECOVERY_SUMMARY.md)。恢复版参数在 `hex_recovery_parameters.m`，使用 `configure_hex_recovery(P)` 应用场景；不要把恢复版参数传给原 `configure_hex_simulink`。

2026-10-25 常用入口同步：在本目录运行 `my_simulation`，现在使用 `hex_recovery.slx`，仿真 24 秒，第 3 秒注入故障。该可编辑示例保留圆轨迹、1 号电机剩余 20% 效能，带噪声、风和 30 ms 反馈延迟，输出总览、六台电机响应和陀螺仪对比图。结果目录为 `output/recovery/recovery_v3_20261023/my_circle_run/`。下文 `open_hex_simulink`、`run_hex_simulink` 的 8 秒设置属于保留的原模型。

此入口已实际跑满 24 秒并通过恢复验收，CSV/MAT 以及八张图的 PNG/FIG 共 18 个输出文件均已检查；记录见 [MY_SIMULATION_RECOVERY.json](validation/MY_SIMULATION_RECOVERY.json)。

第一阶段迁移：**仿真、串级控制、故障注入、传感器及结果绘图**。基线为最新已推送的 `codex/rate-integral-tuning` 分支，提交 `a6b9f1dd3d839416e469755490b8ff7733acd4ee`。原 Python 文件未修改。

2026-10-21 基准调参：保持位置/速度六维接口、原任务几何及时序、物理参数和执行器限幅，重新调整水平位置/速度及滚转/俯仰控制增益。实际 SLX 的 8 秒正常直线/绕圈/升降位置 RMSE 分别为 **0.49 / 2.36 / 3.09 cm**，悬停接近数值精度。13 个当前增益工况、10 项物理/接口检查和 4 个独立 Python 对照通过。当前证据使用 `validation/BASELINE_*.json`；详见 `validation/BASELINE_TUNING.md`。原模型已就地更新并保留用户布局；今后需把该组已验收增益同步到已打开的旧副本时，可运行 `update_hex_baseline_model`。

2026-10-20 调整：**任务指令仅包含期望位置和速度，共 6 个数**，顺序为 `[x;y;z;vx;vy;vz]`。取消轨迹加速度与偏航角速度前馈，控制器继续使用位置、速度、姿态和机体角速度反馈，并在内部通过 `P.control.yaw_hold_rad` 保持固定航向，默认为 0 rad（世界 x 正方向）。控制器仍会计算所需加速度、推力和期望姿态四元数。姿态反馈、IMU 记录和故障注入继续保留。

已经在 MATLAB 中打开旧模型时，运行 `update_hex_pv_model` 可就地更新；它先备份当前内存模型（包括未保存的时钟封装、Display 和布局），再替换任务/控制代码和相关信号维数。`open_hex_simulink`、`run_hex_simulink` 和 `configure_hex_simulink` 也会检测旧接口并执行该更新。完整重建 `build_hex_model` 仍会重新生成布局，日常同步请使用更新函数。

更新函数通过当前模型中 `task_reference`、`controller_step` 的函数代码识别目标块，并使用 `MATLABFunctionConfiguration` 更新代码，支持修改任务/控制子系统和函数块的显示名称；不会搜索或修改其他已加载模型。

## 打开与运行

本机使用 **MATLAB R2025b 与 Simulink**。有界分配采用原 SciPy TRF 算法的 MATLAB 实现，已封装在模块中；无需 Optimization Toolbox、Aerospace Toolbox 或 Python 运行环境。算法来源和许可见 `THIRD_PARTY_NOTICES.md`，许可文本也已保存在模型描述中。

在 MATLAB 中运行：

```matlab
cd('C:\Users\lenovo\Desktop\eVTOL_Simulation\simulink')
open_hex_simulink
```

打开后直接按 **Run**，默认运行 8 秒正常悬停。在 `Flight Recorder` 子系统中查看位置、电机转速和 IMU Scope。双击其他子系统，再打开其中的 `Embedded MATLAB`，即可查看封装在模型中的计算代码。

使用下面的入口会同时运行模型、保存 CSV/MAT、绘制九宫格结果：

```matlab
% 正常绕圈
r = run_hex_simulink(Task="circle", OutputName="normal_circle");

% 3 秒时，1 号电机剩余 80% 推力效能
r = run_hex_simulink(Task="hover", FaultMotor=1, FaultFactor=.8, ...
    FaultTime=3, OutputName="partial_rotor_1");

% 2 秒故障，4 秒恢复，带噪声、风和延迟观测反馈
r = run_hex_simulink(Task="line", FaultMotor=2, FaultFactor=.8, ...
    FaultTime=2, RecoveryTime=4, Noise=true, Wind=true, ...
    ObservedFeedback=true, Delay=.03, OutputName="observed_recovery");
```

`Task` 支持 `hover`、`line`、`circle`、`vertical`、`tracking`、`step`、`saturation`。`FaultMotor=0` 表示健康，`1..6` 对应原 Python 编号 `0..5`。故障效能 `0/.2/.8` 分别对应原项目 Full/Severe/Partial；Full 的所有电机均为零，没有历史 5% 特例。严重故障可触发物理包线早停，保留实际轨迹和终止时间。

## 框图与封装

| 子系统 | 功能 |
|---|---|
| Task Reference | 输出期望位置与速度；悬停、五次平滑直线/绕圈/升降，以及跟踪/阶跃/饱和任务 |
| Flight Controller | 位置—速度—四元数姿态—角速度串级、内部固定航向、导数滤波、积分抗饱和及健康标称有界分配 |
| Fault Injection | 精确故障与恢复时刻，输出给动力学的分段效能 |
| Six Rotor Plant | 19 状态 Newton–Euler、旋翼与机体阻力、电机一阶响应、RK4 积分 |
| Sensors and Feedback | 比力与角速度、安装角与杆臂、噪声和偏置、延迟观测/理想反馈选择 |
| Wind and Body Loads | OU 风输入、附加机体系力与力矩 |
| Envelope Monitor | 位置、速度、角速度与高度边界监控及早停 |
| Flight Recorder | 时序记录与 Scope |

**数值代码已嵌入 `.slx` 的 MATLAB Function 模块**，模型工作区保存总参数快照 `P` 和各模块的专用参数。该 `.slx` 可以单独复制后打开运行默认场景；不会在仿真时调用 Python 或外部 `.m` 数值函数。`blocks/` 和 `functions/` 是可维护源文件。任务/控制源码同步使用 `update_hex_pv_model`，本次已验收增益同步使用 `update_hex_baseline_model`；完整生成器 `build_hex_model` 会重建布局。直接在模型中改代码则保存 `.slx`。

控制器的输入只有观测和任务指令，故障标签与真实效能不进入控制器。效能只通过 `sqrt(eta).*command` 影响电机目标转速；实际推力由机械转速平方计算，避免重复施加故障。

## 参数与时序

世界系 ENU、机体系 FLU、四元数 Hamilton `[x y z w]`（机体到世界），全部为 SI。状态顺序为位置 3、速度 3、四元数 4、机体角速度 3、机械转速 6。

默认控制、传感器和导出均 100 Hz；Simulink 固定离散周期 0.01 秒。动力学内部 RK4 最大步长 0.002 秒（500 Hz）。故障时刻可以落在采样点之间；动力学会按真实边界拆分积分，不会提前或推迟到下一个控制周期。积分器阶段不更新控制器、噪声或风。

`configs/research.json` 是基线参数快照；`hex_parameters` 派生质量、惯量、六电机位置和分配矩阵。各模块仅接收实际需要的参数，控制器参数不包含真实故障或随机输入表。脚本批量运行时：

```matlab
P = hex_parameters();
P.control.rate_i = [30;30;.6];
P.control.yaw_hold_rad = 0;      % 控制器内部固定航向，弧度；任务口不输入偏航角
P.task_family = 1;              % 0悬停 / 1直线 / 2绕圈 / 3升降
P.fault_motor = 3;
P.fault_factor = .8;            % Partial=.8 / Severe=.2 / Full=0
P.fault_start = 3;              % 仿真时钟，秒
P.fault_end = Inf;              % 持续到结束；有限值表示故障解除时间
P.feedback_delay = .03;
P.observed_feedback = true;
P.noise_enabled = true;
P = hex_prepare_inputs(P);
r = run_hex_simulink(Parameters=P, OutputName="custom");
```

若要把这组参数应用到打开的框图后手动按 Run，执行 `configure_hex_simulink(P)`；需要保存默认场景时再保存模型。只编辑模型工作区中的总快照 `P` 不会同步各模块参数。高级用户也可以直接编辑 `P_controller_step`、`P_fault_schedule` 等专用参数。

此版要求传感器和控制器采样率一致；反馈延迟缓冲区容量 1001 点。位置、速度和姿态为直接带噪观测，未实现 EKF。噪声与风的独立随机流在仿真前生成并保存在 `P.exogenous`，不随求解器调用次数变化。MATLAB 与 NumPy 的随机数实现不同，同一 seed 不代表跨语言同一随机序列；跨语言验证注入同一组噪声和风实现。

启用噪声或风时，使用 `run_hex_simulink` 的开关，或修改参数后调用 `hex_prepare_inputs(P)` 再赋给模型工作区；仅在模型工作区切换布尔值不会自动生成随机输入。修改参数后模型需要重新编译，首次运行和批量逐场景检查会花费一些时间。

有界分配保留原问题的轴权重、界限、伪逆可行解分支、零参考正则项，以及 TRF 的反射步、Coleman–Li 缩放、QR/Givens 子问题和停止条件。MATLAB 与 SciPy 的底层 QR/SVD 核不同，饱和欠定问题存在微小的电机分配差异；应检查状态、命令、推力、IMU、积分和终止行为的实际差异。最初尝试的 `lsqlin` 会明显改变强故障响应，已由 TRF 实现替换。

## 输出与验证

`output/<OutputName>/` 中包含：

- `result.mat`：全部信号、参数、仿真结果及基线提交；
- `signals.csv`：带列名的完整时序，电机公共编号 1..6；
- `results.png` 与可编辑的 `results.fig`：轨迹、位置误差、速度、所选首台电机的响应示例、IMU、积分与故障效能；陀螺仪输出同时对照同一传感器坐标系下的真值角速度。
- `motor_<编号>_response.png/.fig`：每台所选电机各有一张独立图，包含实际转速、转速指令，以及“实际减指令”的差值；默认输出 1～6 号电机全部六张。
- `gyroscope_vs_truth.png/.fig`：逐轴比较陀螺仪输出与真值角速度，并显示测量误差及均方根；该页标题、图例和纵轴使用中文，横轴保留 `Time (s)`。

上述绘图标注按 `plot_labels_review.txt` 返稿设置，时间横轴保留 `Time (s)`，速度、角速度等单位使用 `m/s`、`rad/s` 等符号；中文字体使用微软雅黑。

`my_simulation.m` 现在使用 24 秒恢复版，默认 `motor_to_plot=1:6` 一次输出全部电机，可改为 `3` 只看 3 号，或 `[1 3 6]` 查看指定几台。恢复版直接调用 `run_hex_recovery` 后，用 `plot_hex_results(r,'',1:6)` 绘图；原模型支持 `run_hex_simulink(...,PlotMotor=1:6)`。默认共生成八张图（总览、六台电机、陀螺仪对比）。选择电机只影响绘图，MAT/CSV 始终保留六台电机的数据。

扳手向量顺序为 `[总推力 N, Mx Nm, My Nm, Mz Nm]`；IMU 顺序为传感器系比力三轴、角速度三轴。CSV 中的 `observation_state_*` 与 `feedback_state_*` 都按前述 19 状态顺序排列，后者记录控制器实际使用的理想/延迟观测反馈。

当前 CSV 共 128 列，参考指令只有位置 3 列和速度 3 列；期望姿态四元数与内部角速度目标仍记录在控制审计列。2026-10-19 输出的历史 CSV 为 133 列，额外包含旧任务接口的加速度与偏航指令，请按列名读取，不要混用列序号。

当前增益验收在 MATLAB 中运行 `validate_hex_baseline`，独立 Python 对照运行 `python validation/validate_baseline_python.py`。前者使用实际 SLX 并记录误差、边界、姿态归一化、传感器开关和积分步长敏感性；后者把当前明确的增益传入 Git 基线中的原控制器，并使用六维任务适配器。当前报告为 `validation/BASELINE_TUNING_RESULTS.json`、`BASELINE_COMPONENT_RESULTS.json`、`BASELINE_PYTHON_PARITY.json` 和 `BASELINE_DELIVERY.json`。

以下为 **2026-10-20 接口迁移的历史对照流程**，使用冻结的原始增益。独立对照由 Git 提交中的原 Python 源码加显式任务适配器生成，不读取工作目录中较旧或未提交的实现。适配器将轨迹加速度输入置零、偏航角固定为零、偏航角速度置零；原 Python 控制器及动力学源码不改动：

```powershell
python validation/export_python_reference.py
```

随后在 MATLAB 中执行：

```matlab
validate_hex_simulink
```

验收覆盖正常悬停/直线/绕圈/升降、六个电机分别部分失效、严重及完全失效、故障恢复、非网格故障边界、饱和恢复和相同噪声/风下的延迟反馈。检查实际 SLX 的位置、命令、推力、IMU、参考、风、效能、四元数、命令/积分边界、样本数及终止原因。容差为位置 1 mm、命令 0.5 rad/s 且对应推力差小于 5 mN、IMU 0.02（各自 SI 单位）；参考与风误差小于 `1e-10`，效能必须一致。当前数值结果保存于 `output/validation/pv_simulink_validation.json`，交付记录为 `validation/PV_RESULTS.json`；报告还记录与旧版本相比的位置跟踪 RMSE。默认悬停额外检查与旧版本状态一致。

历史迁移记录：初始 0.2 rad/s 的命令筛查在完全失效场景记录到约 0.248 rad/s 的差异，同时位置差仅约 `7.9e-8 m`、IMU 差约 `1.52e-5`，终止时间和样本数一致。因此命令验收改为同时限制转速与实际推力差。`validation/RESULTS.json`、`COMPONENT_RESULTS.json`、`PORTABILITY.json` 与 `DELIVERY.json` 保留为 2026-10-19 的旧接口证据；`PV_` 前缀文件保留为 2026-10-20 六维接口迁移证据。接口简化当时没有调整增益，2026-10-21 的当前增益及验收独立记录在 `BASELINE_` 文件中。

LSTM、WGAN-GP、连续诊断报警和原浏览器动画不属于本次已选的第一阶段范围；现有 Python 功能仍保留。
