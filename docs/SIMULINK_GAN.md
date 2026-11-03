# 原生 Simulink 数据与现有 WGAN-GP 原型

本入口接入现有 `data_process.py → WGAN.py → classify.py`，保留条件 LSTM 生成器、卷积判别器和 LSTM＋注意力分类器。物理正则默认 0；本轮没有修改动力学、PID、旋翼参数、故障强度或飞行验收阈值。

这是仿真域研究原型。代码实现、流程运行、充分训练、GAN 增益和实飞有效性是五个不同结论。本轮只验证前两项。

## 三条入口

|入口|数据/模型|状态|
|---|---|---|
|本文件、`run_simulink_gan.py`|原生 `hex_evtol.slx`，`simulink_observable_v1`|独立于旧 Python 门禁推进|
|`RUNNING_ORDER.md` 的旧 Python 部分|RotorPy、`output_hex_v3`、旧 `combined`|原有造数门禁保留；不混入新数据|
|`uavfd_f550/`|UAV-FD 实飞、独立 F550 模型|calibration/development/locked_test 约定不变，本轮未读取最终测试集|

## 模型、标签与观测链

原生模型为 `simulink/hex_evtol.slx`，控制版本 `control_v3_20261023`。新数据版本为 `simulink_control_v3_observable_20261103_v1`。带容错控制、风、测量噪声及 30 ms 反馈延迟。

故障作用为 `omega_dot=(sqrt(eta)*command-omega)/tau`，实际推力仍为 `kT*omega^2`。因此 eta 是同一指令下的**稳态剩余推力效能**，不是瞬时转速比例或任意桨叶缺损比例。正常 1.0、Partial 0.8、Severe 0.2、Full 严格 0.0；失效后转速按原电机动态衰减。

|标签|故障|新接口编号|
|---|---|---|
|0|Normal|Python -1 / Simulink 0|
|1–6|Full，0%|Python 0–5 / Simulink 1–6|
|7–12|Severe，20%|Python 0–5 / Simulink 1–6|
|13–18|Partial，80%|Python 0–5 / Simulink 1–6|

旧 `scenario_config.py` 中 Python 电机 4、5 的 Full=5% 保持原状，只适用于旧 profile。新编号与 UAV-FD 实机电机位置的对应关系尚未建立。

旧 `combined` 仍是状态真值 6 维＋带噪 IMU 6 维＋指令 6 维，没有改变含义。新 `imu_cmd_rpm` 虽然也是 18 维，但字段与观测链不同，不能复用旧 scaler 或模型断点。

|特征方案|维数|真实 CSV 字段与处理|
|---|---:|---|
|`imu`|6|`accelerometer_sensor_m_s2_1..3`、`gyro_sensor_rad_s_1..3`，按观测延迟因果移位，启动时保持首样本|
|`imu_cmd`|12|上述 IMU＋当前 `rotor_command_rad_s_1..6`|
|`imu_rpm`|12|上述 IMU＋`feedback_state_14..19`|
|`imu_cmd_rpm`|18|IMU＋当前指令＋带噪、延迟后的转速反馈；本轮主方案|

加速度是传感器系**比力**（m/s²），静止悬停包含约 g；陀螺仪为传感器系 rad/s。指令和转速均为 rad/s，没有误当作 rpm。地面系 ENU、机体系 FLU、四元数 xyzw 表示机体到地面；状态真值、姿态四元数、真实效能和所有身份/故障元数据均不进入这些输入。`rotor_actual_rad_s` 仅用于审计带噪转速链，不作为分类特征。指令是当前控制时刻的值，反馈已经包含传输延迟，二者同步相关不代表即时电机响应。

## 采集与来源验证

在项目根目录使用当前可用的 Python 3.11 环境（numpy、pandas、scipy、scikit-learn、matplotlib、PyTorch；本机 PyTorch 2.5.1+cu121），MATLAB R2025b：

```powershell
python run_simulink_gan.py plan
python run_simulink_gan.py collect
```

采集预定绕圈任务、四个扰动种子 20261103–20261106，每组含 19 类，共 76 次独立仿真，每次 24 秒、3 秒注入故障。位置、路径、增益、噪声强度、风强度与原模型一致。只改变注入类别和独立扰动种子。

采集器关闭 Fast Restart：现有 MATLAB Function 的结构体含不可调参数，Fast Restart 曾使新故障请求被忽略。首次尝试完整留在 `failed_fast_restart_attempt/`，清单检查拒绝该批次，未用于训练。采集器现逐次核对实际效能和风；Python 继续核对 CSV/MAT 哈希、完整时间轴、MAT 参数、注入效能、风和带噪/延迟转速链。

原生模型和关键 MATLAB 源码哈希保存在签名采集计划与每条 `run_metadata.json` 中；完整参数和外生输入在 `result.mat`；参数、故障起止、采样、帧约定、噪声/反馈条件、运行状态和飞行验收在 JSON 中。没有用文件名推断标签，也没有给旧 106 个控制工况补造运行来源。

数据完整性与控制验收分别报告：完整有限、注入与来源可信的记录进入分类原型；恢复验收失败记录照常保留并标记，不因分类或飞行结果困难而删除。不能把数据清单的 `complete` 解读为全部飞行验收通过。

## 预处理、分组与小样本预算

```powershell
python run_simulink_gan.py preprocess
# 基础消融使用同一批原生记录：
python run_simulink_gan.py preprocess --feature-set imu
```

默认窗口 100 点（1 秒），步长 100 点；所有类别统一截取 t>4 秒，即故障后 1 秒缓冲。每条 20 个窗口，总计 1,520 个窗口。它们仍然只有四组独立配对扰动条件，不能称为 1,520 次独立飞行。

四折将每组的全部 19 类一起留出。每折三个外层训练组、一个测试组；小样本预算仅从外层训练组中固定随机选两个组，再取每类 24 个窗口，共 456 个原始训练窗口。剩余外层训练组不参与拟合或测试。本轮超参数预先固定，不根据外层测试挑选模型；需要调参时须增加训练组并在外层训练组内设验证组。

预处理仅输出原始量纲。分类器 StandardScaler 与 GAN MinMaxScaler(-1,1) 分别仅拟合预算内训练数据。生成结果先逆变换回物理量纲，再进入分类器尺度。新 profile 不提供也不要求伪造 `predicted_plant_thrust` 或物理目标；尝试开启未验证的物理正则会明确报错。

## 增广、对照与运行

配置：`configs/simulink_gan_prototype.json`。四个对照共享来源、预算、测试集、分类器结构与训练设置：直接训练、类别加权、随机过采样、条件 WGAN-GP。分类器使用相同初始化种子。类别已经均衡时，加权本身应成为无操作对照；不能把随机差异当作权重收益。

```powershell
python run_simulink_gan.py smoke
python run_simulink_gan.py prototype
python run_simulink_gan.py replay
# 或按顺序完成预处理、冒烟、原型和回放：
python run_simulink_gan.py all
```

环境变量接口仍支持旧 `GAN_AUGMENT_MODE=balance` 补齐模式。新增 `ratio` 模式：`GAN_AUGMENT_RATIO=0/0.5/1/2` 表示相对每类**实际预算内原始训练窗口数**的新增数量，向下取整。当前每类 24 个窗口，比例 1 新增 24 个；等量随机过采样也新增 24 个。

改变比例时应复制配置、修改 `augmentation_ratio` 并使用 `--config 新文件.json`。改变数据、代码、特征、预算、随机种子或模型配置后，签名不同的断点不能复用；用新的 `experiment_revision` 或 `results_root`。原目录、权重和日志保留。模型训练可在相同代码与配置下重入上述命令续跑已完成的折/模型；当前不提供 epoch 内优化器恢复。

每折 `augmentation_counts.json` 分别记录原始、计划、实际生成、加入训练和仅质量评价的数量。质量评价另生成每类 8 个窗口，不计入增广。`gan_losses.npz` 记录生成器更新次数及参数变化；最后不足 N_CRITIC 的小批次也执行生成器更新，零更新会报错。

质量检查保留 Wasserstein、KS、频谱差异、TRTS（原始仿真训练/合成测试）和 TSTR（合成训练/原始仿真测试），并记录有限值、训练范围越界、精确重复、以原始跨来源相似度为参照的最近邻/近重复比例、指令与转速关系。范围只指训练观测范围；近似不单独证明记忆训练集。质量结果不用于挑选外层测试表现最好的模型。

原型为一次训练重复，每折 GAN/分类器各 10 epoch，质量分类器 3 epoch。冒烟各 1 epoch。两者均不能作为充分训练或统计显著增益的证据。

## 因果回放

回放仅使用每折完全留出的 19 条完整记录。每次读取当前及过去 100 点，步长 10 点，预先固定置信度 0.5、连续三次相同故障类别才报警。跨注入时刻窗口按窗口末时刻标注，并单独记录跨界标记；标签仅用于审计。报告故障前误报、漏检、检测延迟及首次报警电机定位，延迟包含反馈传输及窗口影响。无可信突发时刻的记录被拒绝用于此入口。

现有转速效能估计器继续负责飞控容错；网络在本轮只做离线故障诊断，不接入控制闭环。两者可后续在同一留出来源上对照，不能将规则式估计器的闭环恢复结果算作神经网络成果。

## 位置与验证

代码职责：`simulink_data.py` 负责原生字段/来源与分组数据接口；`experiment_protocol.py` 负责训练预算、比例数量和质量检查；`data_process.py`、`classify.py` 接入新 profile；`WGAN.py` 修复小批次零更新；`replay_diagnostics.py` 负责独立因果评估；`run_simulink_gan.py` 与配置负责可复现入口；`audit_simulink_gan.py` 核对实际拟合和输出产物。它们复用同一套网络实现。

根目录：`output_simulink_gan/simulink_control_v3_observable_20261103_v1/`。

- `raw/`：签名计划、76 条记录和完成清单。
- `processed/v1_imu_cmd_rpm/`、`processed/v1_imu/`：窗口、标签、来源组和字段/预处理清单。
- `experiments/v1_smoke_*`、`experiments/v1_prototype_*`：逐折 scaler、分类器、G/D、生成样本、独立质量样本、数量清单、折索引、预测、训练日志、配置与源码快照。
- `folds/fold_*/causal_replay/no_gan/`：逐时刻预测与报警、回放报告。

```powershell
python -m unittest discover -s tests -v
python audit_simulink_gan.py
```

实际结果与剩余限制见同目录的 `SIMULINK_GAN_RESULTS.md` 和原型输出；不据仿真交叉验证准确率宣称实飞泛化。UAV-FD 桨叶缺损标签不能直接映射成这里的推力效能标签。
