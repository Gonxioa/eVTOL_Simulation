# UAV-FD：30 个发布文件逐项说明

数据目录：`C:/Users/lenovo/Desktop/eVTOL_Simulation/UAV-FD`。

依据：本地原始 MAT 的变量目录和内嵌列名、11 个原始 MATLAB 脚本、MATLAB 实际读取的预处理表结构，以及 [Zenodo 官方发布说明](https://zenodo.org/records/7648996)。本次只检查结构、类型、列名和文件内容，没有重新计算飞行信号的统计特征、频谱、拟合或评测结果。

官方说明这是六旋翼在桨叶缺损条件下的真实飞行数据，使用定制日志的 ArduPilot。发布内容为 18 个原始飞行 MAT、1 个预处理 MAT、11 个 MATLAB 源代码文件。30 个文件的大小和 MD5 已全部与官方清单核对一致。

`.mat` 是 MATLAB 数据容器，一个文件可同时保存许多矩阵、单元格、表和变量；`.m` 是 MATLAB 源代码，描述如何处理数据。

## 1. 18 个原始飞行文件

下列每个文件都包含整架飞机的多通道日志。故障文件的电机编号指出故障位置，并不表示文件只记录那个电机。健康文件后的 1–6 是记录编号，不是电机编号。

| 文件 | 内容与用途 |
| --- | --- |
| `NO_FAULT1.mat` | 无故障飞行记录 1；提供正常状态基准，可与故障记录对比。 |
| `NO_FAULT2.mat` | 无故障飞行记录 2；补充另一条正常飞行记录。 |
| `NO_FAULT3.mat` | 无故障飞行记录 3；补充正常飞行中的变化。 |
| `NO_FAULT4.mat` | 无故障飞行记录 4；可用于独立于前几条记录的开发验证。 |
| `NO_FAULT5.mat` | 无故障飞行记录 5；项目当前将其留作锁定测试。 |
| `NO_FAULT6.mat` | 无故障飞行记录 6；项目当前将其留作锁定测试。 |
| `FAULT_M1_5.mat` | 1 号电机对应桨叶的故障记录，故障类别标签为 5。 |
| `FAULT_M2_5.mat` | 2 号电机对应桨叶的故障记录，故障类别标签为 5。 |
| `FAULT_M3_5.mat` | 3 号电机对应桨叶的故障记录，故障类别标签为 5。 |
| `FAULT_M4_5.mat` | 4 号电机对应桨叶的故障记录，故障类别标签为 5。 |
| `FAULT_M5_5.mat` | 5 号电机对应桨叶的故障记录，故障类别标签为 5。 |
| `FAULT_M6_5.mat` | 6 号电机对应桨叶的故障记录，故障类别标签为 5。 |
| `FAULT_M1_10.mat` | 1 号电机对应桨叶的故障记录，故障类别标签为 10。 |
| `FAULT_M2_10.mat` | 2 号电机对应桨叶的故障记录，故障类别标签为 10。 |
| `FAULT_M3_10.mat` | 3 号电机对应桨叶的故障记录，故障类别标签为 10。 |
| `FAULT_M4_10.mat` | 4 号电机对应桨叶的故障记录，故障类别标签为 10。 |
| `FAULT_M5_10.mat` | 5 号电机对应桨叶的故障记录，故障类别标签为 10。 |
| `FAULT_M6_10.mat` | 6 号电机对应桨叶的故障记录，故障类别标签为 10。 |

`5` 和 `10` 是文件名及作者 `main.m` 使用的故障标签。现有发布页和代码不足以确认它们对应的缺损几何尺寸、百分比或实际推力损失，不能直接解释为 5%、10% 推力下降。

这些文件不是三台电机同时失效的数据。它们可支持桨叶故障特征研究；三电机故障下的恢复控制仍需要相应的模型和验证。

## 2. 原始 MAT 内部主要记录

18 个文件都有以下核心日志。各条记录的采样行数不同，辅助日志也有差异。

| 变量 | 包含的数据 | 可以用来研究什么 |
| --- | --- | --- |
| `IMU_0`, `IMU_1`, `IMU_2` | 三套 IMU 的三轴角速度、三轴加速度/比力、时间戳，以及温度、状态与内部速率字段 | 机体运动、振动和桨叶故障带来的变化；也可比较三个传感器实例。 |
| `ESC_0`–`ESC_5` | 六个电调的 `RPM`, `RawRPM`, `Volt`, `Curr`, `Temp`, `CTot`, `MotTemp`, `Err` 等 | 电机转速、供电、电流和电调状态。作者预处理将实例 0–5 映射到电机信号 1–6。 |
| `RCOU` | 输出通道 `C1`–`C14`；本机电机 1–6 对应 `C9`–`C14` | 飞控给电机的控制指令；它不是实测推力。DShot 下这是归一命令日志标度，不能简单当作真实 PWM 脉冲宽度。 |
| `RCO2` | 输出通道 `C15`–`C18` | 其他输出通道的日志；不是作者主流程使用的六个电机指令。 |
| `ATT` | `DesRoll/Roll`, `DesPitch/Pitch`, `DesYaw/Yaw` 及误差相关字段 | 期望与估计姿态之间的跟踪关系。 |
| `RATE` | 横滚、俯仰、偏航的目标角速度、实际角速度和控制输出，以及垂向控制相关字段 | 内环跟踪和控制响应。 |
| `PIDR`, `PIDP`, `PIDY`, `PIDA` | `Tar`, `Act`, `Err`, `P`, `I`, `D`, `FF`, `Dmod`, `SRate`, `Limit` | 各控制器目标、反馈、误差和 P/I/D 等输出项。前三者分别对应横滚、俯仰、偏航；`PIDA` 为垂向加速度控制器日志。 |
| `XKF1_0`, `XKF1_1` | 两套 EKF 实例的姿态、`VN/VE/VD` 速度、`PN/PE/PD` 位置、陀螺偏置等 | 飞机的估计运动状态。它们是机载滤波器估计，不是独立外部真值。 |
| `XKF2_*`, `XKF3_*`, `XKF4_*`, `XKF5_0`, `XKFS_*`, `XKQ_*` 等 | EKF 相关状态、创新、状态标志和四元数等 | 检查状态估计的一致性和健康状况。并非全部被作者主特征流程使用。 |
| `VIBE_0`, `VIBE_1`, `VIBE_2` | `VibeX`, `VibeY`, `VibeZ`, `Clip` | 已经过处理的振动指标及裁剪计数。它不是 IMU 加速度原始波形。 |
| `GPS_0`, `GPA_0`, `UBX1_0`, `UBX2_0` | 经纬度、高度、地速、航迹角、卫星/精度及接收机诊断字段 | 定位和 GPS 状态。 |
| `POS`, `AHR2`, `ORGN_0`, `ORGN_1` | 估计位置、高度、姿态和坐标原点等 | 飞行位置与参考原点。不能仅凭这些字段认为已获得完整的任务参考轨迹。 |
| `MAG_0`, `MAG_1`, `MAG_2` | 磁场、偏置和健康相关字段 | 磁罗盘观测和航向估计辅助信息。 |
| `CTUN`, `CTRL`, `MOTB` | 高度/油门控制字段、控制统计，以及电机限制与状态相关字段 | 控制器和电机混控器的运行背景。`MOTB.LiftMax` 等控制内部量不是逐桨推力计测量。 |
| `PARM` | 每条记录 972 项飞控参数，保存为参数名和参数值两列 | 核对 PID、机架、输出功能、动力映射等飞控配置；它不是完整的物理模型参数表。 |
| `MODE`, `ARM`, `EV`, `VER` | 模式、解锁状态、事件和固件版本字段 | 解释飞行状态及日志背景。不能从基础固件版本恢复作者所有定制补丁。 |
| `PM`, `DSF`, `STAK_*`, `IOMC`, `FILE`, `HEAT`, `RAD`, `TERR` 等 | 调度/存储/任务栈/板级状态/温控/通信/地形等辅助日志 | 定位日志与系统运行问题，通常不是首选的桨叶故障特征。 |
| `*_label` | 对应记录矩阵的列名 | 数据字典。例如 `IMU_label` 说明 IMU 各列代表什么；它不是额外传感器测量。 |
| `label_fault` | 六个 `_10.mat` 文件中存在的数值向量；其他 12 条原始记录没有同名变量 | 属于额外标签数据，其精确定义本次未从文档证实；不能当作实测推力损失或故障时刻真值。 |

仅存在 `BARO_label`、`BAT_label` 或 `CMD_label` 等列名说明，并不表示某个文件实际包含 `BARO`、`BAT` 或 `CMD` 记录。本次目录检查以实际非空变量为准。

大部分数值日志前两列是 `LineNo`（日志行号）和 `TimeUS`（启动后的微秒时间），`TimeUS / 1e6` 才是秒。第三列在不同记录中含义不同，例如 IMU 实例、电调实例、EKF 核编号。`PARM` 等特殊表不是这种布局。

以 `NO_FAULT1.mat` 为例：`IMU_0` 为 26,162 行 × 16 列；每一行是一条采样记录。它的内嵌列名为：

```text
LineNo, TimeUS, I, GyrX, GyrY, GyrZ, AccX, AccY, AccZ,
EG, EA, T, GH, AH, GHz, AHz
```

`GyrX/Y/Z` 对应三轴角速度，`AccX/Y/Z` 对应三轴比力。项目已有审计将它们分别按 rad/s 和 m/s²、机体系前右下方向解释。`GHz/AHz` 为内部速率字段，不代表日志实际采样率。各通道采样时刻和频率不一致，不能简单拿相同行号当作同一时刻。

## 3. `data_feature_Table_zoh.mat`：作者已处理的数据与特征

该文件不是额外飞行记录。用 MATLAB 实际读取后，它保存以下 5 个表：

| 变量 | 实际尺寸 | 含义与用途 |
| --- | --- | --- |
| `dataTable` | 18 行 × 37 列 | 每行对应一条原始飞行。36 列信号各自以单元格内的 timetable 保存，另有 1 列 `faultCode`。可直接查看作者同步后的信号。 |
| `data_feature_Table` | 18 行 × 38 列 | 在上述信号与标签之外增加 `FRM_1`，保存每条飞行各窗口的频谱、统计和频谱特征等中间结果。 |
| `feature_Table` | 1,327 行 × 577 列 | 汇总后的窗口级特征。前 4 列是 `EnsembleID_`, `faultCode`, `FRM_1/TimeStart`, `FRM_1/TimeEnd`；其余 573 列为清理后保留的特征。 |
| `feature_Table_Train` | 929 行 × 577 列 | 作者预先划分的窗口级训练数据。 |
| `feature_Table_Test` | 398 行 × 577 列 | 作者预先划分的窗口级测试数据。 |

18 行不等于 18 个采样点：`dataTable` 的每个信号单元格中还有一整段时间序列。1,327 行也不等于 1,327 次飞行，它是 18 条记录切窗之后得到的样本数。

36 个信号如下：

| 列名 | 数量 | 数据 |
| --- | ---: | --- |
| `acc_x`, `acc_y`, `acc_z` | 3 | 三套 IMU 对齐、平均后的三轴比力。 |
| `gyr_x`, `gyr_y`, `gyr_z` | 3 | 三套 IMU 对齐、平均后的三轴角速度。 |
| `pwm_1`–`pwm_6` | 6 | 六个电机的输出命令，来自 `RCOU.C9`–`C14`。变量名为 pwm，但不能据此认定实际输出协议是 PWM。 |
| `esc_1`–`esc_6` | 6 | 六个电调的 RPM 转速。 |
| `cur_1`–`cur_6` | 6 | 六个电调的电流。 |
| `roll`, `pitch`, `yaw` | 3 | 估计姿态角，来源 `ATT`。 |
| `roll_des`, `pitch_des`, `yaw_des` | 3 | 期望姿态角，来源 `ATT`。 |
| `vn`, `ve`, `vd` | 3 | 两套 EKF 对齐、平均后的北/东/地速度。 |
| `vibe_x`, `vibe_y`, `vibe_z` | 3 | 三套 VIBE 对齐、平均后的振动指标。 |

`faultCode` 只有 0、5、10 三类：6 条健康记录都是 0，6 条 `_5` 记录都是 5，6 条 `_10` 记录都是 10。该列本身不区分故障电机位置；若要定位 M1–M6，应额外维护文件到电机编号的映射。

## 4. 11 个 MATLAB 脚本

| 文件 | 输入与输出 | 具体作用 |
| --- | --- | --- |
| `main.m` | 输入 18 个原始 MAT，输出上述 5 个表并保存预处理 MAT | 作者的入口脚本：以 350 Hz 为目标同步数据，构造 36 路信号，添加故障标签，调用完整特征提取，清理缺失值，划分 70%/30% 窗口数据，最后打开 Classification Learner。 |
| `synchronize.m` | 输入文件名、方法名、目标采样率；输出 `Data` | 分发入口：`zero_order_hold` 调用零阶保持路径；`resampling` 调用重采样路径。 |
| `sync_zoh_exa.m` | 输入原始 MAT 和 `Fs`；输出原始选定通道及 `_sync` 字段 | 对齐并平均三套 IMU、两套 XKF1、三套 VIBE，提取六路命令/转速/电流和姿态；按统一时间轴零阶保持，并按作者规则裁剪记录。是 `main.m` 默认使用的路径。 |
| `sync_resample_exa.m` | 输入原始 MAT 和 `Fs`；输出相同类型的 `Data` | 另一条同步路径：通道选择与平均类似，但通过插值、滤波、降采样生成统一采样率的数据。 |
| `resampling.m` | 输入数值、原时间轴、插值时间轴、`Fs`、上采样倍数；输出重采样数值 | `griddedInterpolant` 插值，调用 `lowpass`，再 `downsample`；被重采样路径调用。 |
| `ZOHmatrix.m` | 输入矩阵、原时间轴、目标时间轴；输出对齐矩阵 | 对每个目标时刻调用 `ZOHrow`，取最近已出现的原始记录，使信号保持到下一条记录出现。 |
| `ZOHrow.m` | 输入原时间轴和查询时刻；输出行号 | 用 `sum(times<=t)` 找到截至该时刻的最后一条记录；若时刻早于第一条，则返回首行。其行为依赖时间轴按顺序排列。 |
| `compare2.m` | 输入两张日志矩阵；输出裁剪后的两张矩阵 | 按第 2 列时间戳取交集、删除不共有的行，并检查剩余时间戳序列完全一致；用于两套 XKF1 对齐。它自身不计算平均值。 |
| `compare3.m` | 输入三张日志矩阵；输出裁剪后的三张矩阵 | 对三路记录做同样的时间戳交集检查；用于三套 IMU、三套 VIBE。它自身不计算平均值。 |
| `diagnosticFeatures.m` | 输入带信号/标签的表；输出 `featureTable`, `outputTable` | Diagnostic Feature Designer 导出的特征函数。按 1 秒窗口、1 秒步长处理；36 路信号的统计特征代码启用，但只有 15 路信号的功率谱代码启用，其余部分被注释。 |
| `diagnosticFeaturesfull.m` | 同上 | 作者主流程实际调用的版本；36 路信号的功率谱及统计特征代码均启用，并提取频谱峰值、峰频与频带功率等。名字中的 full 主要体现启用的通道范围，不代表新的一批原始数据。 |

两套特征函数都包含均值、标准差、RMS、峰值、峰度、偏度、波形/峰值/脉冲/裕度因子，以及 SNR、SINAD、THD 等特征代码。功率谱路径使用 10 阶自回归模型，并非简单调用 FFT。频谱特征代码选取 10–150 Hz 内的两个谱峰及频带积分功率；出错时可能填入 NaN，之后 `main.m` 再做缺失值清理。

## 5. 文件如何串起来

```text
18 个原始 MAT
  → main.m
  → synchronize.m
      → sync_zoh_exa.m（默认）
          → compare2.m / compare3.m
          → ZOHmatrix.m → ZOHrow.m
      或 sync_resample_exa.m
          → compare2.m / compare3.m
          → resampling.m
  → dataTable：36 路同步信号 + faultCode
  → diagnosticFeaturesfull.m
  → 窗口统计、功率谱与频谱特征
  → 缺失值清理与窗口级训练/测试划分
  → data_feature_Table_zoh.mat
  → Classification Learner
```

作者保存的预处理 MAT 已包含这些处理结果，所以查看结果不需要重新运行整套入口脚本。重新运行 `main.m` 会重新计算并保存同名文件，还会清理 MATLAB 工作区和打开图形应用；应先复制到工作目录再尝试，以保留官方原件。

## 6. 阅读和复用时的边界

- 原始 IMU、转速和控制输出并非同一采样频率。统一到 350 Hz 只改变数据表示，不会让低频电调遥测获得新的高频信息。
- 同步脚本通过六路命令之和低于 `1450*6` 来裁剪低输出片段，这是作者的启发式规则，不能自动等同于严格识别起降阶段。
- 两个同步脚本中“CUT FIRST 2 SECONDS”下实际执行 `idxcut=2*Fs; X(idxcut,:)=[]`，只删除了一个索引行，并非删除前两秒全部行。本说明区分注释意图与实际代码，没有修改官方脚本。
- 作者在特征行上调用 `cvpartition(...EnsembleID_...,'Holdout',0.3,'Stratify',true)`；这是按飞行编号分层的窗口划分，不保证一条飞行只出现在一侧。不能把该划分直接当作跨独立飞行测试。项目自己的飞行级划分保存在 `uavfd_f550/configs/split.json`。
- 数据中没有本次已确认的逐电机测力计推力通道，也没有已确认的逐时刻真实推力效率系数。桨叶损坏时转速正常并不证明推力正常，不能直接从 RPM 比例推出真实桨叶损坏下的可用推力。
- 本次检查没有解锁项目保留的 11 条最终测试记录，没有重跑作者的特征提取或训练。

## 7. 结构核对记录

本次 MATLAB 实测表结构写入 `output_uavfd_f550/release_table_metadata.json`；18 个原始文件的变量名、矩阵维度和第一条健康记录的完整内嵌列名写入 `output_uavfd_f550/file_guide_inventory.json`。它们是结构记录，没有附加新的信号分析结果。

下面的完整变量目录附录由上述结构记录生成。

### A. 各原始文件的结构数量

顶层变量数包含列名说明；有数据变量数只统计实际非空变量，因此两者不能当作传感器数量。IMU 行数是原始 IMU_0 的尺寸，不是对齐后的共同时间戳数量。

| 文件 | 顶层变量数 | 有数据变量数 | 列名变量数 | IMU_0 尺寸 |
| --- | ---: | ---: | ---: | --- |
| `NO_FAULT1.mat` | 442 | 97 | 345 | 26,162 × 16 |
| `NO_FAULT2.mat` | 445 | 100 | 345 | 57,456 × 16 |
| `NO_FAULT3.mat` | 443 | 98 | 345 | 42,977 × 16 |
| `NO_FAULT4.mat` | 443 | 98 | 345 | 36,855 × 16 |
| `NO_FAULT5.mat` | 442 | 97 | 345 | 34,790 × 16 |
| `NO_FAULT6.mat` | 442 | 97 | 345 | 37,880 × 16 |
| `FAULT_M1_10.mat` | 281 | 99 | 182 | 47,019 × 16 |
| `FAULT_M1_5.mat` | 446 | 101 | 345 | 64,276 × 16 |
| `FAULT_M2_10.mat` | 280 | 98 | 182 | 53,865 × 16 |
| `FAULT_M2_5.mat` | 443 | 98 | 345 | 39,209 × 16 |
| `FAULT_M3_10.mat` | 279 | 97 | 182 | 39,244 × 16 |
| `FAULT_M3_5.mat` | 447 | 102 | 345 | 63,171 × 16 |
| `FAULT_M4_10.mat` | 279 | 97 | 182 | 36,213 × 16 |
| `FAULT_M4_5.mat` | 443 | 98 | 345 | 27,601 × 16 |
| `FAULT_M5_10.mat` | 279 | 97 | 182 | 43,491 × 16 |
| `FAULT_M5_5.mat` | 445 | 100 | 345 | 32,132 × 16 |
| `FAULT_M6_10.mat` | 280 | 98 | 182 | 50,841 × 16 |
| `FAULT_M6_5.mat` | 443 | 98 | 345 | 34,581 × 16 |

### B. 18 条记录共有的非空变量及列名

此处列名来自 `NO_FAULT1.mat` 的内嵌说明，不额外推断字段单位。`_0`、`_1` 等实例共用相应基础列名。

| 共有变量/实例 | 内嵌列名或类型说明 |
| --- | --- |
| `AHR2` | LineNo, TimeUS, Roll, Pitch, Yaw, Alt, Lat, Lng, Q1, Q2, Q3, Q4 |
| `ARM` | LineNo, TimeUS, ArmState, ArmChecks, Forced, Method |
| `ATT` | LineNo, TimeUS, DesRoll, Roll, DesPitch, Pitch, DesYaw, Yaw, ErrRP, ErrYaw, AEKF |
| `CTRL` | LineNo, TimeUS, RMSRollP, RMSRollD, RMSPitchP, RMSPitchD, RMSYaw |
| `CTUN` | LineNo, TimeUS, ThI, ABst, ThO, ThH, DAlt, Alt, BAlt, DSAlt, SAlt, TAlt, DCRt, CRt |
| `DSF` | LineNo, TimeUS, Dp, Blk, Bytes, FMn, FMx, FAv |
| `DU32` | LineNo, TimeUS, Id, Value |
| `ESC_0`, `ESC_1`, `ESC_2`, `ESC_3`, `ESC_4`, `ESC_5` | LineNo, TimeUS, Instance, RPM, RawRPM, Volt, Curr, Temp, CTot, MotTemp, Err |
| `EV` | LineNo, TimeUS, Id |
| `FILE` | LineNo, FileName, Offset, Length, Data |
| `GPA_0` | LineNo, TimeUS, I, VDop, HAcc, VAcc, SAcc, YAcc, VV, SMS, Delta, Und |
| `GPS_0` | LineNo, TimeUS, I, Status, GMS, GWk, NSats, HDop, Lat, Lng, Alt, Spd, GCrs, VZ, Yaw, U |
| `HEAT` | LineNo, TimeUS, Temp, Targ, P, I, Out |
| `IMU_0`, `IMU_1`, `IMU_2` | LineNo, TimeUS, I, GyrX, GyrY, GyrZ, AccX, AccY, AccZ, EG, EA, T, GH, AH, GHz, AHz |
| `IOMC` | LineNo, TimeUS, RSErr, Mem, TS, NPkt, Nerr, Nerr2, NDel |
| `MAG_0`, `MAG_1`, `MAG_2` | LineNo, TimeUS, I, MagX, MagY, MagZ, OfsX, OfsY, OfsZ, MOX, MOY, MOZ, Health, S |
| `MODE` | LineNo, TimeUS, Mode, ModeNum, Rsn |
| `MOTB` | LineNo, TimeUS, LiftMax, BatVolt, ThLimit, ThrAvMx, FailFlags |
| `MSG` | LineNo, TimeUS, Message |
| `MSG1` | cell：导出文本消息容器 |
| `MULT` | LineNo, TimeUS, Id, Mult |
| `ORGN_0`, `ORGN_1` | LineNo, TimeUS, Type, Lat, Lng, Alt |
| `PARM` | 972 × 2 cell：参数名、参数值 |
| `PIDA` | LineNo, TimeUS, Tar, Act, Err, P, I, D, FF, Dmod, SRate, Limit |
| `PIDP` | LineNo, TimeUS, Tar, Act, Err, P, I, D, FF, Dmod, SRate, Limit |
| `PIDR` | LineNo, TimeUS, Tar, Act, Err, P, I, D, FF, Dmod, SRate, Limit |
| `PIDY` | LineNo, TimeUS, Tar, Act, Err, P, I, D, FF, Dmod, SRate, Limit |
| `PM` | LineNo, TimeUS, NLon, NLoop, MaxT, Mem, Load, ErrL, IntE, ErrC, SPIC, I2CC, I2CI, Ex |
| `POS` | LineNo, TimeUS, Lat, Lng, Alt, RelHomeAlt, RelOriginAlt |
| `RAD` | LineNo, TimeUS, RSSI, RemRSSI, TxBuf, Noise, RemNoise, RxErrors, Fixed |
| `RATE` | LineNo, TimeUS, RDes, R, ROut, PDes, P, POut, YDes, Y, YOut, ADes, A, AOut |
| `RCO2` | LineNo, TimeUS, C15, C16, C17, C18 |
| `RCOU` | LineNo, TimeUS, C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14 |
| `SRTL` | LineNo, TimeUS, Active, NumPts, MaxPts, Action, N, E, D |
| `STAK_0`, `STAK_1`, `STAK_10`, `STAK_11`, `STAK_12`, `STAK_13`, `STAK_14`, `STAK_15`, `STAK_16`, `STAK_17`, `STAK_18`, `STAK_19`, `STAK_2`, `STAK_20`, `STAK_255`, `STAK_3`, `STAK_4`, `STAK_5`, `STAK_6`, `STAK_7`, `STAK_8`, `STAK_9` | LineNo, TimeUS, Id, Pri, Total, Free, Name |
| `Seen` | cell：导出记录清单 |
| `TERR` | LineNo, TimeUS, Status, Lat, Lng, Spacing, TerrH, CHeight, Pending, Loaded, ROfs |
| `UBX1_0` | LineNo, TimeUS, Instance, noisePerMS, jamInd, aPower, agcCnt, config |
| `UBX2_0` | LineNo, TimeUS, Instance, ofsI, magI, ofsQ, magQ |
| `UNIT` | LineNo, TimeUS, Id, Label |
| `VER` | LineNo, TimeUS, BT, BST, Maj, Min, Pat, FWT, GH, FWS, APJ |
| `VIBE_0`, `VIBE_1`, `VIBE_2` | LineNo, TimeUS, IMU, VibeX, VibeY, VibeZ, Clip |
| `XKF1_0`, `XKF1_1` | LineNo, TimeUS, C, Roll, Pitch, Yaw, VN, VE, VD, dPD, PN, PE, PD, GX, GY, GZ, OH |
| `XKF2_0`, `XKF2_1` | LineNo, TimeUS, C, AX, AY, AZ, VWN, VWE, MN, ME, MD, MX, MY, MZ, IDX, IDY, IS |
| `XKF3_0`, `XKF3_1` | LineNo, TimeUS, C, IVN, IVE, IVD, IPN, IPE, IPD, IMX, IMY, IMZ, IYAW, IVT, RErr, ErSc |
| `XKF4_0`, `XKF4_1` | LineNo, TimeUS, C, SV, SP, SH, SM, SVT, errRP, OFN, OFE, FS, TS, SS, GPS, PI |
| `XKF5_0` | LineNo, TimeUS, C, NI, FIX, FIY, AFI, HAGL, offset, RI, rng, Herr, eAng, eVel, ePos |
| `XKFS_0`, `XKFS_1` | LineNo, TimeUS, C, MI, BI, GI, AI, SS |
| `XKQ_0`, `XKQ_1` | LineNo, TimeUS, C, Q1, Q2, Q3, Q4 |
| `XKT_0`, `XKT_1` | LineNo, TimeUS, C, Cnt, IMUMin, IMUMax, EKFMin, EKFMax, AngMin, AngMax, VMin, VMax |
| `XKV1_0` | LineNo, TimeUS, C, V00, V01, V02, V03, V04, V05, V06, V07, V08, V09, V10, V11 |
| `XKV2_0` | LineNo, TimeUS, C, V12, V13, V14, V15, V16, V17, V18, V19, V20, V21, V22, V23 |
| `XKY0_0`, `XKY0_1` | LineNo, TimeUS, C, YC, YCS, Y0, Y1, Y2, Y3, Y4, W0, W1, W2, W3, W4 |
| `XKY1_0`, `XKY1_1` | LineNo, TimeUS, C, IVN0, IVN1, IVN2, IVN3, IVN4, IVE0, IVE1, IVE2, IVE3, IVE4 |

### C. 并非每条记录都有的非空变量

| 变量 | 出现在哪些原始文件 |
| --- | --- |
| `D32` | `FAULT_M1_5.mat`, `FAULT_M2_10.mat`, `FAULT_M2_5.mat`, `FAULT_M3_5.mat`, `FAULT_M5_5.mat`, `FAULT_M6_10.mat`, `FAULT_M6_5.mat`, `NO_FAULT2.mat`, `NO_FAULT3.mat`, `NO_FAULT4.mat`, `NO_FAULT5.mat`, `NO_FAULT6.mat` |
| `ERR` | `FAULT_M1_5.mat`, `FAULT_M2_5.mat`, `FAULT_M3_5.mat`, `FAULT_M5_5.mat`, `FAULT_M6_5.mat`, `NO_FAULT1.mat`, `NO_FAULT2.mat`, `NO_FAULT3.mat`, `NO_FAULT4.mat` |
| `MAVC` | `FAULT_M3_5.mat` |
| `MAV_0` | `FAULT_M1_10.mat`, `FAULT_M1_5.mat`, `FAULT_M3_5.mat`, `FAULT_M4_5.mat`, `FAULT_M5_5.mat`, `NO_FAULT2.mat` |
| `STAK_21` | `FAULT_M1_10.mat`, `FAULT_M1_5.mat`, `FAULT_M3_5.mat`, `FAULT_M4_5.mat`, `FAULT_M5_5.mat`, `NO_FAULT2.mat` |
| `TSYN` | `FAULT_M1_5.mat`, `FAULT_M3_5.mat` |
| `label_fault` | `FAULT_M1_10.mat`, `FAULT_M2_10.mat`, `FAULT_M3_10.mat`, `FAULT_M4_10.mat`, `FAULT_M5_10.mat`, `FAULT_M6_10.mat` |
