# 外部 MATLAB / Simulink 项目与当前六旋翼模型对照

核对日期：2026-10-20。目的：确定当前六旋翼仿真能够借鉴的具体内容，以及迁移时需要重做的部分。

后续接口调整：按用户与老师确认的要求，当前模型任务指令已简化为位置、速度共 6 维，取消加速度及偏航角速度前馈，控制器内部保持固定航向，并继续使用姿态与角速度反馈。下文对本项目原 11 维接口的描述保留为本次调研时的历史对照；最新使用方式见 [Simulink 说明](C:/Users/lenovo/Desktop/eVTOL_Simulation/simulink/README.md)。

本次实际访问了下列七个仓库的 README、目录和关键 MATLAB 源码，并下载静态检查了五份 SLX 的内部 XML：Quad-Sim 的 QDMS4CSD 库、INDI 的 controls、Sanassah 的 Drone_Simulation、MathWorks 的 quadcopter_library、Ewing 的 OS4dynamics_01。未在 MATLAB 中执行这些外部模型；兼容性、数值正确性和性能尚未经试跑验证。SLX 仅作归档内容读取，没有触发初始化回调。

当前项目对照依据为本地 simulink/blocks/、simulink/functions/、simulink/build_hex_model.m 和 simulink/configs/research.json。本次仅新增此研究文档，没有修改仿真代码或用户正在编辑的模型。

## 核对版本

| 仓库 | 分支 | 本次解析的提交 |
|---|---|---|
| [dch33/Quad-Sim](https://github.com/dch33/Quad-Sim) | master | [961bc69d4939](https://github.com/dch33/Quad-Sim/commit/961bc69d4939c8d0661eeb9820de668994262f65) |
| [yrlu/quadrotor](https://github.com/yrlu/quadrotor) | master | [79ac78e47064](https://github.com/yrlu/quadrotor/commit/79ac78e470640dceaf1558c421bdc23a41ea229a) |
| [SihaoSun/INDI_Quadrotor_FTC](https://github.com/SihaoSun/INDI_Quadrotor_FTC) | master | [0431290e4edf](https://github.com/SihaoSun/INDI_Quadrotor_FTC/commit/0431290e4edf216bd10f05b33f06fff6cf273d2c) |
| [Sanassah/UAV_VirtualPrototyping](https://github.com/Sanassah/UAV_VirtualPrototyping) | main | [dc5e38387c6d](https://github.com/Sanassah/UAV_VirtualPrototyping/commit/dc5e38387c6d2b03ccc3d9f028d109785ff2f987) |
| [mathworks/Quadcopter-Drone-Model-Simscape](https://github.com/mathworks/Quadcopter-Drone-Model-Simscape) | master | [b2513ee228f2](https://github.com/mathworks/Quadcopter-Drone-Model-Simscape/commit/b2513ee228f26c87c7c862275633f250c5fbb95e) |
| [EwingKang/QuadCopter-Quaternion-Dynamics-in-Simulink](https://github.com/EwingKang/QuadCopter-Quaternion-Dynamics-in-Simulink) | master | [bdd778bb59f9](https://github.com/EwingKang/QuadCopter-Quaternion-Dynamics-in-Simulink/commit/bdd778bb59f94c00c5cc0b64be39a71c331f7259) |
| [petercorke/robotics-toolbox-matlab](https://github.com/petercorke/robotics-toolbox-matlab) | master | [06fddf5228cb](https://github.com/petercorke/robotics-toolbox-matlab/commit/06fddf5228cbfa1132fa63b8e336b43c5ed04899) |

以下“建议”是结合本项目作出的工程判断，不表示外部仓库已经实现或验证了我们的六旋翼方案。

## 1. yrlu/quadrotor：最适合立即对照任务参考与控制方程

实际阅读：

- [control/controller.m](https://github.com/yrlu/quadrotor/blob/master/control/controller.m)
- [control/trajectories/circle.m](https://github.com/yrlu/quadrotor/blob/master/control/trajectories/circle.m)
- [traj_planning/trajectory_generator.m](https://github.com/yrlu/quadrotor/blob/master/traj_planning/trajectory_generator.m)
- [traj_planning/traj_opt7.m](https://github.com/yrlu/quadrotor/blob/master/traj_planning/traj_opt7.m)

它的期望状态明确包含 pos、vel、acc、yaw、yawdot，与我们 Task Reference 输出的五组、共 11 个标量相对应。controller.m 把期望加速度加入位置和速度误差反馈，又用 yaw_des 生成目标朝向，用 yawdot_des 形成角速度参考。因此，我们接口中的加速度和偏航角速度有直接可对照的实现。

可借鉴：

1. 用带字段名称的期望状态解释接口，让用户可以直接看到位置、速度、加速度、偏航角和偏航角速度。
2. 将任务规划从当前悬停、直线、圆形、升降扩展为多航点平滑轨迹。
3. 参考七次多项式轨迹及其解析导数生成方式，明确各段的边界条件与连续性。

对当前项目的判断：我们的直线、圆形、升降已由五次时间函数同时生成位置、速度和加速度，已有前馈功能；可增加的是航点与时间分配。该项目基础控制器采用欧拉角、小角度映射和 PD；我们已有四元数、积分、滤波、限幅、抗饱和和有界分配，建议保留这些能力。其 circle.m 用有限差分算导数，我们现有的对应轨迹使用解析导数，可以继续保留。

对应本地：Task Reference、Flight Controller；源文件 task_reference.m、controller_step.m。

## 2. dch33/Quad-Sim：参数辨识与控制框图组织最有价值

实际阅读了 calculate_CT.m、calculate_CQ.m、calculate_TC.m、calculate_CR_B.m 和 quadcopterDynamicsSFunction.m；静态查看了 QDMS4CSD.slx。相关目录：

- [测量数据分析源码](https://github.com/dch33/Quad-Sim/tree/master/Quadcopter%20Dynamic%20Modeling%20and%20Simulation/Data%20Acquisition%20%26%20Analysis/MATLAB%20Data%20Analysis)
- [模型及测试说明](https://github.com/dch33/Quad-Sim/tree/master/Quadcopter%20Dynamic%20Modeling%20and%20Simulation/Documentation)
- [Simulink 模块库](https://github.com/dch33/Quad-Sim/tree/master/Quadcopter%20Dynamic%20Modeling%20and%20Simulation/Simulation%20Files/Block%20Library)

确认的内容：

- calculate_CT、calculate_CQ 从稳态测量值拟合推力/反扭矩与转速平方的关系，并绘制拟合残差。
- calculate_TC 用油门阶跃后的转速响应，在达到总变化约 63.2% 的时刻估计一阶时间常数。
- calculate_CR_B 拟合油门到 RPM 的关系。
- 库内分开组织电机动力学、姿态控制、位置控制、控制混合、扰动与采样状态估计；状态估计部分含互补滤波结构。

对当前项目最有用的增量是建立“参数—数据—拟合—残差—有效范围”记录。目前 research.json 明确标注为研究设计参数；可按这套流程为 kT、kQ、tau 提供测量依据。现有单一电机时间常数也可以在有数据后检查是否需要按转速区间、升降速方向区分。

单位必须重新统一：该项目 calculate_CT 输出 N/RPM²，calculate_CQ 输出 Nm/RPM²；我们使用 rad/s。若 T=CT_RPM*n² 且 omega=2*pi*n/60，则本项目 kT=CT_RPM*(60/(2*pi))²，kQ 同理。它的 CT 是有量纲拟合系数，不能因名称相同就与其他项目的无量纲推力系数混用。

框图方面，可将我们 Flight Controller 逐层展开为位置、速度、四元数姿态、角速度和控制分配子系统，以便逐块理解。互补滤波可以作为姿态估计入门参考；完整的位置、速度和姿态估计仍需另行设计。

该仓库材料主要面向旧版本 MATLAB，动力学使用 Level-2 S-Function 和欧拉角。建议借鉴辨识方法和框图组织，数值核心继续沿用当前经过对照验证的实现。

## 3. SihaoSun/INDI_Quadrotor_FTC：研究严重失效与控制目标降级

实际阅读：

- [INDI_FTC/allocation_att_indi.m](https://github.com/SihaoSun/INDI_Quadrotor_FTC/blob/master/INDI_FTC/allocation_att_indi.m)
- [INDI_FTC/pseudo_control_att_indi.m](https://github.com/SihaoSun/INDI_Quadrotor_FTC/blob/master/INDI_FTC/pseudo_control_att_indi.m)
- [controlParams.m](https://github.com/SihaoSun/INDI_Quadrotor_FTC/blob/master/controlParams.m)
- [functions/UREstimators.m](https://github.com/SihaoSun/INDI_Quadrotor_FTC/blob/master/functions/UREstimators.m)
- [functions/createAllBusses.m](https://github.com/SihaoSun/INDI_Quadrotor_FTC/blob/master/functions/createAllBusses.m)

静态检查 controls.slx，确认存在位置控制、偏航控制、INDI 分配、滤波/微分相关结构，以及名为 fake estimator 的子系统。

关键发现：allocation_att_indi.m 在失效时移除对应执行器的作用；单旋翼失效时取消偏航控制对应行，双对置旋翼失效时进一步减少受控输出，再根据测得的输出导数增量计算控制更新。它的启发是根据失效后的控制能力调整任务目标。

对当前六旋翼的建议：

- 分析不同失效位置、效能和转速限制下，可实现的总推力与三个轴力矩。
- 在仍可实现原目标时保持原任务；确实不可实现时再设计偏航放松、姿态约束或降落等模式。
- 将 INDI 作为单独的对照控制器，与现有串级控制器在相同飞机、故障、扰动和传感器条件下比较。

当前分配器已有固定轴优先级和转速边界；可增加的是控制能力分析及随模式改变的目标，而不是单纯再加一套固定权重。

不能直接照搬：它是四旋翼、具有不同的质量惯量与旋翼编号，分配矩阵和失效后的受控输出必须按六旋翼重新推导。UREstimators 接收外部 fail_id 并直接写入 state.fail_id，这套最小示例不能被当作已经实现故障诊断的证据。我们当前控制器不接收故障真值；后续若加入重构，需区分已知故障的性能上限实验与带检测延迟、误差的实际闭环。

可立即借鉴 createAllBusses.m 的信号命名方式，让 reference、state、imu 的组成更容易查看。

## 4. Sanassah/UAV_VirtualPrototyping：六旋翼健康分配与电热系统补充

实际阅读：

- [main/ControlLaws_Script.m](https://github.com/Sanassah/UAV_VirtualPrototyping/blob/main/main/ControlLaws_Script.m)
- [main/ReliabilityModel_Script.m](https://github.com/Sanassah/UAV_VirtualPrototyping/blob/main/main/ReliabilityModel_Script.m)
- [main/Drone_Simulation.slx](https://github.com/Sanassah/UAV_VirtualPrototyping/blob/main/main/Drone_Simulation.slx)

SLX 中确认存在六个电机、Weighted Control Allocation、可靠性观察器、电机/ESC 热模型，以及电池相关模块。加权分配子系统包含可靠性矩阵输入、矩阵乘积与逆运算、开方及输出饱和结构。控制脚本给出六旋翼 4×6 控制分配矩阵；推进脚本还包含电气与寿命参数。

可借鉴：

- 为六旋翼分配增加执行器健康、负载或能耗方面的代价。
- 把推进、电池、电热与健康模型组织成独立模块。
- 参考根据响应时间推导控制增益的记录方式，为调参补充计算依据。

迁移需要区分两种量：可靠性描述失效概率或寿命状态，推力效能描述当前指令还能产生多少作用，二者不能直接当作同一个 eta。其矩阵求逆及饱和结构，也不能直接替代我们在优化问题中显式处理推力上下界的方法。

参数尚需逐项复核。例如 ReliabilityModel_Script.m 中仍有 nr=8，以及明确写为 assumption 或 arbitrarily 的寿命/可靠性参数；不能因项目主体是六旋翼，就把所有初始化参数视为已经适配并验证。对本项目，更适合将其作为六旋翼与健康管理的补充参考。

## 5. MathWorks Simscape 项目：推进系统、能耗与批量试验

实际阅读：

- [Scripts_Data/quadcopter_package_parameters.m](https://github.com/mathworks/Quadcopter-Drone-Model-Simscape/blob/master/Scripts_Data/quadcopter_package_parameters.m)
- [Scripts_Data/quadcopter_waypoints_to_trajectory.m](https://github.com/mathworks/Quadcopter-Drone-Model-Simscape/blob/master/Scripts_Data/quadcopter_waypoints_to_trajectory.m)
- [推进测试目录](https://github.com/mathworks/Quadcopter-Drone-Model-Simscape/tree/master/Workflows/Thrust_Test)
- [载荷与风参数扫描](https://github.com/mathworks/Quadcopter-Drone-Model-Simscape/blob/master/Workflows/Param_Sweep/quadcopter_package_delivery_sweep_load_mass_wind.m)

静态检查 quadcopter_library.slx，确认有 Propeller、Thrust and Drag、Package 和投递相关模块。README 与参数脚本还明确包含电机、电池、功率/效率与串级 PID。

最适合借鉴的增量：

1. 推进台架模型：给定转速或控制命令，检查推力、反扭矩、功率与电流。
2. 在现有电机一阶转速模型之外，增加电池电压、功率上限、效率和耗电量。
3. 参考 SimulationInput 与 Fast Restart 组织批量试验，扫描质量、风速和故障效能；并行执行是可选加速方式。
4. 航点轨迹中根据曲率调整速度、接近终点时减速，以及按路径切线设定机头朝向。

我们的电机响应和风扰动已有基础实现；新增价值主要是电能约束与系统试验流程。该项目依赖 Simscape 相关产品，完整引入会增加依赖与计算量。建议先建立独立推进测试模型，完成静态推力与动态响应对照后再决定与整机耦合。

它的 propeller.Kthrust=0.1072 等系数的定义和当前 kT 不同，必须根据所在模块的方程换算，不能直接替换参数数值。

## 6. EwingKang 项目：四元数实现与气动项检查

实际阅读 [OS4dynamics_01_init.m](https://github.com/EwingKang/QuadCopter-Quaternion-Dynamics-in-Simulink/blob/master/OS4dynamics_01_init.m)，静态检查 [OS4dynamics_01.slx](https://github.com/EwingKang/QuadCopter-Quaternion-Dynamics-in-Simulink/blob/master/OS4dynamics_01.slx)。

模型确实分出 Quaternion dot、Quaternion Normalization、Propeller & motor dynamics、ground effect、propeller gyroscopic effect、fuselage aerodynamics 等子系统；初始化参数包含旋翼转动惯量和诱导流时间常数。

适合作为动力学核对参考：逐项检查四元数运动学、旋翼陀螺作用、近地效应和诱导流动态是否与我们的研究工况有关，再决定增加哪一项。我们已使用四元数和归一化，新增复杂气动项需要相应参数或数据支持。

特别注意其初始化四元数为 [q0 q1 q2 q3]，标量在前；我们为 [x y z w]，标量在后。坐标方向、旋翼布局和旋转方向同样需要明确转换，不能直接连接。

## 7. Robotics Toolbox for MATLAB：教材式动力学与动画补充

实际阅读：

- [models/mdl_quadrotor.m](https://github.com/petercorke/robotics-toolbox-matlab/blob/master/models/mdl_quadrotor.m)
- [simulink/quadrotor_dynamics.m](https://github.com/petercorke/robotics-toolbox-matlab/blob/master/simulink/quadrotor_dynamics.m)
- [README](https://github.com/petercorke/robotics-toolbox-matlab)

可借鉴旋翼几何、无量纲气动系数到有量纲系数的组织方式，以及飞行姿态、机头方向与轨迹动画的教学表达。它更适合提供基础方程和可视化对照。

本次核对的 quadrotor_dynamics.m 存在明确限制：任一旋翼转速为零就抛出错误；其状态使用欧拉角，并以向下为 z 正方向。对我们完全失效 eta=0 的研究，不宜直接替换动力学。README 也说明该 MATLAB 工具箱不再积极开发。

## 建议的实施顺序

| 顺序 | 建议工作 | 主要参考 | 当前模型可以保留的部分 |
|---|---|---|---|
| 1 | 给信号加字段名和单位；展开控制层级，方便逐块学习 | INDI Bus、Quad-Sim、yrlu | 现有数值方程与默认试验 |
| 2 | 建立 kT、kQ、tau 辨识与参数来源记录 | Quad-Sim | 现有电机和推力模型形式 |
| 3 | 增加多航点任务、平滑转向与时间分配 | yrlu、MathWorks | 现有 11 维参考接口 |
| 4 | 增加有噪声条件下的状态估计，并明确估计量与真值 | Quad-Sim 状态估计组织 | 现有传感器与延迟接口 |
| 5 | 研究失效后的可实现控制能力、目标调整与控制重构 | INDI、Sanassah | 现有故障注入与试验记录 |
| 6 | 建立电池/电机测试模型，再评估整机电能耦合 | MathWorks、Sanassah | 现有整机基线 |

任何控制或动力学替换应与当前基线使用相同轨迹、故障、扰动和传感器条件。当前模型已经具备积分、抗饱和、故障时刻分段、传感器噪声/延迟和有界分配，外部项目的局部实现不能自动视为整套系统的提升。

## 许可核对记录

这是对实际可见文件的记录，不是法律意见，也不代表已经取得复制所有第三方内容的授权。

- Quad-Sim：README 与源码声明 LGPL-3.0-or-later，仓库有许可文本。
- Robotics Toolbox：README 与上述源码声明 LGPL。
- Ewing：README 包含 MIT 许可文本；目录中没有独立 LICENSE 文件，不等于没有许可。
- MathWorks：[LICENSE.md](https://github.com/mathworks/Quadcopter-Drone-Model-Simscape/blob/master/LICENSE.md) 包含仅配合 MathWorks 产品与服务使用的软件及衍生版本条件。把其源码移入 Python 平台前需单独处理这一条件。
- yrlu、INDI、Sanassah：本次递归目录检查未找到独立 LICENSE/COPYING 文件，已读 README 中也未见通用复用许可。引用要求与源码复用许可应分别核对。

本次结论用于选择学习与研究路径，没有复制外部代码到现有仿真模块中。
