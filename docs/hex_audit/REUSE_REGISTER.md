# 来源与复用台账

原始材料只作为证据。访问日期2026-10-06。源归档字节/SHA256见source_archive_index.json。

## R1 RotorPy 2.1.2 source and paper

作者/版本：Spencer Folk et al.；2023 / v2.1.2。

[RotorPy 2.1.2 source and paper](https://github.com/spencerfolk/rotorpy/tree/v2.1.2)；位置：multirotor.py compute_body_wrench/_s_dot_fn/_pack_state; hummingbird_params.py; simulate.py; environments.py。

local installed files and official version-tag snapshots read; generic Newton–Euler and speed-dependent rotor drag reviewed; scalar Hummingbird values are transplanted priors, not our measurements

## R2 PX4 1.12.3 controller source

作者/版本：PX4 contributors；v1.12.3 commit 2e8918da66af37922ededee1cc2d2efffec4cfb2。

[PX4 1.12.3 controller source](https://github.com/PX4/PX4-Autopilot/tree/v1.12.3)；位置：PositionControl.cpp:108–189; AttitudeControl.cpp:55–108; RateControl.cpp:60–105。

official historical source read; only cascade, derivative-on-measurement, yaw-frame conversion and antiwindup concepts borrowed; SI gains/full-quaternion attitude law are ours, not literal normalized firmware

## R3 PX4 1.12 Controller Diagrams

作者/版本：PX4 documentation contributors；1.12。

[PX4 1.12 Controller Diagrams](https://docs.px4.io/v1.12/en/flight_stack/controller_diagrams)；位置：position/velocity, attitude and angular rate diagrams。

official document read; research SI design explicitly differs from normalized PX4 motor controls

## R4 RflyMAD: A Dataset for Multicopter Fault Detection and Health Assessment

作者/版本：Xiangli Le, Bo Jin, Gen Cui, Xunhua Dai, Quan Quan；arXiv 2311.11340v2, 2024-01-11。

[RflyMAD: A Dataset for Multicopter Fault Detection and Health Assessment](https://arxiv.org/html/2311.11340v2)；位置：§3.2; §4.1。

original text read: modified PX4 1.12.3, real platforms are quadcopters; no direct evidence for our 19-class six-rotor task

## R5 RflyMAD official dataset description

作者/版本：Beihang Rfly/OpenHA；access 2026-10-06。

[RflyMAD official dataset description](https://rfly-openha.github.io/documents/4_resources/dataset.html)；位置：dataset page。

official page archived/read; no data converted into fictitious six-motor channels

## R6 Experimental Validation of a Fault-Tolerant Hexacopter with Tilted Rotors

作者/版本：Juan I. Giribet, Claudio D. Pose, Alejandro S. Ghersin, Ignacio Mas；2018, DOI 10.18178/ijeetc.7.2.58-65。

[Experimental Validation of a Fault-Tolerant Hexacopter with Tilted Rotors](https://www.ijeetc.com/uploadfile/2018/0612/20180612065846780.pdf)；位置：printed p.61 (PDF page4): identified J_X,Y=0.0476, J_Z=0.1094, pole20.06; modified mounting107 degrees。

original PDF text acquired/read; effective identified parameters on tilted platform cannot be treated as an admissible measured inertia tensor for this coplanar design

## R7 Latest protected project working tree

作者/版本：Local project/user working changes；ef5fa5b plus 6 protected files。

[Latest protected project working tree](https://github.com/Gonxioa/eVTOL_Simulation/commit/ef5fa5b0d7e00e63f9a4afe06e78128c1886b82b)；位置：baseline.json; legacy_reference exact copies。

local candidates for thrust7.45e-6, Q/T0.01315 and manufacturer/bench claims are retained but the exact bench table was not acquired in this worktree; current scalar values are explicit design assumptions

## R8 UAV-FD preserved independent package

作者/版本：Università Politecnica delle Marche / prior project audit；v1; existing ef5fa5b package。

[UAV-FD preserved independent package](https://zenodo.org/records/7648996)；位置：uavfd_f550/configs/split.json; docs/uavfd_f550。

preserved byte-for-byte; no new amplitude/PSD reads from 11 locked test flights

## 复用决策

| 模块 | 决策 | 理由 |
| --- | --- | --- |
| control_allocation | 保留并修正正则快捷路径 | 原10测试保留；rho=0行为保留；rho>0须真的优化正则目标 |
| provenance / atomic CSV / 分组与折内缩放 / LSTM / WGAN | 保留并适配v4接口 | 数据版本绑定参数，特征路径随FEATURE_SET，原公平对照保留 |
| RotorPy Newton–Euler/旋转桨阻力 | 方程逐项核对后本地实现 | 六路19状态；无全局patch；不调用其私有pack/inv |
| SE3Control / 上游20状态打包 / 进程级兼容patch | 退出默认链路，隔离参考 | 原代码和24测试保存在legacy_reference，单独进程运行 |
| PX4 1.12.3 | 借鉴结构与指定概念 | 自有SI串级；不声称复刻定制PX4/RflyMAD或ArduPilot |
| kh平移升力 | 不用于本研究停转模型 | 原项独立于Omega，停转仍有0.08136N；适用范围没有覆盖停转 |
| UAV-FD包、文档与冻结划分 | 原样保留 | 独立桨叶缺损任务；不替代当前19类电机实飞证据 |
