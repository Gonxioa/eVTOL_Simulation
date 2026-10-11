# 参数、控制方程与日志台账

## 参数依据

| 项目 | 修改前 | 研究候选与依据 |
| --- | --- | --- |
| 横滚/俯仰 `rate_i`，s⁻² | 1 / 1 | 10来自用户临时试验；20、30为运行前列明的补偿建立速度试验，不标为实机增益 |
| 偏航 `rate_i`，s⁻² | 0.6 | 分别比较0.6和用户候选6；当前筛选保留0.6，完整验证后再决定启用 |
| 横滚/俯仰积分上限，rad/s² | 3 / 3 | 用户候选20；另比较12，来自本模型离线悬停补偿需求约11.56rad/s²的边界试验 |
| 偏航积分上限，rad/s² | 2 | 比较2与用户候选20；离线静态需求约0.284rad/s²，保留2有明确物理余量 |
| 位置、速度、姿态及角速度P/D | 原配置 | 本轮内环筛选保持原值，避免多环同时修改导致归因不清 |
| 竖直速度 `velocity_i[2]`，s⁻² | 0.6 | 内环检查完成后，按预声明0.9/1.2单因素恢复试验，选择通过的较小值0.9；横向积分0.4/0.4和全部限幅保留 |
| 质量、惯量、推力常数、电机时间常数 | 原研究配置 | 本轮没有修改；真实性限制见原 `docs/hex_audit/PARAMETER_AUDIT.md` |
| 10%位置性能余量与长期尾段门槛 | 新增研究门槛 | 本轮离线验收设计，首条候选前固定；不宣称来自厂家、实飞或法规 |

离线估计使用研究几何和当前名义最小范数分配器：
`H=A diag(eta) pinv(A)`，解 `H w_nominal=[mg,0,0,0]`，
再由 `alpha_integral=J⁻¹ moment_nominal` 求悬停所需角加速度积分。
M0/M3的横向需求约11.556rad/s²，其他位置的最大轴分量约10.008rad/s²；偏航绝对值约0.284rad/s²。
该估计仅用于离线选择积分能力，假设静止、无风、未触及分配边界，不是飞行中的故障识别或补偿输入。
完整数值在预声明计划的 `authority_estimates` 中，在线控制器没有读取这个表或eta。

## 保留的在线方程

角速度控制：`alpha=Kp*e_rate+I_rate_used−Kd*filtered_alpha`，
`moment=J*alpha+w×Jw`。每次输出后，积分尝试增加 `Ki*e_rate*dt`。
如果名义分配器的轴力矩残差与该轴误差同向，阻止这一次向外积分；随后按各轴上限夹紧。
速度积分保留原反馈累积和反算项：
`I_velocity_next=clip(I_velocity_used+dt*(Ki*e_velocity+Kaw*(a_limited−a_raw−nominal_acc_deficit)))`。
残差为“请求−名义分配”，实际故障效能、实际推力差和标签不进入积分判断。

结构核对依据为[PX4控制器说明](https://docs.px4.io/main/en/flight_stack/controller_diagrams)：积分能力有限制，导数有滤波。
固定版本代码核对为[PX4 v1.12.3 RateControl.cpp](https://github.com/PX4/PX4-Autopilot/blob/v1.12.3/src/modules/mc_rate_control/RateControl/RateControl.cpp#L71)，
其积分按正/负方向饱和状态约束，并允许反向误差使积分退回。
[同版本PositionControl.cpp](https://github.com/PX4/PX4-Autopilot/blob/v1.12.3/src/modules/mc_pos_control/PositionControl/PositionControl.cpp#L112)
包含速度PID和水平反算抗饱和。
本项目用SI角加速度积分和名义求解器残差，自有实现与PX4的归一化输出及饱和接口不同；没有宣称完全复刻。

## 新增导出日志

全部CSV控制列以 `control_` 开头，仅作审计，不进入诊断特征白名单。

| 字段族 | 含义与单位 |
| --- | --- |
| `position_error_*`、`velocity_error_*` | 控制器实际使用的反馈误差，m、m/s |
| `observed_velocity_*`、`observed_rate_*` | 与反馈时间戳匹配的延迟反馈，m/s、rad/s；不能用当前传感器采样冒充 |
| `attitude_error_vector_*` | `2 sign(q_error.w) vec(q_error)`，无量纲；小角度下近似角度误差 |
| `attitude_error_angle_rad` | 最短姿态误差角，`2 asin(clamp(norm(error_vector)/2,0,1))` |
| `rate_reference_*`、`rate_error_*` | 角速度目标和目标−反馈，rad/s |
| `rate_integral_applied_*`、`velocity_integral_applied_*` | 本次输出实际使用的积分，rad/s²、m/s² |
| `rate_integral_*`、`velocity_integral_*` | 本次更新后、下一输出将使用的积分，明确与applied分开 |
| `rate_integral_requested_step_*`、`allowed_step_*`、`actual_step_*` | 请求积分增量、抗饱和允许增量、限幅后的实际增量，rad/s² |
| `velocity_integral_feedback_step_*`、`antiwindup_step_*`、`actual_step_*` | 反馈增量、反算增量、限幅后增量，m/s² |
| `*_integral_limit_active_*` | 更新后积分位于上限，0/1，比较容差1e−10沿用数值判定，不是物理参数 |
| `*_integral_clipped_step_*` | 本次提议更新被积分上限截断，0/1；与“仍在上限”分开 |
| `rate_integral_blocked_nominal_*` | 本次积分被名义分配残差判定阻止，0/1 |
| `allocator_lower_motor*`、`allocator_upper_motor*` | 各电机命令触及名义上下界，0/1 |
| `nominal_residual_thrust`、`nominal_residual_moment_*` | 名义分配器残差，N、N·m |

`control_log_schema=cascade_audit_v1`写入新导出JSON；v5观测时序契约和六电机编号保持原样。
此前CSV不改写补列；原始数据与新控制日志由版本、配置哈希和生成计划区分。
