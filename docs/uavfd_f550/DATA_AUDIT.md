# 原始数据审计

18 条原始 MAT 的精确字节数与发布 MD5 全部一致；SHA256 另行记录。scripts 的发布比较见 audit_summary.json；作者预处理后的 data_feature_Table_zoh.mat 未用于本轮建模。

| 记录 | 用途 | IMU0整段平均Hz | 中位间隔ms | ESC0整段平均Hz | 共同IMU行数 |
| --- | --- | ---: | ---: | ---: | ---: |
| NO_FAULT1.mat | calibration | 362.793 | 2.581 | 8.963 | 26162 |
| NO_FAULT2.mat | calibration | 234.651 | 2.619 | 4.014 | 57455 |
| NO_FAULT3.mat | calibration | 319.989 | 2.654 | 7.184 | 42975 |
| FAULT_M1_5.mat | calibration | 226.381 | 2.625 | 4.868 | 64271 |
| FAULT_M2_5.mat | calibration | 355.988 | 2.638 | 8.523 | 39203 |
| NO_FAULT4.mat | development | 353.595 | 2.629 | 8.512 | 36852 |
| FAULT_M3_5.mat | development | 208.337 | 2.627 | 4.944 | 63170 |
| NO_FAULT5.mat | locked_test | 234.679 | 2.638 | 5.484 | 34787 |
| NO_FAULT6.mat | locked_test | 331.947 | 2.643 | 8.072 | 37878 |
| FAULT_M1_10.mat | locked_test | 348.737 | 2.661 | 8.470 | 47018 |
| FAULT_M2_10.mat | locked_test | 351.618 | 2.653 | 8.478 | 53863 |
| FAULT_M3_10.mat | locked_test | 348.612 | 2.668 | 8.434 | 39243 |
| FAULT_M4_5.mat | locked_test | 347.847 | 2.668 | 8.605 | 27599 |
| FAULT_M4_10.mat | locked_test | 350.089 | 2.656 | 8.519 | 36211 |
| FAULT_M5_5.mat | locked_test | 355.040 | 2.637 | 8.654 | 32131 |
| FAULT_M5_10.mat | locked_test | 347.928 | 2.663 | 8.571 | 43489 |
| FAULT_M6_5.mat | locked_test | 264.034 | 2.639 | 6.229 | 34578 |
| FAULT_M6_10.mat | locked_test | 353.192 | 2.649 | 8.632 | 50839 |

这些时间统计属于完整性／采样元数据，锁定测试没有计算幅值统计、频谱或特征。开发使用5条标定+2条开发，11条最终记录始终锁定。

18 条均包含972项PARM；FRAME_CLASS=2、FRAME_TYPE=1、MOT_THST_EXPO=0.63、C9–C14 Motor1–6一致。VER均为4.4.0、FWT=0、GH=589333444=0x232083c4。公开基础提交可解析为232083c4d23be21d694d130f639fb6497bd57f98，2022-09-27；这不是作者定制补丁的恢复证据。FWS/MSG和单位字符串在数字MAT导出中未恢复；未声称同固件完全复现。

首条记录的IMU／RCOU／ESC0整段平均频率为362.793／93.344／8.963Hz，与任务线索吻合。IMU GHz/AHz为内部速率字段，不是这些日志频率。停飞缺口使部分记录平均值明显低于局部频率，必须按段处理。

三IMU唯NO_FAULT1完整时间戳相同，其他取交集后平均；逐通道间隔、非递增与长间隔计数见 audit_summary.json。时间缺口的真实丢样数／采样触发原因未知，gap_gt_2_median只是警示计数。PARM、VER、ARM、MODE、RCOU、ESC_0–5、IMU_0–2、ATT、RATE、PIDR/PIDP/PIDY、XKF1_0/1与VIBE有实际记录；label_only未当通道。

| 通道 | 标签字段与单位 | 处理与限制 |
| --- | --- | --- |
| TimeUS | 启动后us | 转换为秒，保留绝对时钟；采样/执行器/滤波延迟未分离 |
| IMU GyrX/Y/Z | rad/s；FRD | 历史发布路径经滤波；定制高频日志位置未知 |
| IMU AccX/Y/Z | m/s²；FRD比力 | 已有日志处理，不再套20Hz固件滤波；高频并非未滤波真值 |
| ESC RPM / RawRPM | rpm | 分别slew后/前；机械转速转换见历史BLHeli，电极数真实性未另证 |
| ESC Volt / Curr | V / A | 通道对齐不外推；不据低频遥测辨识毫秒级响应 |
| RCOU C9–C14 | us-equivalent | DShot模式下日志归一命令标度，不假称实际PWM脉宽 |
| ATT Roll/Pitch/Yaw | deg，估计姿态 | 转rad、解缠；不当运动真值 |
| XKF1 VN/VE/VD | m/s NED | EKF估计，不是独立真值 |
| VIBE | m/s²统计指标 | 5Hz地板、残差平方2Hz平滑、开方的基础算法；不作为原波形 |

固定飞行分组见 uavfd_f550/configs/split.json。候选飞行段、抗混叠方法和ARM初值未知处理见MODEL_SPEC。正式本轮不估计检测延迟。
