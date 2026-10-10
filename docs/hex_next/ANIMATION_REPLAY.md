# 完整任务动画回放

## 运行

在 `hex-tasks-continuous-diagnostics` 分支的项目根目录执行：

```powershell
python view_hex_tasks.py
```

程序只需要 Python 3.9 或更新版本的标准库。自动打开浏览器，默认显示 g006 的健康绕圈记录。
浏览器地址为 `http://127.0.0.1:8765/`；终端按 Ctrl+C 关闭服务。
这里的“完整”指从初始状态播放到最后一个实际保存采样，故障早停记录不补齐到8秒。

```powershell
# 从 g006 的 M0 完全失效记录开始
python view_hex_tasks.py --group 6 --label 1
# 不自动开浏览器；端口被占用时换一个端口
python view_hex_tasks.py --no-open --port 8766
# 检查现有152条记录，不启动播放
python view_hex_tasks.py --check
```

默认数据目录：`output_hex_next_20261009/batch2/dataset_v5`。
默认预测目录：其旁边的 `diagnostics/imu_only_epoch30/predictions`。
可以通过 `--data-dir` 和 `--predictions-dir` 选择其他已有v5记录。
Git不包含这些原始输出，只有源码时需先恢复现有数据或按[运行顺序](RUNNING_ORDER.md)生成新数据。

## 操作

- 选择八个任务组和19种状态：Normal、六个位置的Full、Severe、Partial。
- 播放、暂停、重置，0.25–4倍速度，拖动时间轴或点击曲线定位采样。
- “跳到故障采样”定位第一个已经改变效能的保存采样；另显示实际注入时间，二者不混同。
- 全局视角显示实际轨迹与整段计划轨迹；勾选“跟随放大”可看清机体和六个电机。
- 拖动、滚轮或方向键调整三维视角，“复位视角”恢复默认方向和缩放。
- “同组正常轨迹”加载同一扰动组的健康对照，仅画到当前播放时刻。
- 查看实际位置、三轴速度、总速度、姿态、倾角、角速度及六电机命令、实际转速、推力和效能。
- 分别显示注入真值、已保存网络预测和累计报警；缺少预测时明确显示没有记录。

## 数据依据与显示规则

| 内容 | 依据 | 显示方式 |
| --- | --- | --- |
| 位置、速度、机体姿态、角速度 | CSV的 `truth_*` 列；导出定义见 `hex_research/simulation.py` | 原样读取；最新一个不晚于播放时间的采样，不插值姿态 |
| 三轴速度和总速度 | `truth_vx`、`truth_vy`、`truth_vz`，世界ENU坐标 | 单位m/s；总速度 `sqrt(vx²+vy²+vz²)`，与位置使用同一采样 |
| 命令、实际转速、推力、效能 | `cmd_motor0…5`、`truth_rotor0…5`、`dynamic_rotor_thrust0…5`、`effectiveness0…5` | 分别显示，不根据故障标签重算转速或推力 |
| 机臂和转子尺寸、M0…M5顺序 | 每条JSON中的保存配置；几何定义见 `hex_research/configuration.py:physical` | `r_i=L[cos(iπ/3),sin(iπ/3),0]`，正六边形、同一转子平面 |
| 世界/机体坐标 | JSON的 `ENU_world_FLU_body_xyzw_body_to_world` | 东、北、上；机体前、左、上 |
| 姿态变换 | 保存欧拉角由 `Rotation.as_euler('xyz')` 导出 | `R=Rz(yaw)Ry(pitch)Rx(roll)`；世界位置 `p+Rr_i` |
| 机体倾角 | 机体z轴与世界z轴夹角 | `acos(clamp(cos(roll)cos(pitch),−1,1))`，不等于只看某一个欧拉角 |
| 全段计划轨迹 | 保存task配置；`hex_research/commands.py:ParameterizedTask.at` | 同一五次进度 `s=10u³−15u⁴+6u⁵`，321个显示点；黄色虚线明确标记计划 |
| 当前控制参考位置和偏航 | CSV的 `reference_position_*` 和 `reference_yaw` | 原样显示；不编造期望滚转/俯仰 |
| 故障和结束 | JSON的 `fault_start_s`、`actual_duration_s`、`termination`；末尾真值与保存包络阈值 | 精确注入时间及首次保存故障采样分开；到实际结束时间停止 |
| 诊断 | 30轮开发模型保存的逐窗口JSON | 只读不晚于当前时刻的预测；g002/g004/g006/g008有记录，其他组不替代或重新推理 |

圆盘亮度只是一种实际转速显示，红色表示该采样的注入效能低于1；不绘制没有记录的桨叶旋转相位。
机体中央圆形与机臂为示意外观，尺寸遵循研究配置，不是实机CAD。
相机方向、缩放和绘图采样密度属于显示设置，不进入动力学或控制。
z=0只是坐标参考面，原模型没有地面碰撞。提前结束表示保存的包络检查触发，不自动解释为真实撞地。
模型参数仍是研究设计；网络可靠报警仍未通过。回放不会改善控制、补偿故障或改变已有性能结论。

## 实际验证

机器可读数据核对见 [animation_data_checks.json](animation_data_checks.json)。
浏览器操作核对与截图见 [ANIMATION_VALIDATION.md](ANIMATION_VALIDATION.md)。
只读审计可以重复运行，报告写入独立输出目录：

```powershell
python -m hex_viewer.audit
python -m unittest discover -s tests -v
node tests/check_hex_replay.mjs
```

Node仅用于前端纯函数检查，播放本身不需要安装Node或前端框架。
回放包独立于 `hex_research` 和 `diagnostics`，本次没有修改原有造数、动力学、控制或诊断代码。
生成时源码与当前源码之间已有四处UTF-8读取修复，登记及执行快照保存在
`source_snapshot_generation/register.json`，原始生成记录没有重新签名；新审计如实报告这个既有差异并核对快照。
