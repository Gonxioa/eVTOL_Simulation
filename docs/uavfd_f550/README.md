# UAV-FD／F550 独立模型：中文运行说明

当前代码、真实数据的受限组合标定和实际验证已交付。先看 MODEL_STATUS.md 与 VALIDATION_REPORT.md；完整物理参数和闭环尚未具备实飞验证支持。

在项目根目录执行，Python 3.11 已实测。依赖可单独安装：

```powershell
python -m pip install -r uavfd_f550/requirements.txt
python -m uavfd_f550 sources
python -m uavfd_f550 audit
python -m unittest discover -s uavfd_f550/tests -v
python -m uavfd_f550 validate
python -m uavfd_f550 demo
python -m uavfd_f550 calibrate
python -m uavfd_f550 replay
python -m uavfd_f550 vibration
python -m uavfd_f550 report
```

本轮使用现有 Python，仅补装 h5py 3.16.0 验证 MAT v7.3 读取。依赖版本与每次代码／配置／数据哈希都写入结果。sources 将首选源下载存档，audit 依赖发布元数据 release.json；可在离线环境复用本轮已有 source_snapshots／release.json。未找到原论文全文，IEEE 202 返回未提供论文内容，不能按“访问成功”处理。

原始数据默认 `UAV-FD/`，新输出默认 `output_uavfd_f550/`，与旧 output／runs 隔离。所有命令可传 `--data-dir`、`--out`；信号分析另可传 `--config`。原 MAT 不改写，模型／大量数组／源网页归档不入 Git。小型实测摘要、哈希与台账保存在 docs/uavfd_f550/，可直接审阅。

`python -m uavfd_f550 test` 会明确失败：最终 11 条记录锁定，没有本轮解锁开关。数据层也拒绝使用开发记录做 fit、拒绝读取锁定记录做任何信号分析。audit 仅允许其完整性、时间和配置元数据。

配置：split.json 固定 18 条具体文件名；analysis.json 是预定分析方法假设；demo.json 全部数值仅供解析演示；uavfd_aircraft.json 中缺少的飞机真值为 null。未知完整配置不能导入为真实 Body。正常组合标定及频带增益分别写入 combination_model.json、vibration_model.json；后者并非结构模态标定。

证据入口：SOURCE_REGISTER.md／sources.json 记录出处与访问限制，equation_ledger.json 逐方程对应代码和验证，parameter_ledger.json 逐参数记录单位、证据状态及简化假设。CHANGE_MANIFEST.md 提供本轮改动清单，分支Git历史提供可审查差异；本地还保留 output_uavfd_f550/uavfd_f550_changes.patch。验证报告使用仓库内figures目录的图表，可在GitHub直接查看。

主要输出：audit.json／data_hashes.json、source_fetch.json、analytical_validation.json、normal_replay.json、vibration_validation.json、demo_vibration.json 和三张 PNG。所有段起止、记录 ID、使用目的、条件数与真实失败现象均保留。

新旧对接：旧 main.py、控制器、WGAN 和分类入口保持原路径，新包不 import RotorPy／RotorS 核心。旧 ENU／FLU 数据仅通过 dynamics.old_enu_flu_to_ned_frd 显式转换；不要把旧 rotor_directions 或圆周顺序当 UAV-FD Motor 编号。后续网络接口只允许真实可观测 IMU／PWM／RPM 等通道，不能导入故障真值或未测模态状态作为分类特征。

2026-10-04完成本地开发与验证；2026-10-05用户授权提交并推送独立分支，保留原有未提交修改。未运行旧19类造数／训练管线，完整模型验证、闭环和后续训练仍按MODEL_STATUS中的门槛执行。
