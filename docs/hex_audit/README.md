# 研究六旋翼审计与自有飞控：运行说明

任务日期2026-10-06，基线`ef5fa5b`，本地分支`codex/hex-audit-control`。实际工作树：

```text
C:/Users/lenovo/.codex/worktrees/hex-audit-control/eVTOL_Simulation
```

原桌面工作区的6项修改保持原样。新核心为`hex_research/`，默认入口`main.py`，审计生成器`hex_audit_reports.py`，旧入口与原24项测试位于`legacy_reference/`。独立UAV-FD包及冻结划分原样保留。

先阅读：PARAMETER_AUDIT.md、EQUATION_AUDIT.md、CONTROL_SPEC.md、DATA_CONTRACT.md、VALIDATION_REPORT.md和MODEL_STATUS.md。机器参数/公式台账记录研究设计、推导、移植候选及真实未知身份，不将代码/数学/仿真通过替代实飞证据。

## 最小复现

在工作树根目录执行，Python3.11，当前完整依赖版本见dependencies.lock.txt（版本清单，不是不同平台可无条件安装的通用包集）：

```powershell
python -m unittest discover -s tests -v
python main.py validate
python main.py faults
python validate_hex_data.py
python main.py report
```

需要重新取得参考资料时：`python hex_source_fetch.py`，下载历史版本原文并保存状态与哈希到独立结果目录。第三方官方与安装源码的逐字节比对见third_party_version_check.json；来源与访问限制见REUSE_REGISTER.md。

当前`output_hex_audit/`保留实际正常/故障验证、小数据、两种特征及学习接口结果。其生成清单保存执行当时源码哈希。随后新增校验、适配器和报告入口改变了代码签名；因此再次生成应使用新目录，避免把不同签名续跑到旧来源：

```powershell
python main.py validate --out output_hex_audit_new
python main.py generate --out output_hex_audit_new --groups 2 --duration 6
$env:HEX_OUTPUT_DIR="$PWD/output_hex_audit_new/dataset_v4"
$env:DATA_PROCESS_DIR_OVERRIDE="$PWD/output_hex_audit_new/preprocessed/rotation_combined"
$env:FEATURE_SET="rotation_combined"
$env:HEX_WINDOW="50"
$env:HEX_STEP="50"
$env:HEX_FAULT_BUFFER="0.05"
$env:HEX_MIN_GROUPS_PER_CLASS="2"
python data_process.py
```

`combined`保留旧18列接口，故障窗口接近欧拉奇异点，不具正式特征有效性支持。`rotation_combined`使用观测R前两列、角速度、IMU与命令共21列；只改变观测表示，不读真值姿态。默认陀螺列与机体角速度有冗余。纯WGAN的生成旋转列仍未验证物理正交性；不静默投影修复或声称完整PINN。

## 小规模学习接口

本轮实际用2个独立配对run组、每类2个来源，38个50点窗口。两种特征均完成2折、1epoch，并保留无GAN、类别加权、普通过采样、fold-local WGAN四条路径。指标只用于接口核查；均衡数据下max补齐数量为0，没有证明WGAN样本增强改善。

当前数据`formal_protocol=false`，分类默认拒绝，接口运行必须显式声明：

```powershell
$env:EXPERIMENT_MODE="interface_smoke"
$env:FEATURE_SET="rotation_combined"
$env:DATA_DIR_OVERRIDE="$PWD/output_hex_audit/preprocessed/rotation_combined"
$env:OUTPUT_DIR_OVERRIDE="$PWD/output_hex_audit/interface_recheck"
$env:N_FOLDS="2"
$env:CLASSIFIER_EPOCHS="1"
$env:GAN_EPOCHS="1"
$env:GAN_N_CRITIC="1"
$env:GAN_PHYSICS_LAMBDAS="0"
python classify.py
```

本轮的完整小网络尺寸和执行输出在learning_smoke*.json、learning_checks*.txt；不要求上述命令使用同样尺寸才能判断接口。完整正式训练、实飞迁移、生成姿态物理验证与WGAN效果仍未完成。旧训练结果不迁作新模型结论。

## 运行范围

正常工况包含悬停、小角度/角速度恢复、步进与平滑跟踪、恒定扰动、饱和解除，以及含噪/延迟观测反馈。精确故障边界拆分积分；仿真时长要求对齐导出网格，故障/恢复时刻可非整步。旧参考应在`legacy_reference`目录的单独Python进程中运行，禁止把其全局兼容补丁导入新核心。

物理配置采用一致研究质量分布，推进/气动/观测仍未确认实机；局部质量与几何敏感性同时重算惯量。模型/控制器跟随改变后的名义设计，敏感性结果不能解释为未知参数失配的鲁棒性。原数据、旧输出及11条锁定实飞记录未改变；本轮不合并或推送。
