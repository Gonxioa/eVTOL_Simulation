# V3：12维传感器输入与训练时长独立实验

基线：`f474282d07f9f86de24fcacb6f000b2e09205ef1`。
分支：`codex/simulink-gan-v3-prefix`。
配置：`configs/simulink_gan_v3_prefix.json`。
结果目录：`output_simulink_gan/v3_imu_rpm_prefix_20261105`。
中文结果、CSV、图表与冻结收据：`docs/validation/simulink_gan_v3_prefix_20261105`。

本实验复用现有 `classify.py`、`WGAN.py` 的网络与按优化更新次数训练的函数，复用 `simulink_data.observable_features` 的测量延迟对齐，以及 `replay_diagnostics` 的报警状态机和验证选择目标。V2文件和物理模型不修改。新模块使用独立签名与结果目录；不会读取旧权重或18维缓存作为12维模型。

## 环境与来源

本次实际运行使用 Windows、MATLAB R2025b、Python 3.11、PyTorch 2.5.1+cu121、RTX 4060 Laptop。Python依赖沿用项目环境：numpy、pandas、scipy、scikit-learn、torch、matplotlib。59项自动测试包括本轮新增12项。

先检查分支、未提交修改与原始数据清单。原训练6组、验证3组直接引用原CSV/MAT及哈希。旧测试3组已曝光，仅保留归档。新测试种子预先固定为20261115、20261116、20261117，每组19类，共57次新增仿真；不能试完后挑选表现更好的种子。

原始数据文件较大，未随代码重复提交。运行需要本机既有V2来源及本次新采集目录；原始来源位置和完整哈希见结果中的 `source_manifest.csv`。采集计划记录全部MATLAB模型与源文件哈希，任何模型变化都会使采集入口拒绝执行。

## 执行顺序

在项目根目录运行以下命令。已完成的任务使用签名匹配的优化器检查点续跑，不额外增加已完成任务的更新次数。

```powershell
# 创建专门的三条件组测试采集计划（不同于V1至少四组的入口）。
python run_simulink_v3.py plan

# 在独立MATLAB进程中采集；模型参数沿用原值，CSV/MAT始终24秒。
& 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe' -batch "addpath('simulink'); collect_hex_dataset('output_simulink_gan/v3_imu_rpm_prefix_20261105/raw_new_test3/collection_plan.json')"

# 测试与实际开发训练。
python -m unittest discover -s tests -v
python -u run_simulink_v3.py develop

# 只有全部51个分类器和6个GAN完成，并确认新采集完整，才能冻结。
python run_simulink_v3.py freeze

# 一次统一解锁新测试；权重、scaler、配置、源代码与来源签名均验证。
python -u run_simulink_v3.py test
python run_simulink_v3.py audit
python run_simulink_v3.py report
```

新数据采集已经完成时，可直接从开发训练继续。`all` 阶段按 develop → freeze → test → audit → report 顺序执行，不会重跑仿真。各阶段控制台日志保存在结果目录的 `logs` 子目录，MATLAB采集日志为 `collection_matlab.log`。

不可在既有结果目录中修改配置、采集计划或冻结文件来重选实验。新协议应使用新结果目录，并重新建立相应来源与缓存。

## 预先固定的比较

|编号|输入|实际可用训练区间|原始训练窗口|
|---|---|---|---:|
|A|IMU＋命令＋带噪延迟转速，18维|[0,24)秒|4818|
|B|IMU＋带噪延迟转速，12维|[0,24)秒|4818|
|C|同B|[0,12)秒|2082|
|D|同B|[0,6)秒|714|
|E|同B|从[0,24)秒逐来源等量抽样|714|

E使用独立抽样种子731905，逐条件组抽健康11窗、每条故障记录6窗；无放回抽样且scaler只拟合实际选中的714窗。

采样频率100 Hz，窗口长度100次采样，移动周期0.5秒。用整数样本索引实现前缀边界，再核验实际时间戳。健康记录全部保留；故障记录排除窗口末时刻早于3秒的重复健康前缀。第一条保留故障窗口是2.50～3.49秒。原始仿真不因训练截取而变短。

全部模型统一评价独立来源的前6／12／24秒窗口，并评价故障初期和12秒以后。固定19类argmax为主指标；只含故障的阶段另附“出现类别宏F1”，不能与固定19类宏F1混用。报警阈值下的单窗口指标单列CSV。

所有分类器固定1000次实际优化更新，种子42、43、44。仅在前6／24秒两个端点比较ROS与原类别条件12维WGAN-GP，每个GAN实际3000次G、15000次critic更新。候选增广比例0、0.25、0.5、1.0，逐类新增数量向下取整。

比例按共同完整24秒验证宏F1的三种子均值选择；保留8位小数后同分选较小比例，ROS与GAN分别选择，另保留与所选GAN逐类同量的ROS。训练时长推荐同样按共同完整24秒验证宏F1均值选择，同分选较短时长；规则写在采集和训练前的配置中。比例0表示回退原始数据。

## 报警与冻结

完整24秒因果回放使用100次采样窗口、0.1秒移动周期。每条记录从头初始化，故障时刻仅用于离线标签与评分。推理函数 `causal_window_predictions` 不接收故障时刻；报警不会在第3秒重置。故障前已经激活并持续的报警不能算作新检出。

每模型、每种子在验证集选择阈值及连续次数，保存完整网格。正常误报曝光只统计每组唯一健康来源，3组有效报警观测为69秒。检出、首次定位、最终正确定位分别评价；漏检不从覆盖率分母中删除。

`freeze.json`锁定全部51个分类器、6个生成器、所有scaler、比例、工作点、源代码与新测试来源。测试池、全记录特征缓存及预测缓存读取前均检查授权；新进程检查全部锁定文件哈希，进程内按文件大小和修改时间缓存已核验哈希。测试输出还带冻结签名和来源签名，不能跨协议复用。

## 结果文件

- `development_complete.json`：全部验证候选、实际训练任务、比例和训练时长选择。
- `classifiers/`、`generators/`：检查点、优化器状态、训练曲线、报警网格、质量诊断、正式预测。
- `scalers/*/fit_sources.json`：实际拟合窗口清单与数据哈希。
- `pools/`：特征、训练前缀、来源和代码均签名的窗口池。
- `freeze.json`、`test_access_receipt.json`：冻结与首次授权测试读取凭据。
- `final_test_results.json`：统一新测试结果；主argmax与报警阈值口径分开。
- `audit/`：逐来源数量、完整CSV/MAT哈希、实际Adam更新数、scaler拟合和缓存审计。
- `docs/validation/simulink_gan_v3_prefix_20261105/结果报告.md`：中文结论、限制和主要图表。
- 同目录的 `per_seed_metrics.csv`、`per_condition_metrics.csv`、`paired_differences.csv`、`alarm_per_record.csv` 等保存逐种子、逐条件组及逐记录结果，包括失败和异常。

训练窗口不是独立仿真记录，GAN样本也不增加独立扰动来源；不能将窗口数或三个训练种子当成额外独立飞行实验。
