# Simulink GAN V2：固定来源与分阶段对照

基线为 `6f94612`，工作分支 `codex/simulink-gan-v2`。保留六旋翼模型、物理参数、PID、故障强度、飞行验收门槛及原 LSTM／卷积网络骨架。V1 的原始数据、权重、报告和四折流程保留。

## 来源和固定用途

配置：`configs/simulink_gan_v2.json`。根目录：`output_simulink_gan/v2_12groups/`。

|用途|全局配对组|条件组数|仿真记录数|
|---|---|---:|---:|
|训练|0–5|6|114|
|验证|6–8|3|57|
|最终测试|9–11|3|57|

原 V1 四组通过 `source_registry.json` 引用原文件及 SHA-256，不复制或重写源文件。新增八组完成 152 次原生仿真；新组中的前两组用于训练、中间三组用于验证、最后三组用于测试。共 228 条仿真记录，但统计独立来源是 **12 个配对条件组**。随机种子编号不是采集日期。

仍是绕圈任务、每次 24 秒、3 秒故障注入、原风和噪声强度、30 ms 反馈延迟、原容错控制。没有同时改变任务、物理参数或故障机制。所有完整有限且来源可信的记录保留，飞行恢复验收失败另行标记，不删除困难电机类别。

原生记录仍使用 V1 的已验证导出信号契约；V2 registry 在其上增加固定用途与访问控制。表内记录数不能用作独立统计样本数；训练随机种子重复也不增加独立飞行来源。

## 特征与窗口

`imu`、`imu_cmd`、`imu_rpm`、`imu_cmd_rpm` 分别为 6、12、12、18 维；默认最后一种。IMU 为带噪且按延迟对齐的传感器系比力和角速度；转速用带噪、延迟后的 `feedback_state_14..19`，指令是当前控制时刻的值。单位保持 m/s²、rad/s。真实状态、真实效能、标签、种子、任务编号和故障时刻不进入网络输入。

每个窗口 100 点，步长 50 点。健康来源覆盖全时段；故障窗口按末时刻是否已经到达注入时刻标注。跨界和故障初期单独标记 `transition`，稳定故障段标记 `post_fault`。故障记录中重复的故障前健康窗口不再作为十八份健康数据加入；每组只用一条完整健康来源提供健康窗口。

训练、验证、测试分别有 4818、2409、2409 个窗口，包括 324、162、162 个过渡窗口。窗口高度相关，不能当作独立飞行。StandardScaler 和 MinMaxScaler 仅拟合六个训练组，GAN 的指令条件也仅抽自训练组。

标签仍是 Normal=1.0、Partial=0.8、Severe=0.2、Full=严格 0.0 的剩余推力效能。动力学为 `omega_dot=(sqrt(eta)*command-omega)/tau`；推力由实际机械转速平方计算。Python 0–5 显式对应 Simulink 1–6；没有建立与 UAV-FD 实机位置的对应关系。

## 实施阶段与选择规则

1. **相同步数基线。** 原分类器保留第 150 步短预算对照；所有长预算对照统一 1000 次优化器更新，保存 150、250、500、1000 步状态和验证指标。采用相同学习率、批量大小、初始化种子与步数，不再用相同 epoch 代替相同更新量。
2. **检测头对照。** 保留 LSTM 编码器与 19 类头，新增可选单 logit 检测头。损失 `CE19 + 0.5 × balanced_BCE`，正常与所有故障的检测损失总权重相等。先在验证组比较，原单头始终保留。
3. **生成器预算和条件结构。** 两种 GAN 均实际训练到 3000 次有效 G 更新，每次 G 更新对应五次 critic 更新，保存 500、1500、3000 步。指令条件版本令 G 接收噪声、类别、真实训练指令，输出 IMU 与转速响应；critic 同时接收类别和相同指令，梯度惩罚只作用于可变响应。组合回 18 维输入时指令列保持源窗口原值。
4. **验证选择。** 共同 G 预算按两种 GAN 的验证 TSTR 宏 F1 均值选择，损失值不参与模型选择。比例候选为 0、0.25、0.5、1.0，按验证宏 F1 选择，报警目标和更小比例用于破同分。
5. **同量与最优 ROS 都保留。** 除验证选择的最优随机过采样外，还保留分别与原 GAN、指令条件 GAN 所选比例相同的 ROS 控制。所有分类器均用固定的同一头设计和 1000 次更新。比例 0 不加入生成样本。
6. **冻结后统一测试。** 选定配置、代码哈希、来源划分、scaler 哈希和全部验证工作点写入不可变 `frozen_configuration.json`。冻结后禁止重新运行开发选择；三个训练种子 42、43、44 的全部模型完成后，才开启测试特征，统一评价。

检测头预设采用门槛：阈值 0.5 下验证二分类平衡错误率至少改善 0.005，且宏 F1 不下降超过 0.05。本轮它提高了 19 类分数，但没有达到检测指标改善门槛，因此主增广比较保留单头；双头候选仍进入独立最终对照，不能将它从结果中删除。

验证选择的共同 G 预算为 3000；原 GAN 比例 0.25、指令条件 GAN 比例 0.5、最优 ROS 比例 0.25。最终三个训练种子使用同一套选择，不根据测试重新调整。验证组经过多候选开发，泛化结论以未用于选择的最终测试为准。

## 质量检查

每类每通道记录标准化 Wasserstein、KS 统计量、均值/标准差和频谱差异。按电机和类别检查 0–0.3 秒的指令—转速滞后曲线，记录已知电机时间常数 0.05 秒、反馈延迟 0.03 秒。几乎恒定的指令/转速通道单独标记，不强求高相关。

保留精确重复、训练最近邻、以跨训练来源自然相似度为参考的近重复检查。给定指令的预期复用不能算成响应记忆；条件版本的响应最近邻检查排除给定指令列。响应质量比较也应单独查看 IMU/RPM，不能把“指令直接复制”当成学会指令生成。

TSTR 质量分类器也使用 1000 次优化更新；质量样本每类 32 个，独立于主分类增广样本，不加入主训练数量。范围检查基于训练观测范围，KS、频谱和最近邻均为描述性指标，不据相关窗口计算虚假的独立样本显著性。物理正则保持关闭；指令条件不等于完整动力学约束。

## 报警和机理基线

单头使用全部故障类别概率之和，双头使用检测头输出。持续判故障不要求同一个细分类别连续出现；电机定位使用单独的持续时钟。

阈值候选 0.1–0.9，持续次数 2、3、5，全部由验证组选择。选择目标预先设为：

`漏检比例 + 0.05 × 正常误报事件数/分钟 + 0.05 × 检出条件平均延迟/24`。

报告正常误报**事件数与观测总时长**、漏检、检出延迟分布、首次定位正确率、首次正确定位延迟和最终正确定位覆盖率。检测与定位分别计时；故障前已经存在的报警不计作成功发现新故障。正常观测只统计每组完整健康来源一次，重复故障前前缀不增加分母。

规则基线使用当前指令、转速反馈、已知一阶电机响应和反馈延迟估计跟踪残差，并将速度比例平方换为效能估计。它不读取真实效能、故障电机、标签或注入时刻；注入时刻仅用于评价。规则具有与当前仿真匹配的已知模型优势，不能外推为实飞能力。它和神经诊断均未接入或替换现有飞控容错估计器。

## 运行与续跑

```powershell
python run_simulink_gan.py collect --config configs/simulink_gan_v2.json
python run_simulink_gan.py baselines --config configs/simulink_gan_v2.json
python run_simulink_gan.py heads --config configs/simulink_gan_v2.json
python run_simulink_gan.py gan_train --config configs/simulink_gan_v2.json
python run_simulink_gan.py gans --config configs/simulink_gan_v2.json
python run_simulink_gan.py freeze --config configs/simulink_gan_v2.json
python run_simulink_gan.py final_train --config configs/simulink_gan_v2.json
python run_simulink_gan.py test --config configs/simulink_gan_v2.json
python audit_simulink_v2.py
python maintenance/report_simulink_v2.py
```

已有完整原生数据后，可用 `all --config configs/simulink_gan_v2.json` 依次执行开发、冻结、最终训练和测试。若已冻结，`all` 会跳过开发选择，恢复最终训练并复用一致的最终测试收据。训练检查点含模型、优化器、步数、采样器及随机状态；代码、数据或配置不兼容会被拒绝或进入新的签名作业目录，不覆盖 V1。

```powershell
python -m unittest discover -s tests -v
```

`classifiers/`、`generators/` 保存签名作业和检查点；`scalers/` 保存训练来源清单；`quality/` 保存六个 GAN 检查点的质量样本和指标；`development/` 保存所有候选与选择；`final_training_complete.json` 为全部模型完成门禁；`final_test_results.json`、`test_predictions/` 为统一评价；`feature_access.jsonl` 记录数值来源读取授权；`behavioral_audit.json` 为实际产物审计。

新实验需使用新的 `results_root`。本轮始终只使用仿真域数据，UAV-FD 最终测试集保持封存。跨任务、参数失配、真实桨叶故障和真实域迁移留作独立后续实验。

方法参考：[WGAN-GP](https://papers.nips.cc/paper_files/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html) 使用 critic 输入梯度惩罚改善训练稳定性；[Conditional GAN](https://arxiv.org/abs/1411.1784) 将条件提供给生成器和判别器。本项目的增益由独立实验判断。温度缩放可作为后续校准对照，参见 [Guo 等，2017](https://proceedings.mlr.press/v70/guo17a.html)，本轮未将它当作分类能力改善的替代。
