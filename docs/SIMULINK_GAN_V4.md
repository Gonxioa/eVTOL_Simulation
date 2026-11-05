# V4证据研究：执行顺序与边界

基线：`2e033334637be102e9a28c7ce0955d1f7d1f189f`。
独立分支：`codex/simulink-gan-v4-evidence`。
登记配置：[批次A及共同边界](../configs/simulink_gan_v4_evidence.json)、[增广与检查点规则](../configs/simulink_gan_v4_augmentation.json)。
完整评审修订版：[执行计划](SIMULINK_GAN_V4_EXECUTION_PLAN.md)。

结果根目录为 `output_simulink_gan/v4_evidence_20261105`；该日期是登记版本编号。
V1/V2/V3源码、数据、权重、scaler及冻结收据保持原位只读。

## 实际完成的批次A

171条原有开发记录完成真实信号审计、统计及配对分析。已实际运行46条神经训练轨迹、6个统计特征随机森林。
共同末点初筛四个窗长；最多两个窗口在1000/2000/4000更新预算下比较LSTM与固定CNN，再从0秒开始的原生网格联合复核。
三个训练前缀使用相同完整24秒验证；等量控制采用三个抽样种子与三个训练种子交叉，均只拟合实际选中的训练窗口。

本轮神经网络研究参考选择为1秒窗口、固定CNN、4000次更新、前12秒真实训练数据，2082个窗口。
这不表示GAN最佳训练范围或系统验收。统计特征基线在部分比较中更强，完整保留；4000次更新也不证明优化充分。
具体证据：[批次A结果](validation/simulink_gan_v4_20261105/批次A结果报告.md)。

## 顺序执行

在项目根目录，使用现有Python/PyTorch环境运行：

```powershell
python run_simulink_v4.py prepare
python -m unittest discover -s tests -v
python run_simulink_v4.py audit
python run_simulink_v4.py batch_a
python run_simulink_v4.py report_a
python run_simulink_v4.py batch_b
python run_simulink_v4.py confirm_plan

# 仅在confirm_plan已完成并固定组数和种子后，用独立MATLAB进程采集。
& 'C:\Program Files\MATLAB\R2025b\bin\matlab.exe' -batch "addpath('simulink'); collect_hex_dataset('output_simulink_gan/v4_evidence_20261105/raw_confirmation/collection_plan.json')"

python run_simulink_v4.py freeze
python run_simulink_v4.py confirm_test
python run_simulink_v4.py final_audit
python run_simulink_v4.py report
```

已完成的相同签名任务从原优化器状态续跑，不增加重复计数。改动训练代码、特征或网格会改变签名；冻结后禁止再运行开发选择。
审计、报告和测试具有各自阶段名；写出入口不等于对应阶段已经执行。实际状态应查看结果根目录的 `batch_A_complete.json`、`batch_B_complete.json`、`freeze.json`、`final_confirmation_results.json`、`audit/final_audit.json`。

## 重要约定

- 主输入严格为IMU6路和带噪延迟转速6路，复用V3对齐；不读取命令、真值、绝对时间或来源元数据作为输入。
- 共同末点比较为4794训练窗、2397验证窗；之后原生网格从0秒重建，分别签名，不能混用V3或共同网格缓存。
- 主指标为固定19类argmax宏F1。阶段指标另列实际支持类别宏F1，报警工作点指标单列。
- 神经更新预算1000/2000/4000，固定CNN只有一个结构；不据测试成绩追加网络或预算。
- G1只通过最低非塌缩门槛后作为探索执行，教师不是真值；G2和筛选首轮关闭。
- G0/G1保持原WGAN网络核心，新增隔离的损失分解、固定探针、采样索引及外部教师约束实现；旧WGAN.py和V3行为不改。
- G0检查点选择用独立TSTR的真实验证效果，G1共用同一检查点；全部GAN仍实际训练到3000次G和15000次critic更新。
- 增广比例逐类向下取整，主分类器同预算、同初始化种子；ROS匹配实际加入的逐类数量。
- 新确认只用全新风噪种子，组数按开发阶段预登记的有限规划规则在6～12组内选择；不在看见测试成绩后追加来源。
- 新测试特征读取及缓存访问均在冻结授权之后。历史测试仅保留为已曝光来源，不能作为本轮未见确认。
- 所有窗口识别、连续报警、生成质量和飞行恢复判定各自说明用途；不把测试通过数或教师一致率替代真实识别效果。

完整权重、CSV/MAT、生成银行、优化器、采样索引和日志留在独立结果目录；紧凑审阅证据保存在 `docs/validation/simulink_gan_v4_20261105`。
