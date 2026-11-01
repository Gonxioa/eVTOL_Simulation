# 当前设计与代码总览

2026-11-01 完成版本归档与当前文件命名整理。当前分支：`codex/current-model`；主运行模型：`simulink/hex_evtol.slx`；参数版本：`control_v3_20261023`。

日常入口与结果位置见 [Simulink 当前使用说明](simulink/README.md)。在 MATLAB 的 `simulink` 目录运行 `my_simulation`，或 `report=run_hex_healthy_assessment`。

[GitHub 全版本归档](https://github.com/Gonxioa/eVTOL_Simulation/releases/tag/archive-20261030) 保存整理前的所有代码分支、完整 Git bundle、历史及当前仿真输出。逐文件 SHA-256 与分卷映射见 `maintenance/archive_20261030/manifest.json`。
原始数据集、训练权重与个人文档未上传。

## 当前保留内容

|内容|位置|
|---|---|
|仿真模型和维护源文件|`simulink/hex_evtol.slx`、`simulink/blocks/`、`simulink/functions/`|
|参数、逐次仿真和四健康任务入口|`simulink/my_simulation.m`、`hex_parameters.m`、`run_hex_healthy_assessment.m`|
|当前 v3 仿真数据和图|`simulink/output/simulations/control_v3_20261023/`|
|当前恢复验收和整定证据|`simulink/validation/SIMULATION_SUMMARY.md`、`SIMULATION_TUNING.md`|
|当前增益选值展示材料|`docs/kx_gain_evidence/20261026/`|
|独立 UAV-FD/F550 模块|`uavfd_f550/`、`docs/uavfd_f550/`、本地 `output_uavfd_f550/`|
|现有 Python 诊断管线|`main.py` 及项目根目录的相关源文件|

当前控制系统按稳定姿态和高度、返回原几何轨迹、再继续任务的顺序控制。三类故障为单个电机分别剩余 80%、20%、严格 0% 效能，六台电机逐一测试。
当前 v3 的 106 个工况由 72 个核心故障、4 个健康、18 个理想反馈故障、12 个额外随机种子工况组成。全部飞行验收通过不代表实飞有效性或任意风、延迟、多电机故障保证。

源函数与 SLX 的严格逐电机命令对照未通过，状态及合力/力矩的行为对照通过。当前版本的限制与失败记录继续保留。
独立 F550 实飞组合模型仍未通过整体模型验证，不能与 Simulink 研究模型混为同一套已验证参数。

## 历史版本恢复

历史分支与 `archive-20261030` 标签保留在 GitHub；需要旧模型时使用独立目录检出该标签，不覆盖当前工作目录。
归档附件可按 `archive-manifest.json` 的路径恢复仿真文件，下载后先核对附件 SHA-256。当前内部布局模板是重建模型的依赖，不是另一套日常模型。

当前名称、数据字段和目录已统一；改名前后两个 24 秒原生模型工况的全部记录信号完全一致。迁移映射和重建验证见 `maintenance/naming_20261101/`。
