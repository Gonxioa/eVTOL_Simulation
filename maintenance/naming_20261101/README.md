# 当前模型命名整理

2026-11-01，按用户要求将当前使用的模型、源码、函数、验证入口、日志字段和结果目录统一为普通名称。日常运行不需要执行本目录脚本。

|用途|当前名称|
|---|---|
|框图模型|`simulink/hex_evtol.slx`|
|一键仿真、绘图、保存|`simulink/my_simulation.m`|
|当前参数工厂|`simulink/hex_parameters.m`|
|基础参数与公共模块子集|`hex_base_parameters.m`、`hex_common_block_parameters.m`|
|打开、配置、运行、重建|`open_hex_simulink.m`、`configure_hex_simulink.m`、`run_hex_simulink.m`、`build_hex_model.m`|
|控制器与参考进度|`simulink/blocks/controller_step.m`、`reference_step.m`|
|当前结果|`simulink/output/simulations/control_v3_20261023/`|
|验证报告|`simulink/validation/SIMULATION_*.json`|

完整文件、函数和字段映射见 `mapping.json`。参数结构中的控制策略使用 `P.control.fault_response`，参考审计日志使用 `reference_audit`；恢复时间指标使用 `return_time_s`，定义不变。

## 验证与数据保留

`verification.json` 记录了 MATLAB R2025b 原生运行结果：

- 健康和 1 号电机剩余 20% 效能的绕圈工况各运行 24 秒，均启用风、测量噪声、观测反馈和 30 ms 延迟。
- 两组参数与改名前一致；全部记录信号逐项比较，最大绝对差均为 0。
- 九项观测器检查、已保存模型与源代码核对、从内部模板重建均通过。
- 四个健康任务的已有结果成功读取并重新绘图；输出八张标准图和 24 张健康评估图。
- 新分析入口重新核对了 106 组已有日志中的故障效能、参考轨迹、反馈延迟和风输入，全部通过。

`data_migration.json` 记录了 138 个 MAT 文件的迁移检查，修改字段和文字时逐项断言数值数组不变。`text_data_migration.json` 记录了 117 个 CSV 的数据行校验和及 JSON 迁移。原 106 个验收工况保留，本次没有重新运行全部 106 个工况。

原始版本为 `source_commit.txt` 中的提交；其源代码、校验清单和原始运行报告可在 [命名前版本](https://github.com/Gonxioa/eVTOL_Simulation/tree/5d54456dbea0207adb36c5ad6568a51e6e481470) 核对。当前校验清单已按新文件重算。历史交接、归档记录和三个早期/失败报告保留原文，避免修改当时的证据。

## 维护脚本

`record_before.m`、`migrate_model_and_results.m`、`verify_after.m` 是本次迁移的执行记录，使用临时目录中的改名前后副本。`rename_sources.py`、`move_results.ps1` 和文本转换脚本是一次性操作，已经完成，不能在当前工作目录重复执行。

`finalize_naming.py` 更新当前校验清单并检查名称、报告和文件结构，不运行仿真、不删除文件。
