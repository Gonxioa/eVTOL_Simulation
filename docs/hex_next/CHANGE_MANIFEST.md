# 本轮改动清单

| 模块 | 修改和用途 |
|---|---|
| `hex_research/commands.py`、`configs/continuous.json` | 四类解析平滑任务、参数范围、5路独立种子及明确故障日程；设计参数均有台账 |
| `hex_research/simulation.py` | 控制任务时钟采样/缓存；传感器时钟诊断回调；保存控制参考、初始状态及随机流；保留旧调用 |
| `hex_research/observation.py` | 增加明确FLU机体系观测比力/陀螺字段；既有传感器计算保留 |
| `hex_research/continuous_data.py`、`next_steps.py` | 新正常门禁、152来源配对v5数据、完整性/前缀审查和Windows入口 |
| `diagnostics/features.py`、`continuous.py` | 类型化只读观测包、6/12/18维共享白名单、因果缓冲、报警、回放及离线阶段标签 |
| `diagnostics/preprocessing.py` | 完整连续窗口、纯窗口训练候选、健康去重、窗口继承来源；不全数据拟合scaler |
| `diagnostics/models.py`、`workflow.py` | 保留原LSTM主体，显式模型尺寸、训练group专用缩放、权重加载及实际回放指标 |
| `data_process.py`、`hex_research/data.py` | 新旧schema分流；根预处理共用特征与窗口函数 |
| `classify.py` | 共用LSTM主体；拒绝把v5误送入旧固定故障后区间的正式比较流程 |
| `diagnose_hex.py` | 只读取观测CSV，不读取标签/参考/故障日程的独立诊断命令 |
| `tests/test_continuous_tasks.py` | 导数、时钟、输入隔离、因果/缺样、配对前缀、实际CSV/回调及元信息泄漏检查 |
| `docs/hex_next/` | 请求原文、两批日志、配置/台账、源码执行快照、机器指标、图、逐类/记录失败及交接 |

动力学核心、控制器增益、有界分配算法、WGAN主体、UAV-FD包和旧参考保留。
数据与模型权重按现有规则保留在本地输出，不把大型生成数据提交Git；可按配置复现。
原11条UAV-FD最终记录未解锁，保护检查详见`protected_final_check.json`。
