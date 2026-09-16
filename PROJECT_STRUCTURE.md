# eVTOL_Simulation 六旋翼实验结构

当前主线只包含六旋翼仿真、数据质量审查、分组预处理、Fold内GAN和分类评估。
轴承数据适配代码已移除；历史原始数据与历史结果未删除。

| 路径 | 作用 |
|---|---|
| `hex_physical_characteristics.py` | 仅飞机质量、惯量、几何、电机/旋翼和气动参数 |
| `hex_params.py` | 为RotorPy补充控制器/仿真字段的兼容适配器 |
| `scenario_config.py` | 19类标签、故障电机、严重度、效能因子的唯一配置源 |
| `control_allocation.py` | 常规有界加权最小二乘分配；禁止负推力/超转速且不读取故障真值补偿 |
| `main.py` | 每类默认4次独立仿真，导出带运行元数据与分配审计量的CSV |
| `validate_hex_data.py` | 训练前检查采样、完整性、安全包络、边界、逐时刻故障效能和四维扳手恒等式 |
| `data_process.py` | 匹配时段滑窗、按run_id配对分组、保存期望推力和机体预测推力目标 |
| `WGAN.py` | 折内条件WGAN-GP、动态推力物理残差、小批次生成 |
| `classify.py` | 分组4折、双scaler、多基线、GAN质量与安全指标 |
| `provenance.py` | SHA-256、稳定签名和JSON清单工具 |
| `plot_gan_losses.py` | 汇总正式实验各折/各GAN变体损失曲线 |
| `output_hex_v3/` | 有界名义分配版本的独立重复仿真CSV与生成清单；不会覆盖旧数据 |
| `plots/hex/data_validation/` | 数据质量报告 |
| `plots/hex/data_process/<feature_set>/` | 预处理数组、groups、来源和元数据 |
| `plots/hex/classification/<run>/` | 正式分组评估结果、模型、预测和运行清单 |
| `runs/` | 历史实验归档，当前代码不写入 |
| `docs/` | 项目说明和汇报材料 |

## 数据边界

- 旧 `output/`、旧 `plots/data_process/` 和旧分类结果只作历史记录，不再是默认输入。
- `轴承数据集/`、`plots/cwru/` 及相关历史日志没有被删除，因为它们是数据/结果，
  不是本次要求删除的配套代码；需要彻底清除时应单独确认。
- 正式分类只读 `X_raw.npy`，并强制使用 `groups.npy`；不存在随机窗口划分兜底。
