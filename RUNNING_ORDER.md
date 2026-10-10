# 2026-10-10动画回放

已有v5输出时，在项目根目录运行 `python view_hex_tasks.py` 即可打开完整任务动画。
详细操作、数据依据与验证见[动画回放说明](docs/hex_next/ANIMATION_REPLAY.md)。

# 2026-10-10任务库与连续诊断

最新两批工作以[docs/hex_next/RUNNING_ORDER.md](docs/hex_next/RUNNING_ORDER.md)为准。
新任务/数据/诊断接口已实现并检查；可靠报警仍未通过，详细误报、漏检见[实际报告](docs/hex_next/VALIDATION_REPORT.md)。
下列2026-10-06入口保留用于旧回归，不能代替新机动或连续诊断验证。

# 2026-10-06研究六旋翼运行入口

当前操作和版本边界以[docs/hex_audit/README.md](docs/hex_audit/README.md)为准。不要覆盖旧output_hex_v3或将旧模型训练结果移作新模型结论。

默认核心：

```powershell
python -m unittest discover -s tests -v
python main.py validate
python main.py faults
python validate_hex_data.py
python main.py report
```

38条v4来源和两折一轮学习只是接口检查，formal_protocol=false。正式分类默认拒绝这些数据；具体接口命令、21维旋转选项、旧18维欧拉风险、增广数量为0和生成旋转未通过正交约束均在新说明中记录。

旧运行说明和源码保存在Git基线ef5fa5b及legacy_reference；新代码签名改变时拒绝续跑旧清单，需先用新输出目录验证并生成。11条UAV-FD最终测试不解锁。
