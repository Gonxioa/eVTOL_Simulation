# 2026-10-30 版本归档与本地整理

归档从 2026-10-30 启动，2026-11-01 完成整理。

用户要求先把所有代码版本上传 GitHub，再清理本地旧代码和旧仿真结果，只保留当前版本。

[GitHub 归档发布](https://github.com/Gonxioa/eVTOL_Simulation/releases/tag/archive-20261030)

## 远端保留

- 整理前所有 7 个本地代码分支均推送；最近未提交的健康评估和中文绘图代码单独提交。
- `archive-20261030` 标签指向清理前的完整代码状态。
- `all-code-history.bundle` 保存所有 Git 引用和历史，便于独立恢复。
- 16 个独立 ZIP 附件保存 3,818 个仿真/历史工作目录文件，未压缩约 4.11 GB，上传附件约 2.01 GB。
- `archive-manifest.json` 记录逐文件 SHA-256、附件名和恢复路径。所有附件以 GitHub 返回的 SHA-256 核对，完成后再次查询服务器。

原始数据集、训练权重、个人文档与可重建编译缓存不包含在公开归档中。旧结果内夹带的 84 个训练检查点/序列化模型文件已逐文件核对后保留到本地 `模型/retained_training_artifacts_20261030/`，并从公开 ZIP 中排除。

## 本地保留

当前运行模型 `simulink/hex_recovery.slx`、当前维护源文件、当前 v3 的故障/健康/理想/种子验收数据、最新四健康任务图和日常示例结果。
当前逐环整定、增益比较与研究限制证据保留；原始数据、训练权重、独立 F550 模块和文档材料保留。
`simulink/internal/recovery_layout_template.slx` 是生成器依赖，不是另一个日常入口。

## 恢复历史

代码：在另一个空目录检出归档标签，或执行 `git clone all-code-history.bundle restored-repository`。
数据：下载需要的 ZIP，先核对附件 SHA-256，再解压；路径前缀 `project/` 表示原主项目，`old_worktree/` 表示原历史工作目录。每个 ZIP 独立可读，不需要先拼接。
清单可用于定位某个 `result.mat`、`signals.csv` 或旧模型备份所在分卷。

## 核验记录

- `manifest.json`：源文件和分卷 SHA-256。
- `verified_assets.json`：远端附件地址、服务器 SHA-256 和完成状态。
- `remote_refs_verified.json`：删除旧本地分支前的远端分支核对。
- `latest_layout_validation.json`：隔离副本中的实际 MATLAB 重建、24 秒状态/命令一致性、9 项观测器检查及默认模型回读。
- `deletion_preflight.json`：逐文件校验通过后的精确清理清单。
- `deleted_targets.json`：实际删除记录。

整理不改变飞行器参数、控制增益和主目录已保存恢复模型的文件字节。
