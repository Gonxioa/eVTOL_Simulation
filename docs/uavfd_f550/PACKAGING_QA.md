# 交付补丁复核

补丁以UTF-8／LF字节输出。首次使用Windows文本默认换行写入时，逆向只检查出现CRLF／LF匹配差异；已修正补丁输出方式。

实际执行：

```powershell
git apply --check --reverse --ignore-space-change output_uavfd_f550/uavfd_f550_changes.patch
```

退出0。该命令仅检查，没有应用或撤销补丁；忽略空白差异用于当前工作区的混合行尾。打包脚本再次核对6项原有修改的SHA256，均一致。

补丁包含独立包、维护文档、根目录任务文件和README／PROJECT_HANDOFF新增入口说明。最终内容清单见CHANGE_MANIFEST.md；数值执行日志见checks.txt。
