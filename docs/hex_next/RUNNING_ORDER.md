# Windows复现顺序：任务库和连续诊断

在`hex-tasks-continuous-diagnostics`分支的项目根目录执行，使用本地现有Python3.11。
依赖实际版本见`dependencies.lock.txt`。这些入口不需要原UAV-FD/RflyMAD数据。

已有完整v5输出时，直接执行 `python view_hex_tasks.py` 打开任务动画，不需要再次生成或训练。
播放、倍速、故障跳转和对照操作见[动画回放说明](ANIMATION_REPLAY.md)。

```powershell
python -m unittest discover -s tests -v
python -m hex_research.next_steps library
python -m hex_research.next_steps validate
python -m hex_research.next_steps generate
python -m hex_research.next_steps check
python -m hex_research.next_steps preprocess
python -m hex_research.next_steps interface-fit
python -m hex_research.next_steps replay
```

默认新输出`output_hex_next_20261009/batch2`；日期20261009是需求文件日期，执行日期另写入receipt。
预处理、模型权重和逐记录概率分别在`preprocessed/imu_only`、`diagnostics/imu_only`及其`predictions/`。
正常门禁结果`task_normal_validation.json`；152条来源的元信息在`dataset_v5/dataset_generation_manifest.json`。
源码/配置签名改变后必须使用新的`--out`目录，从正常验证重新执行，禁止重签旧生成记录。

旧回归入口`python main.py validate`和`python main.py faults`保留；建议传新的`--out`，不要覆盖旧输出。

## 同源扩展特征

已生成的v5时序数据支持三种白名单，分别输出，不能仅凭同为18列混用：

```powershell
python -m hex_research.next_steps preprocess --feature-set imu_attitude
python -m hex_research.next_steps preprocess --feature-set imu_attitude_command
```

根目录预处理兼容入口使用同一函数：

```powershell
$env:HEX_OUTPUT_DIR="$PWD/output_hex_next_20261009/batch2/dataset_v5"
$env:DATA_PROCESS_DIR_OVERRIDE="$PWD/output_hex_next_20261009/batch2/preprocessed/imu_only"
$env:FEATURE_SET="imu_only"
python data_process.py
```

`interface-fit`使用原LSTM主体的小网络尺寸、3轮开发拟合，目的为验证实际模型的连续输入和权重加载。
训练/回放配对group分离，缩放仅训练部分拟合；全部是开发记录，尚未固定正式小样本/GAN比较协议。
正式19类识别性能、阈值校准、未见任务家族泛化和扩增改善不能从接口检查宣称成立。

三轮原型的报警检查实际失败。保留该模型和所有预测；另一次30轮开发拟合复现为：

```powershell
python docs/hex_next/run_prototype.py
```

配置`PROTOTYPE_30_EPOCHS.json`，新模型目录`diagnostics/imu_only_epoch30`，不覆盖三轮结果。
已存在的原型目录保持只读；如需另一轮拟合，复制preset并改name。
后续仅训练/回放已有数据时使用独立模型输出，不需要重复未改变的飞行验证；
重新造数或续跑造数时才适用源码/生成计划签名门禁。

## 不含标签、参考或故障日程的独立诊断

输入列契约见DATA_CONTRACT；`imu_only`仅需时间、有效标志、schema及6列机体IMU。
模型接收观测，不要求输入故障时刻：

```powershell
python diagnose_hex.py --input output_hex_next_20261009/batch2/g004_c01_observation_only.csv --checkpoint output_hex_next_20261009/batch2/diagnostics/imu_only_epoch30/interface_model.pt --output output_hex_next_20261009/batch2/new_unlabelled_predictions.json
```

该观测样例由实际g004_c01来源删除全部参考、标签、任务和真值列得到。35次预测与原回放逐项一致。
此命令可以运行未标注的观测记录；输出模型仍为开发原型，可靠报警未通过。
