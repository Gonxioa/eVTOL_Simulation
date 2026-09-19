# 六旋翼正式实验运行顺序（PyCharm或PowerShell）

不要再使用旧 `output/` 和 `plots/data_process/` 跑正式结果。新版默认路径彼此隔离，
并且分类器在缺少 `groups.npy` 时会拒绝运行。

## 0. 代码门禁（当前只执行这一步）

`fix/simulation-physics` 第一批已修 IMU 真值坐标系和基础正弦风幅值，但故障时间边界
P0-03 及新数据版本迁移 P1-04 尚未完成。下方第 1–4 步仍是历史正式流程，**现在不要
运行 `main.py` 生成正式数据，也不要向旧 `output_hex_v3` 续跑**。仅运行不产生正式数据
的静态编译、单元测试及隔离的短时仿真：

```powershell
python -m py_compile .\control_allocation.py .\hex_params.py .\main.py .\validate_hex_data.py .\data_process.py .\WGAN.py .\classify.py
python -m unittest discover -s .\tests -v
```

只有全部通过，且 P0-03、P1-04 完成并经审查后，才更新下方目录及版本并进入正式
数据生成。短时测试的通过不能替代这两个门禁。

## 1. 生成独立重复仿真

默认19类、每类4次，共76个CSV：

```powershell
$env:HEX_REPEATS="4"
$env:HEX_OUTPUT_DIR="$PWD\output_hex_v3"
python -u .\main.py
```

每个CSV只代表一个场景和一个标签，文件名如
`motor2_partial__run003.csv`。同一`run_id`的19类共享轨迹与初始扰动并作为一个
配对组；交叉验证会整组留出，风和IMU随机流仍按场景使用独立seed。

## 2. 先做数据质量审查

```powershell
python -u .\validate_hex_data.py
```

先查看 `plots/hex/data_validation/validation_report.html`。存在 `FAIL`、类别缺失或
每类少于4次独立运行时，脚本会退出，不能进入正式预处理。

## 3. 预处理

默认联合输入：状态6维 + 带噪IMU 6维 + 电机指令6维，共18维；窗口100点，步长50点。
所有类别使用相同的故障后匹配时段，避免飞行轨迹阶段被模型当成类别特征。

```powershell
$env:FEATURE_SET="combined"
python -u .\data_process.py
```

输出到 `plots/hex/data_process/combined/`，其中正式训练必需文件为：

- `X_raw.npy`、`y_all.npy`
- `groups.npy`（每个窗口所属配对`run_id`；同组19类不会跨折拆分）
- `cmd_thrust.npy`（SE3控制器请求值，仅用于审计）
- `physics_thrust.npy`（故障效能和边界作用后的预测推力，可选推力正则目标）
- `feature_names.json`、`source_manifest.json`、`preprocess_metadata.json`

`X_all.npy/scaler.pkl`仅为兼容和查看，正式交叉验证不会使用全量scaler。

## 4. 正式4折实验

默认一次比较：无增广、类别加权、随机过采样和普通WGAN。命令—推力一致性正则
尚未把目标轨迹作为生成器条件，只作为可选实验项，不能称为完整的物理信息GAN。

```powershell
$env:DATA_DIR_OVERRIDE="$PWD\plots\hex\data_process\combined"
$env:OUTPUT_DIR_OVERRIDE="$PWD\plots\hex\classification\formal_combined_run1"
$env:N_FOLDS="4"
$env:CLASSIFIER_EPOCHS="100"
$env:GAN_EPOCHS="600"
$env:GAN_BATCH_SIZE="64"
$env:GAN_GENERATE_BATCH_SIZE="64"
$env:GAN_EARLY_STOPPING="0"
$env:GAN_RESTORE_BEST="0"
$env:GAN_PHYSICS_LAMBDAS="0"
$env:GAN_TARGET_POLICY="max"
$env:CLASSIFIER_RESUME="1"
python -u .\classify.py
```

GAN只补到训练折内最大类别数，不再固定给18个故障类各加500条。GAN使用折内
MinMaxScaler(-1,1)，分类器使用独立的折内StandardScaler；生成样本先还原到原始
量纲，再转换到分类器尺度。

中断后使用相同环境变量和输出目录重新运行即可。代码、数据、划分或超参数改变时，
SHA-256签名会使不兼容的断点失效。

## 5. 特征消融

分别重新执行步骤3和4，`FEATURE_SET`可取：

- `imu`：仅带噪加速度和角速度6维；
- `state`：姿态角和机体系角速度6维；
- `command`：仅6路电机指令；
- `response`：状态+IMU，共12维；
- `combined`：状态+IMU+指令，共18维。

每种特征方案必须使用独立输出目录。若以后显式试验非零推力正则，它需要全部
电机指令列；不含这些列时，代码会自动跳过非零lambda。

## 6. 损失曲线

保持与步骤4相同的输出目录环境变量，然后运行：

```powershell
python .\plot_gan_losses.py
```

## 排错模式的边界

可以临时把epoch改为1以检查接口，但1轮训练、少于4折或不足4次独立仿真均不能
作为实验结果。正式结果必须保留4折、独立来源分组、折内缩放和完整训练。
