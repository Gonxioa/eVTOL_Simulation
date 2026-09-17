# eVTOL_Simulation — Project Handoff

> 用途：给 ChatGPT Work、Codex、Claude 或新的协作者快速恢复项目上下文。  
> 仓库：`Gonxioa/eVTOL_Simulation`  
> 默认分支：`main`  
> 参考代码基线：`f96be8c` (`Initial project import`)  
> 说明：本文件为项目交接文档；后续若仅更新本文件，不代表仿真/训练代码基线发生变化。

---

## 1. 项目目标

本项目面向六旋翼 eVTOL / 多旋翼平台的故障诊断与 AI 可信性验证，核心任务是：

1. 基于 RotorPy 2.1.2 构建六旋翼动力学仿真与故障注入环境；
2. 生成可追溯、可重复、按独立仿真来源分组的数据集；
3. 构建 19 类故障识别任务；
4. 使用 LSTM 分类器、fold-local 条件 WGAN-GP 数据增强与分组交叉验证；
5. 为后续 AI 适航/可信度研究保留完整的数据来源、配置、代码哈希和验证记录。

当前重点不是继续扩展模型，而是先把**仿真物理链路和数据定义修正、固定并重新验证**。

---

## 2. 当前 19 类故障定义

场景由 `scenario_config.py` 统一定义：

- `Normal`：1 类；
- 6 个电机 × 3 个严重度：18 类；
- 合计 19 类。

严重度名义定义：

- `Full`：名义 effectiveness = `0.0`；
- `Severe`：effectiveness = `0.2`；
- `Partial`：effectiveness = `0.8`。

### 重要例外

当前代码存在：

```python
FULL_FAULT_FACTOR_OVERRIDES = {4: 0.05, 5: 0.05}
```

因此：

- M0–M3 的 `Full` 实际为 0%；
- M4/M5 的 `Full` 实际为 5% 残余效能。

这是有意保留稳定故障后数据的折中，不是隐藏 bug；但它会造成不同电机的 `Full` 类物理语义不完全一致。后续必须明确决定：

1. 保留 5% 并在论文/数据说明中明确写为“近完全失效”；或
2. 重新设计 M4/M5 的 0% 完全失效数据采集/截窗策略。

在作出决定前，不要擅自改标签含义。

---

## 3. 主要代码结构

### 仿真与数据生成

- `main.py`  
  总控：RotorPy 兼容补丁、SE3 控制、故障注入、风场、IMU 后处理、单次/批量运行、CSV 导出、续跑完整性检查、数据集 manifest。

- `control_allocation.py`  
  六旋翼 4×N bounded control allocation；控制器使用健康标称分配矩阵，故障 effectiveness 只作用于 plant，不作为控制器 oracle 信息。

- `hex_physical_characteristics.py`  
  六旋翼物理参数的单一事实来源。

- `hex_params.py`  
  将物理参数适配为 RotorPy/控制器所需参数，同时加入 SE3 增益、allocator 参数等非物理配置。

- `scenario_config.py`  
  19 类场景、标签、严重度、effectiveness 定义的单一事实来源。

- `provenance.py`  
  代码哈希、文件 SHA256、稳定哈希、运行时信息与 JSON 写出，用于全链路可追溯。

### 数据验证与预处理

- `validate_hex_data.py`  
  独立重新推导 allocation matrix / effectiveness 等关系，对 CSV 做正交校验；不直接信任生成侧计算结果。

- `data_process.py`  
  以独立仿真来源为 group，截取故障后稳定区间，构造滑窗；正式 CV 使用 raw windows，scaling 在 fold 内完成。

### 生成模型与分类

- `WGAN.py`  
  fold-local conditional WGAN-GP；生成器输出 `Tanh`，GAN 侧使用 fold-local `MinMaxScaler[-1,1]`；可选物理正则。

- `classify.py`  
  正式 19 类评估；`StratifiedGroupKFold`；训练折内 scaler / GAN；比较多种训练策略并输出安全相关指标。

### 测试与文档

优先关注控制分配相关测试、运行顺序说明、项目结构说明及 README。新 agent 在修改代码前应先阅读这些文件（若存在）：

- `test_control_allocation.py`
- `test_controller_allocation_integration.py`
- `README.md`
- `PROJECT_STRUCTURE.md`
- `RUNNING_ORDER.md`

---

## 4. 当前仿真主链路

核心数据生成逻辑：

```text
scenario_config
    ↓
build_scenario()
    ├─ FaultAwareSE3Control
    ├─ FaultInjectionMultirotor / Multirotor
    ├─ TwoDLissajous
    ├─ TurbulentWind
    └─ World
    ↓
RotorPy Environment.run()
    ↓
state / control / plant response
    ↓
IMU generation
    ↓
CSV export
    ↓
validate_hex_data.py
    ↓
data_process.py
    ↓
WGAN.py / classify.py
```

设计原则：

- 控制器**不知道真实故障标签/效能**；
- 故障只在 plant 侧生效；
- 控制器输出与 plant 实际响应分离；
- 目的是保留真实 fault signature，避免控制器提前“知道答案”并掩盖故障特征。

---

## 5. 当前已确认的高优先级问题

### P0-1：IMU 坐标系/真值来源问题

当前 `main.py::generate_imu_data()`：

- 使用世界坐标系速度 `state['v']` 做差分；
- 直接计算 `dv/dt + g`；
- 然后将其作为加速度计真值传给 `RealisticIMU`；
- 同时陀螺真值直接使用 `state['w']`（机体系角速度）。

因此当前 6 维 IMU 存在**加速度世界系、角速度机体系的坐标系混用风险**。

#### 推荐修复方向

优先核对 RotorPy 2.1.2 当前 `Environment/simulate/Imu` 的 `imu_gt` 定义。如果 `imu_gt` 已提供无噪声、机体系 accelerometer / gyroscope ground truth，则优先：

```text
RotorPy imu_gt
    ↓
RealisticIMU
    ↓
叠加项目自定义白噪声 + bias random walk + vibration
```

不要在没有核对 RotorPy 2.1.2 源码/接口的情况下凭记忆实现坐标变换。

#### 数据影响

该问题会改变训练数据的物理含义。修复后，旧 `output_hex_v3` 不应与新数据混合作为同一正式数据版本。

---

### P0-2：`TurbulentWind.base_amplitude` 当前未生效

当前逻辑：

```python
self.base = SinusoidWind()
self.amp = base_amplitude
```

但 `self.amp` 没有进入 `update()` 或传给 `SinusoidWind`。

因此 `build_scenario()` 中传入的 `base_amplitude=0.5` 目前是死参数，实际基础正弦风由 RotorPy `SinusoidWind` 默认配置决定。

#### 修复要求

- 对照本机 RotorPy 2.1.2 的 `SinusoidWind.__init__` 签名；
- 让 `base_amplitude` 真正接入基础风场；
- 不要直接照抄未经核对的参数名；
- 删除/收窄不必要的 `except TypeError` fallback，避免吞掉真实内部错误。

该问题同样属于正式数据生成前必须修复的问题。

---

### P0/P1-3：M4/M5 `Full` = 5% 的语义决策

见第 2 节。此问题不是代码崩溃 bug，而是**数据集类别定义和论文解释问题**。

在决定前：

- 不要私自把 0.05 改成 0；
- 不要把 M4/M5 5% 仍表述为严格 0% 完全失效而不做说明。

---

## 6. 已确认的中优先级工程问题

### P1-1：`FEATURE_SET` 路径未全链路联动

`data_process.py`：

```text
.../data_process/<FEATURE_SET>/
```

但 `WGAN.py` / `classify.py` 默认仍硬编码：

```text
.../data_process/combined/
```

风险：进行 `imu` / `response` 等消融实验时，若旧 `combined` 目录存在，可能静默读取旧数据。

建议：WGAN/classifier 默认目录也由 `FEATURE_SET` 推导；`DATA_DIR_OVERRIDE` 仍保留最高优先级。

---

### P1-2：`classify.py` 写死 `range(6)`

当前：

```python
all(f"cmd_motor{i}" in FEATURE_NAMES for i in range(6))
```

应改为从 `scenario_config` 导入 `NUM_ROTORS`，避免破坏单一事实来源原则。

---

## 7. 已确认的低优先级/架构债

这些问题目前不会优先阻塞正式数据修复：

1. `BoundedControlAllocator` 的 `bound_tolerance` / `max_iterations` 未从 `hex_params.py` 暴露配置；
2. `main.py` 通过全局替换 `np.linalg.inv` 兼容六旋翼非方阵，作用域过大；
3. `WGAN.py` 使用模块级可变 `SEQ_LEN / N_FEATURES / N_CLASSES`，当前串行折训练可用，但不适合未来并行；
4. `WGAN.py` 顶部 `N_FEATURES=18` 是默认字面量，虽然运行时会被真实数据覆盖，但可读性一般；
5. `hex_params.py` 中 `motor_noise_std=0.0` 当前是安全的；若未来改为非零，要重新审查 RotorPy 内部随机数源与项目显式 seed 体系的可复现性。

---

## 8. 当前已经确认“做对了”的部分

不要为了“重构”而破坏这些已验证设计：

- SE3 外环增益通过独立 `se3_*` 键显式覆盖并校验生效；
- 控制分配使用 bounded、非负、速度/推力受限的 nominal allocator；
- ground-truth fault effectiveness 不参与控制器决策；
- 故障时序在 plant 路径统一；
- allocation matrix 泛化到 4×N；
- CSV 原子写入；
- generation plan / manifest / SHA256 / code hash 的可追溯链；
- `validate_hex_data.py` 独立重推 allocation/effectiveness，不直接复用生成侧公式；
- 正式评估使用 `StratifiedGroupKFold`；
- train/test group 显式互斥；
- scaler 只在训练折 fit；
- GAN 只在训练折训练；
- 正式分类读取 raw windows，不使用全量预先 fit 的 scaler 结果。

---

## 9. 正式数据生成规则（当前基线）

当前核心实验约束：

- 六旋翼；
- `SIM_RATE = 100 Hz`；
- 单次仿真 `DURATION = 20 s`；
- `WARMUP = 2 s`；
- 每类至少 4 次独立仿真；
- 固定 base seed，独立控制 trajectory variation / wind / IMU 随机流；
- 同一 `run_id` 下不同类别共享 nuisance-condition variation，便于控制混杂变量；
- CSV 保存场景元数据、随机种子、run_id、trajectory_id、故障时刻、逐样本 fault active 状态及 allocator / plant 审计量；
- 生成结束后必须先跑 `validate_hex_data.py`，不要直接训练。

---

## 10. 数据版本规则

### 关键原则

**物理定义改变 = 新数据版本。**

以下修改发生后，不得把新 CSV 混入旧 `output_hex_v3`：

- IMU 真值/坐标系修正；
- 风场振幅定义修正；
- 故障 effectiveness 定义改变；
- 飞行器物理参数改变；
- 控制分配/plant 故障作用方式发生会改变轨迹响应的修改。

建议修复后使用新的输出目录和新的 dataset version 名称，并重新生成 generation plan / manifest。

---

## 11. Git / 协作工作流

### 当前状态

- 本地仓库与 GitHub 已连接；
- remote：`origin`；
- 默认分支：`main`；
- `main` 用作稳定基线；
- 不应把实验性修改直接堆在 `main`。

### 推荐流程

```text
main
  ↓
创建功能分支
  ↓
修改
  ↓
本地测试/验证
  ↓
commit
  ↓
push
  ↓
审查 diff / PR
  ↓
确认后合并 main
```

### 当前建议的下一分支

```text
fix/simulation-physics
```

第一轮仅处理：

1. IMU 真值/机体系问题；
2. TurbulentWind `base_amplitude` 问题。

**第一轮不要同时修改 WGAN / classify / 类别定义。**

原因：先把物理数据生成层和学习层解耦，便于 diff 审查和问题回溯。

---

## 12. 下一步执行计划

### Phase A — 仿真物理修复

1. 新建 `fix/simulation-physics`；
2. 核对 RotorPy 2.1.2 `Imu` / `simulate` / `Environment` 的实际接口；
3. 修 IMU ground truth 来源；
4. 修 `TurbulentWind.base_amplitude`；
5. 更新/补充对应测试；
6. 运行最小仿真验证；
7. 检查输出 IMU 的坐标系与静态/悬停行为；
8. push 后做代码 diff 审查；
9. 不立刻覆盖旧正式数据。

### Phase B — 类别语义决策

单独讨论并决定 M4/M5 `Full=0.05` 是否保留。

### Phase C — 数据/训练链工程清理

再处理：

- `FEATURE_SET` 路径统一；
- `range(6)` → `NUM_ROTORS`；
- allocator 参数配置化；
- monkey patch 作用域；
- WGAN 维度全局状态等。

### Phase D — 新数据版本

物理层确认后：

1. 使用新 `HEX_OUTPUT_DIR`；
2. 全量重新生成 19 类 × 4 次独立仿真；
3. `validate_hex_data.py`；
4. `data_process.py`；
5. 正式 grouped CV / WGAN / classifier；
6. 保留全部 manifest、hash、配置和结果。

---

## 13. 新 AI / 新协作者启动指令

新会话开始时，先执行以下步骤：

1. 读取本文件 `PROJECT_HANDOFF.md`；
2. 读取当前分支和最新 commit；
3. 阅读与当前任务直接相关的源文件；
4. 对照代码确认本文件描述是否仍然有效；
5. 先给出“已确认 / 已过时 / 有出入”清单；
6. 在确认现状前不要修改文件；
7. 不要仅根据旧聊天记录覆盖当前仓库事实。

推荐给 ChatGPT Work / Codex / Claude 的启动语：

> 先读取 `PROJECT_HANDOFF.md`，再读取当前分支、最新 commit 和与当前任务相关的代码。请逐项核对交接文档是否仍与仓库一致，先输出“已确认 / 有出入 / 已过时”清单，在我确认前不要修改任何文件。

---

## 14. 修改纪律

任何 agent / 协作者都应遵守：

- 不要在没有验证的情况下重写大段已工作的物理链路；
- 不要让控制器读取 ground-truth fault label/effectiveness 作为决策输入；
- 不要混合不同数据版本；
- 不要用全量数据 fit scaler 后再做 CV；
- 不要按重叠窗口随机拆 train/test；
- 不要跳过 `validate_hex_data.py` 直接训练；
- 不要把外部大型数据集、训练输出、模型权重、缓存等提交到 Git；
- 任何会改变数据物理含义的修改，都必须更新 dataset version / provenance；
- 结论优先服从当前仓库代码与可复现实验，而不是旧口头描述。

---

## 15. 交接文档维护规则

当以下事项发生变化时，更新本文件：

- 稳定代码基线；
- 当前工作分支；
- 已确认 bug / 已修 bug；
- 数据版本；
- 故障类别定义；
- 正式训练/验证流程；
- 下一步优先级。

尽量把本文件当作“项目当前事实”，而不是聊天日志。
