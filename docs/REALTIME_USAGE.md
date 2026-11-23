# 实时故障诊断运行说明

本轮主入口为 `run_simulink_realtime.py`，语义版本 `current_state_stream_v2`。旧 `run_simulink_sequence.py` 和 `configs/simulink_sequence.json` 保留为历史记录目标工具，不能把旧 checkpoint 无提示装入新接口。新加载器默认拒绝旧权重；旧目标实验对照需要明确传入 `allow_legacy_control=True`。

## 已有数据与最短命令

无需再次运行Simulink。数据目录为 `C:/Users/lenovo/Desktop/eVTOL_Simulation/output_simulink_gan/control_v4_20261123`。完整记录继续保留。更换机器先修改配置中的 `dataset_root`、`results_root`，原注册表若搬迁可按已有来源目录布局恢复。

在本分支目录执行：

```powershell
python -m unittest tests.test_realtime_diagnosis tests.test_simulink_sequence tests.test_sequence_baselines -v
python run_simulink_realtime.py all --device cuda
```

`all`依次执行审计、真实数据小规模检查、验证候选、12个正式训练、冻结和测试。正式测试阶段会验证全部冻结文件，未完成12个训练、源代码变化或模型文件变化都会拒绝读取测试数值。每个阶段也可单独运行：`audit`、`smoke`、`pilot`、`train`、`freeze`、`evaluate`。已完成且身份相同的训练会跳过；中断训练从 `last.pt` 恢复，包含优化器、采样器与随机数状态。正式预算40次来源遍历，不按测试成绩追加。

## 可调用对象

```python
from simulink_realtime import RealtimeDiagnoser
from simulink_realtime_training import load_realtime_checkpoint
from simulink_sequence_events import EventPolicy
import torch

torch.set_num_threads(1)  # 与本轮在线耗时测量一致；应用程序管理CPU线程池

model, scaler, saved = load_realtime_checkpoint("path/to/best.pt", "cpu")
diagnostic = RealtimeDiagnoser(
    model, scaler, EventPolicy(**saved["best_policy"]),
    sample_hz=100, decision_samples=10, device="cpu"
)

# packets 是采集源产生的迭代器。拿到本批才处理本批，无需后续数据。
for new_measurements, acquisition_times in packets:
    for decision in diagnostic.push(new_measurements, acquisition_times):
        print(decision["time_s"], decision["status"], decision["confirmed_label"])

latest = diagnostic.current()
final = diagnostic.finish()
diagnostic.reset()  # 下一次独立飞行必须复位
```

输入为已经按冻结传感链对齐的 `[N,12]` 测量，按加速度计3、陀螺仪3、六台测量转速排列。故障时刻、电机编号、任务参考、控制指令均不在接口参数内。`decision_samples=1`可逐采样点判断；训练计算段长与判断周期独立。

`push`可以接收任意数量新点，返回该批新产生的判断列表。内部以累计第10、20、30…点判断；样本时间从0开始时，首次判断为0.09秒，此后0.19、0.29…23.99秒。末尾24.00秒的单点推进网络但不额外触发一次判断。读取当前状态不会重复推进网络。缓冲区最多等待不足一个判断周期的数据。

LSTM传递隐藏、细胞状态；CNN传递每层卷积上下文。CNN当前头的感受野61点，缓存并不使它获得无限历史；旧前缀头虽有全历史池，但新目标不使用该头定位。LSTM数值状态跨完整飞行延续，TBPTT梯度在训练段边界截断。

## 原始传感包示例

```powershell
python stream_fault_diagnosis.py --checkpoint "path/to/best.pt" --csv "path/to/signals.csv" --packet-points 7 --device cpu --cpu-threads 1
```

这个示例逐批读取CSV，替换 `csv_packets` 即可接串口或网络采集。`MeasurementAligner`仅把原始IMU延迟3个采样周期，保留已经延迟的 `feedback_state_14..19`，不会再次延迟转速。已经对齐的数据不要再经过该适配器。初始化的前三个时刻使用初始测量保持，与现有仿真链一致。

## 报警与异常输入

故障概率为当前19类softmax输出中的 `1-P(健康)`。检出与同一类别确认独立并行计时。类别暂不确定时输出 `fault_class_pending`，不能把未确认的电机当作定位成功。正式类别变化会再次记录确认时刻。解除只根据持续的低故障概率；物理故障持续时发生解除会在离线报告中记为错误解除。

0.1秒判断周期下，从第一个满足条件的判断到第四个满足条件的判断，首尾相隔0.3秒，才完成0.3秒确认。一次中断就重新计时。`start_s/candidate_at_s`为候选起点，`detected_at_s`为实际报警发出时刻，类别更新中的 `time_s`为实际确认时刻，`cleared_at_s`为实际解除时刻。事后区间可以回填到候选起点，在线延迟不使用回填时刻。

上述秒数都使用采样时间轴，记录的是产生正式输出的判断时刻。实际部署还会受到传输、包积压和计算耗时影响；把很晚才到达的大数据包交给接口不能恢复原本的物理实时性。离线因果回放延迟与本机实测调用耗时分别报告，不能将快速离线回放速度当作完整通信链路的报警延迟。

重复、逆序、非有限值、非预期采样间隔默认原子拒绝，不推进计数或网络。数据缺口默认拒绝；若明确选择 `gap_policy="reset_context"`，缺口后重置网络上下文和持续计时，不补点、不假装观测连续，已经激活的报警不因缺数自动解除。断续采集时应同时复位原始测量对齐适配器。

## 结果目录

默认 `C:/Users/lenovo/Desktop/eVTOL_Simulation/output_realtime_diagnosis_20261123`：

- `data_audit.json`、`source_audit.csv`：228条来源审计，测试数值在冻结后另行验证。
- `development_selection.json`：分段长度、早期权重候选与选择依据。
- `formal/*/history.json`、`validation_epoch_*.json`：实际更新次数、训练损失、分阶段验证与全部报警候选。
- `formal/*/best.pt`、`last.pt`：权重、scaler、配置及恢复状态。
- `freeze_receipt.json`、`test_access_receipt.json`：冻结和测试读取身份。
- `formal/*/{validation,test}`：逐记录、逐组/电机/程度CSV、事件、判断序列、逐时刻指标。
- `online_latency.csv`：真实设备公开接口调用耗时；不同于数据时间轴上的报警延迟。

这里只用原始228条仿真数据，没有GAN或ROS。配对前缀在训练中按记录保留，其重复次数不是独立实验数。数据全部是持续单电机故障，故障解除仅验证了程序逻辑；不能声称已验证真实故障解除后的模型性能。
