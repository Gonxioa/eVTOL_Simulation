"""Preserve the supplied plan and register the accepted V4 implementation details."""
from pathlib import Path
import shutil
from provenance import sha256_file,write_json,stable_hash
from simulink_data import ROOT,read
from simulink_v3_data import immutable_json
from simulink_v4_data import root,snapshot,registry


ADDENDUM = r'''

---

## V4实施登记细则（2026年11月5日评审修订）

本节将原计划中的待定实现选择固定为有限任务表。它记录预先约定，不表示实验已经完成。机器可读配置为项目的 `configs/simulink_gan_v4_evidence.json`，实际完成情况以独立结果目录的日志和报告为准。本节与原文“例如／可选／建议”发生范围冲突时，以本节登记的首轮范围为准。

### 1. 来源和边界

基线固定为2e033334637be102e9a28c7ce0955d1f7d1f189f。分支为codex/simulink-gan-v4-evidence，结果目录为output_simulink_gan/v4_evidence_20261105。复用原六个训练组和三个验证组；既有V2/V3测试均归为已经曝光的历史来源。旧记录、权重、scaler及收据只读，身份写入legacy_snapshot.json。

主输入12维，沿用原IMU和feedback_state_14～19的测量对齐。所有神经输入均不含命令、真值、时间、来源或阶段。24秒、100Hz、2401行和3秒故障保持不变。完整有限但飞行验收失败的记录保留。

### 2. 两种窗口网格分开命名

第一轮窗口对照固定共同末点e=299+50j，e<2400；每组健康43窗、每条故障42窗、每组799窗，六个训练组4794窗、三个验证组2397窗。50/100/200/300点窗口共用相同末点、来源及标签。

在3.49秒结束的窗口，物理故障样本占比依次为100%、50%、25%、16.67%。另记录transport_eligible_fraction，表示经过已知传输延迟后的时间占比，它不证明证据已可辨识。

最终窗长确定前，最多两个入选窗长在原生网格e=W-1+50j上重新做LSTM/CNN联合复核；原生窗口从0秒开始。第6步的前6/12/24秒覆盖比较全部使用原生网格，并重新拟合各自scaler、训练模型。共同网格与原生网格的缓存和优化器不得静默复用。

若保留共同网格，前6/12/24秒会是690/2058/4794窗；这些不是V3的714/2082/4818。原生网格计数按最终窗长重新计算。

### 3. 批次A的有限任务表

|任务|候选及训练种子|实际更新预算|新增神经轨迹上限|
|---|---|---|---:|
|真实小样本记忆检查|每类8个真实训练窗，共152窗，固定抽样种子9042|1000|1|
|共同末点窗口初筛|50/100/200/300点，LSTM，种子42/43/44|1000|12|
|LSTM预算检查|初筛排名前两个窗长，原优化器续跑|保存1000/2000/4000|0，最多续跑6条原轨迹|
|固定CNN对照|两个入选窗长，种子42/43/44|保存1000/2000/4000|6|
|统计特征对照|两个入选窗长，RF200，种子42/43/44|每个一次fit|0，另6个RF拟合|
|原生网格联合复核|两个窗长×LSTM/CNN×三个种子|共同网格选出的更新预算|12|
|前缀覆盖|固定模型和窗长，前6/12/24秒×三个种子|同一选定预算|6，完整24秒模型复用精确匹配的原生模型|
|等量全时段诊断|较有竞争力的6/12秒前缀，抽样种子731905/731906/731907×训练种子42/43/44|同一选定预算|9|

批次A最多46条不同神经训练轨迹、6次RF拟合，神经优化更新总上限163000次。续跑不增加独立重复数。等量抽样完整交叉九次，只做数量／时间覆盖诊断，不挑最好抽样种子替换主训练池。首轮不增加阶段分层抽样或Transformer。

LSTM沿用原两层、隐藏64、注意力池化。CNN固定为Conv1d通道12→64→96→128，核5/5/3、步幅2/2/2、ReLU、AdaptiveAvgPool到8格，随后Linear到64、ReLU、Dropout0.3、Linear到19。Adam学习率0.001，批量32。报告参数量、训练时间和单窗推理耗时，不宣称同更新数即同FLOPs。

统计基线固定RandomForest 200棵树、min_samples_leaf=2、max_features=sqrt、n_jobs=4。输入为12路窗口的均值、标准差、极值、RMS、平均绝对差分、lag1相关、通道间相关及去均值PSD带功率。scaler只拟合实际训练窗口。小样本记忆结果仅检查拟合与流水线，不代表泛化。

### 4. 唯一的主排序口径

每个训练种子先合并三个验证组的共同评价窗口计算固定19类argmax宏F1，再对三个种子等权平均。保留8位小数后同分，依次选择较短窗口、较少更新、较少参数、模型名。关键类别、最差组和成本同时保留；主指标最高只表示研究参考，不是可靠诊断器验收。

训练预算从1000/2000/4000中选择。4000次仍改善时，只能陈述当前预算范围内结果，不能声称已充分收敛。前缀从6/12/24秒中按同一验证规则选择，同分优先较短前缀。阶段指标同时保存实际支持数、固定19类和出现类别宏F1；纯故障18类阶段的固定19类宏F1上限为18/19。

### 5. G1进入条件与有限GAN范围

G0复用原类别条件WGAN-GP核心结构，输出选定长度×12维。G1为ACGAN/GPSC启发的冻结外部教师类别约束变体，不能冒称原论文复现。

G1仅在共同真实基线的三个种子都满足以下最低非塌缩检查时开展探索：19个类别的验证召回均大于0，且健康召回、剩余80%类别的电机＋程度联合正确率均高于1/19。这个门槛只排除明显塌缩教师，不证明标签可信，也不是系统合格线。若未通过，保留失败证据，跳过G1，继续G0/ROS研究。

类别约束权重候选固定0、0.1、1.0；0为G0。首轮G2关闭，生成样本筛选关闭。所有入选合成窗口都检查有限值、实际数量和精确重复；昂贵质量诊断使用每类固定32个探针，明确覆盖范围。质量主要参照所选训练覆盖内的真实验证窗；另做完整24秒下游评价。

生成器种子142/143/144、分类器种子42/43/44、ROS抽样种子242/243/244按序配对，属于三个联合重复，不据此分解生成器与分类器各自的方差。生成器上限3000次G更新，每G步5次critic、梯度惩罚10、学习率各1e-4，保存500/1500/3000。损失分项与固定探针不改变训练随机流。

增广比例0/0.25/0.5/1.0，逐类floor(n×比例)。G0/G1候选与ROS使用同一真实池、scaler和已冻结分类器预算。按三种联合重复的共同完整验证宏F1选择，同分优先较小比例；GAN−同量ROS和GAN−原始结果均保留。质量一致率不由同一教师独自验收。

### 6. 连续报警时钟

原生因果回放使用0.1秒判断周期，每个记录从其首个完整窗口开始；不在3秒重置。50/100/200/300点窗口首次判断依次为0.49/0.99/1.99/2.99秒。3秒窗在注入前只有一次判断，必须报告其预故障曝光局限。正常曝光按唯一健康来源和实际可判断区间计算，不复用V3固定的231次／69秒。

主窗口argmax与概率阈值报警分列。报警阈值0.1～0.9和持续2/3/5仅通过验证选择。任务phase受容错参考调速影响，名义第7秒不保证实际运动结束；phase和恢复状态只用于离线分析。

### 7. 独立确认的预算规则

全部开发完成后，主确认比较固定为所选GAN相对同量ROS。用三个验证条件组上、先跨训练种子平均的配对宏F1差值计算样本标准差s。预算规划取n=clip(ceil((1.96s/0.05)^2),6,12)，即6～12个全新条件组。0.05是本轮预算规划的目标半宽，不是系统误差要求；只有三个开发组的近似正态规划估计不稳，不能承诺达到该精度或统计充分。

在采集前固定n、未用过的连续风噪种子和停止规则；不因测试结果再追加。先完成所有模型、scaler、比例、工作点和来源的冻结，才允许读取新测试特征。确认的独立统计单位是配对条件组，不是重叠窗口，也不是训练种子。

### 8. 文献和状态

补充ACGAN原文：https://proceedings.mlr.press/v70/odena17a.html 。GPSC-GAN作者稿采用独立分类器及真实／生成分类损失，当前冻结教师方案属于自定义变体。TimeGAN包含嵌入、恢复和时间步监督，首轮不将任意平滑项命名为其复现。

执行入口为run_simulink_v4.py。批次A完成后生成实际报告和共同基线冻结收据，再按上述进入条件推进后续阶段。任何未执行或因前提不满足而跳过的变体都须在最终状态中列出。
'''


def prepare(config):
    directory=root(config)/'protocol'; directory.mkdir(parents=True,exist_ok=True)
    original=directory/'provided_original_plan.md'; supplied=Path('C:/Users/lenovo/Downloads/eVTOL_V4_Execution_Plan.md')
    if not original.exists(): shutil.copy2(supplied,original)
    text=original.read_text(encoding='utf-8-sig').rstrip()+ADDENDUM
    target=ROOT/'docs/SIMULINK_GAN_V4_EXECUTION_PLAN.md'
    target.write_text(text.rstrip()+'\n',encoding='utf-8')
    previous=directory/'external_plan_update.json'
    allowed={sha256_file(original)}
    if previous.exists(): allowed.add(read(previous)['updated_sha256'])
    if sha256_file(supplied) not in allowed:
        raise RuntimeError('The supplied plan has new user edits; maintained revised copy is saved in docs without overwriting them')
    supplied.write_text(text.rstrip()+'\n',encoding='utf-8')
    write_json(previous,dict(original_sha256=sha256_file(original),updated_sha256=sha256_file(supplied),
                            external_path=str(supplied),maintained_copy=str(target),config_signature=stable_hash(config)))
    snap=snapshot(config); reg=registry(config)
    write_json(root(config)/'preflight.json',dict(baseline=config['baseline_commit'],branch=snap['branch'],
        original_plan_sha256=sha256_file(original),revised_plan_sha256=sha256_file(target),
        old_snapshot_signature=snap['signature'],source_registry_signature=reg['signature'],
        observed_development_records=171,historical_test_records=114,
        registered_batch_A_max_neural_jobs=config['max_batch_A_neural_trajectories'],
        registered_batch_A_max_updates=163000,stages_started=['prepare']))
    print('V4_PREPARED',target,'legacy pinned',len(snap['files']),flush=True)
