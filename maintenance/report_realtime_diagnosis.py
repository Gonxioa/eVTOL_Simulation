"""Generate the Chinese report, scientific figures and portable trained weights."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from provenance import sha256_file,write_json
from simulink_data import read


def percent(value):
    return "—" if value is None or not np.isfinite(value) else f"{100*value:.2f}%"


def make_report(output):
    output=Path(output)
    read(output/"evaluation_complete.json")
    rows=pd.read_csv(output/"per_seed_results.csv")
    jobs=read(output/"formal_training_summaries.json")
    selected=read(output/"development_selection.json")
    protocol=read(output/"protocol.json")
    config=protocol["config"]
    figures=output/"figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","Arial"],"axes.unicode_minus":False,
                         "font.size":10,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    names={"current_state":"当前状态目标","legacy_record":"旧记录目标"}
    colors={"current_state":"#147c9c","legacy_record":"#db8756"}
    aggregates=[]
    for (role,objective,model),group in rows.groupby(["role","objective","model"]):
        row=dict(role=role,objective=objective,model=model,seeds=len(group))
        for col in rows.select_dtypes(include=[np.number]).columns:
            if col=="seed":continue
            finite=group[col].dropna()
            row[col+"_mean"]=float(finite.mean()) if len(finite) else None
            row[col+"_std"]=float(finite.std(ddof=0)) if len(finite) else None
        aggregates.append(row)
    pd.DataFrame(aggregates).to_csv(output/"aggregate_by_model_objective.csv",index=False,encoding="utf-8-sig")
    training=pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in jobs])
    training.to_csv(output/"training_budget.csv",index=False,encoding="utf-8-sig")
    attempts=[]
    archive=output.with_name(output.name+"_pre_boundary_fix")
    draft=output.with_name(output.name+"_development_range_draft")
    for root,excluded in ((output,False),(archive,True),(draft,True)):
        if not root.exists():continue
        for checkpoint in sorted(root.glob("*/*/last.pt")):
            saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
            complete=(checkpoint.parent/"summary.json").exists()
            attempts.append(dict(phase=checkpoint.relative_to(root).parts[0],job=checkpoint.parent.name,
                excluded_before_boundary_fix=excluded,completed=complete,
                persisted_epochs=saved["epoch"],persisted_optimizer_updates=saved["updates"],
                persisted_forward_points=sum(r["forward_points"] for r in saved["history"]),
                interrupted_epoch_additional_updates_unknown=not complete,
                checkpoint=str(checkpoint)))
    pd.DataFrame(attempts).to_csv(output/"all_training_attempts.csv",index=False,encoding="utf-8-sig")
    write_json(output/"boundary_fix_and_attempts.json",dict(
        reason="Inclusive clear threshold: 1-0.7 can be slightly larger than0.3; added1e-12 double-roundoff tolerance and boundary regression.",
        old_runs_retained=[str(archive),str(draft)],old_runs_used_in_official_results=False,
        attempts=attempts,interruption_scope="For the intentionally interrupted old run only persisted complete epochs are exactly audited; any unpersisted partial epoch is not added to official counts."))
    def save(fig,name):
        fig.savefig(figures/(name+".png"),dpi=180,bbox_inches="tight",facecolor="white")
        fig.savefig(figures/(name+".svg"),bbox_inches="tight",facecolor="white")
        plt.close(fig)
    test=rows[rows.role=="test"]
    group_order=[("cnn","legacy_record"),("cnn","current_state"),("lstm","legacy_record"),("lstm","current_state")]
    labels=[f"{m.upper()}\n{names[o]}" for m,o in group_order]
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,column,title in zip(axes,["state_macro_f1","detection_within_1s","localization_within_3s"],
                               ["逐时刻19类宏F1","故障后1秒内检出率","故障后3秒内正确电机及程度定位率"]):
        for i,(model,objective) in enumerate(group_order):
            values=test[(test.model==model)&(test.objective==objective)][column].to_numpy()*100
            ax.bar(i,values.mean(),yerr=values.std(),capsize=4,color=colors[objective],width=.65,alpha=.85)
            ax.scatter(np.repeat(i,3)+np.array([-.1,0,.1]),values,s=20,color="#172a3a",zorder=3)
            ax.text(i,values.mean()+values.std()+2,f"{values.mean():.1f}",ha="center")
        ax.set(xticks=np.arange(4),xticklabels=labels,ylabel="比例（%）",ylim=(0,110),title=title)
        ax.grid(axis="y",alpha=.2)
    fig.suptitle("新版228条绕圈数据：相同独立测试来源，三个训练种子",fontsize=14)
    fig.tight_layout()
    save(fig,"01_当前状态与旧目标对比")

    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,kind in zip(axes,["detection","localization"]):
        medians=[];p95=[]
        for model,objective in group_order:
            g=test[(test.model==model)&(test.objective==objective)]
            medians.append(g[kind+"_delay_median_s"].mean())
            p95.append(g[kind+"_delay_p95_s"].mean())
        ax.plot(np.arange(4),medians,"o-",label="条件中位数（三种子均值）")
        ax.plot(np.arange(4),p95,"s--",label="条件95%分位数（三种子均值）")
        ax.set(xticks=np.arange(4),xticklabels=labels,ylabel="延迟（秒）",title="故障检出" if kind=="detection" else "正确联合定位")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("延迟仅统计已成功样本；全部漏检和未定位在报告中单列")
    fig.tight_layout();save(fig,"02_条件检出与定位延迟")

    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for family,ax in zip(config["models"],axes):
        job=next(j for j in jobs if j["model"]==family and j["objective"]=="current_state" and j["seed"]==42)
        candidates=job["selected_validation"]["candidates"]
        for candidate in candidates:
            metric=candidate["metrics"]
            ax.scatter(metric["healthy_false_alarms_per_minute"],metric["localization_within_3s"]*100,
                color="#147c9c" if candidate["index"]==job["selected_validation"]["selected"]["index"] else "#c6cdd4",s=70)
            ax.annotate(str(candidate["index"]),(metric["healthy_false_alarms_per_minute"],metric["localization_within_3s"]*100),xytext=(3,4),textcoords="offset points")
        ax.set(xlabel="唯一健康来源误报事件/分钟",ylabel="3秒内正确联合定位率（%）",title=family.upper()+"，种子42，最佳检查点")
        ax.grid(alpha=.2)
    fig.suptitle("验证集报警工作点权衡（候选编号与配置一致；蓝色为所选工作点）")
    fig.tight_layout();save(fig,"03_验证集工作点权衡")

    for family in config["models"]:
        fig,axes=plt.subplots(2,2,figsize=(15,12))
        for row,objective in enumerate(("legacy_record","current_state")):
            job=next(j for j in jobs if j["model"]==family and j["objective"]==objective and j["seed"]==42)
            metrics=read(Path(job["folder"])/"test"/"metrics.json")
            for column,stage in enumerate(("early_0_3s","sustained_after_3s")):
                ax=axes[row,column]
                cm=np.asarray(metrics["states"][stage]["confusion_matrix"])
                norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
                image=ax.imshow(norm,vmin=0,vmax=1,cmap="Blues")
                ax.set(xticks=np.arange(19),yticks=np.arange(19),xlabel="预测类别编号",ylabel="真实类别编号",
                       title=names[objective]+("：故障后前3秒" if column==0 else "：持续故障后段"))
                ax.tick_params(labelsize=8)
        fig.colorbar(image,ax=axes.ravel().tolist(),label="按真实类别归一化的比例",shrink=.7)
        fig.suptitle(f"{family.upper()} 种子42：早期与持续阶段逐时刻混淆矩阵\n类别0健康，1～6完全失效，7～12剩余20%，13～18剩余80%",fontsize=14)
        fig.subplots_adjust(wspace=.25,hspace=.28,top=.89,right=.86)
        save(fig,"04_"+family.upper()+"_早期与后期混淆矩阵")

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for column,family in enumerate(config["models"]):
        job=next(j for j in jobs if j["model"]==family and j["objective"]=="current_state" and j["seed"]==42)
        history=read(Path(job["folder"])/"history.json")
        for phase,label in (("healthy","健康阶段"),("early","故障后前3秒"),("sustained","持续故障后段")):
            axes[0,column].plot([r["epoch"] for r in history],[r["phase_training_CE"][phase] for r in history],label=label)
        valid=[r for r in history if "validation" in r]
        axes[1,column].plot([r["epoch"] for r in valid],[r["validation"]["states"]["all"]["macro_f1"]*100 for r in valid],"o-",label="逐时刻宏F1")
        axes[1,column].plot([r["epoch"] for r in valid],[r["validation"]["selected"]["metrics"]["localization_within_1s"]*100 for r in valid],"s--",label="1秒内正确联合定位率")
        axes[0,column].set(title=family.upper()+"训练阶段损失（种子42）",ylabel="未加权阶段交叉熵")
        axes[1,column].set(title=family.upper()+"完整24秒验证",ylabel="比例（%）",xlabel="完整训练来源遍历次数",ylim=(0,105))
        for ax in axes[:,column]:ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();save(fig,"05_分阶段训练损失与验证曲线")

    # Post-freeze descriptive failure examples, not used for model selection.
    example_rows=[]
    for family in config["models"]:
        job=next(j for j in jobs if j["model"]==family and j["objective"]=="current_state" and j["seed"]==42)
        folder=Path(job["folder"])/"test"
        cases=pd.read_csv(folder/"per_record.csv")
        false=cases[(cases.healthy_false_alarm_events>0)|(cases.prefault_false_alarm_events>0)].sort_values("source_id")
        misses=cases[(cases.label>0)&(~cases["localization_within_3s"])].sort_values("source_id")
        source_ids=[]
        if len(false):source_ids.append((false.iloc[0].source_id,"误报示例"))
        if len(misses):source_ids.append((misses.iloc[0].source_id,"3秒内未正确定位示例"))
        if not source_ids:source_ids=[(cases[cases.label>0].sort_values("source_id").iloc[0].source_id,"固定来源故障示例（没有上述失败）")]
        trace=pd.read_csv(folder/"decision_trace.csv")
        fig,axes=plt.subplots(len(source_ids),1,figsize=(12,3.3*len(source_ids)),squeeze=False)
        for ax,(source_id,title) in zip(axes[:,0],source_ids):
            frame=trace[trace.source_id==source_id]
            ax.plot(frame.time_s,frame.fault_probability,color="#147c9c",label="当前故障概率")
            ax.axhline(job["selected_policy"]["fault_threshold"],color="#db8756",linestyle="--",label="故障阈值")
            ax.fill_between(frame.time_s,0,1,where=frame.alarm.astype(bool),color="#d65151",alpha=.13,label="正式报警激活")
            ax.axvline(3,color="black",linestyle=":",label="故障注入（仅离线标注）")
            ax.set(xlabel="Time (s)",ylabel="概率",ylim=(-.03,1.04),title=f"{family.upper()} 种子42｜{title}｜{source_id}")
            ax.grid(alpha=.2);ax.legend(loc="lower right",fontsize=8,ncol=2)
            example_rows.append(dict(model=family,seed=42,source_id=source_id,purpose=title,selection="Post-freeze first sorted matching example; no retuning"))
        fig.tight_layout();save(fig,"06_"+family.upper()+"_在线报警时序示例")
    write_json(output/"example_selection.json",example_rows)

    # Each source's effective training mass can be checked without a huge point CSV.
    development=torch.load(output/"development_flights.pt",weights_only=False)
    weight_rows=[]
    for f in development:
        if f.role!="train":continue
        y,_=f.targets()
        early=(y>0)&(f.time<f.fault_start_s+selected["selected"]["early_span_s"]-1e-9) if f.label else np.zeros(len(y),bool)
        normal=(~early).sum()
        total=float(normal+early.sum()*selected["selected"]["early_weight"])
        weight_rows.append(dict(source_id=f.source_id,group=f.group,label=f.label,points=len(y),
            healthy_points=int((y==0).sum()),early_fault_points=int(early.sum()),sustained_fault_points=int(((y>0)&~early).sum()),
            ordinary_point_weight=1.,early_point_weight=selected["selected"]["early_weight"],
            raw_weight_sum=total,normalized_ordinary_point_weight=1/total,
            normalized_early_point_weight=selected["selected"]["early_weight"]/total,
            participations_per_point_per_traversal=1,total_traversals=config["max_epochs"]))
    pd.DataFrame(weight_rows).to_csv(output/"training_point_weight_audit.csv",index=False,encoding="utf-8-sig")
    schedule=[]
    train_flights=[f for f in development if f.role=="train"]
    for job in jobs:
        rng=np.random.default_rng(job["seed"])
        for epoch in range(1,job["record_traversals"]+1):
            order=rng.permutation(len(train_flights))
            for start in range(0,len(order),config["batch_flights"]):
                batch=order[start:start+config["batch_flights"]]
                for index in batch:
                    schedule.append(dict(job=Path(job["folder"]).name,epoch=epoch,
                        batch=start//config["batch_flights"]+1,batch_size=len(batch),
                        source_id=train_flights[index].source_id,record_average_factor=1/len(batch),
                        current_loss_sampling="all_valid_points" if job["objective"]=="current_state" else "legacy_decision_points_plus_endpoint"))
    pd.DataFrame(schedule).to_csv(output/"training_batch_schedule.csv",index=False,encoding="utf-8-sig")

    weights=output/"portable_weights"
    weights.mkdir(exist_ok=True)
    manifest=[]
    for job in jobs:
        source=Path(job["folder"])/"best.pt"
        checkpoint=torch.load(source,map_location="cpu",weights_only=False)
        portable={name:checkpoint[name] for name in ("semantics","signature","config","model","scaler","epoch","best_policy","initial_parameters_sha256")}
        target=weights/(Path(job["folder"]).name+".pt")
        torch.save(portable,target)
        manifest.append(dict(path=target.name,sha256=sha256_file(target),source_checkpoint_sha256=sha256_file(source),
                             model=job["model"],objective=job["objective"],seed=job["seed"],epoch=job["selected_epoch"]))
    write_json(weights/"manifest.json",manifest)

    latency=pd.read_csv(output/"online_latency.csv") if (output/"online_latency.csv").exists() else None
    lines=["# 新版飞控数据的实时故障诊断基线", "",
        "本轮复用已验收的228条原生Simulink绕圈记录，没有新运行仿真，没有GAN或ROS。完成2个真实数据小规模检查、12个开发候选、CNN/LSTM×2种训练目标×3个种子的12个正式训练。测试信号在全部正式模型、scaler及报警工作点冻结后统一读取。", "",
        f"补充边界回归时发现0.3浮点减法的包含边界问题，已用1e-12双精度舍入容差修正。修正前的训练保留在{archive.name}，未用于正式结果。其中已完成与主动中止的尝试分别记录在all_training_attempts.csv；中止轮次未落盘的部分更新数不伪称可精确复算。修正后从头重跑了全部正式开发与训练。", "",
        "## 输入、标签和真正的在线处理", "",
        "输入保持加速度计3轴、陀螺仪3轴和六台带噪、延迟测量转速，共12维。转速直接来自feedback_state_14..19；IMU按既有传感链因果延迟3个采样周期，转速不再延迟。对命令、真值、效能及诊断内部列任意改写不影响输入，审计已通过。", "",
        "逐时刻标签：故障前健康，注入后保持对应电机/程度，只有故障本身解除才恢复健康；飞机恢复稳定不改变故障标签。现有数据不含真实解除事件，解除状态机仅有程序回归验证，不能据此宣称已验证故障解除识别性能。", "",
        "RealtimeDiagnoser只需已到达的测量和时间戳，保存LSTM隐藏/细胞状态或CNN逐层缓存、采样计数与事件计时。累计每10点判断一次，首次0.09秒，最后23.99秒；2401个采样点仍全部处理，24.00秒末点不制造额外判断。不同到达批次产生同一判断序列。文件回放使用同一对象。默认拒绝重复、逆序、非有限值和缺口；显式缺口续接会重置上下文及计时，缺数不解除已激活报警。", "",
        "故障概率为1-P(健康)，检出、定位均使用当前头。同一类别的置信度及持续时间不足时，允许故障已检出、类别待确认。候选起点、正式发出时刻、类别确认与解除均分别保存；事后回填不提前在线延迟。", "",
        "## 训练与验证选择", "",
        f"开发候选：连续段100/300点，早期权重1/3，加权范围为故障后前1/3秒；不加权时范围等价、只运行一次，共六个参数组合、12个网络候选训练。每个候选8次来源遍历、开发种子42。共同选中计算段 **{selected['selected']['segment_samples']}点**、早期权重 **{selected['selected']['early_weight']:g}**、加权范围 **{selected['selected']['early_span_s']:g}秒**；其选择采用两个网络的验证排名平均，规则在正式训练前写入计划。分阶段评价范围始终保持共同的故障后前3秒。", "",
        "正式训练均为40次完整来源遍历，batch4，每轮114条/273714个有效点、29次更新，累计1160次更新；训练随机种子42/43/44。每个有效训练点每轮仅推进一次网络。两种目标的遍历次数、前向有效点数、优化更新次数相同。当前目标对全部有效时刻计算加权CE，按记录归一化；旧目标的记录/前缀/当前损失为1/0.25/0.25，其中时间监督按旧判断点取样。旧目标保留完整反传，新目标采用连续段TBPTT，因此比较同时涉及损失与梯度传播方式，不能把全部变化仅归于某一个权重。", "",
        "新目标段间传递数值状态、分离计算图，在一批完整记录处理结束后统一执行一次优化更新。最后短段、变长记录与padding有回归测试；逐来源权重审计在training_point_weight_audit.csv。配对记录的健康前缀按各记录保留，重复参与训练不代表独立飞行来源。", "",
        "CNN沿用四层64通道、核5、膨胀1/2/4/8，当前头感受野61点；LSTM沿用两层64单元。数值历史传递不等于跨段无限反传，CNN当前头也不会因旧历史池存在而获得无限历史。dropout0.3、Adam0.001、梯度裁剪5；float32并禁用TF32。", "",
        "每4轮在相同完整24秒验证来源上比较8个预声明报警工作点。优先满足健康与故障前各自≤1次/分钟误报，再依次按1秒内正确联合定位、3秒内正确联合定位、1秒内检出、逐时刻宏F1等选择；全候选保留。阈值为开发工作点，不是认证安全限制。所有正式选择在读取测试数值前冻结。", "",
        "## 独立测试结果", "",
        "下表为训练种子42/43/44的均值±总体标准差。每种模型都评价相同3个独立条件组、57条记录（3健康、54故障）；重复训练没有增加独立数据来源。定位要求电机及程度同时正确。", "",
        "|网络与目标|逐时刻宏F1|0.5秒内检出|1秒内检出|3秒内检出|0.5秒内正确定位|1秒内正确定位|3秒内正确定位|",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(group,column):
        a=group[column].dropna().to_numpy()
        return "—" if not len(a) else f"{100*a.mean():.2f}% ± {100*a.std():.2f}%"
    for model,objective in group_order:
        g=test[(test.model==model)&(test.objective==objective)]
        columns=["state_macro_f1","detection_within_0.5s","detection_within_1s","detection_within_3s", "localization_within_0.5s","localization_within_1s","localization_within_3s"]
        lines.append("|"+model.upper()+" "+names[objective]+"|"+"|".join(fmt(g,c) for c in columns)+"|")
    lines += ["", "|网络与目标|漏检数/54（各种子）|3秒后才检出（各种子）|从未正确定位（各种子）|首次定位错误（各种子）|健康误报事件（各种子）|故障前误报事件（各种子）|",
              "|---|---|---|---|---|---|---|"]
    for model,objective in group_order:
        g=test[(test.model==model)&(test.objective==objective)].sort_values("seed")
        vals=["/".join(str(int(x)) for x in g[c]) for c in ("always_missed","late_detection_over_3s","never_joint_localized","first_localization_wrong","healthy_false_alarm_events","prefault_false_alarm_events")]
        lines.append("|"+model.upper()+" "+names[objective]+"|"+"|".join(vals)+"|")
    lines += ["", "首次定位错误比例的分母是实际发出首次类别确认的故障数；全部故障的正确定位覆盖率采用54作为分母。", ""]
    for model,objective in group_order:
        g=test[(test.model==model)&(test.objective==objective)].sort_values("seed")
        first="、".join(f"{int(r.first_localization_wrong)}/{int(r.first_localization_count)}" for _,r in g.iterrows())
        repeats="/".join(str(int(v)) for v in g.repeat_alarms)
        clears="/".join(str(int(v)) for v in g.wrong_clears)
        lines.append(f"{model.upper()} {names[objective]}：首次定位错误/已定位分别为 {first}；重复报警 {repeats} 次；错误解除 {clears} 次。健康误报率三种子均值 {g.healthy_false_alarms_per_minute.mean():.3f} 次/分钟，故障前误报率均值 {g.prefault_false_alarms_per_minute.mean():.3f} 次/分钟。")
        lines.append("")
    lines += ["延迟的中位数与95%分位数仅覆盖已成功检出或正确定位的故障，不能代表全部故障。全部漏检、超时及未定位已在上表单列，逐种子、逐电机、逐程度和逐组CSV保留异常。", "",
        "唯一健康来源曝光从首次可用判断到最后判断计算，每个训练种子的测试健康曝光为71.7秒；故障前阶段曝光另列，54条故障记录的前缀高度相关，不能与唯一健康曝光相加充当独立证据。故障前已有正式误报不能直接成为本次故障的成功检出；故障期间后续新报警计为重复，物理故障未解除却清除报警计为错误解除。", "",
        "主逐时刻分类指标来自当前19类输出的argmax，不施加报警阈值。报警指标按所选工作点评价，口径分开。整条记录补充分类采用最后一次在线确认的事件类别，无报警为健康、有报警未定位为-1；新目标不使用未经训练的旧记录头。旧目标另保存原前缀定位+原工作点的结果，便于区分报警逻辑与训练目标的影响。", "",
        "## 实际设备耗时", ""]
    record_table=["## 整条记录分类补充结果", "",
        "|网络与目标|在线事件汇总记录宏F1（三种子均值）|旧训练记录头宏F1（三种子均值）|",
        "|---|---:|---:|"]
    for model,objective in group_order:
        group=test[(test.model==model)&(test.objective==objective)]
        old_scores=[]
        if objective=="legacy_record":
            for job in jobs:
                if job["model"]==model and job["objective"]==objective:
                    metric=read(Path(job["folder"])/"test"/"metrics.json")
                    old_scores.append(metric["legacy_prefix_localization_original_policy"]["record_macro_f1"])
        record_table.append("|"+model.upper()+" "+names[objective]+"|"+percent(group.record_macro_f1.mean())+"|"+
                            (percent(float(np.mean(old_scores))) if old_scores else "关闭训练，不作为正式成绩")+"|")
    record_table += ["", "旧训练记录头直到记录结束才形成整条飞行结果，不能把其宏F1当成故障后0.5/1/3秒的在线报警成绩。", ""]
    position=lines.index("## 实际设备耗时")
    lines[position:position]=record_table
    if latency is not None:
        cpu=latency[(latency.device=="cpu")&(latency.decision_samples==10)]
        gpu=latency[(latency.device=="cuda")&(latency.decision_samples==10)]
        for label,frame in (("CPU",cpu),("GPU",gpu)):
            if len(frame):
                lines.append(f"{label}默认0.1秒判断周期：六个新目标模型的单次调用中位耗时范围 {frame.median_ms.min():.3f}～{frame.median_ms.max():.3f} ms，95%分位数范围 {frame.p95_ms.min():.3f}～{frame.p95_ms.max():.3f} ms，观察到的最大值 {frame.maximum_ms.max():.3f} ms。超出周期调用共 {int(frame.calls_over_period.sum())} 次。")
                lines.append("")
        lines += ["每个模型、设备及判断周期预热60次后测600次，CPU单线程。公开push调用计时包含标准化、网络状态、报警逻辑和结果复制，CUDA前后同步；逐采样点10ms周期也单列。硬件为i7-14700HX、RTX4060 Laptop GPU。该测量是桌面实际观测，不是硬实时保证。报警延迟来自仿真数据时间轴，与这些计算耗时分别报告。", ""]
    lines += ["## 完成范围与限制", "",
        "已采集数据复核：228条，全部控制器版本、文件哈希、19类完整性及来源划分核对；训练标准化仅拟合114条训练来源的273714个测量点。171条开发记录在审计阶段、57条测试记录在冻结后分别完成原生数值传感链检查。测试物理描述性结果曾经查看过，因此这里只称新的分类评价，不声称从未接触测试数据。", "",
        "开发检查预算较正式训练短，开发候选排名仍可能受早期优化波动影响；本轮不据测试结果重新选择分段长度或追加训练。40轮为预先固定的研究基线预算，不声称损失已经严格收敛。训练损失、早期/持续阶段验证和三种子差异全部保留。", "",
        "0.5、1、3秒是观察点，尚未通过控制闭环确认正式安全报警时限。来源只包含当前绕圈任务、单电机持续故障及既定三种强度，不能外推到其他任务、复合故障或真实飞行。", "",
        "## 运行与文件", "",
        "接口与最短命令见仓库docs/REALTIME_USAGE.md；从文件逐批模拟实时采集可运行stream_fault_diagnosis.py。报告目录内有figures、portable_weights、per_seed_results.csv、training_budget.csv、来源审计、开发选型及冻结/测试读取收据。权重保留12个正式训练各自的最佳检查点及scaler。", ""]
    # Explicit quantified changes, without assuming positive results.
    lines += ["## 目标修改带来的实际变化", ""]
    for family in config["models"]:
        current=test[(test.model==family)&(test.objective=="current_state")]
        legacy=test[(test.model==family)&(test.objective=="legacy_record")]
        delta_f1=(current.state_macro_f1.mean()-legacy.state_macro_f1.mean())*100
        delta_loc=(current["localization_within_1s"].mean()-legacy["localization_within_1s"].mean())*100
        delta_fp=current.healthy_false_alarm_events.mean()-legacy.healthy_false_alarm_events.mean()
        lines.append(f"{family.upper()}：统一当前头在线口径下，逐时刻宏F1变化 {delta_f1:+.2f} 个百分点，1秒内正确电机及程度定位率变化 {delta_loc:+.2f} 个百分点，健康误报事件每训练种子平均变化 {delta_fp:+.2f} 次。未将未成功案例从分母删除，也未用测试结果继续调参。")
        lines.append("")
    report="\n".join(lines)
    (output/"实时故障诊断_中文结果报告.md").write_text(report,encoding="utf-8")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",required=True)
    args=parser.parse_args()
    make_report(args.results)
    print("REPORT_COMPLETE",args.results)
