"""Build traceable, Chinese-language figures for the healthy position gains.

Uses archived trials, one matched SLX replay, and published JSON snapshots.
This script does not tune or modify the controller or either Simulink model.
Run the PDF step with a Python environment containing reportlab.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "kx_gain_evidence" / "20261026"
SOURCES = OUT / "sources"
PDF = ROOT / "output" / "pdf" / "kx_gain_evidence_20261026.pdf"
DATE = "2026-10-26"
TEAL, BLUE, AMBER, RED = "#168A7C", "#4477AA", "#DBA149", "#C4515B"
INK, GRAY, GRID = "#183347", "#64748B", "#E2E8EF"
FONT = Path("C:/Windows/Fonts/msyh.ttc")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect():
    import numpy as np
    from scipy.io import loadmat

    SOURCES.mkdir(parents=True, exist_ok=True)
    paths = {
        "kx_candidates": ROOT / "simulink/output/baseline_tuning/robust_screen_3.json",
        "aggressive_candidates": ROOT / "simulink/output/baseline_tuning/candidate_results.json",
        "baseline_validation": ROOT / "simulink/validation/BASELINE_TUNING_RESULTS.json",
        "paired_circle": ROOT / "simulink/output/kx_evidence_20261026/paired_circle_slx.mat",
        "paired_checks": ROOT / "simulink/output/kx_evidence_20261026/paired_circle_checks.json",
        "source_hash_manifest": ROOT / "simulink/validation/SIMULATION_SOURCE_HASHES.json",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Matched SLX replay must finish first: " + ", ".join(missing))
    candidates = read(paths["kx_candidates"])
    selected = [r for r in candidates if r["gains"][0] in (3, 3.5, 4)
                and r["gains"][1:] == [7, 8, 20, .5]]
    assert len(selected) == 12
    assert len({(r["gains"][0], r["condition"]) for r in selected}) == 12
    # Verify that only horizontal position P differs in this comparison.
    controls = []
    for r in selected:
        control = dict(r["control"])
        control["position_p"] = [None, None, control["position_p"][2]]
        controls.append(control)
        assert r["actual_duration_s"] == 20 and r["completed"]
    assert all(c == controls[0] for c in controls)
    aggressive = read(paths["aggressive_candidates"])
    validation = read(paths["baseline_validation"])
    assert validation["passed"] and len(validation["cases"]) == 13
    pair = loadmat(paths["paired_circle"], simplify_cells=True)
    before, after = pair["before"], pair["after"]
    t = np.asarray(before["time"])
    assert np.allclose(t, after["time"], rtol=0, atol=1e-10)
    assert np.allclose(before["reference"], after["reference"], rtol=0, atol=1e-10)
    series = {"time_s": t.tolist(), "reference_xyz_m": before["reference"][:, :3].tolist(),
              "before_xyz_m": before["state"][:, :3].tolist(),
              "after_xyz_m": after["state"][:, :3].tolist()}
    assert abs(t[-1] - 8) < 1e-10 and len(t) == 801
    checks = read(paths["paired_checks"])
    assert checks["same_time"] and checks["same_reference"] and checks["same_exogenous"]
    write(SOURCES / "kx_fixed_inner_candidates.json", selected)
    write(SOURCES / "aggressive_candidates.json", aggressive)
    write(SOURCES / "baseline_validation.json", validation)
    write(SOURCES / "paired_circle_timeseries.json", series)
    write(SOURCES / "paired_circle_checks.json", checks)
    parameter_keys = ["mass", "gravity", "J", "rotor_positions", "kT", "kQ", "yaw_sign", "tau",
                      "omega_min", "omega_max", "duration", "task_family", "task_origin", "heading",
                      "radius", "turn", "motion_start", "motion_end", "fault_motor", "fault_factor",
                      "observed_feedback", "noise_enabled", "wind_enabled", "feedback_delay",
                      "control_dt", "sensor_dt", "integration_dt", "export_dt", "seed", "control"]
    def serial(value):
        if isinstance(value, dict): return {k: serial(v) for k, v in value.items()}
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        return value
    parameter_snapshots = {name: serial({key: run["parameters"][key] for key in parameter_keys})
                           for name, run in [("before", before), ("after", after)]}
    changed_parameters = [key for key in parameter_keys
                          if parameter_snapshots["before"][key] != parameter_snapshots["after"][key]]
    assert changed_parameters == ["control"], changed_parameters
    changed_control = [key for key in parameter_snapshots["before"]["control"]
                       if parameter_snapshots["before"]["control"][key] != parameter_snapshots["after"]["control"][key]]
    assert set(changed_control) == {"position_p", "velocity_p", "attitude_p", "rate_p", "rate_d"}
    write(SOURCES / "paired_circle_parameters.json", parameter_snapshots)
    source_hashes = {}
    for label, path in paths.items():
        source_hashes[label] = {"path": path.relative_to(ROOT).as_posix(),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    sources_manifest = read(paths["source_hash_manifest"])
    mismatches = [name for name, digest in sources_manifest["sha256"].items()
                  if hashlib.sha256((ROOT / "simulink" / name).read_bytes()).hexdigest() != digest]
    assert not mismatches, mismatches
    write(SOURCES / "provenance.json", {
        "report_date": DATE,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": source_hashes, "verified_original_source_hashes": len(sources_manifest["sha256"]),
        "statistical_scope": "One saved run per gain/condition; no confidence intervals or repeated-seed claims.",
        "comparison_scope": "Fixed-inner Kx sweep is separate from whole-controller before/after.",
        "plant_status": "Research model: kT=7.45e-6, Q/T=0.01315 m provisional proxy; not AIR Gear 355 refit.",
        "vertical_gain_status": "Kz=1.4 inherited, validated in selected scenarios, not individually optimized.",
        "fair_window": "All displayed tracking RMSE comparisons use [0,8] s, inclusive, at 100 Hz.",
    })


def chart_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if FONT.is_file():
        font_manager.fontManager.addfont(str(FONT))
        font_name = font_manager.FontProperties(fname=FONT).get_name()
    else:
        font_name = "DejaVu Sans"
    plt.rcParams.update({"font.family": font_name, "axes.unicode_minus": False,
                         "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": GRAY, "ytick.color": GRAY,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": GRID, "grid.color": GRID,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    return plt


def header(fig, title, subtitle):
    fig.text(.065, .953, title, fontsize=20, weight="bold", color=INK, va="top")
    fig.text(.065, .897, subtitle, fontsize=10.5, color=GRAY, va="top")


def foot(fig, line1, line2="单次仿真记录；用于当前模型选值，不表示最优性或实飞有效性。"):
    fig.text(.065, .074, line1, fontsize=9, color=GRAY)
    fig.text(.065, .041, line2, fontsize=9, color=GRAY)


def save_fig(fig, name, plt):
    fig.savefig(OUT / (name + ".png"), dpi=240)
    svg = OUT / (name + ".svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8")
    plt.close(fig)


def create_charts():
    import numpy as np
    plt = chart_setup()
    rows = read(SOURCES / "kx_fixed_inner_candidates.json")
    aggressive = read(SOURCES / "aggressive_candidates.json")
    validation = read(SOURCES / "baseline_validation.json")
    series = read(SOURCES / "paired_circle_timeseries.json")
    t = np.array(series["time_s"])
    ref = np.array(series["reference_xyz_m"])
    before = np.array(series["before_xyz_m"])
    after = np.array(series["after_xyz_m"])
    eb, ea = np.linalg.norm(before-ref, axis=1), np.linalg.norm(after-ref, axis=1)
    before_rmse, after_rmse = np.sqrt(np.mean(eb**2)), np.sqrt(np.mean(ea**2))
    get = lambda k, c: next(r for r in rows if r["gains"][0] == k and r["condition"] == c)
    conditions = ["ideal", "observed_10ms", "noise_wind_10ms", "noise_wind_30ms"]
    condition_names = ["理想反馈", "观测反馈\n10 ms 延迟", "噪声＋风\n10 ms 延迟", "噪声＋风\n30 ms 延迟"]
    values = np.array([[get(k, c)["rmse_first8_m"]*100 for c in conditions] for k in [3,3.5,4]])
    noise_gain_reduction = 100*(1-values[2,3]/values[0,3])
    ideal_gain_reduction = 100*(1-values[2,0]/values[0,0])
    total_reduction = 100*(1-after_rmse/before_rmse)
    selected_final = next(c for c in validation["cases"] if c["name"] == "observed_noise_wind_30ms")
    assert abs(after_rmse - values[2,0]/100) < 1e-10
    a_ideal = next(r for r in aggressive if r["candidate"] == 1 and r["name"] == "circle")
    a_delay = next(r for r in aggressive if r["candidate"] == 1 and r["name"] == "circle_delay_30ms")
    assert a_delay["completed"] and a_delay["actual_duration_s"] == 8

    # Figure 1: controlled comparison; all other control values are identical.
    fig = plt.figure(figsize=(13.3, 7.7))
    header(fig, "01  固定其他增益，比较位置环 Kx", "健康圆轨迹；半径 0.3 m；统一统计前 8 s；每个增益与工况各 1 次记录")
    ax = fig.add_axes([.075,.23,.62,.57])
    x = np.arange(4); width = .23
    for j,(k,col) in enumerate(zip([3,3.5,4],[BLUE,AMBER,TEAL])):
        bars = ax.bar(x+(j-1)*width,values[j],width,color=col,label=f"Kx = Ky = {k}")
        ax.bar_label(bars,labels=[f"{v:.2f}" for v in values[j]],padding=4,fontsize=9)
    ax.set_xticks(x,condition_names); ax.set_ylabel("三维位置 RMSE（cm，越低越好）")
    ax.set_ylim(0,3.6); ax.grid(axis="y",alpha=.7); ax.set_axisbelow(True)
    ax.legend(loc="upper right",frameon=False,ncol=3,fontsize=9)
    ax2 = fig.add_axes([.77,.28,.19,.44])
    tilts = [get(k,"noise_wind_30ms")["maximum_tilt_deg"] for k in [3,3.5,4]]
    ax2.plot([3,3.5,4],tilts,"o-",color=TEAL,lw=2,ms=7)
    for k,y in zip([3,3.5,4],tilts): ax2.annotate(f"{y:.2f}°",(k,y),xytext=(0,8),textcoords="offset points",ha="center",fontsize=10)
    ax2.set_xlim(2.8,4.2); ax2.set_ylim(0,12); ax2.set_xticks([3,3.5,4]); ax2.grid(alpha=.6)
    ax2.set_xlabel("水平位置增益"); ax2.set_ylabel("峰值倾角（°）")
    ax2.set_title("30 ms 工况的代价\n20 s 全段峰值",fontsize=11)
    fig.text(.075,.153,f"Kx 从 3 到 4：理想工况误差降低 {ideal_gain_reduction:.1f}%；噪声＋风＋30 ms 工况降低 {noise_gain_reduction:.1f}%。",fontsize=11,color=TEAL,weight="bold")
    foot(fig,"固定水平内环：速度 P=7，姿态 P=8，角速度 P=20 / D=0.5；竖直位置增益始终为 1.4。")
    save_fig(fig,"01_fixed_inner_kx_comparison",plt)

    # Figure 2: matched SLX before/after; all five horizontal gain groups changed.
    fig = plt.figure(figsize=(13.3,7.7))
    header(fig,"02  整套控制器调参前后的实际轨迹", "匹配的 SLX 仿真：相同物理模型、初始状态、参考轨迹和理想反馈；任务运动区间 1-7 s")
    ax = fig.add_axes([.075,.22,.40,.58])
    ax.plot(ref[:,0],ref[:,1],"--",color=INK,lw=2,label="参考轨迹")
    ax.plot(before[:,0],before[:,1],color=RED,lw=2,label="调参前")
    ax.plot(after[:,0],after[:,1],color=TEAL,lw=2,label="调参后")
    ax.set_aspect("equal",adjustable="box"); ax.grid(alpha=.6)
    ax.set_xlabel("世界系 x（m）"); ax.set_ylabel("世界系 y（m）"); ax.legend(frameon=False,loc="upper right",fontsize=10)
    ax2=fig.add_axes([.565,.25,.37,.51])
    ax2.axvspan(1,7,color=BLUE,alpha=.06)
    ax2.plot(t,eb*100,color=RED,lw=2,label=f"调参前：{before_rmse*100:.3f} cm")
    ax2.plot(t,ea*100,color=TEAL,lw=2,label=f"调参后：{after_rmse*100:.3f} cm")
    ax2.set_xlim(0,8); ax2.set_ylim(bottom=0); ax2.grid(alpha=.6)
    ax2.set_xlabel("时间（s）"); ax2.set_ylabel("三维位置误差范数（cm）"); ax2.legend(frameon=False,loc="upper right",fontsize=10)
    fig.text(.565,.805,f"圆轨迹 RMSE 降低 {total_reduction:.1f}%",color=TEAL,fontsize=16,weight="bold")
    foot(fig,"同时调整了位置、速度、姿态、角速度 P 和角速度 D；87.1% 的改善不能全部归因于 Kx。",
         "旧增益在本次重新运行 SLX，复现历史 RMSE；新增益曲线来自已保存的实际 SLX 记录。")
    save_fig(fig,"02_matched_slx_tracking",plt)

    # Figure 3: a failed robustness choice is kept, not hidden.
    fig=plt.figure(figsize=(13.3,7.7))
    header(fig,"03  选值也要检查延迟下的表现", "两组方案的 Kx=Ky 均为 4；区别在内环增益。两侧 RMSE 均统计完整 0-8 s。")
    panels=[("理想反馈",[a_ideal["rmse_m"]*100,values[2,0]],"位置 RMSE（cm）",3.1),
            ("噪声＋风＋30 ms 延迟",[a_delay["rmse_m"]*100,values[2,3]],"位置 RMSE（cm）",250),
            ("延迟工况峰值倾角",[a_delay["maximum_tilt_deg"],selected_final["maximum_tilt_deg"]],"峰值倾角（°）",55)]
    for i,(title,vals,ylabel,upper) in enumerate(panels):
        ax=fig.add_axes([.08+i*.305,.27,.245,.48])
        bars=ax.bar([0,1],vals,color=[RED,TEAL],width=.6)
        ax.bar_label(bars,labels=[f"{v:.2f}" for v in vals],padding=5,fontsize=12,weight="bold")
        ax.set_xticks([0,1],["高内环增益候选","最终健康基准"],fontsize=9)
        ax.set_ylim(0,upper); ax.set_ylabel(ylabel); ax.set_title(title,fontsize=12)
        ax.grid(axis="y",alpha=.6); ax.set_axisbelow(True)
    fig.text(.075,.165,"候选：速度/姿态/角速度 P = 10 / 16 / 40；最终基准：7 / 8 / 20；两者角速度 D 均为 0.5。",fontsize=10.5)
    foot(fig,"高内环增益候选在理想工况误差更小，但在所测延迟工况明显恶化。",
         "这支持对整套环路进行多场景筛选；不能据此断言单独增大 Kx 就会导致振荡。")
    save_fig(fig,"03_delay_robustness_tradeoff",plt)

    # Figure 4: engineering coverage with scenario-specific RMSE requirements.
    names={"hover":"健康悬停","line":"健康直线","circle":"健康圆轨迹","vertical":"健康升降",
           "tracking":"三维跟踪","step":"位置阶跃","perturbed_hover":"初态扰动悬停",
           "large_reverse_circle":"较大半径反向圆","observed_noise_wind_10ms":"噪声＋风＋10 ms",
           "observed_noise_wind_30ms":"噪声＋风＋30 ms","partial_rotor_1":"1 号电机剩余 80%",
           "partial_rotor_6":"6 号电机剩余 80%","circle_half_integration_step":"积分步长减半对照"}
    cases=validation["cases"]
    fig=plt.figure(figsize=(13.3,7.7))
    header(fig,"04  最终健康基准通过 13 个已保存场景", "图示为位置 RMSE / 各场景预设门槛；门槛为本项目工程要求，非行业认证标准。")
    ax=fig.add_axes([.24,.205,.59,.61])
    yy=np.arange(len(cases)); ratios=[c["rmse_m"]/c["rmse_ceiling_m"] for c in cases]
    ax.barh(yy,ratios,color=TEAL,height=.61)
    ax.axvline(1,color=RED,lw=1.3,ls="--",label="该场景 RMSE 门槛")
    ax.set_yticks(yy,[names[c["name"]] for c in cases]); ax.invert_yaxis()
    ax.set_xlim(0,1.08); ax.set_xlabel("归一化 RMSE（低于 1 表示满足该项要求）")
    ax.set_xticks([0,.25,.5,.75,1]); ax.grid(axis="x",alpha=.5); ax.set_axisbelow(True)
    for y,c in enumerate(cases):
        label="数值精度量级" if c["name"]=="hover" else f"{c['rmse_m']*100:.3f} cm"
        ax.text(1.10,y,label,va="center",fontsize=9,clip_on=False)
    fig.text(.84,.83,"实际 RMSE",fontsize=10,color=GRAY)
    foot(fig,"完整验收还检查峰值/末端误差、完成时长、指令与积分边界等；13/13 为既定场景的结果。",
         "竖直 Kz=1.4 在本轮保持不变；升降 RMSE=3.094 cm，说明通过该测试，不说明 1.4 为最优值。")
    save_fig(fig,"04_validation_coverage",plt)

    # Overview: concise, teacher-facing page with transparent scope.
    fig=plt.figure(figsize=(13.3,7.7))
    header(fig,"健康基准 Kx = diag(4, 4, 1.4) 的选值证据", "当前研究模型的仿真比较 | 证据整理日期：2026-10-26 | 不表示实机标定或全局最优")
    card_x=[.065,.37,.675]
    card_values=[f"{noise_gain_reduction:.1f}%",f"{total_reduction:.1f}%","13 / 13"]
    card_titles=["单独比较 Kx：误差降低","整套控制器：圆轨迹误差降低","已保存场景通过验收"]
    card_notes=["3 → 4；噪声＋风＋30 ms；前 8 s","18.265 → 2.364 cm；理想反馈；8 s","包含延迟、初态扰动及部分失效"]
    for xx,value,title,note in zip(card_x,card_values,card_titles,card_notes):
        fig.text(xx,.805,value,fontsize=30,color=TEAL,weight="bold")
        fig.text(xx,.749,title,fontsize=12,color=INK)
        fig.text(xx,.712,note,fontsize=9,color=GRAY)
    ax=fig.add_axes([.075,.225,.55,.39])
    ax.plot(t,eb*100,color=RED,lw=2,label="调参前：整套原增益")
    ax.plot(t,ea*100,color=TEAL,lw=2,label="调参后：整套最终增益")
    ax.set_xlim(0,8); ax.set_ylim(bottom=0); ax.grid(alpha=.6)
    ax.set_xlabel("时间（s）"); ax.set_ylabel("三维位置误差（cm）"); ax.legend(frameon=False,fontsize=9)
    fig.text(.68,.60,"证据边界",fontsize=14,weight="bold",color=INK)
    fig.text(.68,.545,"• 水平 4 有固定内环的候选对照\n\n• 垂向 1.4 沿用旧值，未单独优化\n\n• 单次记录，不给统计显著性结论\n\n• 动力参数仍为研究快照与代理值",fontsize=10.5,color=GRAY,va="top",linespacing=1.35)
    foot(fig,"两类改善分别统计：固定内环的 Kx 对照，与五组水平增益共同变化的整套调参对照。",
         "所有图、参数、原始汇总快照及来源 SHA-256 随本材料提供，可复核数值。")
    save_fig(fig,"00_teacher_overview",plt)

    metrics={"before_circle_rmse_cm":before_rmse*100,"after_circle_rmse_cm":after_rmse*100,
             "whole_tuning_reduction_pct":total_reduction,
             "kx_3_to_4_ideal_reduction_pct":ideal_gain_reduction,
             "kx_3_to_4_noise_wind_30ms_reduction_pct":noise_gain_reduction,
             "fixed_inner_rmse_cm":{str(k):dict(zip(conditions,values[i].tolist())) for i,k in enumerate([3,3.5,4])},
             "aggressive_ideal_rmse_cm":a_ideal["rmse_m"]*100,
             "aggressive_delay_rmse_cm":a_delay["rmse_m"]*100,
             "final_delay_first8_rmse_cm":values[2,3],"baseline_cases_passed":13,
             "plot_count":5,"paired_samples":len(t),"vertical_gain_retuned":False}
    write(OUT/"metrics.json",metrics)
    return metrics


def create_readme(m):
    text=f'''# 健康基准位置增益的仿真选值证据

整理日期：{DATE}。健康参数版本：`pv_baseline_20261021`。

本材料给出 Kx=diag(4,4,1.4) 在当前研究模型下的仿真依据。水平两个 4 经过候选比较，竖直 1.4 沿用旧值并通过指定升降试验。没有证明全局最优，也没有完成实机验证。

[下载五页 PDF](kx_gain_evidence.pdf)。下方高清图可直接查看，配套 SVG 可用于报告排版。

![证据总览](00_teacher_overview.png)

## 1. 固定其他增益，单独比较位置环

![固定内环对照](01_fixed_inner_kx_comparison.png)

水平速度 P=7、姿态 P=8、角速度 P=20、角速度 D=0.5；所有其他控制参数一致，仅 Kx=Ky 取 3、3.5、4。四种条件均为健康圆轨迹，20 s 试验中统一统计 0-8 s 的三维位置 RMSE。每个组合仅一条已保存记录，不给重复试验置信区间。Kx 从 3 到 4，理想工况误差降低 {m['kx_3_to_4_ideal_reduction_pct']:.1f}%，噪声+风+30 ms 工况降低 {m['kx_3_to_4_noise_wind_30ms_reduction_pct']:.1f}%。30 ms 工况全段峰值倾角从 9.13° 增至 10.01°，体现纠偏要求与瞬态响应的权衡。

## 2. 整套控制器调参前后

![匹配 SLX 轨迹](02_matched_slx_tracking.png)

旧增益本次重新运行实际 SLX；新增益使用已保存的实际 SLX。两者都是 8 s 健康圆轨迹，半径 0.3 m，1-7 s 运动，理想反馈、无噪声、无风。时间轴、参考和外部输入完全一致。RMSE 为 {m['before_circle_rmse_cm']:.3f} 与 {m['after_circle_rmse_cm']:.3f} cm，降低 {m['whole_tuning_reduction_pct']:.1f}%。

五组水平增益同时变化：位置 P 1.2→4，速度 P 2.5→7，姿态 P 4→8，角速度 P 8→20，角速度 D 0.1→0.5。竖直及偏航增益均不变。因此这项 87.1% 的改善不能全部归因于 Kx。

## 3. 保留延迟表现不佳的候选

![延迟下的权衡](03_delay_robustness_tradeoff.png)

两组方案 Kx=Ky 都为 4。高内环候选的速度/姿态/角速度 P 为 10/16/40，最终健康基准为 7/8/20；D 都为 0.5。高内环候选理想 RMSE 更小，但噪声、风和 30 ms 延迟下 RMSE 为 {m['aggressive_delay_rmse_cm']:.2f} cm，而最终参数前 8 s 为 {m['final_delay_first8_rmse_cm']:.2f} cm，两者均完成完整 8 s。

另一组 [5,12,18,45,0.6] 的延迟工况在 3.43 s 提前终止，故未作为完整 8 s RMSE 柱与上述试验混排；失败结果仍保留在 sources/aggressive_candidates.json。该对照说明需要检查内环及延迟，不能据此声称单独增加 Kx 导致振荡。

## 4. 最终参数的多场景验证

![验证覆盖](04_validation_coverage.png)

13 个已保存 SLX 场景满足预设工程验收，包括健康任务、阶跃、初态扰动、较大圆轨迹、10/30 ms 延迟噪声风、1/6 号电机剩余 80% 效能、积分步长减半。不同场景的门槛不同；它们是项目要求，不是适航或行业认证标准。图只显示 RMSE 项，完整判定同时检查峰值/末端误差、时长、限幅、有限性、积分和四元数约束等。

## 5. 可以与老师这样表述

> 在固定内环参数下，比较了水平位置增益 3、3.5、4 在四种反馈/扰动条件下的响应。Kx=Ky=4 在这些候选中降低了跟踪误差，并通过选定多场景工程验收，因此作为当前研究模型的健康基准。垂直位置增益 1.4 为沿用值，通过指定升降测试；本轮未单独优化，不能声称为最优。

## 6. 数据、复现与边界

- 数值定义：RMSE=sqrt(mean(||实际位置-期望位置||²))，先按每个采样点求三维范数，再做均方根。
- [复算指标](metrics.json)、[来源与 SHA-256](sources/provenance.json)、[固定内环候选](sources/kx_fixed_inner_candidates.json)、[两条轨迹采样](sources/paired_circle_timeseries.json)、[前后参数快照](sources/paired_circle_parameters.json)、[配对检查](sources/paired_circle_checks.json)、[最终验收](sources/baseline_validation.json)。
- 脚本：`simulink/validation/build_kx_gain_evidence.py`。有 sources 快照时可独立重画；添加 `--refresh-sources` 则从本地历史输出重新采集。PDF 环境需 reportlab，绘图环境需 numpy/scipy/matplotlib。
- 本轮绘图未更改控制器、动力学参数或 SLX；35 项既有源文件哈希均复核一致。
- 仿真动力模型仍采用 kT=7.45e-6、Q/T=0.01315 m（代理参数），没有切换成 AIR Gear 355 的成套系数。本材料验证的是该模型中的控制器选值，不证明动力参数真实性。
- 图表采用健康基准的历史 8 s 比较窗口；最新故障恢复入口的 24 s 仿真属于另一组试验。
- 每条曲线和柱形来自单次仿真记录，未提供重复随机种子统计结论或实机测试结论。

PNG 适合直接粘贴，SVG 为矢量图，可用于报告排版。PDF 汇总输出到仓库 `output/pdf/kx_gain_evidence_20261026.pdf`。
'''
    (OUT/"README.md").write_text(text,encoding="utf-8")


def create_pdf():
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader
    PDF.parent.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont("MSYH",str(FONT),subfontIndex=0))
    w,h=landscape(A4)
    c=canvas.Canvas(str(PDF),pagesize=(w,h))
    c.setTitle("健康基准 Kx 的仿真选值证据")
    c.setAuthor("eVTOL Simulation - simulation evidence")
    pages=[("00_teacher_overview","健康基准位置增益：仿真证据总览",
            "两类改善分别统计：固定内环 Kx 对照与整套调参对照；不得混同。"),
           ("01_fixed_inner_kx_comparison","固定内环的 Kx 对照",
            "4 个工况，每个增益与工况 1 条记录；RMSE 统一取前 8 s，峰值倾角取全 20 s。"),
           ("02_matched_slx_tracking","整套控制器调参前后的轨迹",
            "实际 SLX 配对，参考/时间/输入一致；同时改变五组水平增益，不能只归因于 Kx。"),
           ("03_delay_robustness_tradeoff","延迟工况与内环增益的权衡",
            "两个 Kx=4 方案在完整 8 s 窗口比较；提前终止的第三候选另保留，不混算。"),
           ("04_validation_coverage","最终健康基准的验证覆盖",
            "13 个指定场景满足既定工程验收；垂直 1.4 为沿用值，尚未单独优化。")]
    for i,(stem,title,note) in enumerate(pages,1):
        c.setFillColorRGB(.094,.20,.28); c.setFont("MSYH",16)
        c.drawString(32,h-34,title)
        c.setFont("MSYH",8.5); c.setFillColorRGB(.39,.45,.55)
        c.drawRightString(w-32,h-32,f"2026-10-26  |  {i}/5")
        image=ImageReader(str(OUT/(stem+".png")))
        iw,ih=image.getSize(); scale=min((w-52)/iw,(h-117)/ih)
        dw,dh=iw*scale,ih*scale
        c.drawImage(image,(w-dw)/2,81+(h-117-dh)/2,width=dw,height=dh,mask="auto")
        c.setFont("MSYH",9); c.setFillColorRGB(.094,.20,.28); c.drawString(32,57,note)
        c.setFont("MSYH",7.5); c.setFillColorRGB(.39,.45,.55)
        c.drawString(32,37,"来源：随附 README、metrics.json 与 sources/ 下可复算快照；完整来源含 SHA-256。")
        c.drawString(32,23,"范围：当前研究模型的单次仿真证据；不表示实机有效性、全局最优或统计显著性。")
        c.showPage()
    c.save()
    shutil.copy2(PDF, OUT / "kx_gain_evidence.pdf")
    print("PDF_OUTPUT",PDF)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--refresh-sources",action="store_true")
    parser.add_argument("--pdf-only",action="store_true")
    parser.add_argument("--pdf-python",default=sys.executable)
    args=parser.parse_args()
    if args.pdf_only:
        create_pdf(); return
    if args.refresh_sources or not (SOURCES/"paired_circle_timeseries.json").is_file(): collect()
    metrics=create_charts(); create_readme(metrics)
    subprocess.run([args.pdf_python,str(Path(__file__).resolve()),"--pdf-only"],check=True)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    print("FIGURES_OUTPUT",OUT)


if __name__=="__main__": main()
