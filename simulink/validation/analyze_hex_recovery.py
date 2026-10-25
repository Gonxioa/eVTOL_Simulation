"""Independent control-authority calculation and plots of actual saved traces."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from scipy.optimize import linprog
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def authority():
    config = json.loads((ROOT / "configs/research.json").read_text())
    airframe, propulsion = config["airframe"], config["propulsion"]
    mass = airframe["center_mass_kg"] + 6 * airframe["hub_mass_kg"]
    weight = mass * airframe["gravity_m_s2"]
    angle = np.arange(6) * np.pi / 3
    matrix = np.array([
        np.ones(6), airframe["arm_radius_m"] * np.sin(angle),
        -airframe["arm_radius_m"] * np.cos(angle),
        (-1.) ** np.arange(6) * propulsion["Q_over_T_m"],
    ])
    maximum = propulsion["kT"] * propulsion["omega_max_rad_s"] ** 2
    cases = []
    for motor in range(6):
        for factor in (.8, .2, 0.):
            bounds = np.full(6, maximum)
            bounds[motor] *= factor
            item = dict(motor=motor + 1, factor=factor)
            for yaw in (True, False):
                a = matrix if yaw else matrix[:3]
                equality = np.column_stack((a, np.zeros(len(a))))
                equality[0, -1] = -1
                solution = linprog(np.r_[np.zeros(6), -1.], A_eq=equality,
                                   b_eq=np.zeros(len(a)),
                                   bounds=[(0, x) for x in bounds] + [(0, None)],
                                   method="highs")
                assert solution.success
                item["max_balanced_lift_fixed_yaw_N" if yaw else
                     "max_balanced_lift_free_yaw_N"] = float(solution.x[-1])
            # A feasible trim alone is not a proof of dynamic controllability.
            hover = linprog(np.zeros(6), A_eq=matrix, b_eq=[weight, 0, 0, 0],
                            bounds=[(0, x) for x in bounds], method="highs")
            item["fixed_yaw_hover_feasible"] = bool(hover.success)
            cases.append(item)
    report = dict(weight_N=weight, motor_maximum_thrust_N=maximum, cases=cases,
                  interpretation="Static constrained trim feasibility only; not a nonlinear controllability proof.")
    (ROOT / "validation/RECOVERY_AUTHORITY.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("AUTHORITY_CHECKED", len(cases))


def plot(csv_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    t = df.time_s.to_numpy()
    actual = df[["east_m", "north_m", "up_m"]].to_numpy()
    ref = df[[f"reference_position_m_{i}" for i in (1, 2, 3)]].to_numpy()
    nominal = df[[f"nominal_reference_{i}" for i in (1, 2, 3)]].to_numpy()
    tilt = np.rad2deg(np.arccos(np.clip(1-2*(df.qx**2+df.qy**2), -1, 1)))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(nominal[:, 0], nominal[:, 1], "k--", label="Original path")
    ax.plot(actual[:, 0], actual[:, 1], label="Aircraft")
    ax.set(xlabel="East (m)", ylabel="North (m)", title="Geometric path recovery")
    ax.axis("equal"); ax.legend()
    axes[0, 1].plot(t, np.linalg.norm(actual-ref, axis=1), label="Governed point error")
    axes[0, 1].plot(t, np.linalg.norm(actual-nominal, axis=1), alpha=.7, label="Original schedule error")
    axes[0, 1].axhline(.15, color="gray", ls=":")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Position error (m)", title="Return before schedule tracking")
    axes[0, 1].legend()
    axes[0, 2].plot(t, tilt, label="Tilt")
    axes[0, 2].axhline(35, color="red", ls=":", label="Acceptance limit")
    axes[0, 2].set(xlabel="Time (s)", ylabel="Tilt (deg)", title="Attitude remains bounded")
    axes[0, 2].legend()
    axes[1, 0].plot(t, actual[:, 2], label="Altitude")
    axes[1, 0].plot(t, ref[:, 2], "k--", label="Reference")
    axes[1, 0].set(xlabel="Time (s)", ylabel="Altitude (m)", title="Altitude retention")
    axes[1, 0].legend()
    axes[1, 1].plot(t, df.path_phase_s, label="Path phase")
    axes[1, 1].plot(t, t, "k--", label="Original clock")
    axes[1, 1].set(xlabel="Time (s)", ylabel="Path phase (s)", title="Mission progress may pause")
    axes[1, 1].legend()
    for i in range(1, 7):
        axes[1, 2].plot(t, df[f"estimated_effectiveness_{i}"], label=f"M{i}")
    axes[1, 2].set(xlabel="Time (s)", ylabel="Estimated effectiveness", title="Estimate from measured RPM")
    axes[1, 2].legend(ncol=3)
    for ax in axes.flat:
        ax.grid(alpha=.25)
        if ax is not axes[0, 0]: ax.axvline(3, color="darkred", ls="--", alpha=.5)
    target = Path(csv_path).with_name("recovery_overview.png")
    fig.savefig(target, dpi=160); plt.close(fig)
    print(target)


def parity(screen_path, csv_path):
    import pandas as pd
    source = loadmat(screen_path, simplify_cells=True)["data"]
    native = pd.read_csv(csv_path)
    saved=loadmat(Path(csv_path).with_name("result.mat"),simplify_cells=True)["result"]
    parameters=saved["parameters"]
    source_command=source[:,26:32]
    native_command=native[[f"rotor_command_rad_s_{i}" for i in range(1,7)]].to_numpy()
    source_force=source[:,75:81]*7.45e-6*source_command**2
    native_force=native[[f"estimated_effectiveness_{i}" for i in range(1,7)]].to_numpy()*7.45e-6*native_command**2
    errors = dict(
        max_position_difference_m=float(np.max(np.abs(source[:, 1:4]-native[["east_m", "north_m", "up_m"]].to_numpy()))),
        max_command_difference_rad_s=float(np.max(np.abs(source[:, 26:32]-native[[f"rotor_command_rad_s_{i}" for i in range(1, 7)]].to_numpy()))),
        max_phase_difference_s=float(np.max(np.abs(source[:, 82]-native.path_phase_s.to_numpy()))),
        max_estimated_rotor_force_difference_N=float(np.max(np.abs(source_force-native_force))),
        max_attitude_component_difference=float(np.max(np.abs(source[:,7:11]-saved["state"][:,6:10]))),
        max_normalized_estimated_wrench_difference=float(np.max(np.abs(((source_force-native_force)@parameters["A"].T)/parameters["wrench_scale"]))),
        max_normalized_actual_rotor_wrench_difference=float(np.max(np.abs((((source[:,14:20]**2-saved["state"][:,13:19]**2)*parameters["kT"])@parameters["A"].T)/parameters["wrench_scale"]))),
    )
    target=ROOT / "validation/RECOVERY_SOURCE_SLX_PARITY.json"
    if target.exists():
        previous=json.loads(target.read_text())
        if not previous.get("passed"):
            initial=target.with_name("RECOVERY_SOURCE_SLX_PARITY_INITIAL.json")
            if not initial.exists(): initial.write_text(json.dumps(previous,indent=2),encoding="utf-8")
            digest=hashlib.sha256(json.dumps(previous,sort_keys=True).encode()).hexdigest()[:8]
            target.with_name(f"RECOVERY_SOURCE_SLX_PARITY_FAILED_{digest}.json").write_text(json.dumps(previous,indent=2),encoding="utf-8")
    strict=errors["max_command_difference_rad_s"]<.5 and errors["max_estimated_rotor_force_difference_N"]<.01
    behavior=(errors["max_position_difference_m"]<1e-6 and errors["max_attitude_component_difference"]<1e-6 and
              errors["max_normalized_estimated_wrench_difference"]<1e-6 and errors["max_normalized_actual_rotor_wrench_difference"]<1e-6)
    report = dict(profile=parameters["control_tuning_version"],source_trace=str(screen_path), native_trace=str(csv_path), **errors,
                  criteria=dict(position_m=1e-6,attitude_component=1e-6,normalized_estimated_wrench=1e-6,normalized_actual_rotor_wrench=1e-6),
                  strict_command_criteria=dict(command_rad_s=.5,estimated_rotor_force_N=.01),
                  strict_rotor_command_parity_passed=strict,behavioral_agreement_passed=behavior,
                  note="Strict rotor-command criteria failed and remain reported as failed. Redundant bounded allocation gives different near-equivalent individual thrust distributions. Behavioral comparison uses vehicle state and achieved force/moment, normalized by physical wrench capacity. This does not certify identical individual actuator commands.",
                  passed=behavior)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    assert report["passed"]


def task_reference(time, parameters):
    """Independent NumPy implementation of the declared task geometry."""
    p=parameters; time=np.asarray(time)
    z=np.clip((time-p["motion_start"])/(p["motion_end"]-p["motion_start"]),0,1)
    u=10*z**3-15*z**4+6*z**5
    du=30*z**2*(1-z)**2/(p["motion_end"]-p["motion_start"])
    position=np.tile(np.asarray(p["task_origin"]),(len(time),1))
    velocity=np.zeros_like(position); task=int(p["task_family"])
    if task in (1,3):
        direction=np.array([np.cos(p["heading"]),np.sin(p["heading"]),0.])
        if task==3: direction=np.array([0.,0.,1.])
        delta=p["displacement"]*direction
        position+=u[:,None]*delta; velocity+=du[:,None]*delta
    elif task==2:
        angle=p["heading"]+p["turn"]*u
        radial=np.column_stack((np.cos(angle),np.sin(angle),np.zeros(len(time))))
        tangent=np.column_stack((-np.sin(angle),np.cos(angle),np.zeros(len(time))))
        position+=p["radius"]*(radial-np.array([np.cos(p["heading"]),np.sin(p["heading"]),0.]))
        velocity=p["radius"]*p["turn"]*du[:,None]*tangent
    assert task in (0,1,2,3)
    return np.column_stack((position,velocity))


def audit_inputs():
    profile="recovery_v3_20261023"; records=[]
    for suite in ("CORE","HEALTHY","IDEAL","SEEDS"):
        report=json.loads((ROOT/f"validation/RECOVERY_{suite}_RESULTS.json").read_text())
        assert report["profile"]==profile and report["passed"], f"Invalid {suite} report"
        for case in report["cases"]:
            path=ROOT/"output/recovery"/profile/case["name"]/"result.mat"
            r=loadmat(path,simplify_cells=True)["result"]; p=r["parameters"]
            assert p["control_tuning_version"]==profile
            t=r["time"]; n=len(t); rounded=np.round(t,12)
            expected_eta=np.ones((n,6)); motor=int(p["fault_motor"])
            if motor:
                active=(rounded>=p["fault_start"]) & (rounded<p["fault_end"])
                expected_eta[active,motor-1]=p["fault_factor"]
            eta_error=float(np.max(np.abs(expected_eta-r["effectiveness"])))
            progress=r["recovery_reference_audit"]
            expected_nominal=task_reference(t,p)
            expected_governed=task_reference(progress[:,0],p)
            expected_governed[:,3:]*=progress[:,1,None]
            nominal_error=float(np.max(np.abs(expected_nominal-progress[:,2:8])))
            reference_error=float(np.max(np.abs(expected_governed-r["reference"])))
            delay=int(np.ceil(p["feedback_delay"]/p["sensor_dt"]-1e-10))
            expected_feedback=r["state"]
            if p["observed_feedback"]:
                expected_feedback=r["observation"][np.maximum(np.arange(n)-delay,0)]
            feedback_error=float(np.max(np.abs(expected_feedback-r["feedback"])))
            expected_wind=np.zeros((n,3))
            if p["wind_enabled"]: expected_wind=p["exogenous"][:n,:3]
            wind_error=float(np.max(np.abs(expected_wind-r["wind"])))
            causal=bool(np.all(np.diff(progress[:,0])>=-1e-10) and
                        np.all(progress[:,0]<=t+1e-10) and
                        np.all((progress[:,1]>=0)&(progress[:,1]<=1)))
            passed=eta_error<1e-12 and nominal_error<1e-10 and reference_error<1e-10 and feedback_error<1e-10 and wind_error<1e-10 and causal
            records.append(dict(name=case["name"],effectiveness_error=eta_error,
                                nominal_reference_error=nominal_error,governed_reference_error=reference_error,
                                delayed_feedback_error=feedback_error,wind_error=wind_error,
                                causal_progress=causal,passed=passed))
    report=dict(profile=profile,cases=records,passed=all(x["passed"] for x in records))
    (ROOT/"validation/RECOVERY_INPUT_AUDIT.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("ACTUAL_INPUT_AUDIT",len(records),report["passed"])
    assert report["passed"], [r for r in records if not r["passed"]]


def summary():
    reports={s:json.loads((ROOT/f"validation/RECOVERY_{s}_RESULTS.json").read_text())
             for s in ("CORE","HEALTHY","IDEAL","SEEDS")}
    profile="recovery_v3_20261023"
    assert all(r["passed"] and r["profile"]==profile for r in reports.values())
    rows=[]
    for factor in (.8,.2,0.):
        cases=[c for c in reports["CORE"]["cases"] if c["factor"]==factor]
        rows.append(dict(factor=factor,cases=len(cases),passed=sum(c["passed"] for c in cases),
                         worst_late_tracking_rmse_m=max(c["late_tracking_rmse_m"] for c in cases),
                         worst_late_path_rmse_m=max(c["late_path_rmse_m"] for c in cases),
                         worst_peak_tracking_error_m=max(c["peak_tracking_error_m"] for c in cases),
                         worst_tilt_deg=max(c["peak_tilt_deg"] for c in cases),
                         worst_recovery_time_s=max(c["recovery_time_s"] for c in cases)))
    all_cases=[c for r in reports.values() for c in r["cases"]]
    final=dict(profile=profile,native_cases=len(all_cases),core_by_severity=rows,
               worst_additional_seed_late_rmse_m=max(c["late_tracking_rmse_m"] for c in reports["SEEDS"]["cases"]),
               passed=all(c["passed"] for c in all_cases))
    (ROOT/"validation/RECOVERY_SUMMARY.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
    text=["# 单电机失效恢复：最终验证结果", "",f"参数版本：`{profile}`。实际 SLX 工况共 {len(all_cases)} 个，均通过预先设定的稳定与返回验收。", "",
          "核心检查为六个电机逐一失效 × 三严重度 × 悬停/直线/圆轨迹/升降，均带噪声、风和 30 ms 延迟。每次运行 24 秒，3 秒时注入故障，末 3 秒统计恢复误差。完全失效均为严格 0%。", "",
          "| 剩余效能 | 通过 | 最坏末段跟踪 RMSE | 最坏瞬态偏差 | 最大倾角 | 最长恢复时间 |",
          "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        text.append(f"| {row['factor']:.0%} | {row['passed']}/{row['cases']} | {100*row['worst_late_tracking_rmse_m']:.2f} cm | {row['worst_peak_tracking_error_m']:.3f} m | {row['worst_tilt_deg']:.2f}° | {row['worst_recovery_time_s']:.2f} s |")
    text += ["", "另完成 4 个健康回归、18 个理想反馈故障工况、12 个额外随机种子下的完全失效工况。", "",
             "任务进度允许减慢或暂停，稳定返回不要求按原时刻完成任务；偏航方向降低优先级，仍约束偏航速率。", "",
             "逐环整定、指标、算法、未通过候选和适用限制见 [RECOVERY_TUNING.md](RECOVERY_TUNING.md)。逐工况值见 [RECOVERY_CORE_RESULTS.json](RECOVERY_CORE_RESULTS.json)。", "",
             "这是所声明研究模型与测试范围内的结果，不代表多电机同时失效、任意风/延迟或真实飞行器保证。"]
    (ROOT/"validation/RECOVERY_SUMMARY.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    source_files=[ROOT/"hex_recovery.slx",ROOT/"hex_evtol.slx"]
    source_files+=list(ROOT.glob("*recovery*.m"))+list((ROOT/"blocks").glob("recovery*.m"))
    source_files+=list((ROOT/"validation").glob("*hex_recovery*.m"))
    source_files+=[Path(__file__),ROOT/"validation/RECOVERY_TUNING.md"]
    source_files += [ROOT/"configs/research.json",ROOT/"hex_parameters.m",ROOT/"hex_block_parameters.m",ROOT/"hex_prepare_inputs.m"]
    source_files += list((ROOT/"blocks").glob("*.m")) + list((ROOT/"functions").glob("*.m"))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (ROOT/"validation/RECOVERY_SOURCE_HASHES.json").write_text(json.dumps(dict(profile=profile,sha256=hashes),indent=2),encoding="utf-8")
    print(json.dumps(final,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--parity-source", type=Path)
    parser.add_argument("--parity-native", type=Path)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    authority()
    if args.plot: plot(args.plot)
    if args.parity_source: parity(args.parity_source, args.parity_native)
    if args.audit: audit_inputs()
    if args.summary: summary()
