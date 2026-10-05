"""Save reviewable evidence, actual metrics and explicit unmet stage conditions."""
from pathlib import Path
import subprocess
import shutil
import numpy as np
from .common import ROOT, CONFIG, read_json, write_json, digest
from .provenance import ANALYSIS_FIELDS, DEMO_UNITS, equations


def run(out):
    docs = ROOT / "docs" / "uavfd_f550"
    docs.mkdir(parents=True, exist_ok=True)
    audit = read_json(out / "audit.json")
    analytical = read_json(out / "analytical_validation.json")
    model = read_json(out / "combination_model.json")
    normal = read_json(out / "normal_replay.json")
    vib = read_json(out / "vibration_validation.json")
    vib_model = read_json(out / "vibration_model.json")
    demo = read_json(out / "demo_vibration.json")
    fetch = read_json(out / "source_fetch.json")
    original = read_json(out / "original_workspace.json")
    protected = {k: {"original_sha256": v, "current_sha256": digest(ROOT/k), "unchanged": digest(ROOT/k) == v}
                 for k, v in original["protected_file_sha256"].items()}
    if not all(v["unchanged"] for v in protected.values()):
        raise RuntimeError("Pre-existing work was modified unexpectedly")
    write_json(docs/"protected_workspace.json", protected)
    write_json(docs/"data_hashes.json", read_json(out/"data_hashes.json"))
    write_json(docs/"source_archive_manifest.json", fetch)
    reference_check = read_json(out/"reference_access_check.json") if (out/"reference_access_check.json").exists() else {"status": "not_run_in_this_output_directory"}
    write_json(docs/"reference_access_check.json", reference_check)
    write_json(docs/"equation_ledger.json", equations())
    write_json(docs/"validation_metrics.json", {"analytical": analytical, "normal": normal,
               "vibration": vib["results"], "vibration_model": vib_model, "demo": demo})
    write_json(docs/"audit_summary.json", {"records": [{k: r[k] for k in
               ("name", "split", "bytes", "md5", "sha256", "release_match", "parameters_count", "VER", "MODE", "ARM",
                "motor_output_map", "imu_timestamps_identical", "imu_common_n", "timing", "loader_warnings",
                "duplicate_variable_names", "firmware_string_preservation")} for r in audit["records"]],
               "scripts": audit["scripts"], "manifest": audit["manifest"]})
    sha = fetch["logged_commit"]["sha"]
    base_url = f"https://github.com/ArduPilot/ardupilot/blob/{sha}"
    sources = [
        {"id": "S01", "title": "UAV-FD dataset v1", "authors": "Baldini, D'Alleva, Felicetti, Ferracuti, Freddi, Monteriù / Università Politecnica delle Marche",
         "year": 2023, "url": "https://zenodo.org/records/7648996", "doi": "10.5281/zenodo.7648996", "version": "v1, published 2023-02-26",
         "location": "Description, individual file sizes/MD5; original main.m, compare3.m, sync_zoh_exa.m", "access": "record/API and local original scripts read"},
        {"id": "S02", "title": "UAV-FD: a dataset for actuator fault detection in multirotor drones", "authors": "Alessandro Baldini et al.",
         "year": 2023, "url": "https://iris.univpm.it/handle/11566/320071", "doi": "10.1109/ICUAS57906.2023.10156060", "version": "ICUAS 2023 pp.998–1004",
         "location": "Crossref metadata; university repository abstract", "access": "Full text not acquired from checked sources: repository lists no attached file; direct retrieval failed, IEEE 202 has no paper content. Supplemental read-only Zotero title enumeration found no matching paper in the current scope (receipt: reference_access_check.json). Physical setup and 5/10 geometry not confirmed."},
        {"id": "S03", "title": "Identification of a Multicopter", "authors": "ArduPilot documentation contributors", "year": None,
         "url": "https://ardupilot.org/copter/docs/systemid-model-development.html", "version": "access snapshot, current document not historical firmware",
         "location": "Model Scope; Model Structure; verification discussion", "access": "read; equation images discussed alongside body definitions. Example quadrotor values not adopted."},
        {"id": "S04", "title": "ArduPilot historical base source", "authors": "ArduPilot contributors", "year": 2022, "url": base_url,
         "version": sha, "location": "AP_MotorsMatrix.cpp:793–799, add_motor; AP_InertialSensor/LogStructure.h:45–61,116–122; AP_InertialSensor_Logging.cpp:39–94; AP_InertialSensor_Backend.cpp:537,713–747; AP_InertialSensor.cpp:1943–1965,1987–1995; AP_BLHeli.cpp:1330–1334,1403–1406,1447–1463; AP_ESC_Telem/LogStructure.h:8–36; AP_Logger/LogStructure.h:54,64,69,1331–1332",
         "access": "source snapshots read; only public base commit is established. Dataset custom logging patch and strings missing."},
        {"id": "S05", "title": "Active Propeller Fault Detection and Isolation in Multirotors Via Vibration Model", "authors": "Alessandro Baldini, Riccardo Felicetti, Alessandro Freddi, Andrea Monteriù",
         "year": 2026, "url": "https://arxiv.org/html/2604.07281v1", "version": "arXiv:2604.07281v1 2026-04-08", "location": "§2 equations(1)–(6), mass/COM and rotation; §3 frequency interpretation",
         "access": "HTML full text read; our E7 follows an explicit support-reaction derivation. No octarotor parameters or active excitation imported."},
        {"id": "S06", "title": "Propeller damage detection, classification and estimation in multirotor vehicles", "authors": "Claudio Pose, Juan I. Giribet, Gabriel Torre",
         "year": 2024, "url": "https://arxiv.org/html/2410.05447v1", "version": "arXiv:2410.05447v1", "location": "§1.3–1.4; frequency-domain methodology; UAV-FD transfer section",
         "access": "HTML read; only diagnostic context, not evidence validating our dynamics or damage geometry"},
        {"id": "S07", "title": "FlameWheel 550 User Manual", "authors": "DJI", "year": 2015,
         "url": "https://dl.djicdn.com/downloads/flamewheel/en/F550_User_Manual_v2.0_en.pdf", "version": "V2.0, 2015.05",
         "location": "printed p.8 Specifications (PDF page index 7)", "access": "spec table read, PDF archived. 550mm frame wheelbase is generic; not proof of dataset hardware or COM radius; 478g is frame only; listed motor/prop combination differs from Air Gear clue."},
        {"id": "S08", "title": "Standard acceleration of gravity", "authors": "NIST / CODATA", "year": 2022,
         "url": "https://physics.nist.gov/cgi-bin/cuu/Value?gn", "version": "2022 CODATA site snapshot", "location": "numerical value 9.80665 m/s2 (exact conventional standard)",
         "access": "read; model convention, not measured local gravity"},
        {"id": "S09", "title": "Legacy project airframe candidates", "authors": "Existing local project", "year": None,
         "url": "local:hex_physical_characteristics.py", "version": protected["hex_physical_characteristics.py"]["original_sha256"],
         "location": "uncommitted working-tree parameter definitions", "access": "reviewed, not accepted as UAV-FD truth; provenance claims unresolved for exact aircraft"},
        {"id": "S10", "title": "AIR GEAR 350 product candidate", "authors": "T-Motor", "year": None,
         "url": "https://store.tmotor.com/product/air-gear-350.html", "version": "requested legacy URL", "location": "not accessible",
         "access": "direct request failed; no manufacturer values adopted"}
    ]
    for entry in sources:
        entry["access_date"] = "2026-10-04"
    write_json(docs/"sources.json", sources)
    ledger = []
    def add(key, definition, unit, value, status, source, location, implementation, evidence, limits, flights=None):
        ledger.append({"key": key, "definition": definition, "unit": unit, "coordinate": "NED world / FRD body unless stated",
                       "value": value, "status": status, "source_ids": source, "source_location": location,
                       "implementation": implementation, "verification": evidence, "limits": limits,
                       "calibration_flights": flights or [], "method": "as described in MODEL_SPEC",
                       "residual_location": "validation_metrics.json" if flights else None})
    formal = read_json(CONFIG/"uavfd_aircraft.json")
    for key in ["mass", "inertia", "rotor_positions", "kT", "kQ", "imu_lever", "imbalance_mu", "modal_frequency", "modal_damping", "motor_time_constant", "fault_geometry_5_10"]:
        p = formal[key]
        add(key, key, p["unit"], None, p["status"], ["S01", "S02"], "Original exact hardware/geometry evidence unavailable",
            "configs/uavfd_aircraft.json", "null enforced by body_from_config", "Do not import legacy or demo values")
    add("gravity", "standard gravity convention", "m/s2", 9.80665, "source_confirmed_convention", ["S08"], "NIST value", "dynamics.py/analysis.py", "freefall/static support", "not site measurement")
    add("motor_count", "six propulsors", "1", 6, "source_confirmed", ["S01"], "hexarotor description; six ESC tables", "data.py/signals.py", "audit_summary.json", "exact geometry unknown")
    add("motor_output_map", "SERVO function Motor1–6", "PWM us-equivalent commands", audit["records"][0]["motor_output_map"], "original_data_confirmed", ["S01", "S04"], "all 18 PARM; SERVOx_FUNCTION 33–38", "audit.py/signals.py", "audit_summary.json", "PWM-equivalent number under DShot, not evidence of physical pulse width")
    add("motor_angles", "base HEXA X bearing clockwise in FRD", "deg", [90,-90,-30,150,30,-150], "derived_from_logged_base_custom_layout_unverified", ["S04"], "AP_MotorsMatrix.cpp:793–799", "analysis.py/ANGLES", "per_motor_wrench_sign", "physical motor locations relative to COM not recovered")
    add("spin_sign", "FRD +z CW rotor spin", "1", [1,-1,1,-1,-1,1], "derived_from_logged_base_custom_wiring_unverified", ["S04"], "same motor array; CW/CCW factors", "analysis.py/SPIN; dynamics.py", "hover/per-motor torque", "software config, not independent visual confirmation")
    add("ESC_RPM", "mechanical logged slew adjusted rotation rate", "rpm", None, "base_source_confirmed_custom_path_unverified", ["S04"], "AP_BLHeli.cpp:1447–1451; AP_ESC_Telem/LogStructure.h:12–13", "signals.py; rpm_to_rad_s", "RPM conversion analytic", "no extra pole division; real pole count and temporal delays not measured")
    add("MOT_THST_EXPO", "ArduPilot actuator normalization parameter", "1", .63, "original_data_confirmed", ["S01"], "all 18 PARM", "audit.py", "audit_summary.json", "not our dimensional kT")
    add("a_z_effective", "slope in -fz=a*sum(Omega²)+b", "(m/s2)/(rad/s)^2", model["vertical"]["coefficients"][0], "flight_identified_effective_combination", ["S03","S04"], "E10 derived; calibration log fields", "analysis.py/calibrate", "normal_replay.json", "intercept large; not physical kT/m", model["calibration_flights"])
    add("b_z_effective", "vertical nuisance intercept", "m/s2", model["vertical"]["coefficients"][1], "flight_identified_nuisance", ["S01"], "E10", "analysis.py/calibrate", "per_flight_estimates", "not known sensor bias; captures model mismatch", model["calibration_flights"])
    for j, axis in enumerate("xyz"):
        add("gamma_"+axis, "effective angular input gain", "(rad/s2)/(rad/s)^2", model["angular"][j]["coefficients"][0], "flight_identified_effective_combination", ["S03","S04"], "E11 integral regression", "analysis.py/calibrate", "NO_FAULT4 development replay", "no unique L,kT,J,kQ or delay; omitted coupling", model["calibration_flights"])
        add("b_omega_"+axis, "angular nuisance intercept", "rad/s2", model["angular"][j]["coefficients"][1], "flight_identified_nuisance", ["S01"], "E11", "analysis.py/calibrate", "normal_replay.json", "not calibrated physical torque", model["calibration_flights"])
    add("healthy_order_band_gain", "1x logged energy/Omega^4", "(m/s2)^2/(rad/s)^4", vib_model["healthy_per_motor_band_gain"], "flight_identified_observation_combination", ["S01","S05","S06"], "E13 support-reaction and variance derivation", "analysis.py/vibration", "vibration_validation.json", "overlapping motor bands; modal/log gains and mu inseparable", [n for n in model["calibration_flights"]])
    add("damage_order_band_increment", "mean of nonnegative effective increments; raw values also retained", "(m/s2)^2/(rad/s)^4", vib_model["shared_damage_increment"], "provisional_not_validated", ["S01","S05"], "E13; raw fault increments in vibration_model.json", "analysis.py/vibration", "FAULT_M3_5 developer metrics", "M1 increment negative; no damage transfer validation; no physical mu or 5/10 mapping", ["FAULT_M1_5.mat","FAULT_M2_5.mat"])
    config = read_json(CONFIG/"analysis.json")
    for key, value in config.items():
        if key not in ("schema_version", "status", "acceptance"):
            unit, definition = ANALYSIS_FIELDS[key]
            add("analysis."+key, definition, unit, value, "method_assumption", [],
                "configs/analysis.json frozen before fit", "signals.py/analysis.py", "sampling/alias analytic checks; limits assessed on development", "not aircraft property or online filtering path")
    demo_cfg = read_json(CONFIG/"demo.json")
    for key, value in demo_cfg.items():
        if key not in ("status", "purpose"):
            add("demo."+key, "analytical demonstration setting", DEMO_UNITS[key], value, "demo_assumption", [], "configs/demo.json", "verification.py", "analytical_validation.json", "never UAV-FD aircraft parameter")
    fixed_choices = [
        ("powered_pwm_threshold", "Author candidate powered interval rule", "us-equivalent", 1450, ["S01"], "sync_zoh_exa.m mean(PWM)>=1450", "Not proof of airborne/hover state or fault onset"),
        ("max_imu_gap_s", "Contiguous common-IMU candidate gap", "s", .03, [], "signals.pieces call", "Selected method threshold; not expected physical sample interval"),
        ("max_state_pwm_gap_s", "RCOU/ATT/XKF interpolation support gap", "s", .1, [], "signals.support calls", "Selected method threshold; unknown processing delays remain"),
        ("nyquist_margin_fraction", "Retained cutoff upper fraction of lower sample rate", "1", .45, [], "vibration.checked_sample/signals.segments", "Explicit margin below Nyquist .5; not original sensor anti-alias reconstruction"),
        ("demo_observation_filter_order", "Demonstration observation Butterworth order", "1", 6, [], "vibration.checked_sample", "Offline forward-backward filter, no aircraft evidence"),
        ("base_vibe_floor_hz", "Historical base VIBE floor cutoff", "Hz", 5, ["S04"], "AP_InertialSensor.cpp:1943–1965", "Requires unavailable pre-log raw acceleration; not applied to exported IMU"),
        ("base_vibe_energy_hz", "Historical base VIBE residual-square smoothing cutoff", "Hz", 2, ["S04"], "AP_InertialSensor.cpp:1943–1965,1987–1995", "Base source reference only; customized path unknown"),
    ]
    for key, definition, unit, value, src, location, limits in fixed_choices:
        add(key, definition, unit, value, "base_source_reference" if key.startswith("base_vibe") else "method_assumption",
            src, location, "signals.py/vibration.py", "MODEL_SPEC; checks.txt", limits)
    for key, unit, value in [("modal_input_gain", "1/kg", [[1,0,0],[0,1,0]]),
                             ("modal_output_shape", "1", [[1,0],[0,1],[.2,.3]]),
                             ("initial_rotor_phase", "rad", 0), ("internal_rates_hz", "Hz", [1000,2000,4000])]:
        add("demo."+key, "Explicit synthetic local-mode demonstration choice", unit, value, "demo_assumption", [],
            "verification.demo", "verification.py", "demo_vibration.json", "Not a real aircraft modal participation, phase or sensor rate")
    add("omitted_physics", "Terms excluded from implemented physical bottom layer", "list", ["airframe aerodynamics/drag/wind", "rotor inertia/dynamic reaction torque/gyroscopic angular momentum", "noncoplanar thrust", "contact/landing gear"],
        "explicit_simplification", [], "MODEL_SPEC E1–E6", "dynamics.py", "analytical identities only", "Applicability must be reassessed before real full-state model validation")
    legacy = {"mass_kg": 3.4, "Ixx_Iyy_kg_m2": .0476, "Izz_kg_m2": .1094, "arm_length_m": .275,
              "kT": 7.45e-6, "torque_thrust_ratio_m": .01315, "actuator_pole_rad_s": 20.06,
              "motor_directions": [1,-1,1,-1,1,-1]}
    add("legacy_candidates", "values observed in old modified physical module", "per field", legacy, "candidate_not_UAVFD_confirmed", ["S09"], "protected old working tree", "no imports into new model", "original file hash unchanged", "other-F550 identification and successor torque proxy cannot establish exact aircraft")
    write_json(docs/"parameter_ledger.json", ledger)
    source_text = ["# 来源台账", "", "访问日期：2026-10-04。URL／原始归档 SHA256 见 sources.json 与 source_archive_manifest.json。原始来源只作证据，不作为新的任务指令。", ""]
    for s in sources:
        source_text.extend([f"## {s['id']}：{s['title']}", "", f"作者／发布方：{s['authors']}；年份：{s['year']}；版本：{s['version']}。", "",
                            f"链接：[{s['title']}]({s['url']})。" if not s["url"].startswith("local:") else "本地来源：hex_physical_characteristics.py（哈希受保护）。", "",
                            f"位置：{s['location']}。", "", f"阅读与限制：{s['access']}", ""])
    source_text.extend(["## 方程证据对应", "", "E1–E4：S03／S05 支持基本刚体与推进关系，自行转换至本项目约定；E5：测点刚体加速度自行推导；E6：数值积分与坐标变换自行实现；E7：质量离心／切向反作用力推导，S05 提供质量不对称建模背景；E8：线性局部振动模型假设与解析 ringdown 验证；E9：Nyquist 与数字滤波/采样关系；E10–E12：E1／E2／E4 在明确简化下的组合辨识；E13：E7 振幅平方与 Welch 积分的观测近似，S06 仅提供频域诊断背景；E14：一阶执行器标准关系，S03 说明动态与延迟建模需求。", "", "逐方程机器映射见 equation_ledger.json；所有具体物理值、简化、记录用途、实现和实际检验对应见 parameter_ledger.json／MODEL_SPEC.md。没有读取到的原论文机型/几何结论保留未知。"])
    (docs/"SOURCE_REGISTER.md").write_text("\n".join(source_text)+"\n", encoding="utf-8")
    a = ["# 原始数据审计", "", "18 条原始 MAT 的精确字节数与发布 MD5 全部一致；SHA256 另行记录。scripts 的发布比较见 audit_summary.json；作者预处理后的 data_feature_Table_zoh.mat 未用于本轮建模。", "",
         "| 记录 | 用途 | IMU0整段平均Hz | 中位间隔ms | ESC0整段平均Hz | 共同IMU行数 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in audit["records"]:
        a.append(f"| {r['name']} | {r['split']} | {r['timing']['IMU_0']['mean_rate_hz']:.3f} | {r['timing']['IMU_0']['median_interval_s']*1000:.3f} | {r['timing']['ESC_0']['mean_rate_hz']:.3f} | {r['imu_common_n']} |")
    a.extend(["", "这些时间统计属于完整性／采样元数据，锁定测试没有计算幅值统计、频谱或特征。开发使用5条标定+2条开发，11条最终记录始终锁定。", "", "18 条均包含972项PARM；FRAME_CLASS=2、FRAME_TYPE=1、MOT_THST_EXPO=0.63、C9–C14 Motor1–6一致。VER均为4.4.0、FWT=0、GH=589333444=0x232083c4。公开基础提交可解析为232083c4d23be21d694d130f639fb6497bd57f98，2022-09-27；这不是作者定制补丁的恢复证据。FWS/MSG和单位字符串在数字MAT导出中未恢复；未声称同固件完全复现。", "", "首条记录的IMU／RCOU／ESC0整段平均频率为362.793／93.344／8.963Hz，与任务线索吻合。IMU GHz/AHz为内部速率字段，不是这些日志频率。停飞缺口使部分记录平均值明显低于局部频率，必须按段处理。", "", "三IMU唯NO_FAULT1完整时间戳相同，其他取交集后平均；逐通道间隔、非递增与长间隔计数见 audit_summary.json。时间缺口的真实丢样数／采样触发原因未知，gap_gt_2_median只是警示计数。PARM、VER、ARM、MODE、RCOU、ESC_0–5、IMU_0–2、ATT、RATE、PIDR/PIDP/PIDY、XKF1_0/1与VIBE有实际记录；label_only未当通道。", "",
         "| 通道 | 标签字段与单位 | 处理与限制 |", "| --- | --- | --- |", "| TimeUS | 启动后us | 转换为秒，保留绝对时钟；采样/执行器/滤波延迟未分离 |", "| IMU GyrX/Y/Z | rad/s；FRD | 历史发布路径经滤波；定制高频日志位置未知 |", "| IMU AccX/Y/Z | m/s²；FRD比力 | 已有日志处理，不再套20Hz固件滤波；高频并非未滤波真值 |", "| ESC RPM / RawRPM | rpm | 分别slew后/前；机械转速转换见历史BLHeli，电极数真实性未另证 |", "| ESC Volt / Curr | V / A | 通道对齐不外推；不据低频遥测辨识毫秒级响应 |", "| RCOU C9–C14 | us-equivalent | DShot模式下日志归一命令标度，不假称实际PWM脉宽 |", "| ATT Roll/Pitch/Yaw | deg，估计姿态 | 转rad、解缠；不当运动真值 |", "| XKF1 VN/VE/VD | m/s NED | EKF估计，不是独立真值 |", "| VIBE | m/s²统计指标 | 5Hz地板、残差平方2Hz平滑、开方的基础算法；不作为原波形 |", "", "固定飞行分组见 uavfd_f550/configs/split.json。候选飞行段、抗混叠方法和ARM初值未知处理见MODEL_SPEC。正式本轮不估计检测延迟。"])
    (docs/"DATA_AUDIT.md").write_text("\n".join(a)+"\n", encoding="utf-8")
    dev = normal["results"]["NO_FAULT4.mat"]
    current = np.array(dev["attitude_rotation_error_rad"]["rmse"])
    baseline = np.array(dev["persistence_attitude_error_rad"]["rmse"])
    stage = {"analytical_demo_pass": analytical["all_passed"], "real_rigid_body_validation_pass": False,
             "attitude_no_worse_than_persistence": bool(np.all(current <= baseline)), "phase_D_enabled": False,
             "formal_parameter_config_ready": False, "large_scale_simulation_enabled": False, "final_test_enabled": False,
             "reason": "No established real-model fidelity threshold; development attitude worse than persistence; physical/custom firmware evidence missing"}
    write_json(docs/"stage_gates.json", stage)
    text = ["# 实际验证报告", "", "本报告来自实际运行结果，不是计划或预计结果。完整数值、每次代码/配置/数据哈希见 validation_metrics.json；实际数组和图位于 output_uavfd_f550。", "",
            "## 解析与软件边界", "", f"{sum(t['passed'] for t in analytical['checks'])}/{len(analytical['checks'])}项预定解析检查通过。所有数值参数为demo，不是UAV-FD标定。", "",
            "检查覆盖静止支撑/比力符号、自由落体、悬停、单桨力矩与反扭矩、已知外力/主轴力矩响应、四元数、旋翼单位转换、IMU离心测点项、一阶执行器阶跃、不平衡离心/切向项与旋向、模态ringdown、积分步长收敛、采样抗混叠与Nyquist拒绝、旧坐标适配。", "",
            f"有机体陀螺耦合的RK4误差（dt=0.1/0.05/0.025s，独立solve_ivp参考）：{analytical['integration_errors']}。", "",
            f"局部模态旋转强迫输入ZOH的1000→2000Hz、2000→4000Hz比较RMS：{demo['modal_internal_rate_step_comparison_rms_m_s2']}m/s²。下降约一半符合输入ZOH的一阶误差；不虚报为连续输入高阶精度。", "",
            "unittest验证MAT字段换序解码、三IMU共同时间戳平均、v7.3/HDF5 cell、重复请求变量拒绝、未知物理参数拒绝和开发/锁定集访问边界。执行记录见checks.txt。", "",
            "## 正常实飞组合模型", "", "NO_FAULT1–3标定，NO_FAULT4开发；每条飞行同等总拟合权重。2Hz离线处理、1s非重叠短时预测，窗口只作误差汇总，不能当独立重复。", "",
            f"有效竖直系数a={model['vertical']['coefficients'][0]:.8g}，截距b={model['vertical']['coefficients'][1]:.6f}m/s²；归一化条件数{model['vertical']['normalized_condition']:.3f}。大截距提示未分离偏置/遗漏作用/相关性，不报告为真实kT/m。", "",
            f"六组独立平方转速设计条件数{model['six_independent_coefficients_diagnostic']['condition']:.3f}，最小/最大奇异值比{model['six_independent_coefficients_diagnostic']['relative_smallest']:.6f}；弱相对激励，没有宣称质量、惯量或六组独立推进系数可唯一辨识。每条飞行估计见parameter_ledger/validation_metrics。", "",
            "| NO_FAULT4指标 | 模型三轴RMSE | persistence三轴RMSE |", "| --- | --- | --- |"]
    for title, key, base in [("角速度rad/s", "angular_rate_rad_s", "persistence_angular_rate_rad_s"),
                             ("姿态旋转向量rad", "attitude_rotation_error_rad", "persistence_attitude_error_rad"),
                             ("EKF速度m/s", "estimated_velocity_error_m_s", "persistence_velocity_error_m_s")]:
        fmt = lambda a: ", ".join(f"{x:.6f}" for x in a)
        text.append(f"| {title} | {fmt(dev[key]['rmse'])} | {fmt(dev[base]['rmse'])} |")
    text.extend(["", f"IMU比力三轴RMSE={dev['specific_force_m_s2']['rmse']}m/s²。x/y模型均为零推力方向比力，真实横向作用/测点影响没有恢复。", "",
                 "结果：角速度有有限改善，姿态所有轴以及水平速度未优于persistence；因此当前组合基线尚不能支撑全状态模型通过，更不能声称长时开环或持续闭环稳定。没有靠调大噪声掩盖差异，也未调参反复追求开发记录的通过。", "",
                 "## 振动实飞开发评估", "", "正常基线只NO_FAULT1–3；损伤增量只FAULT_M1_5/FAULT_M2_5；开发NO_FAULT4/FAULT_M3_5。PSD与相干性针对日志IMU，包含其未知定制滤波作用。", "",
                 f"两条故障标定原始增量：{vib_model['fault_flight_gains']}。M1增量为负，非负约束取0仅为模型边界，不代表故障无振动；没有强制故障能量大于正常。", "",
                 "| 开发记录 | 窗数（仍一条独立飞行） | 同速/同压支持内窗数 | 实测1x带能量均值xyz | 预测均值xyz |", "| --- | ---: | ---: | --- | --- |"])
    for name in ("NO_FAULT4.mat", "FAULT_M3_5.mat"):
        r = vib["results"][name]
        text.append(f"| {name} | {r['windows_n']} | {r['within_speed_voltage_support_n']} | {r['rotor_1x_band_energy_observed_mean']} | {r['rotor_1x_band_energy_predicted_mean']} |")
    text.extend(["", "频带能量单位(m/s²)^2，所有RMSE/轴比例/转频区间见validation_metrics。开发主峰中位数约22–23Hz，明显不同于约66–113Hz旋翼转频；不能将最大峰直接归为损伤1x峰。模态与滤波、转速变化、运动工况及多个转频频带重叠共同限制解释。x-y相干性和全谱图已输出，未凭诊断可分性倒推模型准确。", "",
                 "以上均值覆盖本记录所有合格窗口，并非同速同压子集均值；子集误差单列在validation_metrics的common_support_error。正常能量使用Motor1中心带，故障使用标签电机中心带，均不能解释为独立单桨能量。当前120Hz上限未覆盖开发数据的2x及更高阶次；没有据带外缺测断言真实谐波不存在。", "",
                 "结果：实飞频带观测基线可复现，但两个损伤标定位置不能支持统一损伤增益；FAULT_M3_5三轴能量预测有明显偏差。真实mu、结构模态、传感器传递和缺损几何未标定。", "",
                 "## 通过条件、数据预算与下一阶段", "", "分析配置在首次拟合前固定：解析关系采用数值误差阈值；实飞仅报告与persistence的比较，无有依据的整体高保真通过阈值。后续阈值需按研究用途确定。本轮阶段D、大规模造数、诊断网络和最终锁定测试门槛关闭。", "",
                 "预算：3条正常标定+2条故障标定；2条开发用于评价且计入开发预算；11条最终测试仅审计完整性/配置/时间，不读取幅值或频谱。未来real-only比较必须给相同7条开发实飞额度。", "",
                 "未能确定：真实平台/动力/桨型号与5/10几何；整机m/J/COM/IMU位置；分离kT/kQ和mu/传递增益；高速真实相位；执行器及日志延迟；作者定制滤波/混控/控制实现。详见MODEL_STATUS的准确补充资料清单。", "",
                 "## 本轮执行", "", "2026-10-04执行：git status/branch/remote/log/fetch；git switch -c codex/uavfd-f550-model；sources、audit、calibrate、replay、vibration、demo、validate、report；unittest；最终测试入口拒绝检查；受保护工作区哈希复核。命令、退出状态见checks.txt与工作目录运行记录。当日未合并、推送、运行最终分类或解锁测试集。2026-10-05用户授权提交并推送独立分支，代码发布记录以Git历史为准。"])
    figure_dir = docs / "figures"
    figure_dir.mkdir(exist_ok=True)
    for name in ("normal_replay.png", "vibration_validation.png", "demo_vibration.png"):
        shutil.copyfile(out / name, figure_dir / name)
    text.extend(["", "## 实际输出图", "", "正常实飞开发记录的比力比较：", "",
                 "![NO_FAULT4正常组合模型开发比较](figures/normal_replay.png)", "",
                 "日志IMU的PSD与开发记录x-y相干性：", "",
                 "![实飞频谱开发评估](figures/vibration_validation.png)", "",
                 "以下仅为demo局部模态演示，不是UAV-FD实飞仿真：", "",
                 "![仅demo旋转不平衡及局部弹性观测](figures/demo_vibration.png)"])
    (docs/"VALIDATION_REPORT.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    print("review documents and source/parameter ledgers saved:", docs, flush=True)
