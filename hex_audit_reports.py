"""Generate source-grounded audits and reports from actual local results."""
from pathlib import Path
import json,hashlib,subprocess,shutil,inspect,importlib.metadata,ast
import numpy as np
from provenance import write_json,sha256_file,stable_hash
from hex_research.configuration import physical,matrix,ROOT,DEFAULT,rotorpy_fields

DOCS=ROOT/'docs/hex_audit'

def save(name,text):
    DOCS.mkdir(parents=True,exist_ok=True); (DOCS/name).write_text(text.strip()+'\n',encoding='utf-8')

def source_register(out):
    snap=out/'source_snapshots'
    sources=[
      ('R1','RotorPy 2.1.2 source and paper','Spencer Folk et al.','2023 / v2.1.2','https://github.com/spencerfolk/rotorpy/tree/v2.1.2','multirotor.py compute_body_wrench/_s_dot_fn/_pack_state; hummingbird_params.py; simulate.py; environments.py','local installed files and official version-tag snapshots read; generic Newton–Euler and speed-dependent rotor drag reviewed; scalar Hummingbird values are transplanted priors, not our measurements'),
      ('R2','PX4 1.12.3 controller source','PX4 contributors','v1.12.3 commit 2e8918da66af37922ededee1cc2d2efffec4cfb2','https://github.com/PX4/PX4-Autopilot/tree/v1.12.3','PositionControl.cpp:108–189; AttitudeControl.cpp:55–108; RateControl.cpp:60–105','official historical source read; only cascade, derivative-on-measurement, yaw-frame conversion and antiwindup concepts borrowed; SI gains/full-quaternion attitude law are ours, not literal normalized firmware'),
      ('R3','PX4 1.12 Controller Diagrams','PX4 documentation contributors','1.12','https://docs.px4.io/v1.12/en/flight_stack/controller_diagrams','position/velocity, attitude and angular rate diagrams','official document read; research SI design explicitly differs from normalized PX4 motor controls'),
      ('R4','RflyMAD: A Dataset for Multicopter Fault Detection and Health Assessment','Xiangli Le, Bo Jin, Gen Cui, Xunhua Dai, Quan Quan','arXiv 2311.11340v2, 2024-01-11','https://arxiv.org/html/2311.11340v2','§3.2; §4.1','original text read: modified PX4 1.12.3, real platforms are quadcopters; no direct evidence for our 19-class six-rotor task'),
      ('R5','RflyMAD official dataset description','Beihang Rfly/OpenHA','access 2026-10-06','https://rfly-openha.github.io/documents/4_resources/dataset.html','dataset page','official page archived/read; no data converted into fictitious six-motor channels'),
      ('R6','Experimental Validation of a Fault-Tolerant Hexacopter with Tilted Rotors','Juan I. Giribet, Claudio D. Pose, Alejandro S. Ghersin, Ignacio Mas','2018, DOI 10.18178/ijeetc.7.2.58-65','https://www.ijeetc.com/uploadfile/2018/0612/20180612065846780.pdf','printed p.61 (PDF page4): identified J_X,Y=0.0476, J_Z=0.1094, pole20.06; modified mounting107 degrees','original PDF text acquired/read; effective identified parameters on tilted platform cannot be treated as an admissible measured inertia tensor for this coplanar design'),
      ('R7','Latest protected project working tree','Local project/user working changes','ef5fa5b plus 6 protected files','https://github.com/Gonxioa/eVTOL_Simulation/commit/ef5fa5b0d7e00e63f9a4afe06e78128c1886b82b','baseline.json; legacy_reference exact copies','local candidates for thrust7.45e-6, Q/T0.01315 and manufacturer/bench claims are retained but the exact bench table was not acquired in this worktree; current scalar values are explicit design assumptions'),
      ('R8','UAV-FD preserved independent package','Università Politecnica delle Marche / prior project audit','v1; existing ef5fa5b package','https://zenodo.org/records/7648996','uavfd_f550/configs/split.json; docs/uavfd_f550','preserved byte-for-byte; no new amplitude/PSD reads from 11 locked test flights'),
    ]
    records=[{'id':a,'title':b,'authors':c,'version':d,'url':e,'location':f,'read_status_and_limits':g,'access_date':'2026-10-06'} for a,b,c,d,e,f,g in sources]
    write_json(DOCS/'sources.json',records)
    archive={p.name:{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in snap.iterdir() if p.is_file()}
    write_json(DOCS/'source_archive_index.json',archive)
    lines=['# 来源与复用台账','', '原始材料只作为证据。访问日期2026-10-06。源归档字节/SHA256见source_archive_index.json。','']
    for r in records:
        lines += [f"## {r['id']} {r['title']}",'',f"作者/版本：{r['authors']}；{r['version']}。",'',f"[{r['title']}]({r['url']})；位置：{r['location']}。",'',r['read_status_and_limits'],'']
    lines += ['## 复用决策','','| 模块 | 决策 | 理由 |','| --- | --- | --- |',
      '| control_allocation | 保留并修正正则快捷路径 | 原10测试保留；rho=0行为保留；rho>0须真的优化正则目标 |',
      '| provenance / atomic CSV / 分组与折内缩放 / LSTM / WGAN | 保留并适配v4接口 | 数据版本绑定参数，特征路径随FEATURE_SET，原公平对照保留 |',
      '| RotorPy Newton–Euler/旋转桨阻力 | 方程逐项核对后本地实现 | 六路19状态；无全局patch；不调用其私有pack/inv |',
      '| SE3Control / 上游20状态打包 / 进程级兼容patch | 退出默认链路，隔离参考 | 原代码和24测试保存在legacy_reference，单独进程运行 |',
      '| PX4 1.12.3 | 借鉴结构与指定概念 | 自有SI串级；不声称复刻定制PX4/RflyMAD或ArduPilot |',
      '| kh平移升力 | 不用于本研究停转模型 | 原项独立于Omega，停转仍有0.08136N；适用范围没有覆盖停转 |',
      '| UAV-FD包、文档与冻结划分 | 原样保留 | 独立桨叶缺损任务；不替代当前19类电机实飞证据 |','']
    save('REUSE_REGISTER.md','\n'.join(lines))
    license_file=snap/'rotorpy_LICENSE.md'
    if license_file.exists(): save('THIRD_PARTY_NOTICES.md','# RotorPy 2.1.2 reference\n\n公式参考及隔离参考代码的上游许可证如下；自有串级控制没有复制PX4实现。\n\n'+license_file.read_text(encoding='utf-8'))

UNITS={'model_version':'identifier','parameter_status':'evidence status','frame':'coordinate convention','center_mass_kg':'kg','center_radius_m':'m','center_height_m':'m','hub_mass_kg':'kg','arm_radius_m':'m','rotor_radius_m':'m','gravity_m_s2':'m/s2','kT':'N/(rad/s)^2','Q_over_T_m':'m','tau_s':'s','omega_min_rad_s':'rad/s','omega_max_rad_s':'rad/s','model':'model identity','body_drag_kg_m':'kg/m','rotor_drag_kg_per_rad':'kg/rad',
 'position_p':'1/s','velocity_p':'1/s','velocity_i':'1/s2','velocity_d':'1','attitude_p':'1/s','rate_p':'1/s','rate_i':'1/s2','rate_d':'1','velocity_limit_m_s':'m/s','rate_limit_rad_s':'rad/s','velocity_integral_limit_m_s2':'m/s2','rate_integral_limit_rad_s2':'rad/s2','acceleration_limit_m_s2':'m/s2','tilt_limit_deg':'deg','derivative_filter_hz':'Hz','antiwindup_gain_s_inv':'1/s',
 'wrench_priority':'relative normalized weight','regularization':'dimensionless objective weight','solver_tolerance':'dimensionless numerical tolerance','feasibility_tolerance':'normalized wrench tolerance','bound_tolerance':'N','max_iterations':'iteration count',
 'rate_hz':'Hz','accel_noise_density':'(m/s2)*sqrt(s)','gyro_noise_density':'(rad/s)*sqrt(s)','accel_bias_diffusion':'(m/s2)/sqrt(s)','gyro_bias_diffusion':'(rad/s)/sqrt(s)','additional_accel_sample_std':'m/s2','position_sample_std_m':'m','velocity_sample_std_m_s':'m/s','attitude_sample_std_rad':'rad','rpm_sample_std_rad_s':'rad/s','lever_m':'m','installation_rotation_xyzw':'unit quaternion xyzw S->B','feedback_delay_s':'s',
 'mean_m_s':'m/s','ou_stationary_std_m_s':'m/s','ou_correlation_time_s':'s','update_rate_hz':'Hz','control_hz':'Hz','integration_hz':'Hz (maximum step inverse)','export_hz':'Hz','position_norm_m':'m','speed_m_s':'m/s','body_rate_rad_s':'rad/s','minimum_altitude_m':'m','status':'evidence status','hover_final_position_error_m':'m','tracking_rmse_m':'m','maximum_normal_tilt_deg':'deg','integration_half_step_position_difference_m':'m','control_half_step_position_difference_m':'m','post_saturation_recovery_error_m':'m','seed':'integer'}

def run(cfg,out):
    source_register(out); p=physical(cfg); ledger=[]
    readers={'airframe':'configuration.physical','propulsion':'configuration.physical; dynamics; control; simulation.export','aerodynamics':'dynamics.HexPlant.wrench','control':'control.CascadeController.update','allocator':'control.CascadeController -> control_allocation.BoundedControlAllocator','sensor':'observation.Sensors; simulation feedback queue','wind':'observation.OUWind','timing':'simulation event clock; dynamics.integrate','termination':'simulation.simulate','acceptance':'validation.normal','seed':'simulation.simulate'}
    def walk(value,path=''):
        if isinstance(value,dict):
            for key,val in value.items(): walk(val,(path+'.' if path else '')+key)
        else:
            key=path.split('.')[-1]; group=path.split('.')[0]
            ledger.append({'key':path,'value':value,'units':UNITS[key],'coordinate':cfg['frame'],'category':group,
                           'evidence_status':'research_design_assumption','source_ids':['R7'] if group=='propulsion' else ['R1'] if group=='aerodynamics' else ['R2','R3'] if group=='control' else [],
                           'source_location':'research.json / explicit current design; borrowed structure is not evidence for this numerical value',
                           'reader':readers.get(group,'configuration metadata/signature'),'derivation_or_limits':'See PARAMETER_AUDIT/CONTROL_SPEC; no exact hardware identification or measurement',
                           'verification':'normal_validation.json / tests / audit_metrics.json','identification_flights':[],
                           'range_status':'No measured confidence interval; sensitivity variations are design experiments only'})
    walk(cfg)
    current_modules=list((ROOT/'hex_research').glob('*.py'))+[ROOT/'control_allocation.py']
    for row in ledger:
        key=row['key'].split('.')[-1]
        row['reader_locations']=[f'{file.relative_to(ROOT).as_posix()}:{i}' for file in current_modules for i,line in enumerate(file.read_text(encoding='utf-8').splitlines(),1) if ("'"+key+"'" in line or '"'+key+'"' in line)]
        row['read_identity']='numerical use or recorded configuration metadata; metadata does not always alter physics'
    for key,unit,value,reader in [('maximum_control_interval_s','s',.2,'control.update'),('actuator_step_to_tau_guard','1',.2,'dynamics.integrate'),('zero_force_tolerance','N',1e-10,'math3d.desired_attitude'),('quaternion_zero_norm_tolerance','1',1e-12,'math3d.quaternion'),('supported_num_rotors','count',6,'configuration.physical; shape contracts')]:
        ledger.append({'key':'numerical.'+key,'value':value,'units':unit,'evidence_status':'explicit_numerical_method_assumption','reader':reader,'source_ids':[],'limits':'method guard, not measured physical aircraft parameter'})
    for key,unit in [('mass','kg'),('COM','m'),('inertia','kg m2'),('kT','N/(rad/s)^2'),('kQ','N m/(rad/s)^2'),('motor_tau','s'),('motor_ESC_prop_voltage','hardware identities'),('sensor_pose','m / rad'),('sensor_noise_and_bias_spectra','per channel'),('aero_coefficients','per model')]:
        ledger.append({'key':'real_aircraft.'+key,'value':None,'units':unit,'evidence_status':'unknown','reader':'not instantiated as real-aircraft model','source_ids':[],'limits':'Requires matched original evidence; research values cannot fill this identity'})
    for key,value,unit,derivation in [('mass',p['mass'],'kg','central_mass + 6*hub_mass'),('inertia',p['inertia'].tolist(),'kg m2','central cylinder principal inertia + six point hubs using parallel-axis theorem'),('rotor_positions',p['rotor_positions'].tolist(),'m','arm_radius*[cos(i*pi/3),sin(i*pi/3),0] relative to symmetric COM'),('yaw_sign',p['yaw_sign'].tolist(),'1','body reaction sign; actual rotor spin sign is its negative'),('kQ',p['kQ'].tolist(),'N m/(rad/s)^2','kT * Q_over_T_m')]:
        ledger.append({'key':'derived.'+key,'value':value,'units':unit,'category':'physical','evidence_status':'derived_from_explicit_research_design','source_ids':[],'reader':'configuration.physical; dynamics; control','derivation':derivation,'coordinate':cfg['frame'],'verification':'tests/test_hex_research.py; normal_validation.json','identification_flights':[]})
    legacy=json.loads((DOCS/'baseline.json').read_text(encoding='utf-8'))
    for key,value in [('legacy_candidate_inertia',[.0476,.0476,.1094]),('legacy_candidate_thrust_coefficient',7.45e-6),('legacy_SE3_gains',[544.,46.64]),('legacy_kh',.00339),('legacy_full_overrides',{'4':.05,'5':.05})]:
        ledger.append({'key':key,'value':value,'evidence_status':'legacy_candidate_not_active','source_ids':['R6'] if 'inertia' in key else ['R1','R7'],'reader':'legacy_reference only; excluded from current core','limits':'Original value preserved; not automatically real-aircraft truth'})
    write_json(DOCS/'parameter_ledger.json',ledger)
    eigen=np.linalg.eigvalsh(p['inertia']); hover=np.sqrt(p['mass']*p['gravity']/sum(p['kT']))
    old_source=out/'source_snapshots/hummingbird_params.py'
    names=['c_Dx','c_Dy','c_Dz','k_d','k_z','k_h','k_flap']
    metrics={'mass_kg':p['mass'],'inertia_kg_m2':p['inertia'].tolist(),'inertia_eigenvalues':eigen.tolist(),'triangle_pass':bool(eigen[-1]<=sum(eigen[:2])),'legacy_candidate_triangle_pass':.1094<=2*.0476,
             'legacy_stopped_translation_lift_N':6*.00339*2**2,'hover_rad_s':float(hover),'hover_rpm':float(hover*60/(2*np.pi)),'max_thrust_to_weight':float(sum(p['kT']*p['omega_max']**2)/(p['mass']*p['gravity'])),
             'adjacent_hub_distance_m':cfg['airframe']['arm_radius_m'],'disk_clearance_m':cfg['airframe']['arm_radius_m']-2*cfg['airframe']['rotor_radius_m'],
             'legacy_equal_hummingbird_fields_confirmed':names,'Hummingbird_k_m_is_different':True,'legacy_candidate_inertia_is_effective_tilted_aircraft_identification':True,
             'state_size':19,'allocation_shape':[4,6],'no_measured_real_aircraft_parameters':True}
    write_json(DOCS/'audit_metrics.json',metrics)
    save('PARAMETER_AUDIT.md',f'''# 参数审计

当前数值全部具有研究设计身份，没有确认为UAV-FD或其他实机参数。参数读取、单位和来源状态逐项见parameter_ledger.json；实际配置为hex_research/configs/research.json。

质量来自假定2.8kg中央实体与6个0.1kg轮毂，总计{p['mass']:.4f}kg。中央实体为半径0.10m、高0.06m均匀圆柱，框架/载荷质量集中近似；轮毂为距COM0.275m的点质量。自行用平行轴定理计算J={np.diag(p['inertia']).tolist()}kg·m²。该几何质量分布是研究假设，不是CAD/称重结果，省略杆件分布和轮毂自惯量。对称COM=机架中心仅在该明确对称设计中成立。

导入候选J=(0.0476,0.0476,0.1094)虽正定，但不满足Izz<=Ixx+Iyy。原论文全文确认这些是107度倾斜改装F550上的有效灰箱辨识值和20.06rad/s极点；既不能直接迁移为共面机体真实惯量，也没有本研究3.4kg对应质量证据。原文件保留在legacy_reference。

推力7.45e-6、Q/T=0.01315m和tau=0.05s保留为明确研究设计候选，未取得该动力组合的原始台架表；不标厂家确认。最大转速1100rad/s是研究限值，不能从KV推出。健康悬停{hover:.4f}rad/s，最大推重比{metrics['max_thrust_to_weight']:.4f}。

原c_Dx/y/z、k_d、k_z、k_h、k_flap与RotorPy2.1.2 Hummingbird示例逐项数值相同；原k_m=2.20e-7与示例1.36e-7不同，不能统称完全相同。保留的机体阻力kg/m、旋转桨阻力kg/rad均为移植研究假设。原kh项Omega=0、水平气流2m/s仍给0.08136N正升力；当前停转研究不采用该正常运转桨近似。停转仍有明确机体阻力，不模拟未知静止桨气动力。

中央质量±20%、臂长±10%、kT±20%、tau乘0.6/1.6、Q/T±20%在运行前选定为设计敏感性范围，质量/惯量及几何/惯量一起重新推导。它们不是辨识置信区间，也未使用任何锁定实飞记录。正常响应见validation_metrics，另以同一M0 Partial和50点早期窗口实际检查故障特征，结果fault_feature_sensitivity.json。控制器名义参数也跟随设计，因此不是控制器未知参数失配鲁棒性实验。

旧Multirotor内部抽象增益、旧SE3增益和旧kh/Full特例不被当前主线读取。研究参数、控制设计、传感器随机过程、任务与学习超参数分开管理；source IDs和每项读取模块见机器台账。''')
    formulas=[
     ('E1','pdot=v; vdot=-g e3+R F_B/m','dynamics.HexPlant.derivative','freefall, hover, force response','R1; own ENU/FLU conventions'),
     ('E2','J wdot=M-w cross(Jw)','dynamics.HexPlant.derivative','single motor, angular recovery, integration convergence','R1'),
     ('E3','qdot=.5 q HamiltonProduct [w,0]','math3d.multiply; dynamics','quaternion/antipodal/coordinate checks','own Hamilton xyzw derivation'),
     ('E4','Ti=kTi Omega_i²; Qi=kQi Omega_i²; M=sum(r cross Ti e3+d Qi e3)','dynamics.wrench','six single-motor components and symmetry','R1; independent component derivation'),
     ('E5','Omega_dot=(sqrt(lambda)*Omega_cmd-Omega)/tau','dynamics.derivative; FaultSchedule','exact fault boundary, lag, complete stop, no double scaling','explicit research phenomenological mechanism'),
     ('E6','D=-norm(vrel) diag(cD) vrel; Hi=-Omega_i diag(kdrag)(vrel+w cross ri)','dynamics.wrench','stopped rotors and aero units','R1 adapted; kh excluded'),
     ('E7','u=A f; A_i=[1,yi,-xi,di kQi/kTi]','configuration.matrix; control_allocation','10 retained allocation tests + capability LP','own cross-product reduction; not six-axis actuation'),
     ('E8','fS=R_BS.T [R.T(a+g e3)+alpha cross rI+w cross(w cross rI)]','observation.Sensors','freefall, installation rotation, centripetal lever','own rigid point acceleration'),
     ('E9','v_sp=clip(vd+Kx*(xd-xhat)); a=ad+Kv ev+Iv-Kdv filtered_vdot','control.CascadeController','position step, tracking, saturation recovery','R2/R3 structure; own SI design'),
     ('E10','F=m(a+g e3); qd=thrust/yaw construction; w_sp=clip(KR*2vec(qerr)+Rhat.T e3 yaw_dot)','control.CascadeController; math3d','zero force, collinear,180deg,q/-q, tilted yaw feedforward','R2 concept; full-quaternion research law differs from PX4 reduced-yaw law'),
     ('E11','alpha_c=Kpw(wsp-w)+Iw-Kdw filtered_wdot; M=Jalpha_c+w cross Jw','control.CascadeController','rate recovery, limits, reset, nominal antiwindup','R2 structure; own SI outputs'),
     ('E12','white_std=density/sqrt(dt); bias_increment=diffusion*sqrt(dt); OU a=exp(-dt/tau)','observation.Sensors/OUWind','seed/no-state-write/export-clock checks and feedback flights','own stochastic process convention; no manufacturer PSD claim')]
    write_json(DOCS/'equation_ledger.json',[{'id':a,'formula':b,'implementation':c,'validation':d,'source_and_assumptions':e,'coordinate':cfg['frame']} for a,b,c,d,e in formulas])
    lines=['# 方程、坐标与接口审查','','世界ENU、机体FLU；重力[0,0,-g]，推力沿机体+z。Hamilton四元数xyzw为机体到世界主动旋转，步后归一化。机体omega_body和非负机械Omega_rotor明确分开。机械RPM=Omega*60/(2pi)；当前研究没有电气转速日志，也不推断极对数。','','状态为[x3,v3,q4,w3,Omega6]共19项，风为外部输入；旧含风22项只在隔离参考路径中使用。','','| 方程 | 实际关系 | 实现 | 验证 |','| --- | --- | --- | --- |']
    for a,b,c,d,e in formulas: lines.append(f'| {a} | {b} | {c} | {d} |')
    lines += ['','动力学省略旋翼惯量/动态反扭矩/旋翼陀螺、机架弹性、支撑接触及未确认的气动升力；保留机体惯性陀螺耦合。机体和测点观测不重复加重力或噪声。','', '框架是否可控不能仅由矩阵秩判断；lambda只作用于目标转速，实际推力按当前转速平方计算。','', 'NED/FRD适配同时转换世界向量、机体角速度与旋转矩阵，见math3d.enu_flu_to_ned_frd；没有把四旋翼数据补零为六旋翼。enu_flu_to_ned_frd返回xyzw；to_uavfd_state另将标量移前输出13项wxyz状态和6项Omega，只作坐标/布局适配，不传递参数身份。','', '硬编码扫描：原构造器4xN方阵inv、20状态pack、SE3 super.update已退出默认链路；四元数4、4个直接控制通道保留；上游批处理类不在运行路径，未宣称任意N泛化。']
    save('EQUATION_AUDIT.md','\n'.join(lines))
    mapping=[]
    for i,(r,d) in enumerate(zip(p['rotor_positions'],p['yaw_sign'])):
        mapping.append({'index':i,'document_motor':i+1,'position_FLU_m':r.tolist(),'body_yaw_reaction_sign':int(d),'rotor_spin_about_plus_z':int(-d),'from_above':'CW' if d==1 else 'CCW','command_column':f'cmd_motor{i}','actual_column':f'truth_rotor{i}','measurement_column':f'measured_rotor{i}'})
    write_json(DOCS/'geometry_channel_map.json',mapping)
    save('GEOMETRY_AND_CHANNEL_MAP.md','# 几何与通道\n\n规则共面六边形，第一桨位于机体+x；可称前向顶点构型，不依赖Hex-X名称猜位置。各位置相对该对称设计COM。yaw_sign是机体反扭矩方向，旋翼真实转动符号=-yaw_sign。机体+y向左，+z向上。Motor文档从1计数、代码从0计数；机器表geometry_channel_map.json记录全部6路与列名。当前机型没有真实PWM/ESC日志接线图，因此这些是仿真通道映射。\n\n相邻轮毂距0.275m、桨直径0.2413m，理论净间距0.0337m；高度均0。惯量、安装点、旋向不从UAV-FD配置继承。')
    save('CONTROL_SPEC.md','''# 自有常规串级控制

代码为hex_research/control.py，配置为research.json。位置P产生m/s参考（增益1/s）；速度P/I/D产生m/s²（1/s、1/s²、1），积分态m/s²；姿态四元数误差产生rad/s（增益1/s），角速度P/I/D产生rad/s²（1/s、1/s²、1），积分态rad/s²；乘J并加机体陀螺耦合后才成为Nm。旧544/46.64未被复制成新机型增益。

设计起点在理想执行器、忽略积分时，水平位置/速度闭环约wn=sqrt(1.2*2.5)=1.73rad/s、阻尼比2.5/(2*wn)=0.72；约4/s姿态响应与8/s角速度反馈，考虑0.05s执行器极点。这里只是简化设计推导，不是实际闭环带宽辨识。首次运行前在research.json声明误差、倾角和收敛阈值；实际结果见验证报告。

速度、加速度、倾角、角速度和积分均有显式限幅；总推力投影到观测到的机体z方向。近零力保持上次期望姿态；共线 heading 选最不平行基轴；记录fallback。四元数选择最短旋转，180度采用非零向量符号约定，不用sign(0)。世界z偏航前馈乘Rhat.T，不能直接当机体z参考。

角速度/速度的测量差分经过20Hz一阶低通，alpha=1-exp(-2pi fc dt)；这不是未滤波原始粗差分或真实PX4滤波复现。控制100Hz，传感器/风100Hz，导出100Hz，积分最大步长1/500s；ODE/RK4中间调用不更新控制或随机过程。

速度抗饱和使用本控制器的受限加速度与名义推力缺口，角速度积分按名义分配力矩残差方向冻结；不读取故障lambda或实际故障矩阵。分配加权最小二乘是软优先级，不是严格分层保证。正则默认0，启用时优化零推力参考正则，不能走可行伪逆快捷路径。

每次sim运行新建控制器与滤波状态。时间必须递增，首次步长按声明控制周期；非法四元数、NaN/Inf和非法控制间隔报错。普通诊断控制器只收观测x/v/q/w与任务参考；调度、lambda、故障标签和实际执行器状态不参与控制决策。

ideal_feedback直接使用真值，只用于明确标识的检查；observed_feedback读取传感器队列中的带噪/延迟直接观测（不是EKF），角速度来自真正模拟陀螺读数转换至机体轴。噪声参与实际控制，不是飞行结束后处理。当前不复现PX4完整输入整形/归一化混控/估计器，也不匹配UAV-FD的ArduPilot定制固件。''')
    save('DATA_CONTRACT.md','''# v4数据与学习契约

当前数据版本hex_v4_research_cascade，研究设计参数身份research_design_not_real_aircraft。原v3与UAV-FD数据各自保留；根目录当前入口拒绝把旧数据与新物理配置混合。

物理真值x/v/q/w/Omega只给验证与标注。学习状态6列roll/pitch/yaw/p/q_body/r来自模拟直接观测，IMU6列来自一次传感器误差注入，命令6列cmd_motor0..5是rad/s；combined共18列。默认安装轴FLU，配置改变安装旋转/杠杆臂真实影响IMU。测量转速与真值转速另列，不混入默认特征。偏航解缠，欧拉角为rad且仅声明当前倾角范围，不支持欧拉奇异附近的学习解释。

实际combined窗口最大绝对俯仰约88.7deg、最大相邻yaw变化约1.39rad，接近欧拉退化，不能直接称为正式学习特征已验证。保留combined接口并新增rotation_combined：由同一观测欧拉姿态重建R的前两列6维，再加机体角速度3、IMU6、命令6，共21维；不读取真值姿态。该表示在90deg俯仰附近仍连续，特征维度由清单驱动。默认安装identity时p/q/r与gx/gy/gz重复，因此18或21列都不代表相同数量的独立传感信息。保留此冗余供旧对照，不虚构独立测量。

desired_wrench、nominal_allocated_wrench、steady_fault预测、dynamic_propulsive实际量、含气动total_dynamic力矩分别存储。predicted_plant_thrust只是steady_fault_predicted_thrust的兼容别名，绝不是电机滞后下实际推力。动态量来自同一时刻实际旋翼状态，lambda不再次乘到动态推力上。

Full=0、Severe=0.2、Partial=0.8表示稳态剩余效能；顺序仍0Normal，1–6Full，7–12Severe，13–18Partial。时间日程位于plant之外；故障边界拆分积分，故障验证特意使用3.005s。阶跃衰减/恢复支持，未实现未知物理渐变机理。故障标注不是桨叶缺损率、实际质量损失或电机具体损伤机理。

本轮小数据固定2个配对独立run组×19类，故障3s；所有类统一3.05–3.65s早期区间，50点窗口、不重叠。这是预先指定的接口检查窗口，不宣称稳定段。测量转速采用有符号加性高斯误差，近停转可能为负；它不是物理反转，也不能作为plant输入。健康最小命令由omega_min读取，失效目标仍可为0；当前没有解锁/接触/怠速状态机。Full/Severe失稳与早停保留，在清单记录；若某类区间不足则明确失败，不按类别选好看片段。采样间隔、行数、CSV与sidecar哈希、配置hash、物理参数和标签效能表均过完整性门禁。

学习读取数据清单内的physics_parameters和label_effectiveness，而不是当前代码参数。FEATURE_SET联动预处理/分类/WGAN路径，电机数由一致清单读取。可选物理正则仍是稳态命令—推力一致性，默认lambda=0，不称为动态PINN。按独立run组分折，StandardScaler、GAN MinMaxScaler与GAN训练都只看训练折；保留无GAN、类别权重、普通过采样、WGAN对照。均衡小数据下max补齐策略可以生成0个增广样本，须如实记录。

本轮接口数据formal_protocol=false；正式分类默认拒绝，只有EXPERIMENT_MODE=interface_smoke显式运行小规模接口。1epoch/2fold不是性能结论，不能用于宣称WGAN提升或sim-to-real迁移。没有直接匹配本19类六旋翼电机实飞集；RflyMAD四旋翼不补零，UAV-FD桨叶缺损与lambda不等价；11条锁定实飞记录不解锁。''')
    print('Audit ledgers and specifications saved:',DOCS,flush=True)

def report(cfg,out):
    run(cfg,out)
    normal=json.loads((out/'normal_validation.json').read_text(encoding='utf-8')); fault=json.loads((out/'fault_validation.json').read_text(encoding='utf-8')); cap=json.loads((out/'capability.json').read_text(encoding='utf-8'))
    write_json(DOCS/'validation_metrics.json',{'normal':normal,'faults':fault,'capability':cap})
    lines=['# 实际验证报告','','所有数字来自运行结果。完整指标见validation_metrics.json；阈值是运行前声明的研究要求，不是适航标准或实飞事实。','','## 基线与模块检查','','原工作区24测试通过，输出baseline_tests.txt；原分配器10项保留，新增边界/时序/传感器/输出检查，最新输出见checks.txt。原候选惯量虽基线测试通过仍不满足真实惯量三角约束，说明代码测试与物理证据状态必须分开。','','## 正常闭环','','| 工况 | 位置RMSE m | 最终误差 m | 最大倾角 deg | 反馈 | 判定 |','| --- | ---: | ---: | ---: | --- | --- |']
    for name,r in normal['normal_cases'].items(): lines.append(f"| {name} | {r['position_rmse_m']:.6g} | {r['final_position_error_m']:.6g} | {r['maximum_tilt_deg']:.4g} | {r['feedback_mode']} | {r['passed_research_condition']} |")
    lines += ['', 'saturation_recovery在2–4秒请求不可行的50m高度，同时转速上限降低为950rad/s；其大RMSE反映不可行参考，不用它宣称跟踪通过。判定只使用参考解除后的最终恢复误差，并记录实际饱和比例与积分状态。常量扰动为预定1/-0.5N机体水平力，2–6秒施加。','','### 步长和敏感性','']
    for k,r in normal['step_convergence'].items(): lines.append(f"{k}减半：最大位置差{r['max_position_difference_m']:.8g}m，研究阈值{r['threshold_m']}m，判定{r['passed']}。\n")
    lines += ['','敏感性共10组，质量/惯量与几何/惯量相关联重算；均为设计范围，不是统计CI。每次独立仿真指标和耗时在机器报告中，窗口没有被当作独立重复。','','![正常与观测反馈](figures/normal_validation.png)','','## 故障动力学与能力边界','','| 类别 | 实際终止 | 时长 s | 故障初期稳态/动态推力最大差 N |','| --- | --- | ---: | ---: |']
    if (DOCS/'fault_feature_sensitivity.json').exists():
        sensitivity=json.loads((DOCS/'fault_feature_sensitivity.json').read_text(encoding='utf-8'))
        values=[r['gyro_rms_rad_s'][1] for r in sensitivity['cases'].values()]
        lines.insert(lines.index('## 故障动力学与能力边界'),f'同一M0 Partial的11次设计敏感性运行（基线+10变体）使用相同50点窗口；y陀螺RMS范围{min(values):.6f}–{max(values):.6f}rad/s。配置变化时名义控制模型也跟随，因此不能解释为参数失配鲁棒性。完整三轴比力/陀螺统计见fault_feature_sensitivity.json。\n')
    for name,r in fault['cases'].items(): lines.append(f"| {name} | {r['termination']} | {r['actual_duration_s']:.4g} | {r['max_initial_steady_actual_thrust_difference_N']:.6g} |")
    full=[r for r in cap['cases'] if r['remaining_effectiveness']==0]
    lines += ['',f"6个Full状态LP均零力矩悬停可行：{all(r['zero_moment_hover_feasible'] for r in full)}。秩为4，但所有幸存执行器同时离开边界的最大比例为{[r['surviving_actuator_interior_fraction'] for r in full]}；不能据满秩断言有充分四通道控制裕度。",'',f"120度旋转下分配列对称误差{cap['rotation_120deg_column_symmetry_error']:.8g}。八条控制轴方向LP裕度在capability.json，未冒称最大可达内球。",'', '常规控制器无故障真值，也未按故障位置换增益；Full/Severe早停是该工况实际响应，未把0改成0.05。lambda是执行器目标效能现象，未模拟轴承/绕组/转子具体机理。','', '![Full故障响应](figures/fault_validation.png)','','## 数据与学习接口','']
    lines += ['数据集24条早停的实际触发条件均为z<-2m，详情termination_audit.json。它是本仿真采样包络规则，不是飞机必然无法控制或真实碰地时刻。未实现地面接触；静态可行性与当前有限时间控制结果分开记录。','']
    manifest_path=out/'dataset_v4/dataset_generation_manifest.json'
    if manifest_path.exists():
        dataset=json.loads(manifest_path.read_text(encoding='utf-8')); integrity=json.loads((out/'dataset_v4/integrity_report.json').read_text(encoding='utf-8'))
        lines += [f"实际生成{len(dataset['runs'])}条独立类来源，2个配对run组；完整性检查通过，{integrity['early_stop_count']}条明确早停。所有类按同一3.05–3.65s窗口；38文件不是38独立环境组，公平分组单位为2个run组。",'']
    smoke_path=out/'learning_smoke.json'
    if smoke_path.exists():
        smoke=json.loads(smoke_path.read_text(encoding='utf-8')); write_json(DOCS/'learning_smoke.json',smoke)
        lines += ['学习接口实际检查结果：`learning_smoke.json`，命令输出见learning_checks.txt。1epoch/2fold只判断接口和折内流向，不当成识别性能证据。','']
    continuous=out/'learning_smoke_rotation_combined.json'
    if continuous.exists():
        write_json(DOCS/'learning_smoke_rotation_combined.json',json.loads(continuous.read_text(encoding='utf-8')))
        lines += ['rotation_combined的21维输入也实际完成两折一轮接口，输出learning_checks_rotation_combined.txt。原18维combined仍保留；接近88.7度的欧拉俯仰不具正式学习有效性支持。默认机体角速度与IMU陀螺三列存在冗余。','',
                  '纯WGAN一轮生成的诊断旋转列不满足正交：两折平均列正交误差约1.26/1.23，详见rotation_generator_check.json。没有启用姿态流形投影或相应物理正则，因此不能把这些生成列当成物理姿态已验证。均衡数据下max增广数量为0，WGAN分支分类数字的差异不能证明增广改善。','']
    lines += ['','## 仍未验证的边界','','本研究的真实质量、COM、惯量、推力/扭矩、电机延迟、气动与传感器频谱没有实机匹配。直接姿态/位置/速度误差模型不是完整EKF。完整PX41.12.3定制固件、ArduPilot复刻、大范围翻滚/着陆接触、旋翼陀螺/弹性振动和长时完全失效控制未成立。正式训练和实飞迁移未进行。', '', '初次Normal CSV导出触发可选fault_time字段不属于数字列的KeyError；已修复为仅对该可选列忽略缺省，加入真实Normal导出回归，并保留失败计划快照。没有异常回退到随机数据。']
    figures=DOCS/'figures'; figures.mkdir(exist_ok=True)
    for name in ['normal_validation.png','fault_validation.png']: shutil.copyfile(out/name,figures/name)
    save('VALIDATION_REPORT.md','\n'.join(lines))
    stages={s:{'status':'implemented_and_checked_research_scope','real_flight_support':False} for s in 'ABCDEFGHIJ'}
    stages['F']['normal_research_gate']=normal['base_research_normal_gate']; stages['G']['all_full_stable']=False; stages['I']['formal_training_done']=False
    stages['I']['interface_only']=True
    stages['I']['legacy_euler_feature_singularity_supported']=False
    stages['I']['generated_attitude_physics_validated']=False
    stages['I']['GAN_improvement_demonstrated']=False
    stages['A']['parameter_provenance_traceable']=True
    stages['C']['mathematical_consistency_in_declared_scope']=True
    stages['H']['observed_feedback_executed']=True
    write_json(DOCS/'stage_status.json',stages)
    save('MODEL_STATUS.md','''# 实施状态

当前分支codex/hex-audit-control，基线ef5fa5b；工作树与原工作区隔离，导入6项修改时保留精确哈希与旧代码/测试参考。未合并、推送或删除旧数据。

A：基线、远端、环境版本和24旧测试已核对。B：参数读取、单位、身份与候选限制已台账化；当前是研究设计。C：坐标/19状态/4×6分配/IMU契约已审查。D：新核心无全局补丁，上游批处理不在路径。E：自有SI串级与名义有界分配已实施和模块检查。F：声明的正常工况及步长检查通过。G：19类统一效能、滞后/真实扳手和能力LP已运行；Full/Severe早停保留。H：理想/含噪延迟观测反馈已实际运行，噪声不是后处理；简化观测不是EKF。I：v4小样本导出/哈希/一致窗口/折内学习接口已执行，非正式性能实验。J：审计、实测图、回归与交接完成范围见VALIDATION_REPORT及checks。

机器状态分开记录：来源身份可追溯；数学一致性；代码检查；声明范围仿真正常闭环；实飞支持仍为false。原UAV-FD包、报告和5/2/11划分原样保留，本轮未读取锁定测试幅值或频谱。

需补资料：目标机整机含载荷称重、COM、惯量测量或质量/CAD模型；电机/ESC/桨/电压型号与对应推力/扭矩/动态台架曲线；机体/桨气动系数；安装坐标、传感器噪声/偏置频谱和估计/日志延迟；直接匹配19类定义的六旋翼实飞支持。参数设计、常规飞控和代码修正不能替代这些证据。后续是否进行正式小样本WGAN实验，应另行固定数据预算、样本抽样与完整训练协议。''')
    print('Actual validation report and stage states saved.',flush=True)
