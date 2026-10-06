# 方程、坐标与接口审查

世界ENU、机体FLU；重力[0,0,-g]，推力沿机体+z。Hamilton四元数xyzw为机体到世界主动旋转，步后归一化。机体omega_body和非负机械Omega_rotor明确分开。机械RPM=Omega*60/(2pi)；当前研究没有电气转速日志，也不推断极对数。

状态为[x3,v3,q4,w3,Omega6]共19项，风为外部输入；旧含风22项只在隔离参考路径中使用。

| 方程 | 实际关系 | 实现 | 验证 |
| --- | --- | --- | --- |
| E1 | pdot=v; vdot=-g e3+R F_B/m | dynamics.HexPlant.derivative | freefall, hover, force response |
| E2 | J wdot=M-w cross(Jw) | dynamics.HexPlant.derivative | single motor, angular recovery, integration convergence |
| E3 | qdot=.5 q HamiltonProduct [w,0] | math3d.multiply; dynamics | quaternion/antipodal/coordinate checks |
| E4 | Ti=kTi Omega_i²; Qi=kQi Omega_i²; M=sum(r cross Ti e3+d Qi e3) | dynamics.wrench | six single-motor components and symmetry |
| E5 | Omega_dot=(sqrt(lambda)*Omega_cmd-Omega)/tau | dynamics.derivative; FaultSchedule | exact fault boundary, lag, complete stop, no double scaling |
| E6 | D=-norm(vrel) diag(cD) vrel; Hi=-Omega_i diag(kdrag)(vrel+w cross ri) | dynamics.wrench | stopped rotors and aero units |
| E7 | u=A f; A_i=[1,yi,-xi,di kQi/kTi] | configuration.matrix; control_allocation | 10 retained allocation tests + capability LP |
| E8 | fS=R_BS.T [R.T(a+g e3)+alpha cross rI+w cross(w cross rI)] | observation.Sensors | freefall, installation rotation, centripetal lever |
| E9 | v_sp=clip(vd+Kx*(xd-xhat)); a=ad+Kv ev+Iv-Kdv filtered_vdot | control.CascadeController | position step, tracking, saturation recovery |
| E10 | F=m(a+g e3); qd=thrust/yaw construction; w_sp=clip(KR*2vec(qerr)+Rhat.T e3 yaw_dot) | control.CascadeController; math3d | zero force, collinear,180deg,q/-q, tilted yaw feedforward |
| E11 | alpha_c=Kpw(wsp-w)+Iw-Kdw filtered_wdot; M=Jalpha_c+w cross Jw | control.CascadeController | rate recovery, limits, reset, nominal antiwindup |
| E12 | white_std=density/sqrt(dt); bias_increment=diffusion*sqrt(dt); OU a=exp(-dt/tau) | observation.Sensors/OUWind | seed/no-state-write/export-clock checks and feedback flights |

动力学省略旋翼惯量/动态反扭矩/旋翼陀螺、机架弹性、支撑接触及未确认的气动升力；保留机体惯性陀螺耦合。机体和测点观测不重复加重力或噪声。

框架是否可控不能仅由矩阵秩判断；lambda只作用于目标转速，实际推力按当前转速平方计算。

NED/FRD适配同时转换世界向量、机体角速度与旋转矩阵，见math3d.enu_flu_to_ned_frd；没有把四旋翼数据补零为六旋翼。enu_flu_to_ned_frd返回xyzw；to_uavfd_state另将标量移前输出13项wxyz状态和6项Omega，只作坐标/布局适配，不传递参数身份。

硬编码扫描：原构造器4xN方阵inv、20状态pack、SE3 super.update已退出默认链路；四元数4、4个直接控制通道保留；上游批处理类不在运行路径，未宣称任意N泛化。
