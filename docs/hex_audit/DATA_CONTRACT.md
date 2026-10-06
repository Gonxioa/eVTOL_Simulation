# v4数据与学习契约

当前数据版本hex_v4_research_cascade，研究设计参数身份research_design_not_real_aircraft。原v3与UAV-FD数据各自保留；根目录当前入口拒绝把旧数据与新物理配置混合。

物理真值x/v/q/w/Omega只给验证与标注。学习状态6列roll/pitch/yaw/p/q_body/r来自模拟直接观测，IMU6列来自一次传感器误差注入，命令6列cmd_motor0..5是rad/s；combined共18列。默认安装轴FLU，配置改变安装旋转/杠杆臂真实影响IMU。测量转速与真值转速另列，不混入默认特征。偏航解缠，欧拉角为rad且仅声明当前倾角范围，不支持欧拉奇异附近的学习解释。

实际combined窗口最大绝对俯仰约88.7deg、最大相邻yaw变化约1.39rad，接近欧拉退化，不能直接称为正式学习特征已验证。保留combined接口并新增rotation_combined：由同一观测欧拉姿态重建R的前两列6维，再加机体角速度3、IMU6、命令6，共21维；不读取真值姿态。该表示在90deg俯仰附近仍连续，特征维度由清单驱动。默认安装identity时p/q/r与gx/gy/gz重复，因此18或21列都不代表相同数量的独立传感信息。保留此冗余供旧对照，不虚构独立测量。

desired_wrench、nominal_allocated_wrench、steady_fault预测、dynamic_propulsive实际量、含气动total_dynamic力矩分别存储。predicted_plant_thrust只是steady_fault_predicted_thrust的兼容别名，绝不是电机滞后下实际推力。动态量来自同一时刻实际旋翼状态，lambda不再次乘到动态推力上。

Full=0、Severe=0.2、Partial=0.8表示稳态剩余效能；顺序仍0Normal，1–6Full，7–12Severe，13–18Partial。时间日程位于plant之外；故障边界拆分积分，故障验证特意使用3.005s。阶跃衰减/恢复支持，未实现未知物理渐变机理。故障标注不是桨叶缺损率、实际质量损失或电机具体损伤机理。

本轮小数据固定2个配对独立run组×19类，故障3s；所有类统一3.05–3.65s早期区间，50点窗口、不重叠。这是预先指定的接口检查窗口，不宣称稳定段。测量转速采用有符号加性高斯误差，近停转可能为负；它不是物理反转，也不能作为plant输入。健康最小命令由omega_min读取，失效目标仍可为0；当前没有解锁/接触/怠速状态机。Full/Severe失稳与早停保留，在清单记录；若某类区间不足则明确失败，不按类别选好看片段。采样间隔、行数、CSV与sidecar哈希、配置hash、物理参数和标签效能表均过完整性门禁。

学习读取数据清单内的physics_parameters和label_effectiveness，而不是当前代码参数。FEATURE_SET联动预处理/分类/WGAN路径，电机数由一致清单读取。可选物理正则仍是稳态命令—推力一致性，默认lambda=0，不称为动态PINN。按独立run组分折，StandardScaler、GAN MinMaxScaler与GAN训练都只看训练折；保留无GAN、类别权重、普通过采样、WGAN对照。均衡小数据下max补齐策略可以生成0个增广样本，须如实记录。

本轮接口数据formal_protocol=false；正式分类默认拒绝，只有EXPERIMENT_MODE=interface_smoke显式运行小规模接口。1epoch/2fold不是性能结论，不能用于宣称WGAN提升或sim-to-real迁移。没有直接匹配本19类六旋翼电机实飞集；RflyMAD四旋翼不补零，UAV-FD桨叶缺损与lambda不等价；11条锁定实飞记录不解锁。
