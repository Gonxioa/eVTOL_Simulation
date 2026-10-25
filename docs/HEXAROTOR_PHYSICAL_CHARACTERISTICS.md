# 六旋翼物理特性与参数依据

本说明仅描述飞机本体的质量、惯量、几何、旋翼/电机和气动参数。以下内容明确
不在本文件范围内：仿真频率与时长、轨迹、世界边界、风场、传感器噪声、故障
注入、控制器增益、GAN及分类器设置。

唯一代码来源为项目根目录的 `hex_physical_characteristics.py`；`main.py` 通过
`hex_params.py` 间接使用这组参数。数值分为“实验辨识/台架拟合”“有依据的
代理值”和“工程假设”三类，不能把整套参数称为F550数字孪生。

## 刚体与惯量

| 参数 | 数值 | 单位 | 说明 |
|---|---:|---|---|
| 质量 | 3.4 | kg | 项目目标构型假设，尚无对应实机称重记录 |
| Ixx | 0.0476 | kg·m² | 改装DJI F550闭环灰箱辨识 |
| Iyy | 0.0476 | kg·m² | 同上，论文假设Jx=Jy |
| Izz | 0.1094 | kg·m² | 改装DJI F550闭环灰箱辨识 |
| Ixy、Iyz、Ixz | 0 | kg·m² | 假设主惯性轴与机体系重合 |

惯量来自Giribet等人的实验论文。论文在改装DJI F550上用闭环飞行数据辨识得到
Jx=Jy=0.0476 kg·m²、Jz=0.1094 kg·m²，同时辨识出电机一阶极点
p=20.06 rad/s。这里采用相应的 `tau_m=1/p=0.04985 s`。

这组惯量是实验辨识的有效参数，不是摆振法直接测量值。原论文的机臂/电机以
107°倾角安装，而本项目是旋翼轴竖直的标准平面Hex-X；此外当前3.4 kg质量没有
与论文样机逐项配平。因此它比按质量和臂长缩放Hummingbird更有依据，但仍是
F550级代理参数。

来源：J. I. Giribet, C. D. Pose, A. S. Ghersin, and I. Mas,
“Experimental Validation of a Fault-Tolerant Hexacopter with Tilted Rotors,”
*International Journal of Electrical and Electronic Engineering & Telecommunications*,
vol. 7, no. 2, pp. 58–65, 2018,
[doi:10.18178/ijeetc.7.2.58-65](https://doi.org/10.18178/ijeetc.7.2.58-65)。

## 几何构型

- 六旋翼、标准平面Hex-X布局；
- 质心至各旋翼轴距离：0.275 m，对应F550的550 mm对角轴距；
- T9545直径为9.5 in，代码采用旋翼半径0.12065 m；
- 六个旋翼按60°间隔布置；
- 转向符号依次为 `+、-、+、-、+、-`；
- IMU/参考点暂假设位于质心，即 `rI=[0,0,0]`。

## 旋翼、电机与气动参数

| 参数 | 数值 | 说明与可信度 |
|---|---:|---|
| k_eta | 7.45×10⁻⁶ N/(rad/s)² | AIR2213/920KV + T9545、4S静态推力台数据拟合 |
| k_m | 9.80×10⁻⁸ N·m/(rad/s)² | 临时代理值，尚不是AIR Gear 350直接扭矩标定 |
| k_m/k_eta | 0.01315 m | T-Motor AIR Gear 355/MF9545实测比值 |
| k_d | 1.19×10⁻⁴ | 旧工程估计，未在F550上标定 |
| k_z | 2.32×10⁻⁴ | 旧工程估计，未在F550上标定 |
| k_h | 3.39×10⁻³ | 旧工程估计，未在F550上标定 |
| k_flap | 0 | 当前忽略挥舞项 |
| 电机时间常数 | 0.04985 s | F550闭环辨识极点p=20.06 rad/s的倒数 |
| 最小转速 | 0 rad/s | 电机下限 |
| 最大转速 | 1099.56 rad/s | 10,500 rpm，限制在4S台架数据范围内 |
| c_Dx、c_Dy | 0.005 | 旧工程估计，未在F550上标定 |
| c_Dz | 0.010 | 旧工程估计，未在F550上标定 |

### k_eta拟合

公开的社区台架工作簿记录了AIR2213/920KV、T9545和4S电池在300–10,500 rpm
范围内的35个转速-推力点。按照RotorPy的模型

`T = k_eta * omega²`

对SI单位数据做过原点最小二乘拟合，得到
`k_eta=7.4497×10⁻⁶ N/(rad/s)²`，决定系数R²=0.9985；代码保留三位有效数字。
该数据与UAV-FD所述AIR Gear 350 + T9545组合一致，但测量者是社区用户
Joerg Quinten，并非T-Motor厂家实验室，因此文档不能把它称为“官方标定”。

原始附件：AutoQuad论坛帖子
[T-Motor AIR2213.zip](https://www.forum.autoquad.org/viewtopic.php?p=35977)
中的 `thrustTorqueCalc_AIR2213_T9545_4S.xls`。

### k_m代理值

上述旧工作簿的 `torqueCalc` 页把载荷读数写成规则的1–10序列，量纲和数量级
也与相应推力不相容，不能视为真实扭矩曲线。项目因此没有使用该页拟合 `k_m`。

当前值采用T-Motor官方AIR Gear 355（AIR2213-KV1000 + MF9545）15个台架点
分别拟合 `T=k_eta*omega²` 和 `Q=k_m*omega²` 后的比值0.0131504 m，再与本项目
的 `k_eta` 相乘得到9.80×10⁻⁸。它比按桨盘面积缩放反扭矩系数更有物理依据，
但换用了后继电机/桨叶，仍必须标记为临时代理。

厂家数据：
[T-Motor AIR Gear 355测试表](https://store.tmotor.com/product/air-gear-355-combo-set.html)。

## 数值一致性检查

在标准重力9.81 m/s²下，当前质量和推力系数给出：

- 悬停单电机转速约863.8 rad/s，即8249 rpm；
- 最大数据覆盖转速为1099.6 rad/s，即10,500 rpm；
- 最大单旋翼推力约9.01 N；
- 最大总推力约54.04 N；
- 最大总推力/重量约1.62；
- 悬停转速约为当前上限的78.6%，余量有限。

与RotorS的AscTec Firefly交叉检查时，其公开模型使用1.5 kg质量、0.215 m臂长、
`k_eta=8.54858×10⁻⁶`、`k_m/k_eta=0.016 m`，惯量为
(0.0347563, 0.0458929, 0.0977) kg·m²。当前F550级模型的惯量与反扭矩比在同一
工程量级，没有出现原先 `k_m/k_eta≈0.0395 m` 的偏大值。Firefly不是F550，
这里只把它用作sanity check，不能替代本机标定。

来源：ETH Zurich RotorS
[`firefly.xacro`](https://github.com/ethz-asl/rotors_simulator/blob/master/rotors_description/urdf/firefly.xacro)。

## 可信度边界

当前模型最可靠的是F550级惯量和AIR Gear 350/T9545的推力系数。主要剩余不确定性
依次为：

1. 对目标3.4 kg整机做称重、质心和三轴惯量测量；
2. 对AIR2213/920KV + T9545直接同步测量转速、推力和反扭矩，替换临时 `k_m`；
3. 用电机阶跃试验复核 `tau_m`，因为论文值包含闭环辨识和样机安装影响；
4. 用飞行日志辨识机体阻力、旋翼入流和挥舞参数；
5. 参数变化后重新生成数据集。现有CSV与新参数的代码哈希不同，不能混合续跑。

## Colombo–Giribet论文年份核对

本地保存的 *Learning-Based Fault-Tolerant Control for an Hexarotor With Model
Uncertainty* 不是2022年论文。正式卷期为 *IEEE Transactions on Control Systems
Technology*, vol. 32, no. 2, March 2024, pp. 672–679，DOI为
[10.1109/TCST.2023.3318855](https://doi.org/10.1109/TCST.2023.3318855)。DOI中的
2023对应接收/在线出版阶段，卷期页眉明确是2024。它不是上面0.0476/0.1094惯量
数值的原始来源；惯量来源是2018年的Giribet等人论文。
