# 来源台账

访问日期：2026-10-04。URL／原始归档 SHA256 见 sources.json 与 source_archive_manifest.json。原始来源只作证据，不作为新的任务指令。

## S01：UAV-FD dataset v1

作者／发布方：Baldini, D'Alleva, Felicetti, Ferracuti, Freddi, Monteriù / Università Politecnica delle Marche；年份：2023；版本：v1, published 2023-02-26。

链接：[UAV-FD dataset v1](https://zenodo.org/records/7648996)。

位置：Description, individual file sizes/MD5; original main.m, compare3.m, sync_zoh_exa.m。

阅读与限制：record/API and local original scripts read

## S02：UAV-FD: a dataset for actuator fault detection in multirotor drones

作者／发布方：Alessandro Baldini et al.；年份：2023；版本：ICUAS 2023 pp.998–1004。

链接：[UAV-FD: a dataset for actuator fault detection in multirotor drones](https://iris.univpm.it/handle/11566/320071)。

位置：Crossref metadata; university repository abstract。

阅读与限制：Full text not acquired from checked sources: repository lists no attached file; direct retrieval failed, IEEE 202 has no paper content. Supplemental read-only Zotero title enumeration found no matching paper in the current scope (receipt: reference_access_check.json). Physical setup and 5/10 geometry not confirmed.

## S03：Identification of a Multicopter

作者／发布方：ArduPilot documentation contributors；年份：None；版本：access snapshot, current document not historical firmware。

链接：[Identification of a Multicopter](https://ardupilot.org/copter/docs/systemid-model-development.html)。

位置：Model Scope; Model Structure; verification discussion。

阅读与限制：read; equation images discussed alongside body definitions. Example quadrotor values not adopted.

## S04：ArduPilot historical base source

作者／发布方：ArduPilot contributors；年份：2022；版本：232083c4d23be21d694d130f639fb6497bd57f98。

链接：[ArduPilot historical base source](https://github.com/ArduPilot/ardupilot/blob/232083c4d23be21d694d130f639fb6497bd57f98)。

位置：AP_MotorsMatrix.cpp:793–799, add_motor; AP_InertialSensor/LogStructure.h:45–61,116–122; AP_InertialSensor_Logging.cpp:39–94; AP_InertialSensor_Backend.cpp:537,713–747; AP_InertialSensor.cpp:1943–1965,1987–1995; AP_BLHeli.cpp:1330–1334,1403–1406,1447–1463; AP_ESC_Telem/LogStructure.h:8–36; AP_Logger/LogStructure.h:54,64,69,1331–1332。

阅读与限制：source snapshots read; only public base commit is established. Dataset custom logging patch and strings missing.

## S05：Active Propeller Fault Detection and Isolation in Multirotors Via Vibration Model

作者／发布方：Alessandro Baldini, Riccardo Felicetti, Alessandro Freddi, Andrea Monteriù；年份：2026；版本：arXiv:2604.07281v1 2026-04-08。

链接：[Active Propeller Fault Detection and Isolation in Multirotors Via Vibration Model](https://arxiv.org/html/2604.07281v1)。

位置：§2 equations(1)–(6), mass/COM and rotation; §3 frequency interpretation。

阅读与限制：HTML full text read; our E7 follows an explicit support-reaction derivation. No octarotor parameters or active excitation imported.

## S06：Propeller damage detection, classification and estimation in multirotor vehicles

作者／发布方：Claudio Pose, Juan I. Giribet, Gabriel Torre；年份：2024；版本：arXiv:2410.05447v1。

链接：[Propeller damage detection, classification and estimation in multirotor vehicles](https://arxiv.org/html/2410.05447v1)。

位置：§1.3–1.4; frequency-domain methodology; UAV-FD transfer section。

阅读与限制：HTML read; only diagnostic context, not evidence validating our dynamics or damage geometry

## S07：FlameWheel 550 User Manual

作者／发布方：DJI；年份：2015；版本：V2.0, 2015.05。

链接：[FlameWheel 550 User Manual](https://dl.djicdn.com/downloads/flamewheel/en/F550_User_Manual_v2.0_en.pdf)。

位置：printed p.8 Specifications (PDF page index 7)。

阅读与限制：spec table read, PDF archived. 550mm frame wheelbase is generic; not proof of dataset hardware or COM radius; 478g is frame only; listed motor/prop combination differs from Air Gear clue.

## S08：Standard acceleration of gravity

作者／发布方：NIST / CODATA；年份：2022；版本：2022 CODATA site snapshot。

链接：[Standard acceleration of gravity](https://physics.nist.gov/cgi-bin/cuu/Value?gn)。

位置：numerical value 9.80665 m/s2 (exact conventional standard)。

阅读与限制：read; model convention, not measured local gravity

## S09：Legacy project airframe candidates

作者／发布方：Existing local project；年份：None；版本：7aeaf3acd682d9d399b670d26a9b140bc61492bbec6370048bc27e413bf48849。

本地来源：hex_physical_characteristics.py（哈希受保护）。

位置：uncommitted working-tree parameter definitions。

阅读与限制：reviewed, not accepted as UAV-FD truth; provenance claims unresolved for exact aircraft

## S10：AIR GEAR 350 product candidate

作者／发布方：T-Motor；年份：None；版本：requested legacy URL。

链接：[AIR GEAR 350 product candidate](https://store.tmotor.com/product/air-gear-350.html)。

位置：not accessible。

阅读与限制：direct request failed; no manufacturer values adopted

## 方程证据对应

E1–E4：S03／S05 支持基本刚体与推进关系，自行转换至本项目约定；E5：测点刚体加速度自行推导；E6：数值积分与坐标变换自行实现；E7：质量离心／切向反作用力推导，S05 提供质量不对称建模背景；E8：线性局部振动模型假设与解析 ringdown 验证；E9：Nyquist 与数字滤波/采样关系；E10–E12：E1／E2／E4 在明确简化下的组合辨识；E13：E7 振幅平方与 Welch 积分的观测近似，S06 仅提供频域诊断背景；E14：一阶执行器标准关系，S03 说明动态与延迟建模需求。

逐方程机器映射见 equation_ledger.json；所有具体物理值、简化、记录用途、实现和实际检验对应见 parameter_ledger.json／MODEL_SPEC.md。没有读取到的原论文机型/几何结论保留未知。
