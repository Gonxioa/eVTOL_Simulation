# 当前研究六旋翼物理参数

2026-10-06：当前主线使用明确研究设计，而不是已测量F550或UAV-FD实机参数。详细身份、读取路径、未知项、推导和敏感性见[参数审计](hex_audit/PARAMETER_AUDIT.md)与[机器台账](hex_audit/parameter_ledger.json)。

当前配置为`hex_research/configs/research.json`，质量由2.8kg中央圆柱+6个0.1kg点质量组成，惯量由质量几何推导，满足正定与三角约束。原导入候选及原说明精确保留在`legacy_reference/`；原工作区文件没有覆盖。

旋翼半径用于几何净间距校核，不据此推导kT/kQ。所有推进、气动和观测数值保持研究设计/移植候选身份，真实硬件参数仍未知。
