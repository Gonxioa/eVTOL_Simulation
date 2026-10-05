# 本轮改动与复核清单

建模基线提交：`af9b45ae3e0d77987a051494ba68962964a38832`；独立分支：`codex/uavfd-f550-model`。

2026-10-04完成本地建模及验证；2026-10-05用户授权提交并推送独立分支。发布commit以Git历史为准。

受保护的6项原有修改SHA256均未变；提交范围为独立包、任务文件、证据与验证文档、三张既有报告图，以及README.md／PROJECT_HANDOFF.md入口说明。原始数据和大运行输出保留本地。

数值结果保留执行当时的代码哈希。随后报告生成器增加图表复制、仓库相对链接和历史发布说明；数值模型、数据处理、拟合和分析代码没有变化。最终源码／配置哈希、Git行尾规范化哈希和差异见final_source_snapshot.json。

分支提交提供完整可审查差异；本地还保留output_uavfd_f550/uavfd_f550_changes.patch。原任务文件与Downloads提供文件SHA256一致。

| 新增文件 | 字节数 | SHA256（本地原始字节） |
| --- | ---: | --- |
| CODEX_UAVFD_F550_REBUILD_PROMPT.md | 16081 | ca05f5e6aa52c36eb03599819fc248ac8f96a351fd588a01b971d454c226921f |
| docs/uavfd_f550/audit_summary.json | 248442 | 6ba6cd754c37f21e313d4502d30766a14b1ed4ca2e66403e1ffcff0fa3e96953 |
| docs/uavfd_f550/checks.txt | 3564 | 09c04fe1dc4aaa1b84bc2467906c8d0b99ae4b256b3a444b5d28f0e7be44bf00 |
| docs/uavfd_f550/DATA_AUDIT.md | 3936 | fce661227e8b2de232750fc08f01ab5566f057905c5a2d1e20b3ca5310ce9d8f |
| docs/uavfd_f550/data_hashes.json | 3803 | 75b813acf8c813722f34147cd08bfcf1998e83a84616c509d3e5c4526ed24b24 |
| docs/uavfd_f550/equation_ledger.json | 9032 | 178ff674a2fc9fb38328b406964c6f11760524bc2679e6648470638cd60bff91 |
| docs/uavfd_f550/figures/demo_vibration.png | 98296 | 8262a0c6ad8eda8d6c4f330ff9623de3d04347e8b2ea30937a786fb963ba1c88 |
| docs/uavfd_f550/figures/normal_replay.png | 88576 | 6220f539fd5dbf03ec8840971cb98b587cc2b168141d9f888560aa01b75e8c49 |
| docs/uavfd_f550/figures/vibration_validation.png | 150654 | 1f6f504e0d66f88a1b6b2f2038f115250bacb88b251e211058a5eea7fe7d8092 |
| docs/uavfd_f550/final_source_snapshot.json | 4973 | d81f84646b86c899209da48bd3b93da85676efe578f2ab6ee5a11913faf7c8f7 |
| docs/uavfd_f550/MODEL_SPEC.md | 9504 | 52d7d928c9f4311cd3eba25632981f717e4ebb7a491ec77250818575f9ced6ef |
| docs/uavfd_f550/MODEL_STATUS.md | 2772 | 3bab55b2d7aedebe6bd2efd12fcd96addcaa30f6c7b875045602483445a784f7 |
| docs/uavfd_f550/PACKAGING_QA.md | 680 | ce58c37404795299104cf0b2c9a1da6abb19f10d0eecaf41a12e1b976e360019 |
| docs/uavfd_f550/parameter_ledger.json | 45384 | c8321c935aeab0aadff2dd1149db61e9c3cef26d8a0f64eb11c90fa69fc2bf4b |
| docs/uavfd_f550/protected_workspace.json | 1500 | 0f247eb9aebed56d35027a8bc353d7f91104052c8bd87d13f0aa5f38ae57a502 |
| docs/uavfd_f550/README.md | 3384 | 4125872161c7124656ea6f06ebb21fe1217b837648c8545b4f9b296dbc8f07f1 |
| docs/uavfd_f550/reference_access_check.json | 918 | 15029992e982835f01982f993917698e9b4636ca05d7e6f20d4fd18cad239028 |
| docs/uavfd_f550/source_archive_manifest.json | 13978 | 131b361429b4283364f94e64c7e3dc6a47d105bc27bef0dad285e655e45e75ea |
| docs/uavfd_f550/SOURCE_REGISTER.md | 6325 | 6dabd38be30e7fc60dbb49cf7d0b873d95e5f9fa1b9f9337cd703c21b9a26773 |
| docs/uavfd_f550/sources.json | 5642 | 46de637fab2de9480756b9e9b933682e4acda05e51a4c244ddb19400a07d1576 |
| docs/uavfd_f550/stage_gates.json | 428 | 7de47759b62402af4093ef8f8cbcf8df70f2c0c634d95d97d62bf9b153598734 |
| docs/uavfd_f550/validation_metrics.json | 48485 | feece53318d8f2fe4859523e905c27d650812239505432272468cf22ffe3c90a |
| docs/uavfd_f550/VALIDATION_REPORT.md | 6769 | db99264bf4256a3a81e8c1dd1f6f75366134e01cd31d7b7a31e7220cb843d01d |
| uavfd_f550/__init__.py | 89 | 7b78e8d0fbe09f9a59b63aeb27ddd8c7316f65040286c3f3fabb4f5aa6778b62 |
| uavfd_f550/__main__.py | 1269 | 216aa3250d61dd5085ccd04becf965c352cb4c2f1786ace2a71aaadd18e353ed |
| uavfd_f550/analysis.py | 18485 | aefa0af92a8b4921779d72df5079c1de1a45c25947e9ab5a6b70faf056b283e9 |
| uavfd_f550/audit.py | 4464 | c2c7dcef08d5c7c2a622068f341838f6b201aab492147a7418bbaa98263f09c2 |
| uavfd_f550/common.py | 2269 | 2d43d988213c0268bd2c5cb204e6fd30eb9547e012fee1a407f19af29f39cfce |
| uavfd_f550/configs/analysis.json | 646 | f152f5f6d7ea708ed226618fd01cd8410b5e16cd02b01149e9058c0b1bc73484 |
| uavfd_f550/configs/demo.json | 527 | 27dfcfe13fb75011fc333245cdb754a1b05a3a15a0f2b2f7d54f9d32a288e8ae |
| uavfd_f550/configs/split.json | 611 | ebdee24fb48493729fe5d8dec6ea2e89cb80537e072011571777bb59f26731b5 |
| uavfd_f550/configs/uavfd_aircraft.json | 1301 | 40753b2617e465e1929d42fc03807da531981058a91255f4295224ef8f174b51 |
| uavfd_f550/data.py | 5131 | 1e6aaa04750d87aa8dc7536d42d31a8f8285b4ea14045aa47f43873d0cf9afc5 |
| uavfd_f550/dynamics.py | 7389 | 0486b14d1e8b3e595d1ce3759f00b01d084c41eb196e72d60bc2d493bd8027e0 |
| uavfd_f550/provenance.py | 6048 | cb86708affc500d98628359970e7fedda253bb139651ad8a9e6b863e52bbcf22 |
| uavfd_f550/reports.py | 29897 | fe846464c669d34a6192e5a512ab5b7107e1a4a8dd43392d85e7233bab2fe266 |
| uavfd_f550/requirements.txt | 50 | 09b88dd5bad4314f76660f99dea3ba79006496ba474f5404a6b4e1e2a7699efd |
| uavfd_f550/signals.py | 5795 | e6c2a88290849806485096dc80718eff336f0d414d8bcf5fc2f62be4f3488579 |
| uavfd_f550/sources.py | 4526 | e8a904b8fb04179a2ab675a91dd00bbcc605714f60dfbfe97c54303dc61320c0 |
| uavfd_f550/tests/test_model.py | 4710 | b3c3478effba6676b0544dce0652ec1f0f95249b6aca99170d9ae2429f7cb6e2 |
| uavfd_f550/verification.py | 8327 | a0e81920050f3f70f59ae062cd0805e02f3948033de567485a030b9663d9ac9b |
| uavfd_f550/vibration.py | 5491 | 14df34ba3fdc8faafe26f05806c6e3c02e9584c445e7ae903c4c8d2e713693f7 |

CHANGE_MANIFEST.md不记录自身哈希，以避免循环依赖；Git可能规范化文本行尾。

实测验证：6个unittest通过；23项解析检查通过；18原始MAT+11作者脚本发布校验通过；test预期退出2；完整实飞模型门槛未通过。实际命令结果见checks.txt。

发布准备复核：报告重生成成功；3张仓库图与原运行输出SHA256一致；3个Markdown图片链接均解析到仓库文件；6项原有修改哈希一致。
