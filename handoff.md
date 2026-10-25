# FLASH1022 handoff (2026-10-25)

## 当前状态

- 用户已暂停服务器实验；离线复核已完成，不再补最后N6在线。没有GPU新采集、训练或调参。v2依据为冻结范围离线复核+既有在线验证+真实paired成本，不作全场景准确率保证。
- GitHub代码提交`5cc7cb8347d1efb91ef37a143e1428e82b15086d`；API上传，本地Git HEAD未改，不reset/读旧history。`tmp/v2_github_tree_elements.json`为代码提交SHA256索引。
- 完整离线结果：`stage_forward_v1/postcommit_random_check_v1/offline_v2_review/offline_report.md`、score_brief.json、review_artifact_manifest.json。44 sessions/190raw tiles+16fit，实际public新推170窗；135为aux接受数，不是独立病例。L1小组2/2、L2小组4/4、L3 31/32、旧blind8 8/8、L4 31/32；原32/32上层只是既有在线结果。旧混杂L3 raw32/32标invalid_scope，不用于组件验证。
- 混合历史健康public1/26错GPU，aux34全拒判不能掩盖它；这是采到PROFILE后的条件结果，不是MONITOR误报率。fit/calibration/开发/旧失败producer上下文分别保留，不混成总体准确率。
- 最后N6原/recovery均被容器停止中断(trace103/169)，两partial共3窗无GT预测已锁；latest2例计划目标均命中，512/128+aux48有效，但FMA truth丢失，正式物理验证分母0，不冒充严格2/2或0/2算法失败；余4未评。计划健康监测93步0候选，原重复59仅一致性，不叠独立样本。原所有markers/partial/日志保留。
- 不再诊断容器触发者；已知TERM→10s→KILL，来源未定。server.md只作授权连接配置且gitignored，凭据不输出/上传；只读取回已有宿主日志后临时auth helper已删除，本地无实验进程。没有重启容器或新在线实验。

## 正式模式与结果边界

- 生产no-diag九文件patch `de9beade`，collector仍`57c7cd2d`，模型/阈值/min3不变；显式入口`python3 -m flash_v1.scripts.compute_v2 --plan <完整online计划.json>`启用aux{1,3,7}/48rows+startupGC lifetime；core仍8步512roles/128markers。旧入口默认False，不套用优化成本。
- L1/L2原固定各32/32，原upper6 whole/FWD/BWD 6/6；L3仅FWD开发31/32+独立component blind8 8/8（不合并）；L4仅framed-FWD ATen31/32。
- NEW GCdiag fixed-reg8 strict7/8保留原7；whole2+BWDstage2+norm组件2为6/6，330健康步0误触/错GPU；strict8门false、保baseline7门true。最终no-diag smoke3 public/core3/3、120健康步0检测/错GPU，三窗core512/128+aux48/8TP完整；仅运行差异检查，不冒充新全量准确率。
- PROFILE总体：legacy3.51%CI[2.57,4.39]；PP3runs×2C=6窗总体3.36%[2.25,4.39]。PP cold4.07%[2.28,5.64]仍跨5；cost源1ab4、GCdiag reg8源9072531e及no-diag正式源分别绑定，不称exact同runtime。
- MONITOR每布局2runs共4runs（原268+固定追加268）：完整ABBA平均legacy−.299%CI[−.540,−.064]，PP+.514%[.166,.876]；average_budget_gate均通过。PP BA1.230%[.669,1.840]原严格逐序门仍不通过，原单run/inconclusive全保留；负值不宣称加速。证据`monitor_gc_268_sparse3_draft/tmp_supplement/two_run_*_aggregate.json`。
- 最终报告`flash_v1/COMPUTE_V2.md`；验收限冻结范围和完整ABBA平均预算，不保证任意短窗/负载/总体准确率。

## 保留失败与约束

- sparse4+GC旧reg8 strict5/8保留：d42检测虽在健康时，PROFILE44–51在whole故障43–74内；d219 PROFILE221–228才是1窗纯健康core误定位，不称错GPU2。旧5.5s平台未获因果归因，健康A/B80未复现，源/所有原证据保留。
- 仅读当前flash_v1/results/论文/原Greyhound；不查Git历史/旧FLASH、不恢复删除文件。后处理错误只CPU恢复，不自动重跑GPU；不改分母/阈值逃失败。唯一GPU owner为v2_final_gpu。
