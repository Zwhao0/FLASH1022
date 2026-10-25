# v2 四层计算慢：冻结范围内实测结果

状态：代码提交`5cc7cb8347d1efb91ef37a143e1428e82b15086d`保持不变。用户追加的最后短轮L1/L2已严格4/4通过，之后完成服务器失败开发资源清理；无后续在线实验计划。原N6中断仍保留，不冒充完成。

## 配置与入口

- 2节点16卡，真实TP2×PP2×EP2×Expert-DP2；legacy与node-local PP两布局。
- 模型`bayliner1980/Qwen2.5-0.5B-MoE-A0.3B`，revision `4b89028ef7376d0e92acbbc4a7f560c2374aeec0`。
- `python3 -m flash_v1.scripts.compute_v2 --plan /path/to/approved_online_plan.json`；须提供完整online/core/peer模型绑定。
- 显式模式：core原8步（512roles/128markers），aux相对步`{1,3,7}`（48rows），TP2 matched-work/min3与模型阈值不变；固定模型startupGC lifetime策略。
- 只适用于该固定模型进程生命周期；已有永久代不被解冻，随训练子进程退出。旧入口默认不优化，不套用本模式成本。

## 分层精度（分别报告）

- L1 whole、L2 FWD/BWD stage：原core-only固定集各32/32（100%）；原独立upper6为whole/FWD-stage/BWD-stage共6/6。
- L3 component仅FWD：覆盖attention、routed-expert MLP、MoE block/router、normalization；开发集31/32，独立component blind8为8/8。未验证BWD组件，不能合并成独立32例结果。
- L4：仅framed-forward ATen，31/32；弱注入未触发保留固定分母。不外推BWD算子或全部kernel。
- 以上既有campaign不冒充最终稀疏生产版本的全量重测；使用冻结模型与针对性运行差异检查。

## 新候选与生产运行差异检查

- sparse3+startupGC+GCdiag新固定reg8：strict7/8，原7成功全部保留；strict8/8门仍false，预声明保baseline7候选门true。
- 其中whole2、BWD-stage2、normalization component2合计6/6；不是FWD-stage2。330健康步0误触/错定位。
- 唯一未触为legacy matmul operator rank0，与baseline一致；注入有效，分母8不变。
- [冻结结果](../results/layer_campaign/scope_guard_v1/peer_priority_v1/reg8_control_sparse3_gc_lifetime_v2/candidate_summary.json)含可选GC/iteration诊断，不称与最终no-diag exact runtime相同。
- 本地应用普通no-diag九文件patch `de9beade`，collector仍`57c7cd2d`；19项CPU检查及真实launcher配置路径通过。
- [生产no-diag smoke3](../results/layer_campaign/stage_forward_v1/production_no_diag_smoke3/score_summary.json)：legacy whole r6、BWD r15及PP FWD-stage r13，public/core均3/3；120健康步0检测/错GPU。
- 三窗均core512/128、aux48完整且8个TP pair可比；whole与FWD peer同意，BWD peer正确拒判。仅验证运行差异，不作为新全量准确率。

## 成本与不确定性

- PROFILE：legacy 3.51%，95%区间[2.57,4.39]；PP原1run+固定追加2run共6窗，合并3.36% [2.25,4.39]，六个run/order点均<5%。
- PP cold 4.07% [2.28,5.64]，warm 2.65% [1.30,4.05]；cold区间仍跨5%，保留不确定性。
- [PP合并证据](../results/layer_campaign/stage_forward_v1/sparse3_pp_supplement/aggregate.json)：全部预声明窗口纳入；同输入独立进程成本重复，不是新增盲准确率。
- PROFILE源`1ab4`与GCdiag reg8、最终普通生产源分别绑定；不称三者exact同runtime。
- MONITOR：每拓扑原268+预声明一次268追加，共2runs/layout、总4runs；12warm+ABBA四块64，32真实settle/32measure，完整wall、真实SGD与输入/状态重放核对通过。
- 完整ABBA平均：legacy −0.299%，95%CI[−0.540,−0.064]；PP +0.514% [0.166,0.876]；两者pooled点与CI上界均<1%，average_budget_gate通过。
- PP合并BA单序仍+1.230% [0.669,1.840]，原严格逐序门仍不通过；原run inconclusive及全部窗口保留，不称所有短窗<1%。负值仅表示此配对试验未测出正开销，不宣称加速。
- 聚合使用max16rank完整wall总和比、逐run/order分层配对block2 bootstrap（seed20261024，10000次）；[legacy](../results/layer_campaign/stage_forward_v1/monitor_gc_268_sparse3_draft/tmp_supplement/two_run_legacy_aggregate.json)、[PP](../results/layer_campaign/stage_forward_v1/monitor_gc_268_sparse3_draft/tmp_supplement/two_run_node_local_pp_aggregate.json)。原单run block4结论保留，未改写为通过。
- 本轮按冻结精度范围及完整ABBA平均预算验收：MONITOR<1%、PROFILE<5%的两拓扑总体CI上界均达标；不承诺任意短窗、负载或未来总体性能。

## 最新正式配置短轮L1/L2

- [最后短轮报告](../results/layer_campaign/stage_forward_v1/quick_l1l2_v1/report.md)：两拓扑各L1/L2一例，L2覆盖FWD/BWD；自动检测、public/core与首个实际接受结果均4/4正确，24步FMA及core512/128+aux48全部有效。144健康监测步0候选，不当长期零误报保证。
- 使用已有detached启动器；PP一次启动失败尚无故障测量，只改SID恢复，legacy不重跑。两public封存后才读truth。短故障与小样本不替代32步campaign或总体准确率。
- 取消旧数据扩展前已生成的core-only审计：L1 32/32，L2严格故障内触发口径30/32（2无in-fault触发）；这是旧在线日志+最新core回放，无aux，不能当完整新融合验证，也不是重新运行了最新MONITOR。
- [服务器清理](SERVER_CLEANUP.md)：删除126旧退出容器和334失败路径，实际释放140.376 GiB；FLASH1022、GREYHOUND、所需环境/模型已保全。

## 已完成的离线复核

- [离线报告](../results/layer_campaign/stage_forward_v1/postcommit_random_check_v1/offline_v2_review/offline_report.md)：固定生产模型/融合/采样，全部可用预测先封存再评分；历史已知标签不冒充新盲测。
- 历史L1小组2/2、L2小组4/4、L3 31/32、旧blind8 8/8、L4 31/32。不同来源和重叠病例分开，旧混杂L3 raw32/32标invalid_scope，不用于组件验证。
- 混合历史健康PROFILE public错GPU1/26；aux34窗全部拒判不代表public零误报，也不是MONITOR总体误报率。无trigger的完整历史病例仍在固定分母内。
- 本轮中断前2例计划目标均命中，但FMA truth未落盘，仅辅助观察；余4未评。原/recovery不重复计数，计划健康93监测步0候选。新增在线测试已停止，不再补6例。

## 保留失败与范围

- sparse4+GC旧固定reg8 strict5/8不覆盖：d42虽在健康时检测，PROFILE44–51在whole故障内；d219 PROFILE221–228才是1窗纯健康core误定位。
- 弱例、拒判、漏检和健康错定位不从分母剔除；预测先封存后读truth。
- 结论限冻结拓扑、模型、注入协议和已覆盖scope；不宣称任意负载/硬件降频通用准确率。旧结果见[完整v1首页](README_V1.md)。
