# v2 四层计算慢：冻结范围内实测结果

状态：冻结范围内精度检查与完整ABBA平均成本预算核验完成。GitHub提交由主窗口执行；提交后另做一次4–6例随机抽查，不据小样本作总体保证。

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

## 保留失败与范围

- sparse4+GC旧固定reg8 strict5/8不覆盖：d42虽在健康时检测，PROFILE44–51在whole故障内；d219 PROFILE221–228才是1窗纯健康core误定位。
- 弱例、拒判、漏检和健康错定位不从分母剔除；预测先封存后读truth。
- 结论限冻结拓扑、模型、注入协议和已覆盖scope；不宣称任意负载/硬件降频通用准确率。旧结果见[完整v1首页](README_V1.md)。
