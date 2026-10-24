# FLASH1022 handoff (2026-10-25)

## 当前任务

- 冻结范围内v2结果及完整ABBA平均预算已核验；主窗口准备GitHub原子提交。仅待提交后N∈{4,5,6}随机抽查：两布局各≥2、四层各≥1、rank均匀0..15、L2 phase随机、L3/L4仅已覆盖FWDscope，合理健康校准剂量；提交commit后seed及全设计一次抽取冻结，不重抽。
- 首发随机43..58（12warm后≥30监测健康步）；后续fault结束后≥30恢复再随机延迟0..15；单GPU/32步/无重叠/末尾≥30健康。正常自动MONITOR→PROFILE，不按注入时刻手工请求；预测先封存后truth。草稿`stage_forward_v1/postcommit_random_check_draft/`；synthetic构造/空site保留invalid CPU检查通过，无正式seed。

## 正式模式与结果边界

- 生产no-diag九文件patch `de9beade`，collector仍`57c7cd2d`，模型/阈值/min3不变；显式入口`python3 -m flash_v1.scripts.compute_v2 --plan <完整online计划.json>`启用aux{1,3,7}/48rows+startupGC lifetime；core仍8步512roles/128markers。旧入口默认False，不套用优化成本。
- L1/L2原固定各32/32，原upper6 whole/FWD/BWD 6/6；L3仅FWD开发31/32+独立component blind8 8/8（不合并）；L4仅framed-FWD ATen31/32。
- NEW GCdiag fixed-reg8 strict7/8保留原7；whole2+BWDstage2+norm组件2为6/6，330健康步0误触/错GPU；strict8门false、保baseline7门true。最终no-diag smoke3 public/core3/3、120健康步0检测/错GPU，三窗core512/128+aux48/8TP完整；仅运行差异检查，不冒充新全量准确率。
- PROFILE总体：legacy3.51%CI[2.57,4.39]；PP3runs×2C=6窗总体3.36%[2.25,4.39]。PP cold4.07%[2.28,5.64]仍跨5；cost源1ab4、GCdiag reg8源9072531e及no-diag正式源分别绑定，不称exact同runtime。
- MONITOR每布局2runs共4runs（原268+固定追加268）：完整ABBA平均legacy−.299%CI[−.540,−.064]，PP+.514%[.166,.876]；average_budget_gate均通过。PP BA1.230%[.669,1.840]原严格逐序门仍不通过，原单run/inconclusive全保留；负值不宣称加速。证据`monitor_gc_268_sparse3_draft/tmp_supplement/two_run_*_aggregate.json`。
- 最终报告`flash_v1/COMPUTE_V2.md`；验收限冻结范围和完整ABBA平均预算，不保证任意短窗/负载/总体准确率。

## 保留失败与约束

- sparse4+GC旧reg8 strict5/8保留：d42检测虽在健康时，PROFILE44–51在whole故障43–74内；d219 PROFILE221–228才是1窗纯健康core误定位，不称错GPU2。旧5.5s平台未获因果归因，健康A/B80未复现，源/所有原证据保留。
- 仅读当前flash_v1/results/论文/原Greyhound；不查Git历史/旧FLASH、不恢复删除文件。后处理错误只CPU恢复，不自动重跑GPU；不改分母/阈值逃失败。唯一GPU owner为extended64_protocol。
