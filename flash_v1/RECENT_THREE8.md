# v2 recent-three确认（2026-10-23）

BEAST候选后，实际连续3步耗时median须超过候选baseline×原退化阈值（1.2）；拒绝则还原last_cp。dc89、BEAST参数/30窗、逐步输入、8步PROFILE及采样均不改，contiguous resume保持off。

旧32开发记录的30个fault候选全保留；健康517被过滤、568仍通过。Linux shadow未重现517，因此不能声称已验证消除该现场误触。原独立结果29/32及2/1050健康误触原样保留。

冻结候选后独立抽新8例：2布局×whole/stage/component/operator各1，每布局4不同GPU；component/operator从完整FWD catalog抽，stage随机FWD/BWD。一次time-only实测额外38.7–98.2%，8例均有效FMA/零scope miss，未按效果或定位调剂量。正式40步fault，固定8分母，无truth预测先落盘后独立评分。

结果8/8检出、8/8正确GPU，0错卡/拒判/未触，健康0/330；布局各4/4、四类各2/2。检出延迟2–3步；8个完整8步窗均在fault内，每窗512角色行+128完成标记，16rank/epoch/comm union全部通过。这是小规模开发回归，不能证明普遍达到四档准确率目标或绝对不退步。

运行层默认仍false，v1及缺少开关的旧冻结plan语义不变。新v2_collect.plan/online_validation_plan、v2_benchmark.benchmark_plan生成入口默认开启，允许显式false或CLI --no-monitor-confirm-recent-three对照；默认接入15项CPU/等价测试通过，不重复GPU。

新增确认仅CPU候选处理，无新GPU event/collective/常驻采样；本地fake-candidate微测增量约0.70µs，不含原BEAST。总开销继续引用MONITOR_COST_V3.md的漂移限制，不宣称稳定≤1%。

证据：results/recent_three8/{summary,cases,freeze,design,cleanup}.json；原始会话v2s_recent_three8_*。四会话全部完成，双节点owned进程/Redis17422/shm清空，GPU交回root；不再扩测。
