# v2 离线定位回放与既有在线日志复核

服务器实验已暂停；本次只CPU重推/评分，模型、阈值、固定aux1/3/7、min3和融合均未改。两部分均先封存全部可用新预测，再读标签；历史开发标签此前已知，不称新盲测。

## 本轮既有在线日志复核（N6计划全部保留）

原attempt只有legacy103/270步，recovery169/270步；PP两次均未启动。同case预声明recovery优先一次，original只做一致性。完整实际故障窗/连续trace决定覆盖，不能因无PROFILE排除漏检。

| 计划case | 状态 | 记录检测/计划目标public | core/peer | 证据与物理有效性 |
|---|---|---|---|---|
| legacy-0，L3/rank0 | recovery故障32步trace完整 | 检测、public rank0 | core拒判；peer rank0 | PROFILE71–78，512角色/128marker、aux48完整；FMA truth缺失 |
| legacy-1，L1/rank7 | recovery故障32步trace完整 | 检测、public rank7 | core/peer rank7 | PROFILE137–144，512/128、aux48完整；FMA truth缺失 |
| legacy-2，L1/rank3 | 尚未到达 | 未评 | 未评 | 未评 |
| PP-0，L4/rank3 | PP未启动 | 未评 | 未评 | 未评 |
| PP-1，L2/rank14 | PP未启动 | 未评 | 未评 | 未评 |
| PP-2，L2/rank7 | PP未启动 | 未评 | 未评 | 未评 |

两例仅可报计划目标条件命中2/2、core1/2；不能作为物理注入验证后的正式准确率。两attempt所有whole_compute_truth文件缺失、injection日志均0字节，故正式物理验证分母为0。原scorer严格成功字段为0，表示缺truth无法通过，不解释成0/2算法失败。全部3个可用PROFILE重新调用生产diagnose（original1、recovery2），与记录public一致。

recovery健康连续非warm/nonfault监测93步、候选0；纯健康PROFILE为0窗，因此没有健康定位率分母。original健康59步候选0仅一致性，不独立叠加。两例host2 rank0 wall非PROFILE中位数相对相邻健康固定前后各最多16步为+23.19%、+56.51%；这是观察到的wall差异，不能替代FMA真值或PROFILE开销。

## 历史数据离线定位回放

固定旧全集44 sessions：190 raw tiles（186完整、4不完整尾窗），另16 fit/calibration tiles。实际新production public推理170窗、仅aux可推32窗（含canonical16）、尾窗4个保留。135是aux接受窗口数，不能当135独立case。完整186 raw aux新推断与旧fixed3 rank/status无变化。

| 历史case ledger | 本次public命中/固定分母 | 边界 |
|---|---:|---|
| L1 scope revalidation | 2/2 | 本次仅小组覆盖，不是原32/32重测 |
| L2 scope revalidation | 4/4 | 同上 |
| L3 scope revalidation | 31/32 | 1无结果保留 |
| 旧blind8 | 8/8 | 当时独立、现在已知历史集 |
| L4 | 31/32 | 1无结果保留 |
| 旧reg8 | 6/8 | 2无结果保留 |
| reverted reg8 | 7/8 | 1无结果保留 |

全部12 ledger/138去重session-case key另存；来源/重复组分别报告，不合成总体准确率。旧peer L3原始回放32/32属于已确认的scope-confounded混杂组（`peer_v1/l3/injection_scope_confound.json`），标记`invalid_scope`，只保留原始观察，不用于组件准确率或合格验证。

健康数据混合历史fit/calibration、开发与旧regression/失败候选producer采集上下文；不代表当前正式生产采集器的无条件健康误报率，全部1与26均保留。健康raw42 tiles：38完整+4尾窗。aux结构可评34窗全部拒判，另4 legacy fit结构无效也拒判；不能用aux拒判掩盖public错误。完整public可复核26窗中有1次纯健康错GPU：旧`v2s_peer_priority_reg8_node_local_pp_01` epoch4，PROFILE219–226，原健康interval199–228；core rank1、aux拒判，生产融合保留core。该失败区别于已知旧sparse4 PROFILE221–228/rank4，未被排除。其余12完整健康窗（fit8、calibration4）同时缺core evidence与markers，不能冒充public通过。canonical16只aux可推且均拒判。

这1/26是已采健康PROFILE的条件定位结果，不是连续BEAST监测误报率；离线回放没有仿真未来触发、调度或分布变化。当前结果支持冻结范围内v2复核，不能仅据此代表全系统或总体准确率。成本仍引用既有正式paired预算，不用故障wall冒充开销。

证据入口：`score_brief.json`；A全证据`partial_full_audit.json`、冻结`partial_prediction_freeze.json`；B全部ledger/覆盖`complete_dataset/score_brief.json`、`case_ledger.json`、`health_coverage_audit.json`和`prediction_freeze.json`。没有GPU重采集、训练、调参或剔除失败。
