# 分层独立测试（2026-10-23）

L1通过：固定32（legacy/localPP各16、每GPU单独一次），32/32严格端到端正确，无错卡/拒判/未触。32次均故障期内触发，完整8步PROFILE全部在40步故障内，16rank×8step×4角色及完成标记齐全，实际FMA有效。健康误触3/1050（全localPP），保留，不据准确率通过声称无误触。

当前dc89模型、BEAST30窗/8步PROFILE、recent-three确认、MONITOR v3冻结；新seed随机顺序与强度，比例额外真实FMA覆盖主FWD+BWD（不含Optimizer），这是计算负载模拟，不是硬件降频。目标估计整步额外40–80%，一次time-only实测37.7–93.6%、无低于30%例；正式非PROFILE实测45.1–135.7%。无改剂量/删例/换例。每布局1177步正式+177步time-only，全部完成后独立评分。

故障首步为0：detect中位3/最大11，PROFILE start中位5/最大13、end中位12/最大20。下一层冻结前可据此考虑32步故障+30步健康，不能视为全场景延迟保证。L2结果见下。

证据：results/layer_campaign/l1/{design,freeze,plan_ready,execution_lock,cases,summary,manifest_source_audit,cleanup}.json；预测先于truth评分落盘。双节点运行源/模型一致；freeze另含未部署未执行的成本脚本monitor_cost_v3.py，源审计明确保留此辅助文件差异。四会话completed；owned进程/Redis17422/shm清空。旧独立32=29/32及所有旧结果不变。

## L2

未通过32/32门槛：严格端到端30/32，两layout各15/16，FWD/BWD各15/16；30次fault内触发均定位正确且完整8步PROFILE在32步fault内，0错卡/拒判，32次实际FMA有效且0miss/drift。健康4/1050误触（legacy3/localPP1），辅助非PROFILE健康1025步。

两失败均保留：legacy BWD rank2在167–198步未触发，正式非PROFILE退化40.8%、此前健康均值3394ms；localPP FWD rank0故障787开始，被健康784预触发的PROFILE786–793跨入，之后无fault内新触发，退化65.4%。前者检测漏，后者检测状态/窗口覆盖；未出现fault内触发后的局部证据拒判或错卡，不能仅按已触发30次全正确宣布通过。两例均非time-only弱例。

每布局每rank一次、FWD/BWD各8且按PP/TP/EP及host平衡。新独立seed，冻结算法不改，随机强度估计整步额外40–80%，单次time-only实测6.6–184.0%（2例<30%原样保留），正式非PROFILE8.3–101.6%。固定1049步formal+177步time-only/layout，无调剂量/换例/缩窗。检测延迟1–19、PROFILE end offset10–28；30个有效定位窗完整。

证据results/layer_campaign/l2/{design,freeze,cases,summary,manifest_source_audit,cleanup}.json。4节点completed且全部冻结源/模型一致，双节点owned进程/Redis17422/shm清空。GPU停止等待root优化决策，不进L3，不重跑覆盖原32。L3/L4以后仅在上层gate过后新抽设计，按真实语义族分层、保持已验证FWD范围。

后续CPU诊断纠正：上述legacy r2的40.8%使用混合先前低背景均值，非BEAST baseline；最后10个非PROFILE健康步wall median5384ms，fault median5075ms（−5.7%，mean+11.2%），背景提前约3.4倍变慢。追加FMA有效，不代表故障发生时满足≥30%明显整步阶跃；历史telemetry空，不能归因具体外部活动。原30/32与全部病例保留。

## continuation_v1后续独立验证

上层L1抽4=4/4；新L2固定32=31/32，健康触发1，错卡/拒判0，实际continuation child=0，不能宣称GPU子窗路径已验证。唯一漏legacy r6 BWD被健康预触发窗跨入；目标证据满足续采条件，但其他rank未知旁支触发全局否决。新CPU最小修复在results/layer_campaign/continuation_v2独立审计，尚无修后GPU精度。PP中断原件保留，完整恢复除session_id外语义一致；结果见results/layer_campaign/continuation_v1/gates/l2/summary.json。owned资源已清，不进L3。

## continuation_v2独立验收

仅修正未知旁支全局veto并补metadata边界，dc89/BEAST/reset/阈值不改。新L1随机2/2、健康0；新L2固定32严格32/32，FWD/BWD各16、布局各16，32个fault内完整8步PROFILE且实际FMA有效。健康1/1050且拒判，无错健康定位；自然child=0，人工hint真实单child控制流测试另列，不能混作自然候选覆盖。旧30/32及31/32不改。源/模型四节点核验、owned与telemetry清空。结果results/layer_campaign/continuation_v2/gates/l2/summary.json；限制与冻结见results/layer_campaign/continuation_v2/README.md。暂不L3，先补首层4role之外的全stage可观测性。
