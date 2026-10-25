# Campaign round1：三组六例

固定6例：strict 6/6，public 6/6，core 5/6，peer 5/6，first actual accepted 6/6。

| 组 | 布局 | 类型/阶段 | GPU | 检测步 | public/core/peer | FMA24/512+128/aux48 |
|---|---|---|---:|---:|---|---|
| group1 | legacy | L1 whole | 2 | 48 | correct_rank/rejected/correct_rank | 全有效 |
| group1 | legacy | L2 fwd | 6 | 103 | correct_rank/correct_rank/correct_rank | 全有效 |
| group2 | legacy | L1 whole | 3 | 47 | correct_rank/correct_rank/correct_rank | 全有效 |
| group2 | legacy | L2 fwd | 2 | 102 | correct_rank/correct_rank/correct_rank | 全有效 |
| group3 | node_local_pp | L2 bwd | 8 | 48 | correct_rank/correct_rank/rejected | 全有效 |
| group3 | node_local_pp | L1 whole | 4 | 102 | correct_rank/correct_rank/correct_rank | 全有效 |

g1 L1 GPU2 core拒判由同窗有效peer补足；g3 L2 BWD GPU8为有效aux拒判，沿用既有BWD允许门，未改阈值/融合。

独立健康监测217步（73+71+73）0候选；mixed0，纯健康PROFILE0窗，不报告定位比例。每例24步注入、8步PROFILE；全部后续3轮计划在本轮GT评分前已全部冻结。此6例不与此前4例合并，最终campaign固定24例；尚未完成的轮保留待评。

3组全部public封存后才读truth评分。全部FMA24、512角色/128markers和aux48有效；source47e07b1c与生产算法不改。source边界为工作站106提交字节核验、远端实际执行subset/catalogs，不声称未部署便捷scripts/tests已部署。

详见score_brief.json、score_summary.json全audit及GT前scoring_rule_before_truth.json。
