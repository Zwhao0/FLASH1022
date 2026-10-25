# Campaign round2：三组六例

固定6例：strict 6/6、public 6/6、core 6/6、peer 5/6、first actual accepted 6/6。

| 组 | 布局 | 类型/阶段 | GPU | 检测步 | public/core/peer | FMA24/512+128/aux48 |
|---|---|---|---:|---:|---|---|
| group1 | legacy | L2 fwd | 8 | 47 | correct_rank/correct_rank/correct_rank | 全有效 |
| group1 | legacy | L1 whole | 7 | 102 | correct_rank/correct_rank/correct_rank | 全有效 |
| group2 | node_local_pp | L1 whole | 11 | 48 | correct_rank/correct_rank/correct_rank | 全有效 |
| group2 | node_local_pp | L2 fwd | 1 | 103 | correct_rank/correct_rank/correct_rank | 全有效 |
| group3 | node_local_pp | L1 whole | 6 | 47 | correct_rank/correct_rank/correct_rank | 全有效 |
| group3 | node_local_pp | L2 bwd | 12 | 103 | correct_rank/correct_rank/rejected | 全有效 |

独立健康监测218步，候选0；mixed0、纯健康PROFILE0窗，无定位比例。BWD peer有效aux拒判沿原允许门，未改阈值/融合。

三组全部public封存后才读truth评分；FMA24、core512角色/128marker、aux48全部有效。四轮24计划在任何campaign GT评分前已预冻结；后轮不依据本结果调整。本轮6例不与此前4例合并。

source47e07b1c；工作站106提交字节，远端实际执行subset/catalogs。详见score_brief、全audit score_summary与GT前scoring_rule。
