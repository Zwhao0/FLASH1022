# Campaign round3：三组六例

固定6例：strict/public/core/first actual accepted各6/6；peer 4/6。

| 组 | 布局 | 类型/阶段 | GPU | 检测步 | public/core/peer | FMA24/512+128/aux48 |
|---|---|---|---:|---:|---|---|
| group1 | legacy | L1 whole | 11 | 48 | correct_rank/correct_rank/correct_rank | 全有效 |
| group1 | legacy | L2 bwd | 13 | 104 | correct_rank/correct_rank/rejected | 全有效 |
| group2 | node_local_pp | L1 whole | 13 | 47 | correct_rank/correct_rank/correct_rank | 全有效 |
| group2 | node_local_pp | L2 bwd | 13 | 102 | correct_rank/correct_rank/rejected | 全有效 |
| group3 | legacy | L2 fwd | 0 | 48 | correct_rank/correct_rank/correct_rank | 全有效 |
| group3 | legacy | L1 whole | 8 | 104 | correct_rank/correct_rank/correct_rank | 全有效 |

独立健康监测218步，候选0；mixed0、纯健康PROFILE0窗，无定位比例。两例BWD peer为有效aux拒判，沿原允许门；没有调整融合/阈值。

三组全部public封存后才读truth评分；FMA24、core512角色/128marker、aux48全部有效。四轮24计划在任何campaign GT评分前预冻结，后轮无调参或重抽；不与此前4例合并。

source47e07b1c；工作站106提交字节，远端实际执行subset/catalogs。详见score_brief、全audit score_summary与GT前scoring_rule。
