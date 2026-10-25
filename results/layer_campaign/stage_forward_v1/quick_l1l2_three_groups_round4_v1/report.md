# Campaign round4: three groups, six fixed cases

Strict/public/core/first actual accepted: 6/6. Peer: 4/6; two BWD peers validly reject under the unchanged BWD rule.

| Group | Layout | Tier/phase | Rank | Detection step | Public/core/peer |
|---|---|---|---:|---:|---|
| group1 | legacy | L2 fwd | 2 | 48 | correct_rank/correct_rank/correct_rank |
| group1 | legacy | L1 whole | 0 | 103 | correct_rank/correct_rank/correct_rank |
| group2 | node_local_pp | L2 bwd | 6 | 47 | correct_rank/correct_rank/rejected |
| group2 | node_local_pp | L1 whole | 12 | 102 | correct_rank/correct_rank/correct_rank |
| group3 | node_local_pp | L2 bwd | 12 | 47 | correct_rank/correct_rank/rejected |
| group3 | node_local_pp | L1 whole | 1 | 101 | correct_rank/correct_rank/correct_rank |

All six: valid24-step FMA truth, complete512 roles/128 markers, valid48-row auxiliary. Healthy monitoring216 steps, zero candidates/mixed; pure healthy PROFILE0, no ratio.

All three public snapshots sealed before truth. All four rounds were frozen before any campaign scoring. No algorithm/model/threshold change. Source47e07b1c: workstation106 submitted resources, server actual execution subset/catalogs. Previous four-case check excluded.
