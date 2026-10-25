# 最后四轮 V2 L1/L2 在线复核：24固定新例

四轮各6/6严格通过，共24/24自动检测、public正确、首个actual accepted正确；FMA24及完整PROFILE证据全部有效。core23/24、peer18/24，不与此前4例或旧core-only样本合并。

| 分层 | strict/public | core | peer |
|---|---:|---:|---:|
| L1 whole | 12/12 | 11/12 | 12/12 |
| L2 FWD | 6/6 | 6/6 | 6/6 |
| L2 BWD | 6/6 | 6/6 | 0/6（有效aux拒判） |
| legacy | 12/12 | 11/12 | 11/12 |
| node_local_pp | 12/12 | 12/12 | 7/12 |

唯一L1 core拒判为round1/group1 GPU2，由同窗有效peer补足；六个BWD aux完整有效但拒判，沿既有BWD门由core定位，未降低阈值或改变融合。没有以晚窗正确覆盖早先错GPU。

每例24条truth、24有效执行步，selected calls和物理FMA launch为正，无scope miss/drift。每窗完整8步、512角色、128 completion markers，aux48行与TP/context绑定有效，PROFILE均在故障内。

12个独立完成session的有效非warm/nonfault监测步869（四轮217/218/218/216），健康候选0、mixed0。纯健康PROFILE条件分母为0窗，因此定位比例为null；不报0%定位误报。连续监测候选观察也不保证全负载总体误报率。

首次读取本campaign真值前已一次冻结全部4轮、12组、24目标。每组whole L1和stage L2各一，布局各12例，L2 FWD/BWD各6例。rank、阶段分配、目标extra .5–.7、onset与seed均未按结果重抽。

每轮三组public全封存后才评分；最早in-fault epoch为primary，首个actual accepted按因果顺序独立保留。固定24分母，无剔除、缺失或无trigger漏报。

每组130–134步，warm12、真实健康监测及组内恢复至少30步、fault24、tail8–10。这是有限短轮验证，不替代原32步campaign或总体准确率。故障wall不用于PROFILE/MONITOR成本。

算法、模型、阈值保持提交`5cc7cb8347d1efb91ef37a143e1428e82b15086d`的生产v2，source freeze `47e07b1c`。工作站106个提交代码/模型资源逐字节核验；服务器检查实际执行subset和catalogs一致。已知13个便捷scripts/tests未部署、3个非实际链test版本差保留，不声称服务器全部106均部署。

`batch_ready.json`锁定全部目标；`score_brief.json`提供分层、健康及全部case结果；`case_ledger.json`全列24个ID。各轮保留brief/report、public/index/completion、GT前scoring_rule和prediction_seal_and_label_refs。

`upload_manifest.json`列建议上传的小复现文件。各轮`score_summary.json`全audit及raw rank truth/logs保留本地。
