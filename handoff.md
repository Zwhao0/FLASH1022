# FLASH1022 handoff (2026-10-25)

## 当前完成状态

- v2生产代码`5cc7cb8347d1efb91ef37a143e1428e82b15086d`不变，API上传，本地Git HEAD未同步，不reset/查旧history。最新结果见flash_v1/COMPUTE_V2.md及下方报告；当前无待执行在线任务。
- 用户最新重复流程已完成：4轮/12组/24新例，batch quick_l1l2_four_round_campaign_v1；strict/自动检测/public/首accepted均24/24，L1 whole12/12、L2 FWD/BWD各6/6，两布局各12/12；全部FMA24、core512/128、aux48有效。core23/24（1拒判由peer补足），peer18/24（6个BWD有效拒判由core定位）；算法/阈值/模型/采样不改。
- 最新报告results/layer_campaign/stage_forward_v1/quick_l1l2_four_round_campaign_v1/report.md、score_brief.json/case_ledger.json；各轮3public封存后才GT，全24目标在首次GT评分前预冻结，未重抽/加样本。此前4例quick_l1l2_v1单独4/4，不混总。
- 新批次12个完成session健康869监测步0候选，纯健康PROFILE0窗不报条件比例；不作总体/长期零误报保证。两host无本批训练进程/shm/17422，临时askpass helper已删，项目容器保留running，server.md原文件/gitignore不动。

## 已完成清理及保护

- 用户已授权清理执行：126旧退出容器/334失败路径，df可用空间实增140.376GiB，0错误/跳过。FLASH1022整目录/模型/源码、保留容器ID/镜像/Env/挂载、GREYHOUND必要子树前后一致。flash_v1/SERVER_CLEANUP.md；私有receipt在tmp/server_cleanup_receipt.json。
- GREYHOUND/Greyhound、dataset、checkpoints与results各s4-reference-exact已整理。FLASH1/runtime是实际DockerRootDir，原地保护，旧四bind路径留兼容symlink。保留flash1022-data16/run及greyhound-s4-reference-exact-node0/node1；镜像保留，不动外部服务、不全局prune。greenhound/opensource/xpu和oss-xpu-build范围未明确暂留。

## 结果边界与旧记录

- offline_v2_review已重推旧190raw+16fit，public170窗；L1小组2/2、L2小组4/4、L3 31/32、旧blind8 8/8、L4 31/32。旧混杂L3 raw32/32无效，来源/重叠病例不合总体准确率。135仅aux接受窗数。
- 原64case core-only扩展32/32与严格30/32（旧L2两例无in-fault触发），无aux，不作完整融合或最新MONITOR重跑。旧四层随机N6两partial缺FMA truth，仅计划目标2/2辅助观察，余4未评；原失败数据全部保留。
- 混合历史健康public1/26错GPU保留；aux34拒判不代表public零误报，也不等于MONITOR率。最新869健康观察与该条件PROFILE样本分开。
- MONITOR完整ABBA平均legacy−.299%、PP+.514%（总体CI上界<1）；PP BA单序1.230%严格逐序门未过。PROFILE总体3.51%/3.36%（总体CI上界<5）；PP cold区间跨5保留。负值不作加速，结论限平均预算/已测范围。
- 正式模式aux{1,3,7}/48+startupGC lifetime、core8步、model/min3不变。workstation106提交资源匹配，server实际执行subset/catalog一致；13非链便捷脚本/tests未部署+3test版本差预存在，清理前后无变化，不声称全106部署。

## 约束

- AGENTS.md简洁；不读旧FLASH/Git history、不恢复删文件，不改分母/阈值逃失败。server.md只授权连接，凭据不输出/上传。科学失败记录在FLASH1022保留；只清用户明确范围的旧开发资源。不写下一轮推测。
