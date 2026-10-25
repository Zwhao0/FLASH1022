# FLASH1022 handoff (2026-10-25)

## 当前完成状态

- v2生产源码提交`5cc7cb8347d1efb91ef37a143e1428e82b15086d`，离线报告提交16ac596；API上传，本地Git HEAD未同步，不reset/查旧history。全部详细结果见flash_v1/COMPUTE_V2.md。
- 用户最后追加短轮已完成：quick_l1l2_v1，两拓扑各L1/L2一例、L2 FWD/BWD；strict/public/core/首accepted均4/4，24步FMA及512/128+aux48全有效，144健康监测步0候选。两public先封存再truth；PP仅一次startup无测量失败后fresh SID恢复，legacy未重跑。源/model及阈值未改，无后续在线任务。
- 最新短轮报告：results/layer_campaign/stage_forward_v1/quick_l1l2_v1/report.md；原N6两partial和缺FMA证明状态仍保留，计划命中2/2不当正式验证；不混入新4例。
- 用户清理已执行完成：两host删除126旧退出容器/334名单路径，df可用空间实增140.376GiB，0错误/跳过。FLASH1022整目录/模型/源码、保留容器ID/镜像/环境/挂载、GREYHOUND必要子树均前后一致。详见flash_v1/SERVER_CLEANUP.md；私有完整receipt在tmp/server_cleanup_receipt.json。
- GREYHOUND/Greyhound、dataset、checkpoints与results各s4-reference-exact已整理。FLASH1/runtime是实际Docker数据根，原地保护；旧四个bind路径留兼容symlink。保留flash1022-data16/run及greyhound-s4-reference-exact-node0/node1；全部现有镜像保留。不动外部服务，未全局prune。greenhound/opensource/xpu及oss-xpu-build范围未明暂留。

## 验证边界

- 离线报告offline_v2_review：190raw tiles+16fit，public重推170窗；135是aux接受窗，不是独立病例。L1小组2/2、L2小组4/4、L3 31/32、旧blind8 8/8、L4 31/32；旧混杂L3 raw32/32无效，分组不合总体准确率。
- 原L1/L2 core-only扩展中断前已生成结果：32/32与严格30/32（2无in-fault触发）；无历史aux，不当完整最新fusion或最新检测全量重测。旧32/32原报告口径不能覆盖该严格审计。
- 混合历史健康public1/26错GPU仍保留；aux34全拒判不代表public无误报，也不等于MONITOR率。
- MONITOR完整ABBA平均legacy−.299%、PP+.514%（总体区间上界均<1）；PP BA单序1.230%严格逐序门仍未通过。PROFILE总体legacy3.51%、PP3.36%（总体CI上界<5）；PP cold区间跨5保留。负值不作加速声明，验收限平均预算/已测范围。
- 正式入口compute_v2启用aux{1,3,7}/48+startupGC lifetime；core8步、模型/min3不变。服务器实际执行源码/模型SHA匹配v2；106本地提交资源中的部分便捷脚本与tests未部署/版本不同，预存在baseline，清理前后未改变。

## 约束

- AGENTS.md简洁；不读旧FLASH/Git历史、不恢复删文件，不重抽/篡改失败/改分母逃结果。server.md只供授权连接且gitignored，凭据不输出/上传。失败科学记录均在FLASH1022内保留；本轮只删用户明确名单的旧开发资源。
