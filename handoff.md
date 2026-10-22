# FLASH v1交接（2026-10-22）

PP节点内、DP跨节点布局快验完成，代码及测试结果发布到GitHub main。遵守AGENTS.md；只读当前flash_v1/results/论文/原Greyhound，不读旧FLASH/旧交接/Git历史、不恢复删除。

- 新布局显式参数`--parallel-layout node_local_pp`（默认legacy）：每节点TP2×PP2×EP2，跨节点Expert-DP2/dense-DP4；冻结模型rank0/9/14/7在线4/4正确，均2步检测，163健康步零触发，4窗零丢失。16卡FP32梯度/SGD验证通过。见flash_v1/LAYOUT_LOCALPP.md及results/layout_localpp/report.json；本次未重测开销。
- 本轮注入GPU stream `_sleep`：每微批反向累计450ms，注意力/本卡路由专家各半；2微批每步名义900ms，rank0/9/14/7各单独25步，不改通信/时钟。450ms单次实测445.60–446.59ms，迭代均值增加74.10%–102.82%，属于较强故障，不能外推轻微故障。见results/layout_localpp/injection_summary.json。
- 当前真实2节点16卡TP2×PP2×EP2×Expert-DP2，完整Qwen派生MoE，seq16/2微批。生产入口：工作站`python3 -m flash_v1.scripts.online16 --session 全新ID`。
- MONITOR已改native关闭时直接转发、mmap逐步队列/CPU控制器、复用WORLD2float控制与异步D2H；去掉Python整数setitem隐含同步H2D。BEAST参数、完整8步PROFILE和定位阈值不变。双节点drain ACK后才关Redis，修复退出竞态。
- 最终真无探针ABBA平均差−1.173%，配对−0.068/−2.307%，59项检查通过；200健康步零触发、loss差0。同次训练交替增量−1.096%（A共享探针/测量，非总成本）。仅本次平均点估计<1%，负差不称加速，不保证每个短窗/长期。
- rank6/10在线故障回归2/2，158健康步零触发；两8步窗完整16rank覆盖、零丢失。匹配故障PROFILE差−0.621%。冻结定位源码/模型hash未改；此前离线盲测16故障+4健康全正确。9项CPU回归及探针记录/事件等价检查通过。
- 结果见flash_v1/MONITOR_OPTIMIZATION.md及results/monitor_optimization/final_summary.json。首轮+9.835%和第二轮收尾失败均保留；最终是全新四组，不替换失败。
- 快验泛化仅score层：blind20向量×100随机rank重排，2000/2000等变；固定阈值缩放0.5x/1x/2x分别17/20、20/20、19/20正确，其余inconclusive。见generalization_audit.json。这不证明跨模型/并行规模/硬件/故障类型泛化；当前实现严格固定TP2/PP2/16rank/24层。
- GitHub Zwhao0/FLASH1022 main已更新新布局代码与测试证据；经GitHub工具发布，不上传本地旧Git历史。模型权重/完整原始PROFILE本地保留；本轮发布回执results/layout_localpp/github_upload.json。
- 评分仅CPU：`python3 -m flash_v1.scripts.monitor16_healthy_report`及`python3 -m flash_v1.scripts.monitor16_crossover --session monopt_crossover_01`。后者仅显式--run才启动GPU；生产不启benchmark。
- 当前无后台实验/遗留Redis17422/shm，16卡维持1980MHz，未改他人任务，证据results/layout_localpp/cleanup.json。停止新实验仅touch本session的STOP，勿kill他人/改时钟。
- 远端/home/campus/zhangweihao/FLASH1022，容器/workspace/FLASH1022。host2 flash1022-run，host1 flash1022-data16；Docker /opt/greyhound-docker/docker，DOCKER_HOST=unix:///run/greyhound-docker.sock；host1 sudo -n env。NCCL Socket、CUMEM/NVLS off、channels4/buff1MB；host1可用约2.3GB/card，Boost缺，native so在host2编译复制。
