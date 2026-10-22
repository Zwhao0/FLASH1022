# FLASH 第三章复现 v1

当前2节点16卡采用真实TP2×PP2×EP2×Expert-DP2。新算法已接入自动检测→8步PROFILE→在线定位；rank6/10故障回归2/2正确、158健康步零触发；常驻MONITOR已优化，最终无探针ABBA平均耗时差−1.173%（本次平均开销点估计<1%，不作加速或长期保证），200步健康零误报，见[MONITOR优化与对照](flash_v1/MONITOR_OPTIMIZATION.md)。原接入过程见[在线报告](flash_v1/ONLINE16.md)。此前冻结离线盲测故障16/16、健康4/4正确，见[数据研究报告](flash_v1/DATA16_STUDY.md)。旧健康差分在线入口保持停用。

从论文与GREYHOUND源码重新开始；未读取旧交接或旧FLASH实现。

- [运行与实现](flash_v1/README.md)
- [实测结果与边界](flash_v1/RESULTS.md)
- [论文机制规格](flash_v1/PAPER_SPEC.md)

真实Qwen派生0.5B MoE；4GPU原生Megatron完成检测与双标签定位。8GPU PP2首次误定位后，修正组计分并加入健康PROFILE校准，[独立新实测定位rank7](flash_v1/PP_OPTIMIZATION.md)。这些为4/8卡历史结果；当前16卡TP2与跨节点训练已通过数值验证。旧误定位、真实降频未触发、OOM、模板法失败及采集中止均保留。不宣称通用准确率，不包含Fail-Stop或第四章恢复。
