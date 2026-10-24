# FLASH v2 当前任务

冻结范围内四层计算慢结果、显式sparse3+startupGC模式及完整ABBA平均预算已核验，见[结果与边界](flash_v1/COMPUTE_V2.md)。当前由主窗口提交GitHub；随后按提交commit一次冻结N∈{4,5,6}及seed/两布局/rank/scope/剂量/随机onset，自动MONITOR→PROFILE→定位，完成小数量随机抽查。

- 原逐序严格门、弱例/漏检/拒判/失败不改写；平均预算通过不保证所有短窗或总体性能。
- 每次只一GPU故障；健康预热与恢复≥30步，先封存公开预测后读truth，不重抽逃结果。
- 只读当前flash_v1/results/论文/原Greyhound，不查旧FLASH或Git历史，不恢复删除文件；不扩Fail-Stop/通信故障/第四章SHIFT。
- 不干扰共享服务器其他任务；GPU执行由唯一owner协调，保留既有原始证据与失败。继续工作先读handoff.md。
