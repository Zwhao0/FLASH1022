# FLASH第三章 v1 当前状态

已完成：真实2节点16卡TP2×PP2×EP2×Expert-DP2；数据驱动定位接入自动检测/8步PROFILE/在线定位；常驻MONITOR优化和验证。结论与范围见flash_v1/MONITOR_OPTIMIZATION.md，继续工作先读handoff.md。

当前用户要求：上传v1代码到Zwhao0/FLASH1022，然后只讨论算法普遍性/泛化能力；可用现有数据快速验证，不改算法/阈值，不新增GPU实验或优化。

约束：
- 仅当前flash_v1/results/论文/原Greyhound，不读旧FLASH、旧交接或Git旧历史，不恢复已删文件。
- 不做Fail-Stop、通信故障或第四章SHIFT；保留真实失败/未触发/inconclusive，不通过改阈值制造成功。
- 共享服务器保留其他任务，不改其时钟/进程；每次只运行一个本实验训练作业，完工清理自身资源。
- 遵守简洁AGENTS.md和handoff.md；当前没有后台训练或待续实验。
