# MONITOR成本复核（2026-10-23）

冻结dc89定位、BEAST阈值、逐步输入与8步PROFILE；独立scoped32仍为29/32，不重跑、不替换。原成本报告均保留。

CPU细段对照发现root训练与尾等待互补：bare为752.66+3.85ms，原online为599.53+163.02ms；完整wall为757.03/762.97ms。等待主要反映跨rank进度，不能当作新增控制成本。root准备/collective enqueue/D2H enqueue/发布/日志各均值<0.25ms，无40ms常驻CPU缺陷证据。

实现仅减少冗余：非root只D2H复制epoch，16rank总payload从128减为68字节/步；末PP用一次stack替换zero+loss copy；预建GPU视图。复用原WORLD collective、读取epoch的同步边界及8步窗口不变。显式plan.monitor_cost_timing才记录CPU细段，无新增CUDA同步/event。源快照见results/monitor_cost_v3/source_before。

真实SGD ABBA：两布局各4×60步，每组全部48测量步；bare A无probe/证据/controller，健康方案无注入hook或scoped/whole dispatch。legacy完整wall合并−1.296%，相邻+1.837%/−4.501%；localPP合并−1.762%，相邻−2.331%/−1.213%。两布局全部身份/序列/消费/数值门禁通过，loss最大差0。三故障开发回归及一次既有268步零LR交叉补充见results/monitor_cost_v3/{regression_report,crossover_report}.json；所有步骤与失败保留，负差不解释为加速。

开发回归3/3正确（FWD6/expert13/whole2），检测差2/3/2步；每窗512角色行+128完成标记，16rank同epoch且comm union完整，205步loss最大差0、无健康误触、结束回MONITOR。24步PROFILE配对合并−2.214%，三窗−1.696%/−2.406%/−2.612%；匹配whole工具的共同dispatch税保留，不能作为纯MONITOR基线或≤5%置信保证。

短ABBA只有两相邻配对；配对范围是漂移描述，不是置信区间。零LR交叉只衡量生产metric/control及BEAST增量，不能代替真bare总成本。小结构改动未被证明造成端到端提速；不以均值负差宣称稳定≤1%。CPU控制/队列8项测试通过。

同进程semantic-only交叉268步全部完成/消费、无native/触发/PROFILE，零LR同位置loss一致。增量均值+0.549%，四配对+0.427/+0.284/+0.259/+1.233%；A共享controller/发布/证据对象，不能替代真bare总开销。13个GPU会话全部完成；两节点owned进程、Redis17422、shm均为空，GPU所有权交回root。汇总results/monitor_cost_v3/summary.json。首次CPU评分将无事件文件误当缺件，已保留失败并仅修复CPU评分，GPU未重跑。
