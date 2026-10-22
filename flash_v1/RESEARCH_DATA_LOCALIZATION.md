# 数据驱动定位研究（2026-10-22，采集前审查）

以下保留采集前的假设与建议；真实TP已实现，最终采样、失败方法及冻结盲测结论见[DATA16_STUDY.md](DATA16_STUDY.md)。本文件中“现实现/未验证”均指审查时状态。

范围：本轮根目录 `results/`、第三章中文论文、`Greyhound-master`、现实现/测试；未读旧FLASH、旧交接或Git历史，未改代码/启动实验。新路线是2节点16卡 TP2×PP2×EP2×ExpertDP2；须先实现并验证真实TP，现适配器明确TP=1，现8卡样本不能证明新拓扑有效。

## 为什么双标签没有解决问题

- 中文论文PDF46–47/50页：同标签clique按最大平均通信时延筛组，再在组内取平均时延最小rank；多标签一致是置信证据，未证明其独立或充分。原Greyhound的DP/TP标签含PP stage，当前统一dense_DP/Expert_DP遗漏stage。
- 同步通信近似 `duration_r = transfer + max(arrival)-arrival_r`。慢计算rank晚到、通信短；但PP回压也能令健康rank晚到。两个标签共享计算调度/PP等待，不能把“双标签”当独立两票，更不能据此证明根因。
- 现数据不同stage的通信尺寸/调用次数明显不同：v2健康PROFILE中rank0–3每rank Expert_DP1457条+dense_DP893条，rank4–7分别仅47+47条。完整group_id的多条operation还可能共享同一Event耗时。逐行独立统计或直接累计微秒会高估证据。
- 已停止矩阵实际运行的是参考差分扩展。rank2事件中真rank2有双标签共6146us，健康stage的rank4单Expert_DP达7556us而获胜；rank3与其PP链对端7都出现双标签。根因候选与传播症状可同时得高分，调成“只收双标签”也不能解决后一例。
- 现算法另有任一尺寸冲突全局拒绝、PP链冲突拒绝规则；这些能降低覆盖率，不会凭空增加根因信息。关闭健康差分后，对本地已有PROFILE只读重放：rank0不确定，rank2→2、rank3→3；旧rank7/v2仍不确定。此重放不是新盲测，更不能声称矩阵完成。
- rank1监控在iteration43触发，故障首步46，20秒PROFILE跨越故障前后。必须单列故障前误触发，不能把该PROFILE简单作为纯健康或纯故障训练样本。

## 可辨识性与采集的最小补充

现PROFILE有rank、成员、clique、count/dtype、group_id、主机timestamp_us、buffer_addr；缺显式iteration/microbatch/phase、通信实例全局匹配、GPU到达时刻、PP边时序与计算区间。旧记录可验证统计指纹是否稳定，不能由聚合通信时延重建PP根因。主机API时间不是GPU实际开始时刻；跨节点时钟也不能直接相减。

新采集建议保留原始记录再离线聚合：

1. 固定拓扑与训练配置，报告实际TP/PP/EP/ExpertDP成员和rank坐标、每组真实operation；dense DP=EP×ExpertDP=4，world=TP×PP×DP=16。TP×EP组合组不能误标纯TP。
2. 每条记录增加iteration、microbatch、forward/backward/optimizer阶段、实际op、count/dtype、stream、comm局部序号、group内op索引。稳定key用语义/成员/phase/序号，不能跨run依赖地址或unique-id哈希值。
3. 使用communicator+operation序号匹配同一次collective，各rank核对计数一致；group_id仅进程内计数，不能直接当跨rank全局ID。Group共享耗时只计一个观测块。
4. 为PP区分“本rank慢计算”与“等上游”，在PROFILE增加少量计算段GPU Event和PP Send/Recv依赖/等待记录；至少一个不包含PP等待的本地计算段。事件插入需遵循stream依赖，不把CPU排队/异步等待算成计算。保留不增加插桩版本作开销对照。
5. 同一配置采独立健康run及16张卡逐卡持续故障run；健康/故障随机交错，记录强度、实测时钟、外部负载、配置hash与恢复结果。GPU stream延迟和真实降频分开报告，不能混作同一故障分布。

## 建议最小离线算法

先保持小模型与明确拒判，不上复杂神经网络；健康差分仅作新离线研究方法，现在线禁用入口保持关闭。

- 固定窗口，按稳定key计算每个完整迭代/Group的median时延和组内相对时延 `median_peer(duration)-duration_r`；用独立健康run的median/MAD或经验分位数标准化。尺度下限由健康训练数据确定；不能把大量相关operation当独立样本估标准误。
- 将标准化通信残差、计算段异常、PP邻居阶段异常组成向量；每种标签/phase限权，保留有符号值。明确缺测mask，不能用0代表健康，也不能让某rank缺失特征成为标签暗号。
- 候选A：只用健康标定的计算异常+同步等待证据，按rank等权融合；候选B：用逐卡开发故障构造16个稳健指纹，采用尺度归一后的最近原型/余弦距离，最多一个全局拒判阈值及第一第二名间隔阈值；健康作为第17种状态。不要同时调十几个融合门槛。
- 在独立开发run先检查每对rank原型距离是否大于run间噪声，重点检查PP链对端、TP伙伴、同节点。若两卡特征不可分，报告候选集合/不确定并补充计算或PP观测，不能靠rank号打破平局。
- 固定论文原两阶段算法作为基线，A/B分别报告，做通信-only、加计算、加PP消融。指纹法只声称固定训练配置下已知16卡单计算慢故障识别；不能外推未知拓扑、未知故障类别或多故障。

## 防泄漏与验收

- 划分单位是完整独立run/采集session，不是记录行、连续迭代或重叠窗口；同一run的所有窗口同一分区。健康标定、尺度、特征选择和原型只用训练分区，阈值只用验证分区。
- 开发集覆盖16卡且需要独立run重复才能量化稳定性；盲测另起训练进程/会话，随机rank、注入时间、强度并混入健康；留出至少一种未用于调参的强度。只有每卡一条轨迹时，不能评估跨run泛化。
- 推理入口不提供manifest/injection/fault_rank、目录名、buffer/comm地址、温度/时钟注入标记等答案线索；这些只由评分程序读取。合法拓扑rank坐标可作为输出映射，但禁止硬编码某卡先验故障。
- 提前锁定算法/配置/hash和盲测计划；盲测全量预测落盘后才揭示真值。盲测失败若改算法，原盲测转开发集，必须新采独立盲测，不能循环重放直到成功。
- 单列oracle故障窗口的离线定位和实际检测→PROFILE→定位端到端结果；真值切窗仅能用于前者。必须计健康误报、故障前误触发、错定位、拒判、漏检；报告全16卡混淆矩阵、定位正确数/全部故障run、precision、coverage、每卡表现及开销。未启动/中止样本列出，不能补成成功或失败。

## 原始研究来源与可借鉴路线

1. [GREYHOUND，ATC25 §4.3](https://www.usenix.org/system/files/atc25-wu-tianyuan.pdf)：先按等通信量分组排查，再对候选组做独立GEMM和链路验证。可借鉴“通信只是筛查、独立计算证据确认”，但该文验证会暂停训练；本路线优先离线计算段证据，不能直接宣称无暂停复现其验证。
2. [CCL-D，PPoPP26 §4–5](https://arxiv.org/html/2605.04478v1)：跨主机/内核状态、communicator+operation计数Trace ID、区分晚进入与慢传输；其最小时延判计算慢也依赖异常类型判断。可借鉴精确通信实例对齐及多维证据；现AllReduce Event不能声称等价CCL-D，完整移植成本较高。
3. [SnailTrail，NSDI18](https://www.usenix.org/conference/nsdi18/presentation/hoffmann)：从执行trace依赖图计算critical participation。可借鉴PP等待向源头追溯，作为指纹不可分时的下一阶段；只有聚合平均时延时不能恢复因果依赖。

以上是适配建议与可检验假设，未验证在本项目16卡拓扑上的有效性。

## 独立论文对照实现

`paper_localize.py` 是 **paper-consistent disclosed baseline**，不接入`analysis.py`或在线控制器。按原样保留的真实tag（应含PP stage）及kind/count/dtype比较，执行clique最大平均值筛选/CV0.1、rank最小平均值/CV0.1及`(mean-min)/mean>=.2`门槛，累加无量纲rank score；不用健康参考，不强制双标签，不对PP链/其他尺寸冲突全局否决。

论文没有完整规定多尺寸融合；本实现明确取舍为同tag/name/成员实际group只保留最大归一化score一次，等最大值指向不同rank则仅该group拒判。没有证据时“不确定”不等于证明健康。缺kind/count/dtype报错，不擅自把未知操作假定AllReduce；缺失PP stage也不会被推断补齐。以上均随结果输出歧义注释，不能称不明部分已严格复现。

CLI复用`offline.py`的episode目录与session分区检查，只在全部预测生成后读取真值评分；不拟合参数。示例：`python3 -m flash_v1.paper_localize --dataset results/data16 --split validation --output results/paper-validation.json`。报告故障episode定位准确率、全部episode拒判率、健康FP及分母；健康无报警单列为不确定，不能混作准确定位。新增纯逻辑测试仅防方向、可比集、融合和评分错误，不代表GPU有效性。
