# FLASH v1 论文规格（从零提取）

来源：根目录《中文论文.pdf》第三章；以下页码为印刷页码，PDF 页码=印刷页码+16。
范围：计算 Fail-Slow、GPU 降频、小型真实 MoE；跳过 Fail-Stop、网络故障、第四章 SHIFT 恢复。
已渲染核对 PDF 第46页（印刷30页）的图3-4与两阶段算法。

## 1. 必须保留的机制（23-35页）

- 数据面：`LD_PRELOAD` 拦截 NCCL；简化 communicator 记录只需地址、unique-id 哈希、global rank。
- 同 unique-id 哈希的各 rank communicator 组成一个 clique；不能用进程内地址跨 rank 分组。
- 初始化上报 parallel states（组成员及 DP/TP/PP/EP/Expert_DP 等语义），按成员识别稳定通信组标签。
- MoE 动态 EP AllToAll 负载不稳定；主要使用 attention/dense DP、TP、Expert_DP/TP 等稳定通信证据。
- 控制平面：节点内共享内存；节点间 Redis；每节点本地控制器、一个全局控制器。
- 状态 0=MONITOR，只记调用时间戳/元数据，不启用 CUDA Event。
- 状态 1=PROFILE，启用 GPU CUDA Event；采样完成、定位后恢复 MONITOR。
- 框架仅需迭代时间与组成员上报回调，不重构训练计算。
- 第三章没有 checkpoint、offload、故障恢复或缓解；这些属于第四章 SHIFT，不是 FLASH v1 必需项。

## 2. 迭代时间监控（25-27、33、44页）

- Rank 0 完整训练步墙钟时间覆盖前向、反向、梯度同步、优化器更新。
- Redis list `iter_time_{rank}`，元素 `iteration:time_ms`，历史最多500条。
- 默认最近30个迭代；跳过最初3个预热迭代；在 BOCD 前移除 z-score >3 的异常点。
- 论文称 BOCD 使用 Rbeast，计算变点后验 `P(r_t=0 | x_1:t)`，阈值0.8。
- 退化阈值0.2，即变点后的平均迭代耗时比基线增加超过20%。
- 变点不得落在窗口最右侧；右保护 k=1或2，候选 cp ∈ [1,W-k]。
- cp 左序列为 xl；右序列 xr 包含 cp；xr' 是 xr 去掉首点。
- b=mean(xl)，d=mean(xr)，q=mean(xr')。
- `recovery_ratio = (d-q)/(d-b)`；>=0.75 则视为瞬时抖动并过滤。
- 相邻变点间距<2迭代只保留第一个；须保留原始迭代编号，不能让去异常点改变物理距离。
- 右侧点不足、空分段、分母为0时不能凭空报告退化，继续收样。
- PROFILE 触发后本地控制器轮询 Redis 100-500ms，写本地原子状态；Probe 在 Group 结束检查。
- `profile_period_seconds=max(latest_iter_seconds*2.2,20.0)`。
- 不要把 milliseconds 直接代入秒制下限20。

## 3. CUDA Event 与统计（28-29、32-34页）

- 同 NCCL stream 在实际调用前 record(start)，实际提交后 record(stop)。
- Group 包含多个通信时，仅首调用记录 start，最外层 GroupEnd 后记录 stop。
- 论文用 `cudaEventSynchronize(stop)` 再 `cudaEventElapsedTime`，包含 GPU kernel 内 waitPeer 等等待。
- 同 Group 的各操作可共享该组耗时，必须保留 Group ID；注意这并非每个操作独立耗时。
- 本地控制器按 communicator 聚合 duration 的 min/max/avg/std。
- 监控模式不能每次操作 CUDA 同步；否则破坏低开销目标。
- 晚到同步点的计算慢 rank 通常测得更短通信耗时，快 rank 的 Event 区间包含等慢 rank 的等待。
- 因此最小时延是“计算慢”的线索，绝不能以“时延短”排除计算型故障。
- 对比应限定同标签、同类型/同消息大小；动态消息量混合会造成伪短时延（46-47页承认此限制）。

## 4. 两阶段定位（30-31、34页）

1. 过滤单 rank clique；每个标签独立比较。
2. 每 clique 计算 `max_avg_latency=max(comm.avg_latency)`。
3. 某标签只有一个 clique：跳过组间筛选，直接进入 rank 阶段。
4. 多个 clique：计算上述指标的 `CV=std/mean`；CV<0.1 则该标签没有可靠组间差异，跳过。
5. 否则选 `max_avg_latency` 最大的 clique。
6. 候选 clique 内按 rank 聚合其 communicator 记录，得到各 rank 的 avg_latency。
7. 各 rank avg_latency 的 CV<0.1 则无可靠差异，跳过。
8. 否则取 avg_latency 最小 rank；设计章节另述默认20%离群阈值（可明确定义为 `(mean-min)/mean>=.2`）。
9. 汇总不同标签的候选与证据；多个标签指向相同 rank 提高置信度。
10. 报告 rank、标签、clique成员、各rank时延、CV、分数；无证据时报告 inconclusive。

实验展示分数（41-42页）：
- Clique score=`(t_clique-mean_same_type_cliques)/mean_same_type_cliques`。
- Rank score=`(mean_clique-t_rank)/mean_clique`。
- 多并行标签累积分数为参与各组得分之和（44-45页）。
- 注意展示用 clique 平均时延得分，与筛选用 max_avg_latency 是不同统计，不能混淆。

## 5. 论文实验协议与快速 v1（36-45页）

- 原环境：2节点×8 H200，NCCL2.26.5，CUDA12.9，PyTorch2.8a，Megatron-core0.12.1。
- 原模型 Qwen13.3B/A1.4B：36层、hidden1024、16头、128专家、top8、1共享专家。
- 原并行 EP8/TP1/PP1/CP1/Expert_DP2；GBS4096、MBS8、seq4096、muon、lr1e-3。
- 降频档：正常1965→1600/1200/900/600/345 MHz；不同硬件必须先查询支持频率，不能照抄。
- 第一组实验文字称限制一个“节点”的GPU频率；混合并行实验明确限制一张GPU至600MHz；v1单rank注入应明确记录。
- 原吞吐实验250迭代；AllReduce微基准4-128MB、每档10000次、每批100次、预热500。
- 原准确率比较故障强度5/25/45/65/85%，并与GreyHound比较；v1无需跑满这个矩阵。
- 快速v1最小有效实验：健康基线、监控模式健康、单rank持续降频、恢复频率后训练继续。
- 核实真实MoE路由、专家MLP、EP token dispatch与combine、真实梯度更新；不能仅给dense网络取MoE名字。
- 至少保留dense DP与Expert_DP两个稳定标签，便于交叉定位；EP=2且Expert_DP=2一般需4个rank。
- 实测报告正常/异常迭代时间、检测触发迭代、剖析时长、预测rank、真值、证据与额外开销。
- Precision=TP/(TP+FP)，Recall=TP/(TP+FN)；健康无触发也必须单列，不可伪造样本规模。
- 不能将CPU mock或sleep注入的成功说成真实GPU降频复现；硬件结果和本机验证分开标识。

## 6. 论文歧义及实现取舍必须声明

- 40页称“持续超过20迭代”，与26-27/44页“右保护1-2迭代”、检测延迟1-2迭代不一致；建议以设计章节的右保护实现。
- 28页剖析至少20秒且约2.2迭代，与44页“采样1迭代、总延迟2-3迭代”不完全一致；保留公式并记录实测。
- Rbeast 是论文指定库，但未提供先验/采样参数；不能声称任意简化阈值或标准BOCD与其数值完全一致。
- 31页20%离群阈值与34页“CV过阈值后直接取min”不完全一致；配置暴露并注明选取规则。
- 30页图内示意纯通信120ms，与同页正文慢rank3ms不一致；仅作原理示意，不能当测量预期。
- 论文开销/准确率数字只是原论文报告，不是本次v1复现实测；v1不可继承其100%结论。
