# PP节点内、DP跨节点快验（2026-10-22）

已在真实2节点16卡上通过4个计算慢案例。完整Qwen派生MoE、seq16、2微批、真实SGD；原冻结定位源码/模型/阈值均未改。本次验证布局迁移，不重新测量开销，也不代表所有故障/配置覆盖。

新选项`parallel_layout=node_local_pp`：物理rank不变（host2为0–7，host1为8–15），每节点TP2×PP2×EP2，节点间Expert-DP2；普通参数dense-DP4含节点内EP副本和跨节点副本。

| 分组 | 实际成员示例 | 位置 |
|---|---|---|
| TP | [0,1]、[4,5]、[8,9] | 节点内 |
| PP | [0,4]、[3,7]、[8,12] | 节点内 |
| EP | [0,2]、[4,6]、[8,10] | 节点内 |
| Expert-DP | [0,8]、[4,12]、[7,15] | 跨节点 |
| dense-DP | [0,2,8,10]、[4,6,12,14] | 跨节点 |

`rank_layout.py`仅在Megatron初始化时用模块局部torch代理重排成员，恢复真实torch并映射rank缓存；原Greyhound源码未改。CPU直接运行vendor分组逻辑4项检查通过。实际16卡FP32四层参考模型的logits/aux、全部梯度、绑定embedding及1/2微批SGD验证全部通过。

预注册`results/layout_localpp/protocol.json`，在线session `layout_localpp_01`共275步。4张代表卡依次单独注入，各25步：GPU计算流中`torch.cuda._sleep`分散到反向pre-hook，225ms分给注意力、225ms分给本卡持有的路由专家，即每微批反向累计450ms；2微批使每迭代名义累计900ms。没有延迟通信函数或改时钟，不对应某个固定降频比例。

450ms单次CUDA Event校准实测445.60–446.59ms；各故障25步均值（含8步PROFILE）相对紧前健康段，迭代由737.59–767.41ms升至1306.72–1497.64ms，增加74.10%–102.82%。这是较强故障，不能据此声称轻微慢故障精度；通信/计算重叠使端到端增量不必等于注入时长。完整校准与逐案例强度见`results/layout_localpp/injection_summary.json`。

| 目标rank | 物理GPU | PP阶段 | 在线预测 | 检测延迟 |
|---:|---|---:|---|---:|
| 0 | host2 GPU0 | 0 | rank:0 | 2步 |
| 9 | host1 GPU1 | 0 | rank:9 | 2步 |
| 14 | host1 GPU6 | 1 | rank:14 | 2步 |
| 7 | host2 GPU7 | 1 | rank:7 | 2步 |

4/4正确；163健康迭代零触发。PROFILE窗口50–57、110–117、170–177、230–237均完整8步覆盖16rank，各1792原始记录，零丢失、训练结束前完成定位；最小第一/第二候选分数比15.62（冻结门槛2）。完整证据`results/layout_localpp/report.json`及`topology_numeric_check.json`。

工作站入口（新session会启动GPU，默认仍为原legacy布局）：

```bash
python3 -m flash_v1.scripts.online16 --session 全新ID --parallel-layout node_local_pp
# 只评分本次已有数据
python3 -m flash_v1.scripts.online16 --session layout_localpp_01 --score-only
```

默认CLI为两个随机单GPU案例；本次4个代表性案例的完整plan另行预注册保存。两节点实验进程、Redis17422和本session共享内存已清理，16卡时钟保持1980MHz；未改其他任务。见`results/layout_localpp/cleanup.json`。
