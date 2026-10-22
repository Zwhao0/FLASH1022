# 16 GPU 数据驱动定位研究

范围：2节点×8 H800，TP2×PP2×EP2×Expert-DP2，每次仅一张GPU注入计算流延迟。优先精度，采样开启期开销目标约5%。这是第三章定位机制的扩展研究；原论文对照、旧在线实现与新离线方法分别保留。

后续进展：已接入自动检测→按需8步PROFILE→在线定位，两个随机故障快验通过，见[ONLINE16.md](ONLINE16.md)。以下保留此前离线研究及其适用边界。

## 已验证的基础

- 全部24层的社区Qwen派生0.5B MoE，23层32专家/top8＋共享专家；随机tokens、真实路由/分发/反向/SGD，不评估语言质量。
- 真实attention、专家/共享MLP、词表TP切分；原Megatron PP/DDP，独立EP组。16rank小配置FP32输出/aux/逐参数梯度/1及2微批更新均通过（`results/data16_sessions/verify16_06_vocab`）。完整模型首轮OOM保留，词表切分后正常运行。
- 两节点统一NCCL Socket，CUMEM/NVLS关闭，channels4、buffer1MB；不代表论文原H200/RDMA环境。其他任务和GPU时钟未更改。

## 有上限的采样

- 窗口8迭代；attention反向按communicator调用序号stride8轮换，加各组首调用。保留实际组/消息量、rank、iteration、layer/phase、op index、group id和节点内时间戳。
- phase在已有TP collective入口写共享内存，不新增GPU Event、同步或模块hook。14字段GREYHOUND ring持续drain，压缩原始记录，核查覆盖与事件丢失。最终读取器复用生产者mutex，仅复制新增记录，不调用GPU API。
- pilot每窗口1664条稳定记录，192个attention特征完整；首版约9千条。正式4个独立会话共80窗口，采样量固定，不持续无限采集。

## 从数据发现的问题

同层Q/K/V反向有3个TP归约；首调用吸收迟到等待，后两次已对齐。pilot中两个故障均在全部12层×2微批首调用出现约16–25ms对端等待，后两次约10–12us。直接取六调用中位数会丢掉2/6真正信号。已有op index允许离线恢复首调用，无需加采样；PP对端未呈现同等强一致模式，但尚不能据此宣称泛化成功。

首调用选择仅适用于当前非融合QKV、2微批、stride8配置；每group/phase需完整两个连续三调用块，双rank覆盖不足则拒判。轮换样本来自不同iteration，不能假装同一迭代完整trace。

## 固定协议与入口

- 正式train两个独立session（每卡400/650ms每反向、不同token seed）；validation第三session每反向500ms。每session16故障段＋4健康段，每段2settle＋8采样。
- `offline.py`：健康MAD标定＋故障通信模板＋拒判；比较原median、分阶段median和首调用特征，仅用validation选择。`paper_localize.py`保留披露歧义的论文两阶段对照。
- 模型/阈值/特征配置及源数据hash冻结后，才采独立seed和随机强度的blind_test；不能把同session episode拆跨split，不能看盲测后调参仍称盲测。
- 同时报告正确/错卡/拒判/健康误报与分母；定位窗口实验不等于MONITOR→检测→按需PROFILE的完整在线闭环。真实GPU stream延迟模拟也不等于真实硬件降频。

入口：`python3 -m flash_v1.scripts.data16 --session <唯一名> --role train|validation|blind_test --seed <数> --delay-ms <数>`。原始会话在`results/data16_sessions`，打包episode在`results/data16`，pilot独立存储不参与正式fit。

## 开发验证结论（非盲测）

| 方法 | 故障正确/16 | 错卡 | 拒判 | 故障判健康 | 健康强制定位误指/4 |
|---|---:|---:|---:|---:|---:|
| 披露适配取舍的论文规则 | 10 | 6 | 0 | — | 4 |
| 全调用median＋模板 | 0 | 0 | 3 | 13 | 0 |
| attention median＋模板 | 0 | 0 | 2 | 14 | 0 |
| attention首调用＋模板 | 3 | 0 | 7 | 6 | 0 |
| **局部首调用等待差** | **16** | **0** | **0** | **0** | **0** |

健康强制定位误指不是完整系统的健康误报率：本轮人为采集PROFILE，没有通过原检测器触发。原始失败报告全部保留在`results/data16_models`。

选定`wait_localize.py`：同组、同迭代/层/首调用配对，计算`peer_duration-own_duration`；每层跨2微批取median，每卡跨12层取median。保留有符号微秒值，不将其他GPU的全局噪声塞入模板距离。只用3个独立会话中的12个健康窗口校准H=1412.25us，健康上限1.25H，故障上限2H=2824.5us，最高/次高至少2；中间区间或缺证据拒判。

算法、阈值、依赖源码哈希在`results/data16_models/freeze.json`冻结；验证20/20。随后才生成blind16_s4：16卡随机顺序、每反向100–650ms随机强度、独立token seed、4健康段。实际抽取107.818–641.340ms；每步2微批，注入总量为两倍。时间/强度是预先切窗的条件定位实验，不代表检测器能触发所有这些故障。

## 冻结后的独立随机实测

| 方法 | 故障正确/16 | 错卡 | 拒判 | 健康窗口指向故障卡/4 |
|---|---:|---:|---:|---:|
| 披露适配取舍的论文规则 | 16 | 0 | 0 | 4 |
| **冻结的局部首调用等待差** | **16** | **0** | **0** | **0** |

新方法20/20正确；故障最高分4598.75–27346.25us，最高/次高比至少4.92；健康最高分至多978.5us。仅一个新session、每卡一个故障窗口，不能声称跨模型、拓扑、故障类型或长期准确率100%。原规则验证曾6次错卡，本轮故障均对；健康强制定位误指仍不是完整系统误报率。

成功会话为`blind16_s4_retry2`，两节点零ring丢失、每窗口16rank覆盖完整，实际注入hook/cycles检查通过。此前`blind16_s4`因新ring初始化竞争在训练前失败，`blind16_s4_retry`因全ring乐观快照busy在第22步失败；均保留为基础设施失败，未出定位结果。只修读取器，三次的seed、顺序、强度完全一致；冻结推断及模型哈希前后核验一致，见`results/data16_models/blind_audit.json`。

结果：`results/data16_models/wait_blind.json`（逐窗口分数与混淆矩阵）、`paper_blind.json`、`freeze.json`；源代码归档、原始压缩记录与失败日志在`results/data16_sessions`。可复评：

```bash
python3 -m flash_v1.wait_localize evaluate --dataset results/data16 --model results/data16_models/wait_frozen.json --split blind_test --output results/data16_models/wait_blind.json
```

## 开销与边界

最终读取器对照：`blind16_s4_retry2`的4个健康窗口，与`monitor16_overhead_04_snapshot`相同代码、模型、二进制、配置、token seed和迭代位置配对，共32步：

- **完整迭代墙钟增量+1.85%**，4窗口分别+1.44%～+2.30%，单步配对p95为+3.23%。
- 每步16rank中最大训练耗时的总和增量+1.91%。rank0单独计时反而-2.63%，不能把负值解释为加速；因此采用包含后续跨rank同步的墙钟与最慢rank口径。
- 结果与输入hash在`results/data16_sessions/snapshot_overhead.json`；本配置本轮采样增量低于约5%目标，但不是FLASH相对无探针训练的总体开销或长期保证。两次串行运行及共享GPU的背景干扰限制了结论。

历史对照保留：最初全阶段稀疏采样约+10.8%；缩减阶段后，旧读取器的rank0训练计时32步约+0.40%、单步p95约19%（`final_overhead.json`）。口径不同，不与最终墙钟数字混用。

目前不能据此认定论文算法有错：此前真实TP未就绪、跨stage可比性和调用聚合都会影响定位；原规则的多尺寸融合也未充分规定。新方法保留FLASH的“慢计算晚到、通信等待较短”思路，显式匹配通信实例并加入健康边界，属于扩展研究。本节离线实验不证明端到端准确率；后续闭环快验见ONLINE16.md。未验证真实降频或多故障；若扩展后出现PP根因混淆，再讨论少量本地计算段/PP等待观测。

验证依据：16rank FP32完整输出/梯度/SGD对照通过；29项定位/拓扑CPU检查通过，新增CPU快照并发测试通过，正式盲测零丢失。本轮实验进程、私有Redis与共享内存均已清理，GPU时钟仍1980MHz，其他任务未改动（`cleanup_check.json`）。
