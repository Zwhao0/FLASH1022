# 16 GPU 自动闭环接入与快速验证

2026-10-22：新定位已接入实际16卡训练中的自动流程，独立随机快验通过。此前16/16是离线冻结盲测，本轮新增端到端快验为两个随机故障，不混用分母。

## 流程与PROFILE变化

1. MONITOR输入改为完整迭代墙钟，包含已有WORLD loss同步；不再仅用rank0同步前的局部训练耗时。独立CPU进程保留30步BEAST、概率0.8、退化严格大于20%，排除前12步启动及PROFILE/汇总期间样本。
2. 检测后，由rank0在已有指标AllReduce中附带下步采样epoch，16rank统一边界。PROFILE固定完整8步，沿用attention反向stride8轮换及每组首调用；保留iteration、layer/phase、communicator调用序号。替代旧流程至少20秒的时长规则。
3. 两节点复用生产者mutex持续drain，先读WORLD已完成iteration，再复制/发布新增记录，最后发送iteration ACK；不比较主机时钟。冻结`wait_localize.py`在线读取完整窗口并输出结果，训练继续运行。
4. PROFILE到8步即关闭；数据汇总最多等10秒，缺覆盖/丢记录拒判，不延长采样。汇总后丢弃期间积压监测数据并重建基线，避免采样扰动重复触发。每次运行默认最多3个窗口，可通过plan的max_profiles配置诊断预算；达到预算后停止新诊断。

推断只接收configuration、实际topology和profile rows，不读取注入计划/真值。健康校准阈值与推断源码保持原冻结哈希。现入口针对固定2节点、TP2×PP2×EP2×Expert-DP2训练适配，不是任意分布式训练作业的通用接入。

## 真实随机快速验证

会话`results/data16_sessions/online16_quick_01`：195步，独立随机seed，真实完整24层社区Qwen派生MoE训练；GPU stream延迟模拟，未改时钟。每步2微批，下表延迟按每反向计。

| 真故障 | 延迟ms | 故障迭代 | 自动检测 | 自动PROFILE | 在线结果 | 检测至结果 |
|---|---:|---|---:|---|---|---:|
| rank2 / PP0 | 479.770 | 56–90 | 58 | 60–67 | rank2正确 | 14.21s |
| rank15 / PP1 | 576.952 | 136–170 | 139 | 141–148 | rank15正确 | 16.50s |

- 两次故障均自动触发并定位正确，检测分别晚2/3个迭代；均在训练结束前完成，无强制PROFILE或事后离线替代在线输出。
- 113个预热后的健康迭代零检测触发；恢复阶段未反复采样。仅一次会话，不能估计长期误报率。
- 两窗各8步、16rank覆盖完整、1664条语义匹配记录（原始计时1792条），ring零丢失；实际注入hook/cycles与计划核验通过，结束时采样开关为0。
- 汇总：`online_report.json`；原始在线事件`host2/events.jsonl`，定位输入`host2/online_profile_1/`和`online_profile_2/`。本轮没有重跑全部16卡端到端矩阵，也未验证真实降频、弱于检测门槛或多故障。

## 短开销对照

`online16_control_01`保持相同代码/配置/模型/seed及前90步故障计划，关闭在线控制与PROFILE，保留原MONITOR探针。前90步loss一致（误差<1e-4）。按包含跨卡同步的wall_iteration_ms配对：

| 区间 | 匹配步数 | 总墙钟增量 |
|---|---:|---:|
| 去预热后的完整闭环13–90 | 78 | +2.75% |
| 健康监测13–55 | 43 | +3.72% |
| 故障中PROFILE 60–67 | 8 | +1.73% |

本轮均值低于约5%目标；整体单步配对p95约16.50%，健康阶段约25.02%，不能保证每步或长期低于5%。串行短运行、共享GPU背景负载及注入校准存在噪声；PROFILE区间还含故障，且墙钟会包含前一步日志/Redis尾部开销，不把+1.73%当纯采样因果开销。此对照不是无探针训练基线，也不与此前健康采样增量1.85%混用。详见`overhead_quick.json`。

## 入口与验证边界

```bash
# 自动注入两个随机故障的快速验证；必须使用唯一session名
python3 -m flash_v1.scripts.online16 --session <唯一名> --timeout 900
# 仅重新评分已经完成的会话
python3 -m flash_v1.scripts.online16 --session online16_quick_01 --score-only
```

`train_pipeline --online`及node_collection plan的online=true启用新流程；非online采集与旧4/8卡控制器保留。注入计划用于本实验驱动与独立评分，不负责触发PROFILE。核心实现为`online.py`、训练中已有指标通信的控制slot、drain的有界Redis发布。

5项CPU控制测试通过，覆盖精确8步、双节点ACK、采样期间排除/基线重建、配额，以及训练尾部不足8步/还没开始采样/最后一行后schedule才到达的处理；尾部拒判修复在GPU快验后加入并经CPU验证，未额外跑GPU。冻结推断未改，匹配对照验证控制slot没有改变loss。异常时只清理本session进程/Redis/shm，不操作其他作业。
