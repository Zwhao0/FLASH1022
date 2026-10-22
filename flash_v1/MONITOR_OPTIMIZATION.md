# 常驻 MONITOR 优化（2026-10-22）

目标：PROFILE 平均开销约5%以内；常驻 MONITOR 平均开销小于1%，保留逐步检测和原定位精度。范围为当前2节点16卡 TP2×PP2×EP2×Expert-DP2、真实完整Qwen派生MoE、seq16/2微批、随机token系统实验；不是其他模型/负载的开销保证。

## 实现

- `native/probe.cpp`：关闭PROFILE时直接转发NCCL，缓存函数指针；不创建记录、CUDA计时事件或同步，不写ring。仅PROFILE绑定phase/op上下文，MONITOR关闭逐rank诊断日志。
- `monitor_channel.py`：训练线程使用512条定长mmap队列和短flock；CPU控制器逐条处理并批量转Redis。满队列/序列缺口报错，不丢步、不降检测频率。
- `train_pipeline.py`：沿已有WORLD loss collective携带控制epoch，无新增collective。预分配2float GPU/pinned CPU缓冲；非root异步D2H在下步已有同步后读取；root一次D2H读取loss/epoch，CPU除法替代GPU loss除法。
- 已确认当前Torch的CUDA `value[1]=Python整数`引入每步同步H2D；改为零值沿用`zero_()`，非零命令使用GPU `fill_()`。
- MONITOR空ring低频休眠；PROFILE仍完整采8步。收尾增加controller完成和双节点drain ACK屏障，再关本实验Redis，修复节点结束竞态。

BEAST窗口30、概率0.8、退化20%、逐步输入、8完整迭代cohort及冻结定位模型/阈值均未改变。`wait_localize.py`、`offline.py`、`wait_frozen.json`的hash与`results/data16_models/freeze.json`一致。

## 真正无探针对照

预注册`results/monitor_optimization/protocol_stage3.json`：A1/B1/B2/A2，各100步真实SGD、同seed/配置；A不加载probe、不启动controller/drain/Redis。两方保留相同注入框架、普通训练日志与墙钟测量；全部13–100步纳入，排除12步预热，不包含启动/结束时间。

| 运行 | 模式 | 平均完整迭代/ms |
|---|---|---:|
| monopt3_a1_baseline | A无FLASH | 856.219 |
| monopt3_b1_online | B MONITOR | 855.639 |
| monopt3_b2_online | B MONITOR | 814.994 |
| monopt3_a2_baseline | A无FLASH | 834.243 |

合并B/A耗时差 **−1.173%**；相邻配对 **−0.068% / −2.307%**。59项身份/序列/收尾等检查通过，loss最大差0，两个B共200步健康零触发、队列全部消费、零丢失、采样关闭。

本次平均开销点估计小于1%；负差不解释为加速。共享环境存在时间漂移，8步块差范围约−2.61%到+1.36%，同位置聚合比p95约+4.25%；不能保证每个短窗口都小于1%，也不提供长期统计保证。完整结果`results/monitor_optimization/report_stage3.json`。

同次训练交替对照亦通过：增量平均差 **−1.096%**，四相邻配对为−1.856%/−0.280%/−1.555%/−0.678%，loss最大差0，268条测量全部消费、零触发/PROFILE；128个B步骤均保留。负差同样不解释为加速。完整结果另见`results/data16_sessions/monopt_crossover_01/monitor_crossover_report.json`。预注册12步预热+8×32步 A/B/B/A/B/A/A/B、lr0和固定32位置token；A共享native快路径和测量通道，仅衡量生产metric/control和BEAST的增量，不能代替上述无探针总成本。

## 精度和PROFILE回归

- `monopt2_b1_online`真实SGD：同一组预定rank6/10故障回归 **2/2正确**，分别在故障开始后2/3步检测；158健康步零触发。两次8步采样均覆盖16rank，1792原始记录/窗、零丢失，训练结束前输出定位；不是新一轮16卡逐GPU盲测。
- 冻结来源不变，先前离线独立盲测16故障/4健康结果保留在`DATA16_STUDY.md`。
- CPU探针mock与优化前源比较：8步PROFILE的576记录/事件顺序完全一致；MONITOR百万调用零新增分配/探针计时API。此微测试不代表端到端GPU开销。控制/队列/收尾9项CPU测试通过。
- 相同seed/故障的无FLASH `monopt2_c_profile_control`：实际PROFILE105–112相对差 **−0.621%**，135步loss完全一致；未观察到5%目标退化，但短窗口点估计仍受漂移影响。见`results/monitor_optimization/profile_control_stage2.json`。

## 保留失败与复现

首轮完整ABBA点估计+9.835%，未达标，见`report_stage1.json`。第二轮点估计−7.755%，但B2完整训练100步后遭收尾Redis竞态，不能标为验证通过；原失败、日志和`stage2_failure.json`均保留。第三轮是全新四组对照，未替换第二轮失败运行。

```bash
python3 -m unittest flash_v1.tests.test_online flash_v1.tests.test_monitor_channel flash_v1.tests.test_node_finish
python3 flash_v1/tests/probe_cpu_mock.py --baseline results/monitor_optimization/probe_before.cpp
python3 -m flash_v1.scripts.monitor16_healthy_report
python3 -m flash_v1.scripts.monitor16_crossover --session monopt_crossover_01
```

命令在工作站项目根目录执行。上述评分命令只读现有实验；crossover仅显式`--run --session 全新ID`启动GPU作业。当前生产在线入口仍为`python3 -m flash_v1.scripts.online16 --session 全新ID`。

完成后两节点实验进程/Redis17422/本实验共享内存均已清理，GPU时钟维持1980MHz，未改其他任务；证据`results/monitor_optimization/cleanup.json`。
