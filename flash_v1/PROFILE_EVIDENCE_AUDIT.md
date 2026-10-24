# PROFILE证据审计（开发数据，2026-10-23）

**结论：部分足够，先离线复用现有PP/DP证据；尚不能证明通用计算慢可定位。** 只读两布局mixed32与whole32共63窗，未使用故障rank生成特征、未拟合模型、未运行GPU。原32例现均按开发数据处理；新盲测须另采。

- 四会话raw共112,896行；每窗1,792行。稳定profile每窗1,664行：TP attention-BWD 1,152，TP FWD/MLP-BWD各64，TP/DenseDP/ExpertDP phase0各128。全部稳定配对完整、零重复。FWD/MLP-BWD仅覆盖单侧PP阶段的首调用，并非各层×微批全覆盖。
- **raw另有每窗128行PP组AllReduce**，8个PP clique、phase0、配对完整；被`localization=false`从profile排除。此项可直接离线加入，不增加采集成本。DP/ExpertDP已有明显跨rank等待差，但等待传播使幅度本身不能证明源GPU。
- `native/probe.cpp`仅拦AllReduce，phase2按stride8采样，另保护每comm的operation_index0；计数为thread_local，不能视为全comm唯一首调用。`drain_collection.py`写raw前丢弃duration<=0，所以未选FWD/专家调用的时长及时间戳已不可恢复。没有EP Send/Recv、PP P2P、计算阶段时长、线程ID；跨host的steady_clock时间戳不可直接比较。
- 冻结wait定位器只用TP phase2，未利用PP/DP互补。现有`offline.py`可用稳定DP，PP需从raw按实际组成员配对。legacy已有独立健康train两会话各4例、validation4例；导出目录未发现新布局独立健康校准集。

下一步：先离线融合TP/PP/DenseDP/ExpertDP，用独立健康会话校准拒判并保持检测阈值。若仍不足，再在FWD与MLP-BWD各层/微批选确定的首同步点，语义配对；粗略上限新增48次定时/iter/rank（有重叠），禁止直接全采。CUDA event同步成本须健康AB实测，不能由样本数宣称PROFILE≤5%；MONITOR无需新增证据。再冻结方案、另采随机故障盲测。

机器证据：[summary.json](../results/profile_evidence_audit/summary.json)。范围及限制同上。
