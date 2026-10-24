# 通用计算慢：最小增量证据

**审计结论。** 现63窗稳定组配对完整，但每窗TP注意力BWD1152条、FWD/MLP-BWD各64条，后两者仅偶然首调用覆盖；额外128条是PP首AllReduce，已复用。简单TP/DP/PP融合未解决专家传播归因，不代表任何算法都不可解。原结果及模型保留；用过的mixed32/whole32均属开发集。

**最小实现。** v2已选择显式semantic-only后端（独立重采验证中）：不加载旧native，不开启旧phase mmap标签；v1保持原路径。`evidence.py`显式opt-in，仅PROFILE临时给一个注意力和一个owned expert叶模块装前后向hook，每步4跨度，固定同slot重复8步；模块前hook用prepend包住已有注入hook，不移动故障。注意力hidden_states从关键字传回同一首位置参数，使PyTorch完整BWD边界有效。每跨度完整记录内部`_all_reduce_with_phase`的同stream起止Event，结束时减通信区间union；通常每步11对Event。采叶专家以避开EP dispatch/return，不把整个含EP的MLP当本地计算。无TorchDispatch、无新同步；临时包装与hook在PROFILE结束恢复，MONITOR零hook/Event/patch。不同selection_index可轮换层/专家，但不能按truth选择。

**最小字段。** 测量行为`rank,iteration,epoch,layer,phase,slot,module_name,shape,dtype,stream_id,module_elapsed_us,communication_intervals,cpu_enqueue_us,duration_us,coverage_complete`；duration是模块跨度减完整通信union的**本地stream残余**，含主机入队空洞，不是纯kernel时间。通信次数必须符合当前TP实现：注意力FWD1/BWD3，专家FWD1/BWD2；流变、区间不完整、Event未ready均拒用。每PROFILE drain写`iteration_complete/rank/iteration/epoch/rows_count/expected_rows=4/event_pairs`，即使缺测也有完成标记。只在已有迭代CUDA完成边界后query并读Event，不额外等待。

**可解决与盲区。** 源模块前排队的真实GPU工作留在残余，传播造成的内部TP等待被扣除，可补旧FWD/专家阶段盲区。不能直接减旧native稀疏duration；跨host绝对timestamp不得排序。只采一个层/专家，未覆盖的单层/单专家故障可能拒判；重叠流、模块外EP/PP等待、CPU调度抖动仍需诊断，不能凭单值强归因。若此证据不足，先分析失败，再补语义首同步点或稀疏EP证据，不开全量profiler。

**验证与公平开销。** CPU检查区间union、缺失拒判、hook顺序、MONITOR无事件；真GPU先原/候选同seed比loss和全部梯度，再最小健康及旧FWD/专家注入窗，先确认4阶段完整可观测。开销同布局/输入/批次/时钟配对交错ABBA：A真实训练、B MONITOR、C原PROFILE、D候选PROFILE；报B/A≤1%、D/B≤5%目标及D/C增量、置信区间、最慢rank wall和完整8步wall（含drain）。先无注入测监测成本；模拟注入hook/校准/审计两侧同开，另报其自身成本。负点估计不称加速。

**冻结与盲测。** 先独立健康fit/calibration及每类单rank开发故障；固定源码/so/模型/阈值/预算/环境后，新seed随机rank/起点/强度、两布局与各模式重采。推断仅见观测，truth在结果落盘后评分，失败不替换；分别报检出、定位、拒判、健康误触发及开销，不以whole成功代替阶段泛化。
