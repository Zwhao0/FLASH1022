# 通用计算慢v2采集小阶段

当前：旧健康PROFILE ABBA legacy4.148%、localPP21.742%未达5%保留。semantic-only旧raw标not_applicable，新域数值/四健康完整；对数候选896c新开发6/10（4拒判）保留。健康事实否定PP分组，原时长MAD候选dc89已锁（6σ/2margin/≥3不变），正采新独立dev02两布局及健康控制，见`results/v2_semantic/active.json`。未正式冻结/新盲测。GPU执行唯一负责人为采集agent；其余agent只做CPU。

- `results/v2_collection/preflight_01.json`：双节点64源码、模型、3原生库一致；未改GPU时钟、未触碰外部vLLM。
- `results/v2_collection/numeric_01.json`：原路径/候选同seed4步，16rank全部梯度SHA与4步loss精确一致、全部有限。会话`v2_numeric_legacy_{baseline,candidate}_01`均完整保留。
- 首轮候选256条span中192有效；attention BWD 64条缺内部通信。已修复kwargs导致的backward scope提前结束；`candidate_02`四步复验16rank全梯度SHA/loss精确一致，256/256有效、64完成marker齐。旧bug结果保留，见`numeric_02.json`。
- 两布局五mode各112步、7168/7168有效，开发10例source正确。legacy独立Hfit/Hcal健康拟合候选ef624；新布局另12健康窗均拒判、全rank可比，不回填/挑优。四健康会话各60步。
- `v2_online_node_local_pp_01`完整145步：FWD/expert 2/2检出正确、0/83健康误触发；2窗各128完成marker/512有效测量，均训练结束前定位。
- 接入仅plan `evidence_enabled=true`启用；whole外、evidence内，既有CUDA同步后drain。`numerical_audit_steps`仅显式测试，全梯度拷贝/哈希耗时不用于开销结论。
- 开销：纯健康PROFILE与含whole工具的matched5fault分开；ABBA同模型/seed/SGD/输入，固定每rank注入单位及whole完整FMA参数，并核对实际cycles/rounds。idle Whole派发工具税单列，不能降低监控成本口径；MONITOR另测无FLASH总成本。离线强制PROFILE不含在线CPU部分，需另核完整在线匹配窗。
- 后续：公平开销→完整源码/模型/协议冻结→新随机32盲测。检测/BEAST/8step/MONITOR结构不变，truth/mode/剂量不传localizer。source/model/native跨节点一致；只STOP owned会话，不触碰外部vLLM/时钟，不重跑覆盖旧ID。
