# 最后短轮 V2 L1/L2 在线复核

固定新4例全部严格通过：MONITOR自动检测4/4、public目标4/4、core4/4、first actual accepted4/4。每布局L1/L2各1例，L2全局覆盖FWD/BWD。模型、阈值、固定aux1/3/7、min3与融合均未改变。工作站106项源码/验证资源/模型匹配提交`5cc7cb8347d1efb91ef37a143e1428e82b15086d`；服务器实际执行的生产源码与模型SHA一致，source freeze `47e07b1c`。部分工作站便捷脚本与测试未部署或版本不同，不在本次执行链；清理前后服务器现存源码状态完全一致。

| 布局 | 层/阶段 | GPU rank | 检测步（故障起点延迟） | public/core | FMA24及证据 | peer |
|---|---|---:|---:|---|---|---|
| legacy | L1 whole | 5 | 46（+3） | 正确 | 有效 | 正确 |
| legacy | L2 BWD | 2 | 99（+1） | 正确 | 有效 | 拒判（BWD允许） |
| node_local_pp | L2 FWD | 14 | 48（+3） | 正确 | 有效 | 正确 |
| node_local_pp | L1 whole | 6 | 104（+3） | 正确 | 有效 | 正确 |

全部4窗都有完整512角色记录/128 completion markers、aux48行及有效TP/context绑定，PROFILE完整8步均在故障内。FMA每例24条truth、24有效执行步、物理launch及selected calls均正、无scope miss/drift，4/4有效。BWD peer为有效aux窗口下拒判，沿用既有BWD门；未改为强行peer成功。首个真正接受的public verdict全部正确，没有以后续窗口覆盖错误。

健康有效非warm/nonfault监测步legacy70、PP74，共144；检测候选0、mixed PROFILE0。纯健康PROFILE为0窗，因此不宣称健康PROFILE定位率0/N。这里是本短轮的连续监测观察，不是全负载误报率。

训练130/134步，故障每例24步；健康warm/故障间恢复至少30步，末尾仅8–10步。这是用户授权的最后短样本，不替代原32步campaign、原N6或总体准确率。一次随机冻结rank/阶段/时序与基于既有健康校准的目标extra .5–.7；未依据旧pass/fail挑例或加大剂量。host2 rank0的非PROFILE wall相对固定邻近健康median分别+58.27%、+35.74%、+47.88%、+118.54%；观察wall不是纯注入因果效应，也不是PROFILE开销。成本仍只引用既有paired预算。

执行使用现成detached launcher的短RPC/exec-d，未用长attached exec。legacy130完整后public SHA保持`a8606d0893a09ba3fe0051d70d24afe151abd7ebe5c3a7331135ebdef45e7be3`。原PP只到startup且两host无train/events，已标`startup_failed_no_fault_measurements`，原SID/session/manifest及父execution_failure保留；唯一授权恢复仅将PP SID改为`_startup_recovery1`，seed、case IDs、rank、dose、onset、134步、配置和源码全不变。原startup失败不混入4case分母，legacy未重跑。新PP public SHA `8f9ad1b5f304bfa760c44ba85fbb41916e6e1ae886e5af1fb464c00b3d49ad16`；两public全封存后才读取truth评分。

入口：`score_brief.json`为精简结果，`score_summary.json`保留全audit；`scoring_rule_before_truth.json`为GT前因果/24步门规则，`pp_startup_recovery_amendment.json`与`execution_complete.json`记录恢复边界。原N6两次partial及64case core-only结果均保留且不与本轮合成准确率。
