# PP定位修正与验证（2026-10-22）

**当前状态：用户已停止实验，健康PROFILE差分定位入口已停用。** 下文仅保留扩展思路和历史实测，不代表当前启用策略，也不等同论文第三章忠实复现。跨stage标签与附加定位规则仍待核对。

固定8卡PP2×EP2×ExpertDP2，新持续慢实验已自动定位真值rank7；健康对照0自动报警。仍处复现阶段，未扩卡或添加并行配置，原失败完整保留。

## 修改与原理

- 分离真实通信标签和count/dtype，严格同尺寸比较；每个真实组只贡献一份尺寸平均得分，避免bucket重复加权。成员缺失、并列、尺寸冲突保留证据，不强报唯一源rank。
- 可选健康PROFILE校准，跨run按真实成员、family、count、dtype匹配，不按变化的communicator地址/hash，不读取故障真值。
- 对每rank计算 `delta=当前平均时长-健康平均时长`，候选取delta最小rank，新增同伴等待残差为 `max(delta)-min(delta)`。常量传输成本及共同偏移先在同尺寸差分内消除，之后累计的是等待残差，不是跨尺寸原始时长。
- 残差须超过 `max(健康平均时长×0.2, 3×最大差分均值标准误)`；各真实组独立累计，组内有效尺寸均分。保留全部候选、逐rank基线/当前值、样本数及未校准结果。
- `--profile-at`手动健康采样（manual_profile不算报警），`--reference-profile`引用完成的健康运行；校验并行/训练/bucket/NCCL配置，保存参考文件SHA256。新增原始调用时间戳与buffer地址，不改训练或NCCL调度。

这是参考论文第三章迟到rank/同伴等待原理的**校准扩展**，不冒充论文原始绝对时延算法。原退化检测阈值、原生bucketing、模型、注入rank/模块/每反向125ms及2微批均未改变。

## 实际验证

健康85有效步，34步手动采20秒PROFILE，共9,776条，0自动报警。新故障110有效步：rank7于46–105步持续注入，106步撤销，训练至113步（含3步预热）；48步自动报警、变点46，约1.80秒触发，PROFILE采7,280条，最终在线定位rank7，恢复段完成。

| 数据 | 原算法重放 | 校准修正结果 | 性质 |
|---|---:|---:|---|
| 原8卡失败profile，7,144条 | rank3 | rank7 | 开发回放，不计新实测成功 |
| 独立新8卡profile，7,280条 | rank0 | rank7 | 规则冻结后运行；在线与最终代码重放一致 |
| 新健康运行 | 0自动报警 | 0自动报警 | 手动采样不算检测事件 |

原算法来自本轮保存的 `results/validation/final_source.tar.gz`；新记录先恢复旧count标签再输入，未混尺寸，源码hash保存在重放JSON。本轮正确定位不是新数据碰巧也让原算法选对。

新故障累计残差：rank7=145,970µs（dense_DP及Expert_DP）、rank6=133,016µs（Expert_DP）、rank0=336µs。未校准的修正计分本轮也选7，但旧7,144条仍有PP歧义，健康参考使两轮均可定位。

新故障正常/异常非PROFILE/恢复均值416.74 / 560.18 / 395.49ms；共享条件下不是受控开销。单段延迟校准123.99ms；每步250ms仍为2微批配置值，未冒充实际hook执行总时长。

容器40项测试通过：覆盖尺寸隔离、每组一票、成员完整性、格式兼容、rank置换、偏移抵消、并列/冲突、4卡已有样本及两轮8卡回放。4卡双标签正例仍rank3；原attention-only冲突样本现保留rank2/3并返回不确定，不再计作确定成功。训练数学语义未变，未重复昂贵梯度验证。

## 边界与证据

参考负载与通信时序必须可比。同count可能包含不同bucket位置，均值/标准误受采样组成、自相关及共享任务影响；这不是置信度保证。定位仍须由全局退化检测触发：健康PROFILE前后半互相比对会产生35–816µs弱候选，但未触发真实监控报警；健康自对照不代替独立健康对照。

尚未完成逐微批PP等待/bucket就绪因果追踪。本轮只有一个独立新持续故障，证明固定配置可定位rank7，不宣称通用准确率；真实降频、其他故障rank、TP、多节点或更多卡不在本轮验证范围。

原始记录：`results/stage3_pp_v2_healthy_profile/`、`results/stage3_pp_v2_sustained/`；汇总 `results/stage3_pp_v2_summary.json`；测试 `results/validation/pp_v2_unit.log`。旧 `stage3_pp_diagnosis.json` 不变。

```bash
python3 -m flash_v1.scripts.audit_pp_profile results/stage3_pp_v2_sustained --reference-profile results/stage3_pp_v2_healthy_profile/profile_1.json --legacy-archive results/validation/final_source.tar.gz --output results/stage3_pp_v2_new_replay.json
```

两轮重放为 `stage3_pp_v2_old_replay.json`、`stage3_pp_v2_new_replay.json`。服务器进程均结束，8卡时钟1980MHz；未修改其他任务。
