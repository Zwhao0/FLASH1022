# FLASH v1 实测（2026-10-22）

最新：[PP定位修正与验证](PP_OPTIMIZATION.md)。独立新8卡故障在线定位rank7，原算法在同数据误报rank0。以下阶段1–3为修正前原实测，失败记录保留，不覆盖成功口径。

阶段1结论：第三章核心检测→按需剖析→rank定位链路在真实小型Qwen派生MoE上跑通；本节仅一个模拟计算慢正例，不能视为完整复现。阶段2补充见文末。

## 模型与环境

- 公开模型：`bayliner1980/Qwen2.5-0.5B-MoE-A0.3B`；社区Qwen2.5转换，非Qwen官方发布。
- 固定revision `4b89028ef7376d0e92acbbc4a7f560c2374aeec0`；权重SHA256 `6c1673e92522af47907eb8e1825142061a367ec5993e4060554fbce271f683be` 已校验。
- 494,712,832总参数；全部24层，23层MoE、32路由专家/top8及共享专家；每rank保有368,097,280参数，未把模型裁成玩具配置。
- host2物理H800 GPU4/5/6/7，共享其他任务，EP2×Expert-DP2；BF16全参数SGD，lr1e-3，batch1。
- PyTorch2.7.0a0/NGC25.03，CUDA12.8，NCCL2.25.1，Transformers4.53.2，Rbeast0.1.25。
- 随机token做系统性能测试；不评估语言能力。训练首步实测目标embedding行最大更新0.00134277，确有优化器更新。

## 快速实验

所有正常步统计均排除最初3步预热；计算延迟统计另排除剖析区间和跨注入边界的步。

| 实验 | 有效训练步 | 结果 |
|---|---:|---|
| 无probe基线，seq32 | 60 | 平均273.38ms |
| 健康监控，seq32 | 80 | 平均249.32ms，0报警 |
| GPU7真实1980→345MHz，seq32 | 160 | 注入前265.94ms，注入期间266.71ms；未达20%门槛，0报警 |
| GPU7真实降频，seq128 | 160 | 注入前264.01ms，注入期间280.93ms；约6.4%变化，0报警 |
| seq256，原初始化/提前预热各一次 | 0 | 首步NCCL分配显存失败；未注入故障，失败日志保留 |
| rank3 GPU计算stream延迟模拟，seq32 | 160 | 注入前260.02ms，异常非剖析步473.86ms；自动检出且定位rank3 |

真实降频由nvidia-smi采样确认345MHz；共享负载和小模型计算占比使其未形成足够大的全局退化，不把这些运行记为定位成功。
健康两轮的负开销差值来自未受控共享条件和不同运行长度，不能据此宣称监控提升性能或给出可靠开销百分比。

## 成功正例细节

- 故障为经用户授权的GPU计算延迟模拟：仅rank3在反向计算后、梯度同步前执行校准后的`torch.cuda._sleep`；没有CPU sleep，没有延迟NCCL函数。
- 请求250ms，CUDA Event校准实测247.985ms；第46–125步生效，第126步撤销；全程训练完成至第163步（含3步预热）。
- 第49步自动报警，变点为第46步，Rbeast后验1.0；从实际开始模拟到报警约1.85秒（观察到4个故障步）。
- 检测窗口基线260.54ms→467.79ms，退化79.55%；使用原定20%阈值，没有为得到结果降低阈值。
- 按`max(迭代秒数×2.2,20)`剖析；含轮询/收尾约21.03秒，采集29,032条完整clique通信记录。
- dense_DP同消息大小的rank平均通信时长：r0=2161.85µs，r1=2129.18µs，r2=2185.36µs，r3=87.91µs；rank CV=0.547，r3得分0.946。
- 自动预测rank3与真值一致；Expert_DP组间CV约0.079，小于0.1，被正确跳过，没有伪造交叉验证。
- 注入开始至定位完成约23.11秒；本实现保留20秒剖析下限，未复现论文声称的2–3迭代总延迟。

## 验证与复用边界

- 7项分析/控制测试通过；真实Rbeast数值例验证1-based变点坐标转换。
- 四CPU rank核对原始HF MoE与EP版本输出、输入梯度、参数梯度，含空专家分配：通过。
- 四GPU原生NCCL采集：真实AllReduce值、正Event时长、跨rank相同unique-id哈希：通过。
- GREYHOUND的14字段Record、RecordBuffer、NcclRecordStorage和配置直接编译复用；原文件仅17行新增、3行删除用于命名隔离、锁初始化、清理。
- 新probe针对本后端的AllReduce，Group共用并复用Event，MONITOR不创建/同步Event；使用公开NCCL接口与unique-id哈希，不解析私有结构。
- 当前仅单节点4rank固定拓扑，未覆盖多节点/TP/PP/完整NCCL算子矩阵。dense_DP仅1 clique；组间筛选在单测验证，未在本正例形成有效多clique证据。
- 梯度通信在反向末按dense_DP→Expert_DP串行，前一次同步会消除后续部分迟到差异；未验证论文多标签独立交叉定位。后续优化应优先对接原生Megatron重叠梯度通信。
- GPU stream延迟只模拟计算迟到，不等同真实降频对功耗、带宽和共享任务的影响；两类结果已分别保留。

原始日志、剖析数据和汇总在根目录`results/`：`summary.json`、`fault_gpu_delay/`、两个真实降频目录及两次OOM目录；验证日志在`results/validation/`。服务器实验进程已结束，GPU时钟恢复1980MHz。

## 阶段2：原生梯度通信与真实组标签

保留完整HF Qwen权重、已验证的EP dispatch/combine，直接调用原始GREYHOUND Megatron的parallel_state、DDP、ParamAndGradBuffer及反向bucket就绪钩子。不是Megatron GPT模型移植。两类梯度都按dense_DP大小4缩放，SGD显式使用main_grad；39个dense和59个expert bucket在反向计算中异步发起AllReduce。拓扑由每rank实际ProcessGroup成员上报，按unique-id clique观测成员匹配，不再按组大小猜标签；缺失/多义上报排除并保存诊断。

| 4卡EP2×ExpertDP2实验 | 正常/异常/恢复平均ms | 检测、定位 |
|---|---|---|
| 健康监控80步 | 246.29 | 0报警 |
| attention反向计算延迟250ms，160步 | 249.35 / 437.14 / 269.55 | 49步报警，cp46；最终rank3正确，但Expert_DP候选rank2，标签冲突保留 |
| 单步同延迟，90步 | 251.81 / 397.53(1步) / 247.06 | 0报警；不能据此把z-score去异常的效果归为恢复比例过滤 |
| 固定总延迟250ms，attention与本地路由专家各半，160步 | 279.44 / 423.10 / 269.88 | 48步报警，cp46；dense_DP与Expert_DP均指向rank3 |

均保留3步预热；异常均值排除本地profile起止前后0.5秒，正常/恢复为完整对应步。共享负载有波动，均值不代表受控开销或准确率。两次持续测试均46–125步生效、126步撤销，训练完成至163步。校准实测分别248.09和248.00ms；延迟分配依据同一校准cycles，未修改检测阈值或通信函数。第二次对照是预先限定的一次计算覆盖范围检查，不能用它抹掉第一次冲突。

第一例15,820条profile记录、3个clique，Expert_DP组间CV 0.322选[0,2]并指向rank2；该rank与真值rank3同属EP[2,3]，与计算迟到经EP传播相符，但缺少精确bucket身份，不能断言因果。第二例Expert_DP大bucket组间CV约0.780，选真值所在clique并定位rank3；dense_DP多项同指rank3，形成真实多标签及多clique证据。两次仍保留20秒profile下限。

GPU4rank tiny HF块校验通过：输出、输入梯度、全部owned参数梯度、两步SGD及空专家，且finish前所有bucket已发起通信。禁用TF32后按FP32容差校验；首次TF32形状舍入差异失败日志保留。13项单测在容器通过。原始日志及统计：`results/stage2_native_*/`、`results/stage2_summary.json`、`results/validation/stage2_*`。

资源限制：原生路径默认NCCL缓冲首次反向OOM，记录在stage2_native_smoke_v2；使用NCCL_MAX_NCHANNELS=4、NCCL_BUFFSIZE=1048576后跑通，配置进入后续manifest。最初一次启动因旧Megatron API命名差异失败并修复，日志同样保留。此阶段仍TP1/PP1，尚不能外推TP/PP、多节点或论文规模。

## 阶段3：8卡原生PP与失败边界

完整同一494,712,832参数checkpoint，PP2×EP2×ExpertDP2，物理GPU0–7。每stage保留12层，用原始Megatron非交错1F1B、DDP及finalize_model_grads，绑定embedding/head梯度经原生embedding组求和。HF层/权重/EP保持原定义；可微router充分统计随FP32 hidden payload传至末stage，保留全层aux loss。仅无padding、固定长度、no-cache；FP32 P2P及额外统计是适配简化，不是原论文性能配置。

8GPU完整24层tiny FP32核对通过：logits、全层aux及其router导数、所有stage-owned参数梯度、绑定权重、1及2微批平均、两轮SGD。最大参数梯度绝对误差1.49e-7、最大梯度相对L2误差3.85e-6、参数更新后最大绝对误差5.97e-8。原BF16检查因绑定梯度分段累加舍入超预设逐元素容差而失败，整体相对误差约0.77%；保留失败，不放宽容差。FP32严检不能替代完整BF16逐元素等价证明。

| 实验 | 有效步/平均迭代ms | 结果 |
|---|---|---|
| 原attention hook健康对照 | 80 / 357.16 | 0报警 |
| 与持续慢同attention+experts hook配置健康 | 80 / 410.34 | 0报警；包含故障模拟hook成本及共享负载差异，不能当低开销证明 |
| rank7持续计算延迟及恢复 | 160；前428.43 / 异常562.78 / 恢复407.73 | 48步检出cp46，最终错误预测rank3 |
| rank7同配置单步抖动 | 90；前415.86 / 单异常557.98 / 后419.90 | 第46步生效、第47步撤销，0报警 |

故障为每次反向请求125ms、校准124.05ms；2微批每训练步配置总延迟250ms，第46–125步生效、第126步撤销。异常统计排除profile区间前后0.5秒；共享负载使各运行均值不能作为受控开销估计。

20秒剖析采集7,144条稳定通信记录，真实2个dense_DP及4个Expert_DP clique；另4个PP成员组的绑定权重AllReduce按真实拓扑排除，没有因size=2错标为Expert_DP。末stage dense_DP指向真值rank7（各rank均值约9,202/8,192/9,213/3,590µs），但前stage多个尺寸的dense和expert证据指向其PP同伴rank3，累积分数2.093对0.524，最终误定位。保留所有原始候选，不能以存在一个正确子证据把最终结果改记成功。

原始Megatron在PP rank>0禁用bucketing：前stage多bucket、后stage整buffer，导致两stage消息尺寸不同。按相同count/dtype比较时，没有可直接对照的dense_DP跨stage尺寸；各尺寸按单clique进入rank阶段。2026-10-22复核发现，同一clique不同count还被当作独立标签累加，属于适配计分问题；仅消除重复票仍会误定位。EP/PP依赖传播是待验证解释，现有profile缺少时间线，不能直接把本例确认为论文适用边界。详见[PP诊断](PP_DIAGNOSIS.md)。原负结果保留，未搜索注入点、修改阈值或添加同步来制造成功。

## 收束口径与复现边界

| 第三章机制 | 本轮证据 |
|---|---|
| 完整训练步上报、30点/Rbeast、20%退化门槛 | 4/8GPU真实训练自动报警，cp与实际注入开始一致 |
| z-score、右保护、恢复比例过滤 | 代码及边界单测；GPU单步抖动不报警；不把z-score效果冒充恢复比例的独立实证 |
| MONITOR/按需PROFILE/回MONITOR | 每轮events、local、原始profile；20秒下限保留；定位完成后重新建立监控基线 |
| GREYHOUND数据面复用、unique-id clique | 原始共享内存14字段结构；native probe GPU校验；真实多clique记录 |
| 实际parallel state标签 | 原生Megatron组成员全rank上报，完整性/多义检查；同size PP组实际被排除 |
| 按标签两阶段定位、多标签汇总 | 4卡两稳定标签一致正例；4卡标签冲突与8卡最终误定位均保留 |
| 原生梯度通信与训练正确性 | 原始DDP/buffer/反向hook；4GPU FP32及8GPU PP FP32完整梯度、更新核对 |
| 恢复与短抖动 | 同一训练继续运行恢复段；短抖动控制见各实验目录 |

原生DP/EP与PP2已有正负证据，不能宣称通用准确定位或完整复现论文数字。按用户最新要求，当前仍处复现阶段：先解决8卡PP适配计分和归因缺口，再讨论更多卡与并行配置。此前直接以“方法边界”收束PP的结论由[PP诊断](PP_DIAGNOSIS.md)修正；继续扩16卡不能替代该核查。

明确未验证/简化：TP、跨节点本地/全局控制部署、16GPU、通信/Fail-Stop故障、完整NCCL算子矩阵、跨stream NCCL Group、真实语料质量及论文规模性能；PP Send/Recv本身不参与定位。Rbeast先验/采样参数论文未给出，本轮固定参数/seed使用trend.cpPr，不能声称与完整在线run-length BOCD数值等同。单尺寸证据按tag累计，多个bucket尺寸不是多个独立并行标签，可能增加某类通信的票重；所有冲突和候选必须一同解释。GPU stream延迟覆盖指定模块，仍不是硬件降频；真实345MHz未触发和OOM结果不可删除。

最后修复PROFILE关闭与定位落盘之间的收尾竞态：明确等待processed_iteration及profile_completed_epoch，超时记failed而非假completed；新增延迟/空样本回归测试，容器14项单测通过。已有实验都有完整profile_end及localization，可按原结果采信。原始结果在results/stage3_pp_*，验证及失败日志在results/validation/stage3_*。

可重复汇总：`python -m flash_v1.summarize results/stage2_native_* results/stage3_pp_* --output results/final_summary.json`。按实际注入迭代分段的统计在by_injection_iteration，状态/错误、全部检测和最终定位均保留。最终源码及SHA256快照在results/validation/final_source*，不依赖旧Git历史。
