# 尚未接入：全两MB FWD辅助证据

只新增独立collector与CPU测试；当前59c9/runtime/四role/dc89不变。不得将此文件存在视为GPU数值、覆盖或开销通过。

- `StageForwardEvidence(torch, stage, schedules, dist, rank, expected_collectives=..., stage_metadata=...)`；每iteration围住原schedule：`with aux.iteration(iteration, enabled=PROFILE, epoch=epoch): ...`。默认关闭不安装hook/wrapper/Event。现有CUDA完成边界后`aux.drain()`返回独立schema记录；由lead另接独立运输，不能混入原四role测量/完成marker。
- 必须先改`train.ep_forward`一个插点：给count交换和紧随两次`.tolist()`整体加`with aux.ep_metadata():`，从`dist.all_to_all_single(receive_count, send_count, ...)`之前到`send, receive = ...tolist()`之后；`bincount`及后续dispatched索引compute在外。可由独立可空观察器或小helper传入；禁止拿故障truth驱动观察。**本文件未应用这个插点；未接入时样本必拒用。**
- 临时包装原Megatron `schedules.forward_step`（含loss）及dist的`all_reduce/all_to_all_single/all_gather`，每个MB独立计时、内部通信和metadata跨度取union后加总，排除MB间PP/BWD。仅同stream、恰好两MB、每MB恰一stage调用/每owned及shared MLP恰一次、所有Event ready和完整通信manifest才可用；不额外sync、不改Tensor/梯度。不能证明隐藏内部stream或未拦截的其他通信API，换实现须重新审计。
- 当前TP2/EP2：设实际Qwen2MoeMLP叶总数为`M`、local层数`L`、具`_ep_group`的真实MoE层数为`E`，每MB预期`all_reduce=L+M+int(stage.pre_process)`、`all_to_all_single=4*E`、`all_gather=int(stage.post_process)`、metadata跨度`E`。PP0为L12/M204/E12，PP1为L12/M188/E11（层23是dense单MLP，输入保留[B,S,H]）。从模型census导出清单，禁止用观测计数反定预期；collector调用必须显式`metadata_spans=E`。Event对总数为`2*(1 + sum(expected_collectives.values()) + E)`。
- 原四role不变。辅助只收各MB实际stage输入shape/dtype、全层MLP输入shape/dtype及调用次数；stage_metadata另传真实PP/TP/模型宽度/端点。健康fit/cal采完再设计work条件基线，未知work拒辅助，禁止预先用总token除时长/外推/拿truth拟合。
- GPU交接后先比loss/全部梯度/SGD，再验证第二MB独有functional site、非首层/norm/head和健康数据；完整8步ABBA含drain核PROFILE<5%、MONITOR<1%。输出是含CPU空洞、内存/资源竞争的stream残余，不是纯kernel时间或已认证定位模型。
