# FLASH 第三章 v1

当前路线：2节点16卡、真实TP2×PP2×EP2×Expert-DP2。`wait_localize.py`已通过`online.py`接入自动检测与有界8步PROFILE，rank6/10故障在线回归2/2正确；常驻MONITOR已优化，真无探针ABBA平均耗时差−1.173%（本次平均开销点估计<1%），见[优化与实测边界](MONITOR_OPTIMIZATION.md)，接入命令见[在线报告](ONLINE16.md)。此前冻结离线盲测故障16/16、健康4/4正确，见[数据研究报告](DATA16_STUDY.md)。离线采集入口`python -m flash_v1.scripts.data16`保留，模板法失败与论文对照保留，旧`--reference-profile`入口仍禁用。下方4/8卡命令与校准结果为历史记录。

从论文第三章与原始 GREYHOUND 重新实现；不读取或沿用旧 FLASH 实现。
仅检测/定位计算 Fail-Slow；支持真实 GPU 降频与经用户授权的GPU计算stream延迟模拟；不包含 Fail-Stop、通信故障或第四章 SHIFT。

- 论文规格与歧义：[PAPER_SPEC.md](PAPER_SPEC.md)。
- 模型：`bayliner1980/Qwen2.5-0.5B-MoE-A0.3B`，revision `4b89028ef7376d0e92acbbc4a7f560c2374aeec0`。
- 社区基于 Qwen2.5 转换的公开 MoE；非 Qwen 官方发布。完整494,712,832参数、24层，23层32路由专家/top8及共享专家。
- EP2×Expert-DP2，4GPU；真实可微 token dispatch/combine、全参数 SGD；随机 token 仅测系统性能，不评估语言质量。
- GREYHOUND 直接复用 `detector/shm_storage.{hpp,cpp}`、`config.hpp`：14字段记录、共享内存环形队列与锁。
- 原代码仅对共享内存加实验命名隔离、锁初始化与退出清理修复。新 NCCL 适配层用公开API取得rank/组ID，替代版本敏感私有结构。
- 分析预处理/BEAST流程沿用 GREYHOUND 思路，按 FLASH 改为迭代时间输入、恢复过滤、动态剖析和两阶段定位。
- 单节点本地/全局独立控制器经Redis通信；稳定dense_DP/Expert_DP分组。组标签由实际成员上报匹配，拒绝缺失/多义组；未验证多节点、TP或通用NCCL调用覆盖。
- 只拦截本实验使用的AllReduce与communicator/group初始化；动态EP Send/Recv不作为定位证据。

```bash
python flash_v1/scripts/download_model.py
bash flash_v1/native/build.sh
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=4 flash_v1/train.py --device cpu --verify-only --output /tmp/flash_ep_verify
python -m unittest discover -s flash_v1/tests -p 'test_*.py'
python -m flash_v1.run --mode baseline --steps 80 --output results/baseline
python -m flash_v1.run --mode monitor --steps 80 --output results/monitor
python -m flash_v1.run --mode fault --fault-kind gpu-delay --delay-ms 250 --fault-steps 80 --steps 160 --inject-at 45 --output results/fault_delay
# 真实降频对照（本次未达到20%全局退化门槛）
python -m flash_v1.run --mode fault --fault-kind clock --steps 160 --frequency 345 --output results/fault_clock
```

`run`默认物理GPU4,5,6,7，故障rank3对应GPU7；可用`--gpus`调整。
需要root或GPU时钟控制权限、redis-server、CUDA/NCCL开发文件、g++和Boost开发库。
每次独立Redis端口/共享内存，异常退出恢复GPU自动时钟；保留独立延迟恢复进程。
共享GPU上其他负载使误报/定位不确定性上升，报告必须保留未成功结果，不能继承论文准确率。

结果：`manifest.json`、`train.jsonl`、`injection.jsonl`、`events.jsonl`、`profile_*.json`。

阶段2使用原始GREYHOUND Megatron的parallel_state/DDP/梯度buffer与反向就绪钩子，保留同一完整HF模型及已核对的EP。SGD使用原生main_grad，两类梯度按dense_DP大小平均；不是完整Megatron GPT权重移植。

```bash
# 共享显存环境下固定NCCL配置；各对照保持一致
export NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=1048576
python -m flash_v1.run --mode monitor --gradient-backend megatron --steps 80 --output results/native_healthy
python -m flash_v1.run --mode fault --fault-kind gpu-delay --gradient-backend megatron --delay-placement attention-experts --delay-ms 250 --fault-steps 80 --steps 160 --output results/native_sustained
python -m flash_v1.run --mode fault --fault-kind gpu-delay --gradient-backend megatron --delay-ms 250 --fault-steps 1 --steps 90 --output results/native_jitter
torchrun --standalone --nproc-per-node=4 -m flash_v1.tests.megatron_verify --output results/validation/native_verify
```

`serial`保留阶段1复现入口；`megatron`为阶段2反向重叠梯度通信。延迟默认放在attention反向前；`attention-experts`把总cycles各半分给attention及本rank所有路由专家。两种放置都是真实GPU stream延迟而非通信故障，不能等同真实降频。阶段2两种放置的定位差异已保留在RESULTS中。

阶段3用原始Megatron 1F1B schedule和finalize_model_grads运行8卡PP2×EP2×ExpertDP2。Stage wrapper保留24层及绑定embedding/head；P2P额外携带可微router统计以保留全层aux loss，payload为FP32，模型参数BF16。仅固定长度、无padding/no-cache，TP1；这是明确的适配简化。PP全模型FP32小配置的logits、aux、逐参数梯度和1/2微批SGD已核对。首轮误定位保留；后续修正组计分并加入健康参考，独立实测定位rank7，见[PP验证](PP_OPTIMIZATION.md)。

```bash
torchrun --standalone --nproc-per-node=8 -m flash_v1.tests.pipeline_verify --output results/validation/pp_verify
python -m flash_v1.run --mode monitor --gradient-backend pipeline --gpus 0,1,2,3,4,5,6,7 --delay-placement attention-experts --steps 80 --output results/pp_healthy
# 健康PROFILE参考；手动采样不算检测报警
python -m flash_v1.run --mode monitor --gradient-backend pipeline --gpus 0,1,2,3,4,5,6,7 --delay-placement attention-experts --steps 85 --profile-at 34 --output results/pp_reference
# 同配置校准；每反向125ms、2微批，真值rank7
python -m flash_v1.run --mode fault --fault-kind gpu-delay --gradient-backend pipeline --gpus 0,1,2,3,4,5,6,7 --fault-rank 7 --delay-placement attention-experts --delay-ms 125 --fault-steps 60 --steps 110 --reference-profile results/pp_reference/profile_1.json --output results/pp_calibrated
```

本次结果：[RESULTS.md](RESULTS.md)。GPU stream延迟经CUDA Event校准，阶段1放在目标rank反向计算后、梯度同步前；阶段2放在反向模块内；不延迟通信函数，不用CPU sleep。它不模拟降频的功耗/带宽变化。

服务器入口：host2 `/home/campus/zhangweihao/FLASH1022`，容器 `flash1022-run`，镜像 `flash1022-runtime:v1`。
通过 `DOCKER_HOST=unix:///run/greyhound-docker.sock /opt/greyhound-docker/docker exec -it flash1022-run bash` 进入后 `cd /workspace/FLASH1022`。
