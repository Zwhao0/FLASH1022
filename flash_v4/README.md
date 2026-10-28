# FLASH v4

**工程发布门槛已通过；最终凭据见 `releases/v4/release_status.json`。历史失败完整保留，不表示所有尝试全通过。**

面向计算慢故障的 rank/GPU 定位版本。保留 v1 训练源码，通过运行时适配实现 BOCD/Rbeast MONITOR → 有界 PROFILE → 冻结定位器。当前范围为两台机器、每台 8 GPU、16 rank 的小型 Qwen MoE，支持 `legacy` 和 `node_local_pp` 两种拓扑。

## 版本配置

- MONITOR：10% 持续退化门槛，原概率/持续/恢复过滤，每 8 步检查。新运行显式设 `r92_store_v1`；缺省配置仍走原 Gloo，另保留 `r92_gloo1_v1`。
- Store：rank 0 发布唯一 run/iteration 命令，peer 有界等待；不等待 peer ACK。同步/失联语义变更已获得用户授权。仅验证监督的 16 rank、PyTorch 2.7.0a0 / NGC 25.03；不承诺任意版本或无限长会话。
- PROFILE：R86，小 GEMM stride 64、大 GEMM 全采、kernel 256、8 步窗口；使用 `r68_async_finalize_v1`。逐次通信详细计时代码注释保留，真实 NCCL 转发和计数保留。
- 定位：冻结 r11 模型及规则。故障真值由注入/事后评分读取，不传入定位器。

## 环境与构建

已测试环境：Linux / NGC `pytorch:25.03-py3`，Python 3.12、PyTorch 2.7.0a0、CUDA 12.8，Hopper GPU。Python 依赖版本见 `flash_v1/requirements.txt`（transformers 4.53.2、numpy 1.26.4、Rbeast 0.1.25 等）；还需要现有 Megatron-Core 环境、模型文件、CUDA toolkit、NCCL headers、C++17 编译器和 NVIDIA MPS。

从仓库根目录执行：

```bash
python3 -m pip install -r flash_v1/requirements.txt
CUDA_HOME=/usr/local/cuda NCCL_INCLUDE_DIR=/usr/include \
  bash flash_v4/native/build.sh flash_v4/native/libflash_v4_observer.so
python3 releases/v4/prepare_runtime.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python3 -m unittest discover -s flash_v4/tests -p 'test*.py'
```

`prepare_runtime.py` 校验冻结模型和六个推理源文件的原始字节/哈希，只为当前克隆目录生成路径迁移凭据；不会拟合模型或重跑验证。原冻结凭据完整保存在 `releases/v4/frozen/r11_kernel_match_v3/`。生成的 `results/` 仍为本地运行数据。已有不同凭据时脚本拒绝覆盖，请用新克隆。

C++ CPU 测试可逐个编译运行，不需要 GPU：

```bash
for name in event_pool identity_cache output_buffer r55_caches; do
  c++ -std=c++17 -O2 "flash_v4/tests/test_observer_${name}.cpp" -o "/tmp/v4_test_${name}" && "/tmp/v4_test_${name}"
done
```

模拟慢计算需额外构建 `flash_v4/native/build_stream_delay.sh`；生产定位不依赖此故障注入器。该脚本默认 CUDA 架构 90，需与实际 GPU 匹配。

## 在线入口

`flash_v4.shared_node` 管理每节点的私有 MPS、8 个训练进程、时限和清理。两节点同一运行必须使用相同唯一 run ID、master 地址/端口与 seed，node rank 分别为 0、1；每次运行选择新的输出目录。模型目录须提前存在。

在两节点分别设置 `NODE_RANK`、`MASTER_ADDR`、`MODEL_DIR`、`OUTPUT_DIR`、唯一 `RUN_ID`（`v4_` 加小写字母/数字/下划线，总长不超过 63），再运行：

```bash
FLASH_V4_MONITOR_CONTROL=r92_store_v1 \
FLASH_V4_MONITOR_RUN_ID="$RUN_ID" \
FLASH_V4_MONITOR_DEGRADATION=0.1 \
FLASH_V4_PROFILE_SAMPLE_STRIDE=64 \
FLASH_V4_PROFILE_FINALIZE=r68_async_finalize_v1 \
FLASH_V4_PROFILE_FINALIZE_SUPERVISED=r68_async_finalize_v1 \
FLASH_V4_FORCE_PROFILE_AT=0 \
FLASH_V4_ALLOCATOR_BUDGET=available \
PYTHONPATH=. python3 -m flash_v4.shared_node \
  --run-id "$RUN_ID" --layout legacy --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 27120 \
  --model "$MODEL_DIR" --output "$OUTPUT_DIR" \
  --steps 112 --seq-len 1024 --seed 20261028 --timeout 180 \
  --observe-mode online --observer-library "$PWD/flash_v4/native/libflash_v4_observer.so" \
  --transport ib
```

另一拓扑使用 `--layout node_local_pp`。这条命令启动有界观测训练；训练中的自动收集与 CPU 推理由 `online_driver` 控制器负责。仓库中的 `online_harness` 是本项目两台服务器的适配器，依赖站点 Docker 配置和外部私有 `/tmp/flash_v4_remote.py`，不能直接用于其他服务器。部署到新集群需实现其 transport/容器/路径适配；凭据始终留在仓库之外。

已有收集数据可单独运行冻结推理；输入目录下需有 `node*/observations/rank_*.jsonl`，输出文件采用新建语义：

```bash
PYTHONPATH=. python3 -m flash_v4.online_infer \
  --input-root /path/to/collected/window --topology legacy --case example
```

## 验证与边界

最终测试结果见 `releases/v4/evidence/final_test_summary.json`，版本结论见 `releases/v4/release_status.json`。原始结果留在实验工作区；发布的证据摘要和哈希不等于完整原始数据包。

R93 MONITOR 成本 10/10 有效：Store 相对原始 OFF 端点，完整运行最大正增量 +0.1711%，训练步时总和 +0.2139%，作为本轮 <1% 工程观察。OFF 本身有漂移，不宣称稳定 0.1% 或绝对零。R86/R87 PROFILE 已按 ≤5% 工程目标接受，端到端 1% 未稳定达到；CPU/RSS 也不保证稳定下降。

历史失败分母保留：原随机矩阵 30 成功/2 超时；R88–R91 代表快测 8/11 正确、3 未触发；R94–R96 Store 回归共 5 次尝试，2 正确完整、3 采集/控制失败（其中一场曾注入前报错目标）。最后两次有效回归不能替换前三次失败。详见发布的历史摘要。

R97–R99 共 13 场独立尝试，严格口径 9 场合格、4 场异常保留：z02 的 seq1024 模拟仅完成 28/32 个注入步；z04、z05 在训练启动前遇到 rendezvous 端口冲突；z07 虽正确检测和定位且完成恢复，但 watcher 在 after105 报 stale progress，严格时钟区间资格未通过。原失败不覆盖，也不从累计结果中删除。

R100 只验证 watcher 控制链修复，不改生产 MONITOR 或定位规则：legacy rank8 严格通过；PP rank4 因 180 秒训练超时，MONITOR 结束仅 11/16，虽 PROFILE 窗口完整却报 rank9（目标为 rank4），watcher 在第49步因 stale progress 提前恢复，严格结果为 `collection_failure`，错目标报警另列，不能归为纯测试问题。同期共享 vLLM SM 观测最高 99%，只能作为并发负载证据，不能认定为唯一原因；共享高负载下超时和错误报警风险仍未排除。

R101 新 PP rank12（新 release/seed、端口27235、同一 watcher修复、96 steps）已严格通过。修复后的真实故障验证共3次尝试：legacy rank8、PP rank12共2场完整通过，PP rank4的1场失败仍保留。两拓扑均完成有效修复验证，工程发布门槛通过；全部累计16次尝试，11场严格合格、5场异常。原矩阵与R100原组仍不全通过，不构成正式32/32或总体FPR证明。

本版本只涉及计算慢的 L1 rank/GPU 定位。最终代表快测不等同正式盲测、32/32 全覆盖、总体误报率 ≤5% 或任意训练/模型/集群泛化保证。共享 GPU 活动只做采样记录，不扣估计噪声。真实降频测试须记录物理绑定、实际退化、完整注入区间和时钟恢复；不得重放已消耗验证或用结果回调冻结规则。

修复后的真实降频测试 watcher 为 `releases/v4/clock_watch_r100.py`：仅在完整计划区间结束且成功恢复后停止读取训练进度，finally 恢复与watchdog/stream清理仍执行。相关CPU安全回归可运行 `PYTHONPATH=. python3 releases/v4/watcher_cpu_r100.py`（15项，进程和GPU命令均为测试替身）。选取训练端口时，应避开系统临时端口范围，并在两节点执行bind预检；预检不保证之后没有竞争。
