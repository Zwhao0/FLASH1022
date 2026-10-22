# 离线指纹研究工具

本文件记录开发阶段模板法；验证仅3/16定位正确，保留作为失败对照。当前选定方法是`wait_localize.py`的局部首调用等待差，算法与冻结结果见[DATA16_STUDY.md](DATA16_STUDY.md)。

`offline.py` 是独立研究路径，不接入现有在线定位或 controller。每个 episode 放在 `results/data16/<run_id>/`，包含 `manifest.json`、`topology.json`、`profile.json`。manifest 需有 `session_id`（或 `session`）、`role`（`train` / `validation` / `blind_test`）和监督标签 `fault_rank`；健康样本的标签为 `null`。同一 session 的所有 episode 必须属于同一 split。训练要求健康样本和每个故障 rank 都来自至少两个独立 train session。

只纳入 topology 中 `localization: true` 或明确 `stable: true` 的多 rank 组。每个 iteration、稳定组和 rank 先取 log duration 中位数，再计算 `median(peer log duration) - rank log duration`，最后跨 iteration 取中位数。特征身份由实际成员、语义标签、group name、kind、count、dtype 和观测 rank 构成；clique、group ID、时间戳、通信地址和故障标签不进入特征。健康 train run 按特征估计 median/MAD 尺度；验证 split 选择健康边界、模板距离和类别间隔拒判阈值。特征缺失保留为缺失 mask，覆盖低于默认 70% 时拒判。

```sh
python3 -m flash_v1.offline fit --dataset results/data16 --output results/offline-model.json
python3 -m flash_v1.offline evaluate --dataset results/data16 --model results/offline-model.json --split validation --output results/validation-report.json
python3 -m flash_v1.offline evaluate --dataset results/data16 --model results/offline-model.json --split blind_test --output results/blind-report.json
```

模型保存 train/validation 输入文件哈希、特征配置、拒判阈值和冻结模型哈希；fit 不读取 blind profile 或其故障标签。blind session 不得与 fit 使用的 session 重叠。报告以 episode/run 为分母，列出准确率、拒判、健康误报、覆盖率、混淆矩阵和每 run 的缺失特征。`flash_v1/tests/test_offline.py` 的数据完全合成，只验证代码逻辑，不代表 GPU 实验或算法效果。
