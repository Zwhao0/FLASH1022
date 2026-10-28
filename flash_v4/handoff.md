# v4 发布交接入口

当前版本入口为本目录 `README.md`、`releases/v4/release_status.json` 和 `releases/v4/evidence/release_gate.json`（路径均相对仓库根目录）。根目录历史交接不代表本次 v4 的运行状态。

- 当前配置：Store `r92_store_v1`、BOCD10%、每8步检查、R86 stride64/8步PROFILE、冻结r11定位器；训练源码未改。
- 最新修复验证：legacy rank8与PP rank12完整通过；真实降频watcher完整恢复后停止读取进度，保留finally恢复与清理。部署测试端口避开系统临时范围并做两节点bind预检。
- 累计16次尝试11严格合格，5异常保留；原矩阵未改为全通过。共享高负载下出现过超时和错目标报警，风险未排除。
- 两个有效健康场无触发/无报警；本工程发布不等同32/32正式盲测或总体FPR证明。
- 模型路径迁移工具与本地部署步骤见README；私有SSH transport、凭据、模型权重和GPU二进制不随本版本发布。
