# FLASH v2 四层计算慢

2节点16卡，真实TP2×PP2×EP2×Expert-DP2；覆盖legacy与node-local PP两布局。模型为`bayliner1980/Qwen2.5-0.5B-MoE-A0.3B`（revision `4b89028ef7376d0e92acbbc4a7f560c2374aeec0`）。

显式v2启动（完整online plan含core/peer模型绑定）：

```bash
python3 -m flash_v1.scripts.compute_v2 --plan /path/to/approved_online_plan.json
```

该入口启用aux固定采样`{1,3,7}`与fixed-model startupGC；core保持完整8步。旧入口默认不变，实测成本只属于显式配置模式。

- [离线复核与误报边界](results/layer_campaign/stage_forward_v1/postcommit_random_check_v1/offline_v2_review/offline_report.md)
- [v2结果与边界](flash_v1/COMPUTE_V2.md)
- [运行与实现](flash_v1/README.md)
- [完整旧v1首页与报告链接](flash_v1/README_V1.md)
- [论文机制规格](flash_v1/PAPER_SPEC.md)

冻结范围内实测：MONITOR完整ABBA平均legacy −0.299%、PP +0.514%；PROFILE总体legacy3.51%、PP3.36%，对应总体95%CI上界均在预算内。PP逐序严格MONITOR门仍不通过；负值不作加速声明。精度、失败与适用范围见v2报告。

仅计算Fail-Slow；不包含Fail-Stop或第四章恢复。提交后4–6例随机抽查待执行，不将有限实测外推为总体保证。
