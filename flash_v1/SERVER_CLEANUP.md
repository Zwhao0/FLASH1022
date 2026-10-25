# 服务器清理结果（2026-10-25）

最后短轮V2测试及证据回收完成后，仅按用户授权的失败开发名单清理。

| 主机 | 删除退出容器 | 删除名单路径 | 可用空间实际增加 |
|---|---:|---:|---:|
| host1 | 68 | 203 | 110.539 GiB |
| host2 | 58 | 131 | 29.837 GiB |
| 合计 | 126 | 334 | 140.376 GiB |

保留两台FLASH1022目录、其项目容器/镜像/环境/模型；GREYHOUND参考容器配对与全部现有镜像均保留。源码SHA、模型元数据、保留容器ID/ImageID/Env摘要/挂载前后验证通过，删除错误/跳过均0。

GREYHOUND中已整理Greyhound源码、dataset、checkpoints/s4-reference-exact、results/s4-reference-exact及安全容器清单。FLASH1/runtime内是正在使用的Docker数据根，原地保留；四个旧绑定路径保留轻量兼容链接。

名称范围不明确的greenhound、opensource、xpu目录及oss-xpu-build暂留。默认Docker服务未修改，未停止其他服务、未全局prune，未启动GREYHOUND旧任务。

服务器106清单中预存在的工作站脚本/测试部署差异单独记录；清理没有补写或改变这些文件。私有详细清单与回执保留在本机tmp/server_cleanup_plan.json和tmp/server_cleanup_receipt.json，凭据不写入公开报告。
