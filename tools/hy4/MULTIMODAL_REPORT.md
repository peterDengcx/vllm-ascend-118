# HY4 随机视觉权重多模态验证

代码提交：`274ccff`。所有时间为北京时间（UTC+08:00）。

本次使用 `/mnt/weight/Hy4-preview-FP8` 的实际 MXFP8 文本和 MTP 权重，
仅对缺失的视觉参数做随机初始化。结果用于验证图像/视频执行链路，
不代表模型具有图像/视频理解能力，也不评判图片内容识别精度。

## 实现与环境

- 修改范围为本仓库。vLLM checkout 的 diff 与开始时逐字节一致；原有依赖修改保留。
- 在 Ascend 插件中组合 HY4 文本模型、视觉编码器、投影、图像/视频处理器和三维 MRoPE。
  SFA、indexer 和 MTP 均传递独立的旋转位置；因果 KV 位置保持单独处理。
- 使用 GPU 参考代码的视觉默认配置：27 层、hidden 1152、16 heads、head dim 72、
  patch 16、spatial merge 2、newline、BF16 视觉计算。MRoPE sections 为 `[12,10,10]`，interleaved。
- 缺失的 337 个视觉参数张量使用显式种子 20261020，按参数名派生独立种子。
  TP 从同一全量参数生成分片；已有检查点参数保持加载值。未显式设置种子时缺少视觉权重会报错。
- 文本检查点的模板使用 `--chat-template-content-format string` 接收 vLLM 媒体占位符。
- 测试机：8×Ascend950DT（96 GiB/卡），CANN 9.1，torch 2.10，torch-npu 2.10.0.post4，
  transformers 5.14.1，vLLM 0.30.0。PCP=1、DCP=1。
- graph 为语言解码器的 `FULL_DECODE_ONLY` ACLGraph；视觉编码器保持 eager。
- 本轮请求限额为 2 张图片/1 个视频；图片最多 65536 pixels，视频最多 4 帧。

## 分阶段结果

每阶段通过真实 curl 发送 5 项文本和 6 项多模态请求；多模态请求包含单图、
两张交错图片、非正方形图、四帧 MP4 和带图长上下文。检查 HTTP 状态、
可读输出，以及实际视觉编码和融合。随机视觉权重不用于图像语义准确率评判。

| 阶段 | 完成时间 | 文本 | 多模态 | 运行证据 |
| --- | --- | --- | --- | --- |
| eager；无 async/MTP/PD | 2026-10-20 00:42:35 | 5/5 | 6/6 | 编码 32 次、融合 48 次；均有限，图片不同则特征不同 |
| + async | 2026-10-20 00:46:38 | 5/5 | 6/6 | 编码 32 次、融合 48 次；均有限 |
| + decode graph | 2026-10-20 00:49:58 | 5/5 | 6/6 | 实际 FULL 回放；API 就绪后的回放另有审计记录 |
| + 1-token MTP | 2026-10-20 00:54:51 | 5/5 | 6/6 | 27 draft / 25 accepted，graph 回放正常 |
| + PD（P/D 各 TP4） | 2026-10-20 01:02:50 | 5/5 | 6/6 | 5911 external KV tokens、44 次 rank 传输；41 draft / 25 accepted；graph 正常 |
| 快速单实例（1024-token 预填） | 2026-10-20 01:14:04 | 5/5 | 6/6 | 28 draft / 25 accepted；编码、融合和 graph 正常 |

编码/融合次数按所有 rank 的日志统计，包含媒体缓存复用，不能等同于请求数。
PD 的实际请求记录为 16 次编码、24 次融合，均在 P 侧完成；D 使用传入的 KV。
P 使用 NPU 0–3，D 使用 NPU 4–7，经 MooncakeConnectorV1 传输 KV，代理端口为 9000。
每 3 层卸载 2 层；其余单实例验证每 5 层卸载 1 层。预留 KV cache 均为每卡 2 GiB。
快速评测最初沿用文本模型的每 6 层卸载 1 层配置，在 1024-token 预热时显存不足；
多模态快速配置已改为每 5 层卸载 1 层，保留 1024-token 批次及 async/graph/MTP。
该失败尝试的完整启动输出和后续重测仍保留在统一日志。
eager 带图长请求为 2748 个 prompt token，超过 index TopK=2048。
该请求回答 `2`；图片请求示例回答 `测试完成。`，视频请求示例回答 `好的，这是一段测试视频。`。

331 项 CPU 回归测试通过，3 项实际 NPU 测试通过。覆盖随机初始化、图片 patch 顺序、
MRoPE 轴选择、文本位置等价、视频时间戳桥接、SFA/indexer 元数据、MTP 量化排除项，
以及实际 NPU 视觉编码边界和 MRoPE 图捕获/回放。
另在 4096 个文本位置上验证原始 HY4 RoPE 和 MRoPE 的缓存逐位相同。

## GPQA Diamond 快速精度

完成时间：2026-10-20 01:15:32+08:00。完整 198 题，正确 102 题，准确率 **51.52%**，
评测耗时 **87.59 秒**，请求错误 0，无法解析答案 0。
该耗时不包含模型启动、预热和前置 smoke；总输入 51855 token，输出 198 token。

单轮、0-shot、选项置换种子 0、temperature 0、并发 8，
`reasoning_effort=no_think`，structured choice 限定 A/B/C/D，最多 1 个输出 token。
这是快速直接作答口径，不能直接与公开思维链或多轮结果比较。
数据 SHA256：`41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`。
逐题完整输入和响应在统一日志中；汇总另存 `gpqa-summary.json`。

与此前文本快速评测使用相同数据和请求；此前为 97/198，本轮为 102/198，
其中 177/198 个预测一致。本轮使用多模态组合模型，权重卸载分组也有变化；
这一次分数变化不能单独证明精度改善。

## 日志与复现

统一日志：`.vllm_ascend/hy4_multimodal/validation.log`。
每条 JSONL 包含 `completed_at`、`stage`、`status`、完整输入和输出；
curl 请求保留完整图片/视频 data URL 和服务响应。另有独立的服务原始日志，
以及 `media/` 中的测试媒体。失败的调试尝试和修复后重测结果均保留在统一日志中。

启动及重复测试命令见同目录 `README.md`；所有阶段使用
`--multimodal --directory .vllm_ascend/hy4_multimodal`。
真实视觉权重到位后，应使用其匹配配置和处理器参数，去掉随机初始化覆盖，
重新验证图像/视频语义精度。

最终快速服务保留于 `127.0.0.1:8000`，model name 为 `hy4`，健康检查通过。
释放设备可执行 `python tools/hy4/validate.py stop --directory .vllm_ascend/hy4_multimodal`。
