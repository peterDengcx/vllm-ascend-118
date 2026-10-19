# HY4 MXFP8 验证

适配直接复用当前 vLLM 的 HY4 文本模型，通过 Ascend worker 补丁替换注意力执行入口。
无需修改 vLLM 源码。支持检查点的共享 TopK、interleaved RoPE、attention sink、gate 和 iHC。
MXFP8 KV 投影在加载时恢复为浮点矩阵，再由 SFA 吸收。

目标权重 `/mnt/weight/Hy4-preview-FP8` 是文本检查点，不包含视觉模块。
插件内提供图像/视频处理器、视觉塔、投影和三维 MRoPE 的兼容实现，
使用 GPU 参考代码的视觉配置默认值。当前实现要求 PCP=1、DCP=1。

## 多模态链路测试

`--multimodal` 将文本检查点组合为 `HYV4VLForConditionalGeneration`，
通过显式 `ascend_random_vision_seed` 仅初始化缺失的视觉参数。
文本和 MTP 参数继续加载实际 MXFP8 检查点；不修改检查点或 vLLM 源码。
该文本检查点的模板不处理图像/视频内容项，因此启动脚本显式使用
`--chat-template-content-format string`，由 vLLM 将媒体转换为模型占位符。
固定种子默认 20261020，TP 分片从相同全量参数初始化。
随机视觉权重仅验证结构和执行链路，不具有可靠的图像/视频理解能力。
后续拿到完整 VL 检查点时，应使用其真实配置、处理器配置和权重，
去掉随机初始化覆盖，重新验证语义精度。

```bash
python tools/hy4/validate.py launch --multimodal --stage eager --directory .vllm_ascend/hy4_multimodal
python tools/hy4/validate.py wait --directory .vllm_ascend/hy4_multimodal
python tools/hy4/validate.py smoke --directory .vllm_ascend/hy4_multimodal
python tools/hy4/validate.py metrics --directory .vllm_ascend/hy4_multimodal
python tools/hy4/validate.py stop --directory .vllm_ascend/hy4_multimodal
```

后续 `async`、`graph`、`mtp`、`pd_prefill`、`pd_decode` 和 `fast` 的
`launch` 均追加相同的 `--multimodal --directory .vllm_ascend/hy4_multimodal`；
其他操作追加相同目录。代理会继承 P 的多模态状态。
全部完成时间、命令、请求和响应追加到该目录的单一 `validation.log`。

多模态 smoke 在五项文本请求后执行六项实际 curl：单张红图、蓝图、
非正方形图、两张交错图片、四帧 MP4 视频、带图长上下文。
日志保留图片/视频的完整 data URL 和响应，并检查编码特征有限、
不同图片产生不同特征、视觉 token 实际融合到语言模型。
媒体内容使用随机视觉权重，不评判颜色识别或画面语义。
诊断由 `ascend_vision_diagnostics` 显式启用，会同步设备读取少量统计值，
性能测量可关闭该项。`graph` 捕获语言解码器，视觉编码器使用 eager。

## 分阶段测试

在仓库根目录执行。需要 8 张空闲 NPU；本次环境为 Ascend950DT、CANN 9.1、
torch 2.10、torch-npu 2.10.0.post4、transformers 5.14.1。

```bash
set -e
for stage in eager async graph mtp; do
    python tools/hy4/validate.py launch --stage "$stage"
    python tools/hy4/validate.py wait
    python tools/hy4/validate.py smoke
    python tools/hy4/validate.py metrics
    python tools/hy4/validate.py stop
done
```

`eager` 关闭异步调度、图、MTP 和 PD。`async` 增加异步调度，`graph` 再增加
FULL_DECODE_ONLY ACLGraph，`mtp` 再增加 1 token MTP。
每一轮包含中英文、算术、自我介绍和超过 TopK 长度的输入，通过真实 curl 请求检查。
默认并发 4 个请求，预留 2 GiB KV cache，使用每 5 层卸载 1 层的权重预取配置。
关闭 NZ 转换以支持 pinned 权重的图捕获。

所有阶段共用 `.vllm_ascend/hy4_adaptation/validation.log`，采用 JSONL 格式，
逐项记录北京时间的完成时间、命令或请求、完整输出和状态；请求与命令执行另有耗时。
启动的详细输出另外保留在带时间戳的 server 日志中；启动失败时完整输出也写入统一日志。
`run --label NAME --command COMMAND...` 可将额外的测试结果记录到同一日志。

## PD 分离

先停止单实例服务。P 使用设备 0–3，D 使用设备 4–7，每 3 层卸载 2 层，
保留异步调度、decode graph 和 MTP。CPU 需要容纳两个模型的卸载权重。
本机为 Ascend950 UB 网络，脚本在 PD 阶段默认设置
`ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'`，供 Mooncake 使用 HIXL 1.3 通信资源。
其他硬件或网络应按部署环境配置该变量。

```bash
python tools/hy4/validate.py launch --stage pd_prefill --port 8100
python tools/hy4/validate.py launch --stage pd_decode --port 8200
python tools/hy4/validate.py wait --stage pd_prefill
python tools/hy4/validate.py wait --stage pd_decode
python tools/hy4/validate.py proxy --stage pd --port 9000
python tools/hy4/validate.py wait --stage pd
python tools/hy4/validate.py smoke --stage pd
python tools/hy4/validate.py metrics --stage pd
python tools/hy4/validate.py stop --stage pd
python tools/hy4/validate.py stop --stage pd_prefill
python tools/hy4/validate.py stop --stage pd_decode
```

PD 使用本仓库的 MooncakeConnectorV1 和示例代理。检查 P/D 日志及传输指标，
不能仅凭代理返回 200 就认定发生了 KV 传输。`metrics` 会同时检查 D 的
`external_kv_transfer` token 计数大于零，以及实际完成传输的日志。

## 快速 GPQA

停止 PD 两端后，启动快速单实例配置并先复测，再执行精度测量：

```bash
python tools/hy4/validate.py launch --stage fast
python tools/hy4/validate.py wait
python tools/hy4/validate.py smoke
python tools/hy4/validate.py metrics --stage fast
python tools/hy4/gpqa.py --concurrency 8
```

`fast` 使用异步 + graph + MTP，预填批次上限为 1024，
每 6 层卸载 1 层，减少传输和预填轮数。

使用 [simple-evals 引用的 GPQA Diamond 数据](https://github.com/openai/simple-evals/blob/main/gpqa_eval.py)，
完整 198 题、单轮、0-shot、选项置换种子 0、temperature 0、使用 structured choice 约束为 A/B/C/D、最多 1 个输出 token、
`reasoning_effort=no_think`，直接输出选项字母。逐题输入输出和汇总写入统一日志，
汇总另存 `gpqa-summary.json`。这种快速测量不能直接与公开的思维链或多轮结果比较。
