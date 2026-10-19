# HY4 MXFP8 验证

适配直接复用当前 vLLM 的 HY4 文本模型，通过 Ascend worker 补丁替换注意力执行入口。
无需修改 vLLM 源码。支持检查点的共享 TopK、interleaved RoPE、attention sink、gate 和 iHC。
MXFP8 KV 投影在加载时恢复为浮点矩阵，再由 SFA 吸收。

目标权重 `/mnt/weight/Hy4-preview-FP8` 是文本检查点，不包含视觉模块。
真实多模态适配和验证需要完整视觉权重。当前实现要求 PCP=1、DCP=1。

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
逐项记录北京时间的完成时间、命令或请求、完整输出、状态和耗时。
启动的详细输出另外保留在带时间戳的 server 日志中；启动失败时完整输出也写入统一日志。
`run --label NAME --command COMMAND...` 可将额外的测试结果记录到同一日志。

## PD 分离

先停止单实例服务。P 使用设备 0–3，D 使用设备 4–7，每 3 层卸载 2 层，
保留异步调度、decode graph 和 MTP。CPU 需要容纳两个模型的卸载权重。

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
不能仅凭代理返回 200 就认定发生了 KV 传输。

## 快速 GPQA

启动最快的已验证单实例配置后执行：

```bash
python tools/hy4/gpqa.py --concurrency 8
```

使用 [simple-evals 引用的 GPQA Diamond 数据](https://github.com/openai/simple-evals/blob/main/gpqa_eval.py)，
完整 198 题、单轮、0-shot、选项置换种子 0、temperature 0、最多 16 个输出 token、
`reasoning_effort=no_think`，直接输出选项字母。逐题输入输出和汇总写入统一日志，
汇总另存 `gpqa-summary.json`。这种快速测量不能直接与公开的思维链或多轮结果比较。
