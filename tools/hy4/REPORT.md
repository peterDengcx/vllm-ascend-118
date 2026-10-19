# HY4 MXFP8 文本适配验证结果

验证日期：2026-10-19。以下完成时间均为 UTC+08:00。
模型：`/mnt/weight/Hy4-preview-FP8`，ModelOpt MXFP8，约 757.88 GiB（含 MTP）。
硬件：单机 8 × Ascend950DT，单卡 96 GiB；主机约 2.25 TiB RAM。
环境：CANN 9.1.0、torch 2.10、torch-npu 2.10.0.post4、transformers 5.14.1。
vLLM 基线：`ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
vLLM Ascend 基线：`99eeba9`。

## 代码改动

- `c4714a5`：通过 Ascend worker 补丁接入 HY4 的共享 TopK、interleaved RoPE、
  attention sink、gate 和上游 iHC；复用 Ascend SFA 执行路径。
- MXFP8 专家和线性层保留量化执行；仅将需要被 MLA 吸收的 `kv_b_proj` 恢复为浮点矩阵。
- 权重预取释放多余的 NPU buffer 引用，并以相同 dense storage 的方式拷贝转置权重，
  支持 ACLGraph 捕获与重放。
- `1ab1448`：加载后处理复用原来的 pinned CPU storage，避免同时部署 P/D 时重复占用
  两份卸载权重的主机内存。
- 未启用稀疏 KV 卸载时延迟导入 memfabric，避免无关的初始化副作用。

所有提交的源码改动均位于本 vLLM Ascend 仓库。范围审计确认 vLLM 源码 diff 与任务开始时
逐字节一致；用户原有的 `requirements/common.txt`、本仓库 `pyproject.toml` 和
`requirements.txt` 改动保留且未纳入适配提交。

## 分阶段服务验证

每轮使用真实 curl 请求测试算术、中英文首都问答、中文自我介绍和长输入。
各轮结果均为 `2`、`北京`、`Paris`、完整可读的中文介绍和 `北京`，未出现替换字符或乱码。
最长输入 2734 token，超过 index TopK=2048。

| 配置 | curl 结果 | 最后一条请求完成时间 |
| --- | --- | --- |
| Eager；异步/MTP/PD 关闭 | 5/5 | 2026-10-19T21:57:01.101442+08:00 |
| Eager + 异步调度 | 5/5 | 2026-10-19T22:02:12.040885+08:00 |
| 异步 + FULL_DECODE_ONLY ACLGraph | 5/5 | 2026-10-19T22:07:13.542616+08:00 |
| 异步 + graph + 1-token MTP | 5/5 | 2026-10-19T22:15:35.690482+08:00 |
| 异步 + graph + MTP + PD（TP4 P + TP4 D） | 5/5 | 2026-10-19T23:06:08.281328+08:00 |
| 快速单实例（TP8，batch 1024，卸载 1/6 层） | 5/5 | 2026-10-19T23:10:55.957437+08:00 |

图模式在运行日志中出现 `ACL graph replay is active` 和实际 `FULL` 执行计数。
MTP 单实例短用例产生 11 个 draft token，接受 10 个；这只是功能测试，不作为广泛接受率估计。

PD 通过 MooncakeConnectorV1 和仓库示例代理：

- P 设备 0–3、D 设备 4–7；各 TP4 + EP，每 3 层卸载 2 层。
- P/D 两端保留异步、decode graph、1-token MTP。
- D 实际接收 2,871 个外部 KV token；
  4 个 rank 共记录 20 次成功传输，排除了仅靠重算返回结果的情况。
- D 的 MTP 产生 16 个 draft token，接受 10 个；确认实际图重放。
- 本机 UB 网络需要 `ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'`。
  缺失该配置时，旧通信路径报 rank table 缺少 `rank_count`；配置后小张量逐项比对及真实模型传输均成功。
  参数依据为 [CANN HIXL Initialize 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hixlug/llmcrv2_42_25.html)。
- 代理绕过本机 HTTP 代理，以确保请求到达本地 P/D 服务。P/D handshake 端口分别从 20001、21001 开始。

## GPQA Diamond 快速精度

完整 198 题、单轮、0-shot direct answer、选项置换种子 0、temperature=0、
`reasoning_effort=no_think`、structured choice 约束为 A/B/C/D，输出上限 1 token、并发 8。
这是无思考的约束选项测量，不能直接与公开的 CoT 或多轮分数比较。

- 正确 **97/198，准确率 48.99%**。
- HTTP/请求错误：0；无法解析的回答：0（计为错误）。
- 实际请求耗时 **83.53 秒**，完成于 `2026-10-19T23:15:15.747104+08:00`；耗时不含服务启动。
- 输入 token：51,855；输出 token：198。
- 运行配置：TP8 + EP，异步 + graph + 1-token MTP，max batch tokens=1024、max sequences=8，
  每 6 层卸载 1 层，KV cache=2 GiB，NZ 关闭。
- 数据来源：[OpenAI simple-evals 使用的 GPQA Diamond CSV](https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv)。
- CSV SHA256：`41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`。

## 回归、日志和边界

相关 CPU 回归 **228 passed**；实际 NPU 算子测试 **12 passed**。
NPU 测试覆盖 attention sink 对 dense 参考结果、4096 长度的 TopK/RoPE，
以及 BF16/FP8/uint8 转置权重的 pinned storage 复用和图捕获/重放。

完整命令、环境配置、逐项输入输出、失败尝试及修复后的结果均在统一 JSONL 日志：
`/home/q00939541/qlf_hy4/vllm-ascend-hy4/.vllm_ascend/hy4_adaptation/validation.log`。
每步记录带时区的完成时间、输入、输出和状态；请求与命令执行另有耗时。
逐题 GPQA 输出也在此日志，汇总另存 `gpqa-summary.json`。
最终测量 run_id：`a57df47127074d70a38a83a6f4ed536c`。先前的自由输出尝试因解释文本被截断而中止，
已记录 `gpqa_abort`，不计入本次 198 题分数；已修正 Unicode 分行符的日志读取方式，并保留原始备份和诊断更正记录。
复现命令见 [README](README.md)。

当前交付为文本适配。按用户确认，视觉权重未就绪，多模态适配和实测留待完整权重提供后进行。
当前要求 PCP=1、DCP=1；PD 覆盖单机 4+4 卡，服务上下文上限为 16K，最长实测输入为 2734 token。
此机器需使用权重卸载，测试吞吐不代表权重全驻留设备时的性能。
