# HY4 CANN 9.3：run09 实测代码与通信修复

本目录保存 2026-11-23 在 Ascend950PR 上完成的修改组实验。
128/128 请求成功；正式输入 128000、输出 1024、并发 32、90% 前缀命中。
整机八卡输出吞吐 856.8797 token/s，平均 TPOT 12.6619 ms/token。

这是实际运行版本的补丁及证据归档。仓库顶层运行代码不自动启用这些改动。
本轮没有同环境对照组，不能把与旧 CANN 9.1 结果的差值归因到单项优化。
650E 尚未复测；请求成功和长度正确不等于模型精度认证。

## 代码与版本

- 基础仓库：`ShySummer/vllm-ascend-hy4-flash`。
- 基础提交：`2c123f517504fa551379f7f5c7f3c4ac0e8b5686`。
- `patches/cann93-run09.patch`：运行快照相对基础提交的完整 Python 内容差异，8 个文件。
- `patches/communication-fix.patch`：其中的 KV 通信修复，供单独审阅；不要重复应用。
- `source-manifest.json`：变更文件哈希；`results/source-sha256.json`：全部 2267 个运行快照文件哈希。
- `reference/`：实际使用的启动、压测、指标收集、端点导出和八 worker 连通性验证脚本。
- `results/`：正式指标、逐请求拆解、配置、通信验证及资源回收记录。

补丁包含 Engram 的 CANN RMS 路径、iHC decode pre-tail 融合及 post 融合开关、
C8 decode 批量读取、IndexPool 调整、DFly DP 图实验路径和通信修复。
DFly DP 图代码虽保留在快照中，run09 的开关为 **false**；不要把它记作本轮启用的优化。
旧 `20261123-indexpool-lmhead` 的短图和 FP32 LMHead 缓存没有用于本轮。

在上述基础提交的独立源码目录中应用一次：

```bash
git apply --check /path/to/20261123-cann93-run09/patches/cann93-run09.patch
git apply /path/to/20261123-cann93-run09/patches/cann93-run09.patch
```

生成补丁时忽略了基础仓库与运行环境的 CRLF/LF 差异；内容变更后的文件哈希
须与 `source-manifest.json` 一致。完整原始日志及源码压缩包留在本地实验归档中，
其 SHA-256 记录在 manifest；本目录不包含模型权重、请求文本或构建产物。

## CANN 9.3 通信问题及修复

新 HIXL 自动资源生成只枚举 worker 可见的四张卡。
P 看到 0–3，D 看到 4–7，跨组 Mesh 端点缺失；即使优先指定 PEER2PEER，
仍出现 `no_matching_pair` 并回退到无法建链的 PEER2NET/CLOS。
拓扑文件已挂载且与旧容器一致，此次不是漏挂文件。

同一新容器、同一原装组件、256 KiB 数据的验证结果：

| 可见范围与资源配置 | 读回结果 | 建链及读回时间 |
| --- | --- | --- |
| 八卡可见，Mesh 优先，自动资源 | 正确 | 0.6192 s |
| P/D 各四卡，Mesh 优先，自动资源 | 失败 | 10.1083 s |
| P/D 各四卡，Mesh 优先，完整资源 | 正确 | 0.6227 s |

修复由两个配置共同生效：

```bash
export HY4_MOONCAKE_LOCAL_COMM_RES_DIR=/absolute/path/to/generated/resources
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.link_priority":"PEER2PEER"}'
```

在八卡可见的受控导出步骤中，从当前 HIXL 实际生成的端点导出每卡 `0.json` 至
`7.json`。每卡含 7 个直接对端和 1 个 CLOS 端点。正式 worker 仍保持 P/D 分组可见，
通过已有平台 API 将逻辑卡映射为物理卡，初始化 TransferEngine 前装载对应文件。
不设置目录变量时保留原有行为。协议仍为 `ub_ctp`，没有替换 CANN、HIXL 或 Mooncake。

该变量为实测快照中的实验开关：默认未设置，值为本地目录路径，不含凭据。
若将此补丁提升为通用生产功能，应按仓库规范集中注册并审阅环境变量；本归档保留
已验证版本的原始名称及实现。不要把另一台机器的端点地址照搬到当前服务器。

端点导出脚本见 `reference/transport-resource-export-20261123-01/`。
八 worker 验证脚本见 `reference/transport-worker-validation-20261123-01/`。
真实分组可见下，正式 GlobalTE 入口的 32 个 P↔D 有向组合均通过逐字节校验。
这解决了本实验 Mesh 路径，未单独证明 CLOS 网络故障已修复。

## 启动配置和复现依赖

实测 CANN/HIXL 9.3.0、torch_npu 2.10.0.post7、Mooncake 0.3.14。
P：卡 0–3、TP4/DP1、eager。D：卡 4–7、TP1/DP4、EP、DFly5、FULL_DECODE_ONLY。
BF16 计算、FP8 KV，Engram CPU offload，D 使用共享 pageable 表。

| 参数 | run09 值 |
| --- | --- |
| max_model_len | 131072 |
| max_num_seqs | 128 |
| max_num_batched_tokens | 4096 |
| gpu_memory_utilization | 0.85 |
| hy4flash_cann_engram_rms | true |
| hy4flash_fused_ihc_post | true |
| hy4flash_bulk_c8_decode_reads | true |
| hy4flash_dfly_dp_graph | false |
| enable_mlapo / enable_fused_mc2 | false / 0 |

`reference/run09/run.py` 是实测控制器；它向 docker exec 注入上述两个通信环境变量。
仅执行 `start_prefill.sh` 或 `start_decode.sh` 不会自动获得这两项控制器注入配置。
先独立预热 115200 token 共同前缀，再执行 AISBench C32/N128，seed 1024，不开 profiling。

这些脚本是原环境快照，保留了机器路径、容器名、端口及租约管理接口依赖。
迁移时需准备匹配的 HY4 vLLM 源码、CANN 算子构建、权重、输入数据、
`host.mindie_npu_coordination`、共享内存及对应挂载，并检查路径和端口。
本包的源码快照仅覆盖 vLLM-Ascend Python；不把未打包的外部 vLLM/构建依赖宣称为可独立复现。
输入 SHA-256：`8b46b164426cf173c2315bc20e270c15cb2cab42b12a493d65472cf2d420b9a2`。

## 实测结果与口径

| 指标 | 数值 |
| --- | --- |
| 成功 / 失败 | 128 / 0 |
| 正式统计时长 | 152.9643 s |
| 整机输出吞吐 | 856.8797 token/s |
| P 端平均 prefill，不含排队 | 1.78398 s/request |
| P 端平均排队 | 18.59083 s/request |
| 客户端平均 TTFT | 21.82669 s |
| 客户端平均 TPOT | 12.66193 ms/token |
| TPOT p50 / p90 / p99 | 13.06005 / 13.44229 / 13.49532 ms/token |
| D 端平均 decode | 13.00052 s/request |
| DFly 草稿 token 接受率 | 99.4186% |
| P 正式前缀命中率 | 90% |
| preemption | 0 |

每请求命中 115200 token，新增 prefill 约 12800 token；该 prefill 时间不是
完整 128K 冷请求的计算时间。独立共同前缀预热不计入正式指标。
DFly 每次流事件可包含多个 token，75.3125 ms 的平均流事件间隔不能当作 TPOT。
本轮结束后服务已停止，八卡租约释放。

## 归档验证

`verify_archive.py` 校验归档哈希、补丁还原后的源码、Python 语法、通信配置边界及结果一致性。
在本机有基础仓库时运行：

```bash
python verify_archive.py --base-repo /path/to/vllm-ascend-hy4-flash
```

硬件验证使用本轮已有的 32 方向通信和正式 128 请求结果，整理归档没有重跑性能实验。
归档检查已通过。仓库全量 `bash format.sh ci` 因整理机器缺少 `pre-commit`
未能执行；完整 NPU 单元测试集未在本次归档阶段重跑。
