# HY4 precision fixes — hold upstream integration

**精度问题已修复；暂不合入主仓。** 本次归档来自容器
`hy4-pr22839-363c1616b-1127`，追加到昨天的私有补丁归档分支。
修复范围为 Target Attention、MoE combine/norm 与 IndexPool 的已定位数值和候选集问题。
尚无完整 AIME2026 最终分数，不能把组件修复表述为整网测评分数已恢复。
本次不更新或合入主仓 PR 22839，也不启用自动合并。

## 修复内容

- Target Attention：C8 reader 保留 FP32 KV，FP32 QK 后使用融合 sink/softmax/PV。
  该路径替换了不能满足这些精度边界的原 CANN MLA FIA 调用。
- MoE：恢复 BF16 expert sum 的中间舍入，再调用 CANN AddRMSNorm。
- IndexPool：保留 pool/norm 融合，恢复 key 位置采样、RoPE 和 FP8 编码；
  为短上下文补齐合法 dense 历史候选，padding 行保持无效。
- 保留其余 Decode/Draft 融合、DFLY5 和 Draft 图模式；本轮没有修复
  Engram rstd 或 iHC head 的残余数值差异。

## 获取和应用

本分支保存补丁归档；请在独立 HY4 源码仓库应用，不要对本归档仓库根目录应用。
[固定基线源码](https://github.com/LostFox11/vllm-ascend/tree/363c1616b1dbbdcd12399301dfaa50b48753adc9)
是容器候选的来源；`source/` 包含本轮全部 10 个变更文件（含测试和 release note）。
`manifest.json` 保存精确基线和文件/补丁 SHA256。

在基线 `363c1616b1dbbdcd12399301dfaa50b48753adc9` 上只应用增量补丁：

```bash
git apply --check /path/to/hy4-precision-fix.patch
git apply /path/to/hy4-precision-fix.patch
```

从昨日回退基线 `edc7c505fd5be74a99fc68173ede69ce4318da9b` 开始时，
改用 `hy4-complete-fixed.patch`，它包含到固定基线的集成及本轮修复：

```bash
git apply --check /path/to/hy4-complete-fixed.patch
git apply /path/to/hy4-complete-fixed.patch
```

两种方式选择一种。不要叠加本目录两份补丁，也不要再叠加旧的 Decode 补丁、
旧重构补丁或 Prefill overlay。完整补丁包含固定基线已有的 Prefill 集成。
继续使用匹配的 vLLM 源码及 `patches/hy4flash-vllm-ba15685.patch`、
CANN 9.3、torch-npu 2.10.0.post7 和对应自定义算子。
权重、原生二进制及机器专属通信资源不在本次归档内。

## 已有验证及边界

- NPU 组件记录：34 项通过；MoE 的 24 组及图回放逐位一致，4 组 C8 reader
  对照逐位一致。Attention 相对 L2 约 0.001%–0.005%，图回放约 0.0064%，并非逐位一致。
- IndexPool 新增 11 项回归在修复前全部失败、修复后通过；联合相关测试
  `73 passed, 72 skipped`。跳过项不计作 NPU 检查。
- 新容器内 6 个核心实现文件已在本次交付时核对 SHA256；其中 decode_fusions.py
  仅有 CRLF/LF 换行差异，统一为 LF 后与归档源码一致，其余 5 个文件字节一致。
  容器内是源码导出目录，没有 `.git`；基线来自本地工作树及既有交付记录。
- `validation/` 保留组件结果、IndexPool 修复前后日志和校验记录。
  IndexPool 两项修复尚无独立 NPU 数值对照；完整 AIME2026 与性能验收仍需最终结果。

组件 NPU 记录对应修复过程中的较早源码（原始 SHA256 见
`validation/integration-done.json`），不是对当前最终候选全部文件的复测。
最终候选的 Attention tile 为 64；不能把这些历史组件结果当作最终整网验收。

本次交付验证：两份补丁均通过独立临时 Git index 应用校验，结果树完全一致，
10 个变更文件的补丁结果与候选源码在统一换行后全部一致。
本地执行 `bash format.sh ci` 因缺少 pre-commit 未能运行完整检查；
保留已有 Ruff/回归记录，不将其宣称为本次全仓检查通过。

本次仅交付已定位精度修复及可复核证据，保持暂缓合入主仓。
