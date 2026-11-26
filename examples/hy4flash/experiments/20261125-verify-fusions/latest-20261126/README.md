# Complete HY4 Decode fusion patch after withdrawal

This archive contains the cleaned Decode/Draft patch from prepared source commit
`39e5b83b83e07abdfcaf87e80a6d363cd4ca11d1`. It supersedes the earlier patch in this directory.
The private branch remains an archive: framework source files at repository root
are not replaced by the separately maintained HY4 source checkout.

The direct-source counterpart is
[`peterDengcx/vllm-ascend:codex/hy4-cann93-contribution`](https://github.com/peterDengcx/vllm-ascend/tree/codex/hy4-cann93-contribution),
whose checked tip was `8e10cca921622ef494af89731395206a4c3106cc`.
The prepared `39e5b83b8` source is its child. This upload updates only the
`peterDengcx/vllm-ascend-118:codex/hy4-verify-fusions-20261125` patch archive.

## Correspondence with PR 22839

The withdrawn [integration commit](https://github.com/vllm-project/vllm-ascend/pull/22839/changes/3dc154cdf812a7d6e524bc12fd61c0a2956a643c)
is `3dc154cdf812a7d6e524bc12fd61c0a2956a643c`: 41 changed files based on
`25757d5f18681e4fda9cb867a6235e5259b5a87e`.
Revert commit `edc7c505fd5be74a99fc68173ede69ce4318da9b` restores exactly that parent's tree.
The PR head was the revert commit when checked on November 26; this archive
does not restore the withdrawn commit unchanged or update the upstream PR.

In this private repository, the original runtime changes were archived by
commit `576bf0a2c` as `../hy4-tested-runtime.patch`, based on HY4 commit
`2c123f517504fa551379f7f5c7f3c4ac0e8b5686`. That patch covers 34 of the withdrawn
commit's file paths. The withdrawal integration also supplied packaging,
shared-CPU support, documentation and tests. Commit `7f204ac16` later placed
the 48-file complete integration patch in this directory.

The current complete patch has 62 files. Forty of the withdrawn commit's
41 paths still have changes against the restored base. `vllm_ascend/envs.py`
now matches the base: experimental environment controls were removed.
The other 22 paths provide later fixes, runtime dependencies, regression tests,
launch examples and documentation. Relative to the previously archived
complete patch, 26 files differ. See `scope-map.json` for every file and its
relationship to the withdrawn submission.

## Included implementation

- Target Decode: native MLA/FIA context with learned sink, stable C8 candidate
  compaction and gather, graph-time KV lengths, output gate, IndexPool
  RoPE/scoring/candidate assembly, precise iHC and MoE fusion integration.
- DFly: normalization, residual and quantization fusion, target KV/top-k reuse,
  fused graph-external query/metadata preparation and model-only graph replay.
- Engram and verify preparation: batched/triton hashes, lookup/normalization/
  residual fusion, optional native CPU row lookup, pinned staging/scatter,
  target embedding and RoPE/hash preparation and host mask dispatch.
- Dependencies: graph registration and task updates, worker/model-state
  integration, native-source packaging and Mooncake initialization/resource
  handling through the existing connector resource-path setting.
- Supporting tests, current launch entry points, configuration migration notes
  and release notes. Existing Prefill implementation in the base is retained;
  this patch adds no separate Prefill optimization files.

Only `hy4flash_dfly_dp_graph` remains as a HY4 optimization boolean; it defaults
to false. Operator enable/rollback switches and HY4 experiment environment
variables are removed. Supported paths are selected by device/layout capability.
Mooncake uses the existing connector key
`kv_connector_extra_config.ascend_local_comm_res_path`.

## Exact base and application

Apply `hy4-decode-complete.patch` once to the separate HY4 checkout at
`edc7c505fd5be74a99fc68173ede69ce4318da9b`, not to this archive branch. Its tree is also identical
to `25757d5f18681e4fda9cb867a6235e5259b5a87e`. The prerequisite commits
`b7824a634f8baa5c2de8f3ee4c1927fe6695f747` and
`25757d5f18681e4fda9cb867a6235e5259b5a87e` are already included in that base.

```bash
git apply --check /path/to/hy4-decode-complete.patch
git apply /path/to/hy4-decode-complete.patch
```

Do not stack the older runtime/verify patches or the withdrawn commit on top.
The diff is `62 files changed, 5129 insertions(+), 442 deletions(-)`. SHA256:
`3f4a4c846708f9a183acbb073b814f2a7e47cb8b2f1456df4905d346945bca56`.
`source-sha256.json` records the SHA256 of each patched file's Git contents.
Exact-base application reconstructs prepared tree `3d51f5b8c161bdacd3a28360eaf63b307fe440f2`;
all 62 file hashes were checked against the prepared source.

Use matching HY4 vLLM sources and the existing companion
`patches/hy4flash-vllm-ba15685.patch` from that source checkout. Recorded runtime
dependencies include CANN 9.3, torch-npu 2.10.0.post7 and matching CANN transformer
custom operators. Weights, binary custom operators, native shared libraries and
per-device transport descriptors are not included. Build the packaged optional
`engram_row_lookup_cpu.cpp` for the deployment host if native lookup is needed.

## Manual launch

`runtime/start_prefill.sh`, `runtime/start_decode.sh` and `runtime/bin/vllm`
are copied from the prepared source. They preserve server-118 paths: update
the model, CANN, source roots and transport paths for the destination checkout.
The explicit entry registers torch-npu before Triton detection and checks source
selection/RoPE launchability. Start P and all four D ranks in separate terminals:

```bash
bash runtime/start_prefill.sh
bash runtime/start_decode.sh 0 8200 4 0 127.0.0.1 13445 1 1
bash runtime/start_decode.sh 1 8201 4 1 127.0.0.1 13445 1 1
bash runtime/start_decode.sh 2 8202 4 2 127.0.0.1 13445 1 1
bash runtime/start_decode.sh 3 8203 4 3 127.0.0.1 13445 1 1
```

P uses physical devices 0-3, TP4/DP1 eager; D uses physical devices 4-7,
TP1/PCP1/DP4, DFly5 and Draft graphs. A separate proxy must route P port 8100
and D ports 8200-8203; P/D startup alone does not start that proxy.

## Validation of this revision

- 166 focused CPU unit tests passed, including DFly graph/preparation, FIA
  lengths, Decode fusions, host mask scope and Mooncake resource forwarding.
- 72 Triton CPU interpreter cases passed. These are not NPU kernel tests.
- Ruff 0.14.0, repository-configured spelling, Python syntax, shell syntax and
  whitespace checks passed for the cleanup change. All 50 Python
  files in the complete patch also passed syntax compilation.
- Exact-base application and complete result-tree equality passed. Logs and
  details are in `validation/` and `validation.json`.

NPU model accuracy/performance was not rerun for this cleaned revision; the full
repository pre-commit suite was not run. Earlier hardware results apply to their
recorded revisions and do not certify this patch. This upload is an archive,
not an upstream merge or a new claim of model quality or speedup.
