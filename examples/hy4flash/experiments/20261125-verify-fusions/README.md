# HY4 verify input preparation fusions

This archive contains the integrated tested-runtime patch and a narrower patch
for the HY4 target verify input preparation experiments. Both are based on the separately maintained HY4 source
commit `2c123f517504fa551379f7f5c7f3c4ac0e8b5686`; it is not directly
applicable to the parent of this archive commit, which does not contain HY4
model sources. That base commit is not reachable from this repository's
existing parent branch. Obtain the matching HY4 source separately before
applying the patch; cloning this branch alone is insufficient.
`hy4-tested-runtime.patch` includes the target Decode data
fusion, Draft graph and fusions, Engram batching, verify preparation, MoE
AddRMSNorm pair handling, Mooncake communication setup, fixes, and their
runtime dependencies as present in the captured serving source.
The prior Decode data fusion patch is archived separately in
`../20261124-decode-data-fusions/`. These patches overlap: select a single
appropriate patch for a given base; do not apply them on top of one another.
`verify-fusions.patch` is the narrower eight-file verify preparation delta.

The patch contains independently guarded target embedding and RoPE/hash
preparation, fused hash kernels, and CPU-offloaded Engram row preparation with
native CPU lookup, reusable pinned staging, and device scatter. Original
paths are retained for unsupported inputs. Environment controls include
`VLLM_ASCEND_HY4_TARGET_EMBED_FUSION`,
`VLLM_ASCEND_HY4_TARGET_ROPE_HASH_FUSION`, and
`VLLM_ASCEND_HY4_ENGRAM_ROW_PREP`. See the patch for default and compatibility
semantics. Only the row preparation path was enabled in the 16k/1k zero-prefix
profiling referenced below; do not attribute that run to the other controls.

## Apply and build

On an appropriate HY4 checkout, first verify its full source and installed
runtime dependencies against the experiment environment; base commit alone is
not proof that the whole serving environment matches. Review the patch and run:

```sh
git apply --check /path/to/hy4-tested-runtime.patch
git apply /path/to/hy4-tested-runtime.patch
g++ -O3 -std=c++17 -shared -fPIC -fvisibility=hidden \
  vllm_ascend/models/hy_v4_flash/engram_row_lookup_cpu.cpp \
  -o vllm_ascend/models/hy_v4_flash/engram_row_lookup_cpu.so
```

The shared library is built for the deployment host and is not included. The
row optimization falls back if the library is unavailable. Do not overwrite
an existing library without preserving the previous artifact.

The measured Decode configuration sets `HY4_DFLY_PAGEABLE_ENGRAM=1` and loads
`pageable_engram_probe` from the experiment directory. The matching module is
preserved at `runtime/pageable_engram_probe.py` (SHA-256
`293b7a7fd1b15a04dc816e3996209325a755bc65ec3d1681c9de86d306a58ee2`). Add
this archive's `runtime` directory to `PYTHONPATH` for that configuration.
This is a runtime shared-CPU-buffer dependency, not profiler instrumentation.
The environment also used a separate vLLM source tree; matching this archive
alone does not establish compatibility with other vLLM versions.

The 16k/1k and 128k/1k captured Decode runs used the following relevant
configuration: `HY4_DFLY_PAGEABLE_ENGRAM=1`,
`HY4_TARGET_PREP_EXTRA_FUSIONS=0`, `HY4_ENGRAM_TRITON_HASH=0`,
`HY4_ENGRAM_HASH_GRAPH=1`, `VLLM_ASCEND_HY4_ENGRAM_ROW_PREP=1`, DFLY5,
`FULL_DECODE_ONLY`, and `hy4flash_dfly_dp_graph=true`. They enabled
`hy4flash_decode_data_fusions`, `hy4flash_cann_engram_rms`,
`hy4flash_fused_ihc_post`, and `hy4flash_bulk_c8_decode_reads`. Their
`PYTHONPATH` put a separate vLLM source tree before the vLLM-Ascend checkout.
Installed package metadata reported Python 3.12.13, vLLM `0.33.0+empty`,
vLLM-Ascend `0.19.1rc2.dev3793+g7a7ed073c`, torch `2.10.0+cpu`, and
torch-npu `2.10.0.post7`; package labels alone do not identify all source
changes. The separate vLLM tree and deployed CANN/custom binaries are not
contained in this archive.

## Evidence and limitations

The integrated patch covers 34 changed source files compared with the exact
base. A server audit compared 890 Python, C++, header and YAML files under
`vllm_ascend` against that base; all 34 changed or added files are included.
A broader audit of 2,346 relevant code and configuration files under
`vllm_ascend`, `csrc`, and `cmake` found no additional changes or missing
base files in those file types. This audit does not cover other file types, other source repositories,
or every independent historical candidate. The patch was generated from the
source used by the verify-gap experiment;
all 47 files recorded in its service gate matched their captured SHA-256
values, and its 45 Python files passed syntax compilation. On the same server,
the native CPU library was rebuilt from the archived source and loaded
successfully; its SHA-256 matched the experiment binary. The shared binary is
excluded. This captures the tested configuration, not a claim that every
optional branch was exercised or that every change is a speedup. The retained
source includes previous fixes and fallbacks; it is not a collection of every
abandoned experimental candidate.

The row preparation differential gate passed 80 cases with zero maximum
error. On the serving workload, real-weight row comparisons passed three times
per D rank. A 16k/1k, C32, zero prefix-repeat run captured four-rank steady
Decode for a three-second control window with Draft graph and DFLY5 enabled;
the capture was stopped early and is not a complete N128 performance result.
The source patch also contains independent embedding and RoPE/hash controls;
their combined differential gate passed 20 cases in addition to 38 previous
cases. Three profile variants were captured, but their per-rank shape matching
was not established for all comparisons, so no serving speedup is claimed.

No complete model accuracy benchmark or profiler-free, matched 16k A/B is
included. The archived patch is a review and porting artifact; applying it to
the exact base does not substitute for build, runtime, precision and performance
validation on the destination environment.
