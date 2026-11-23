# HY4 IndexPool / LMHead experimental source

This directory preserves the code used for the 2026-11-23 experiment against
`ShySummer/vllm-ascend-hy4-flash` commit
`2c123f517504fa551379f7f5c7f3c4ac0e8b5686`.

**650E retest is pending. Do not merge or enable these experiments by default.**
The measured source system reports **Ascend950PR**, not 650E.

## Code layout

- `patches/indexpool-lmhead.patch`: the measured candidate changes: context-aware
  IndexPool graphs, a shared FP32 LMHead weight cache, and the companion change
  to the external vLLM `logits_processor.py`.
- `patches/runtime-compatibility.patch`: the common CANN/image compatibility
  changes applied to both baseline and candidate. These are environment-specific
  fallbacks, not validated 650E adaptations.
- `reference/`: exact copies of the experiment's data generator, controller,
  result collector, profiling preparation, collection, and offline parser.
- `source-manifest.json`: source hashes, pinned upstream revision, and retest gate.

The reference scripts retain their original machine paths and deployment
assumptions so the measured implementation is traceable. They are **as-run
snapshots**, not a portable deployment entry point. Prepare an isolated runtime,
review paths/container/ports, provide the existing coordinator and compatibility
helpers, and reserve the intended devices before any new workload.

The optimization patch uses the original `overlay/` layout containing the vLLM
and vLLM-Ascend Python packages. The compatibility patch applies inside that
overlay. Neither patch is applied automatically by importing this directory.

## Interpretation and retest gate

The completed C32/N128 experiment used 128000 input tokens, 1024 output tokens,
115200 prewarmed prefix tokens, P TP4/DP1 and D TP1/DP4 with DFly 5.
The single-run whole-service output throughput was 130.4530 token/s for baseline
and 133.6128 token/s for candidate (+2.42%). This is not evidence of stable gain.

The 2048-token short-graph limit does not cover the formal 128k requests. The
upstream NPU head path may bypass the cached FP32 weight. The extra cache costs
about 1.383 GiB per D worker and the observed D KV capacity fell about 10.4%.
Successful requests and token lengths do not certify model accuracy.

Before any merge, repeat baseline/candidate performance and precision validation
on 650E, verify actual code paths, and record hardware/runtime versions and the
common compatibility treatment. The current profiling capture is candidate-only
C8/N8, using one repeated 128k input with an experiment-local prefix-read cap to
hold the measured P cache hit rate at 90%. Parsing/download status is tracked
separately and must not be reported as complete until verified.
