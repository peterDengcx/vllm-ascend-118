# HY4 experiment results — Ascend950PR, 650E retest pending

## Scope and labels

These measurements were taken on eight **Ascend950PR** devices on 2026-11-23.
They are **not 650E results**. The proposed changes remain experimental and
must not be merged before the 650E retest described in [RETEST_650E.md](RETEST_650E.md).

`annotations.json` labels the hardware, revision, topology, runtime, workload,
code switches, invalid attempts and comparison limits. `evidence-manifest.json`
records SHA-256 hashes of the published evidence. The source revision is
`2c123f517504fa551379f7f5c7f3c4ac0e8b5686` for both arms.

## C32 / N128 paired performance

Both arms use the same frozen 128 synthetic requests. Each request has 128000
input tokens and 1024 output tokens. A separate 115200-token prefix warmup is
excluded from formal statistics. P uses TP4/DP1 and D uses TP1/DP4, with EP,
DFly 5, BF16 compute and FP8 KV. Both arms use the same image-compatibility
fallbacks, so this is not a measurement of the unmodified upstream environment.

| Metric | Baseline C01 | Candidate C02 |
| --- | ---: | ---: |
| Successful / failed requests | 128 / 0 | 128 / 0 |
| Measured P prefix hit rate | 90.00% | 90.00% |
| Duration, seconds | 1004.7454 | 980.9839 |
| Whole eight-card output throughput, token/s | 130.4530 | 133.6128 |
| Output throughput divided by all eight cards, token/s/card | 16.3066 | 16.7016 |
| Mean TTFT, ms | 213075.8 | 207633.2 |
| Mean TPOT, ms | 8.1 | 8.2 |
| Mean ITL, ms | 48.4 | 48.7 |
| Mean E2E, ms | 221404.8 | 216008.7 |
| P99 TTFT, ms | 241809.5 | 235622.0 |
| P99 E2E, ms | 250138.7 | 243997.6 |

The single-run whole-service output throughput difference is **+2.42%** and
mean TTFT difference is **-2.55%**. TPOT/ITL did not improve. Mean TTFT accounts
for approximately 96% of mean E2E. This does not establish repeatable gain or
attribute the difference to a particular optimization.

Each arm generated 131072 output tokens. Per-request input hashes match across
the two arms. The original AISBench aggregate JSON and latency CSV are under
`results/{baseline,candidate}/`; the recomputed comparison is in
`results/comparison.json`.

## Applicability and cost

- The short IndexPool graph limit is 2048 tokens; the formal 128k requests use
  the long-context path. Startup/warmup replay logs do not prove formal short
  graph use.
- The upstream Ascend head can return from its BF16-input/FP32-output matrix
  multiplication before reaching the cached FP32-weight fallback. Cache
  allocation is not proof of cache use on the main path.
- The FP32 head cache adds 1484783616 bytes (approximately 1.383 GiB) per D
  worker. D KV capacity changes from 339136 to 303744–303808 tokens per worker,
  a reduction of approximately 10.4%.
- D graph capture memory changes from 3.92 GiB to 6.06–6.07 GiB. This and the
  earlier KV-budget log are different accounting stages and are not simply
  added together as an extra-memory total.
- Successful requests and correct lengths do not certify model accuracy.

## Candidate-only C8 / N8 profiling

One identical 128000-token input was replayed eight times with concurrency 8;
all eight generated 1024 tokens successfully. A P-only task-local cache-read
cap of 115200 tokens holds the actual P prefix hit ratio at exactly 90%:
921600 hits / 1024000 queried tokens. This workload-control patch is separate
from the model optimization and was not used for the C32/N128 comparison.

The capture covers P and D with Level1 + PipeUtilization. The first capture,
which lacked the read cap and reached 98.70625% hit rate, is excluded. Native
worker-daemon parsing required an independent offline parser. Four D workers
were parsed; P parsing was resumed on a separate filesystem after the original
output volume exhausted available space. **Full validation and full download
are still pending in this publication snapshot.**

`profiling/decode/dp*/` contains the four parsed D aggregate CSVs and profiler
metadata. `profiling/capture-annotations.json` describes this partial evidence;
multi-GB raw data, timelines and duplicate DB exports are not stored in Git.
Profiler timings must not be compared to the unprofiled C32/N128 throughput.

## Validation status

The six archived Python scripts compile. Git whitespace checks pass with
unified-diff context whitespace explicitly preserved. The repository command
`bash format.sh ci` was attempted but could not run because `pre-commit` is
not installed locally; a full lint pass is not claimed. The hardware experiment
is the recorded Ascend950PR run, not a new NPU run for this archival change.

650E performance, precision and path-use validation remain **not run**.
