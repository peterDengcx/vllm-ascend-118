# 650E retest plan — not run

The current record is from Ascend950PR. No result here establishes 650E
compatibility, accuracy, memory use or performance. Keep the PR in draft and
do not merge until the following evidence is reviewed.

## Environment and matching conditions

1. Record the 650E product name, available devices, memory, driver, firmware,
   CANN, torch, torch_npu, vLLM, vLLM-Ascend, tokenizer and model revisions.
2. Pin the same upstream code and experimental patch hashes. Rebuild native
   binaries for the target; do not copy the Ascend950PR binaries to 650E.
3. Check the actual operator support and choose any necessary compatibility
   treatment before measuring. Apply the same treatment to both arms and record
   every difference from the current environment. The archived image fallbacks
   are not a universal 650E configuration.
4. Reserve eight devices. Retain P TP4/DP1, D TP1/DP4, EP, DFly 5, BF16 compute,
   FP8 KV, max model length 131072 and the same scheduler settings. If a target
   constraint prevents a match, label the new case separately.

## Repeat the performance comparison

1. Reuse the frozen synthetic dataset hash in `annotations.json`; validate the
   actual token lengths with the target tokenizer before sending requests.
2. Cold-start each arm and warm only the 115200-token common prefix outside
   formal statistics. Measure formal cache hit rate from P counter deltas.
3. Run C32/N128, input 128000/output 1024, temperature 0, ignore EOS and streaming.
   Require 128 unique successful requests, correct lengths and AISBench exit 0.
4. Run repeated interleaved baseline/candidate pairs. Report per-run metrics and
   variation; do not choose a single favorable run or assume an acceptance
   threshold that the user has not specified.
5. Report whole eight-card output throughput, duration, mean and percentile
   TTFT/TPOT/ITL/E2E, KV capacity and memory/capture cost.

## Precision and actual-path gates

- Compare baseline/candidate fixed-input outputs and logits using the agreed
  numerical tolerances; cover prefill, decode, long contexts and mixed batches.
- Verify target and DFly reuse of the cached head and any early-return path.
- Verify the short-graph selection boundary and the long/mixed fallback before
  considering the IndexPool path validated. Keep any supplemental boundary
  checks separate from the primary 128k performance case.
- Record NaN/Inf, unsupported operators, failed requests and memory errors.
  Request success alone is not a precision gate.

## Candidate-only profiling and delivery

Repeat the separate C8/N8 candidate-only capture with one repeated 128k input,
1024 output tokens and the explicit 115200-token P cache-read cap. Keep common
prefix warmup and graph compilation outside the capture. Use Level1 +
PipeUtilization, parse outside daemon workers, verify all selected P/D workers,
and reserve sufficient output space before collecting the full window.

Publish completed parsing/download status, checksums and measured P cache rate.
Keep raw/large artifacts outside Git and link their manifests. A reviewer must
confirm the 650E results and precision/path gates before marking the draft PR
ready; creating the PR is not authorization to merge it.
