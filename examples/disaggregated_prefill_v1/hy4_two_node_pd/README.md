# HY4 two-host Mooncake PD with DFly

P: 141.61.141.22:18010; D: 141.61.141.26:18020; proxy: P:18000.
Each host runs DP1/TP8/EP8. Override MODEL, NIC, ASCEND_REPO, VLLM_REPO,
P_IP and D_IP as needed. Run each role inside its own Ascend container.

- Eager: start_p.sh / start_d.sh.
- Graph: start_p_graph.sh / start_d_graph.sh; FULL_DECODE_ONLY target graphs,
  DFly enabled with three draft tokens. The DFly draft network remains eager.
- Start start_proxy.sh after both /health endpoints return 200.
- smoke_test.py exercises chat streaming and nonstreaming via PD.
- extended_test.py exercises a 297-token prompt and concurrent requests, and
  reports direct-vs-PD exact text equality separately from transport success.

Prefetch offload selects only expert parameters: one layer out of each four,
one-step lookahead. Graph requires pinned host weights; set
VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=0. Pageable H2D copies are not
capturable in CANN. The resulting 107025 unjoined-stream error is secondary.
Use weight_nz_mode=0 to avoid the separate ND-to-NZ conversion limitation.
The small graph_offload_repro.py pinned test captures the circular prefetch
stream and verifies replay observes changing host weights. It uses less than
10 MiB device memory. Run with `python3 graph_offload_repro.py pin` and pinned
weight offload enabled. The unpin variant intentionally reproduces CANN failure.

Current coverage is PP1, text-only, this checkpoint and matching TP8 layouts.
Exact direct/PD text equality was not established in the eager baseline; a
first-token logprob difference existed before subsequent drafting steps.
Graph success alone does not establish accuracy parity or long-run stability.

## Validation on 2026-11-20

- Seven offloader unit tests passed.
- Pageable micro-repro failed at H2D with CANN's explicit unsupported-capture
  message, followed by secondary capture_end 107025.
- Pinned micro-repro passed capture and replay after changing CPU weight data.
- Both TP8 services captured FULL_DECODE graphs with DFly still enabled.
- PD chat nonstream and stream completed, including SSE [DONE].
- D workers logged ACL graph replay is active alongside successful KV pulls
  and nonzero accepted draft tokens.

This validates target-model graphs plus eager DFly drafting, not captured
DFly draft-network execution. The latter remains disabled by the HY4 proposer.
