# Latest HY4 Decode patch and manual launch entry

This directory updates the November 25 archive with the November 26 Decode
candidate. It adds only the reviewed patch, its source manifest and the manual
launch entry. The surrounding repository's framework sources are unchanged.

## Exact patch base

Apply `hy4-decode-complete.patch` to the separately maintained HY4 Ascend base
`edc7c505fd5be74a99fc68173ede69ce4318da9b`. This archive branch itself does not
contain that base or the HY4 model source, so do not apply the patch to this
archive checkout. Obtain the matching source separately.

```bash
git apply --check /path/to/hy4-decode-complete.patch
git apply /path/to/hy4-decode-complete.patch
```

The patch has 48 changed files, 3596 insertions and 246 deletions. It includes
the retained Decode/Draft implementations, automatic capability selection,
tests, documentation and the restored `use_eager_prefill_rotary` initialization.
Its SHA256 is
`5e92f3e63d02990e82012d29dce275627831720b36d826c17498fe6c8bb0cff2`.
All 48 patched files matched the recorded server source after normalizing
line endings to LF; `source-sha256.json` records their hashes. The patch passed
an application check and actual application on its exact clean base.

This candidate supersedes the earlier archive's optimization flags: only
`hy4flash_dfly_dp_graph` remains as a HY4 optimization boolean. General Ascend
options, Engram storage configuration and transport settings remain. Do not
stack the older archive patches or restore their experimental flags on top.
The two prerequisite commits `b7824a634f8baa5c2de8f3ee4c1927fe6695f747` and
`25757d5f18681e4fda9cb867a6235e5259b5a87e` are ancestors of the exact base.

## Runtime dependencies and manual launch

Use matching HY4 vLLM sources and the existing companion
`patches/hy4flash-vllm-ba15685.patch` from the HY4 source checkout. The recorded
environment used CANN 9.3 and torch-npu 2.10.0.post7. Weights, external custom
operator binaries, native shared libraries and transport descriptors are not
included here. The optional CPU lookup C++ source is included in the patch;
build the shared library for the destination environment when using that path.

`runtime/start_prefill.sh`, `runtime/start_decode.sh` and `runtime/bin/vllm`
preserve the manual startup correction. The shell scripts call their adjacent
entry explicitly instead of resolving the system `vllm` command. The entry
prepends the selected source roots, imports torch-npu before Triton detection,
and checks source selection and RoPE JIT launchability before weight loading.
The container's editable-install `.pth` can otherwise select an older source
even when PYTHONPATH names the candidate. All paths in these scripts are for
server 118; adjust the model, source, CANN and transport paths elsewhere.

Run one service per terminal inside the prepared service container:

```bash
bash runtime/start_prefill.sh
bash runtime/start_decode.sh 0 8200 4 0 127.0.0.1 13445 1 1
bash runtime/start_decode.sh 1 8201 4 1 127.0.0.1 13445 1 1
bash runtime/start_decode.sh 2 8202 4 2 127.0.0.1 13445 1 1
bash runtime/start_decode.sh 3 8203 4 3 127.0.0.1 13445 1 1
```

Start all four D ranks before waiting for distributed initialization. P uses
devices 0-3 with TP4/DP1 eager; D uses devices 4-7 with TP1/PCP1/DP4, DFly5 and
Draft graphs. A separate proxy must route port 8100 and ports 8200-8203 before
clients can access port 19383. Starting P/D alone does not start the proxy.

## Validation boundaries

The completed eight-card PD run had 128/128 successful requests, no failures,
1024 output tokens per request, and four-rank Draft capture/input preparation/
replay evidence. It recorded no preemptions and did not collect profiling.
After restoring the initialization, CPU tests passed 982 cases; prerequisite
checks passed 480 cases with 11 skipped. No paired speedup or model-quality
certification is claimed.

The later manual launch failed in upstream RoPE preparation with a plain
Python function where a Triton launchable kernel was expected. The precise
trigger in those failed workers remains unconfirmed. The corrected entry
passed parent and spawned-child import checks. All four manually started D
APIs subsequently returned HTTP 200, as did P and the separately started proxy;
this is readiness evidence, not another completed end-to-end workload result.
Shell syntax and whitespace checks passed. The full local `format.sh ci`
check could not run because pre-commit was unavailable.
