# Clean-container HY4 smoke reproduction

This directory stacks Prefill commits `821c4a1a4bbbe8314a4aae5b4d75643dccd7250c`
and `bbec6137496e8e2aba801146cec98789e7e7c3f0` on the complete Decode patch in
the parent directory. The second commit depends on the first. Apply the resolved
overlay once; do not also cherry-pick the original commits. Conflicts preserve
the Decode implementation and the layer-specific Prefill metadata fix.

The parent Decode patch remains unchanged. The supplied Prefill commits carry
their own Prefill controls; they are separate from the cleaned Decode patch.

## Environment and sources

The validation uses a new container `hy4-118-clean-smoke-stack-1126`, an independent
source directory `/workspace/repro`, and eight Ascend 950PR devices on server 118.
It reuses image `hy4-cann93:hy4-source-bound-1123`, image ID
`sha256:43f75a3f18b1fa05cc5e0950669e87b228e08867f6a5017502d3b2dbfd60b860`.
The image is a local prerequisite, not a publicly downloadable image.

Dependencies: Python 3.12.13, CANN 9.3, torch 2.10.0+cpu,
torch-npu 2.10.0.post7, triton 3.6.0, triton-ascend 3.6.0.post2,
transformers 5.19.0, mooncake-transfer-engine-npu 0.3.14 and
cann-ops-transformer 1.0.0. Native libraries and custom CANN operators are reused
from this image; this is a clean Python-source validation, not a native rebuild.

Use Ascend base `edc7c505fd5be74a99fc68173ede69ce4318da9b`, the parent
`hy4-decode-complete.patch`, and this directory's `prefill-stack-resolved.patch`.
Use vLLM `ba15685ab7f4f5fbed8630bbb7441f8837e76fbe` plus the resulting Ascend
checkout's `patches/hy4flash-vllm-ba15685.patch`.

`prepare.sh` installs these sources without replacing the image's native
libraries. The actual isolated run extracted exact-commit source archives
instead of cloning; tracked Ascend file contents were checked against Git blob
hashes. No old runtime directory is mounted in the new container.

Do not label vLLM ba15685 as the released `0.33.0`: this development revision
uses main-branch APIs. A release label selects incompatible Kimi and CUDA graph
configuration paths. `prepare.sh` supplies a development version explicitly.

## Host prerequisites

Reserve all eight devices and stop conflicting services before starting.
Create the container with host networking, private IPC, 256 GiB shared memory,
unlimited memlock, device nodes for all eight NPUs and the matching host driver,
DCMI, URMA and HCCL configuration mounts. Bind a new host directory at
`/workspace/repro`; mount the model directory read-only at
`/home/d00943518/weights/Hy4-flash-FP8`.

The model weights and image are external prerequisites. Mooncake also requires
host-specific per-device transport JSON files `0.json` through `7.json` under
`/workspace/repro/communication-resources`. This run copies the already exported
descriptors for the same eight physical devices. Do not reuse those descriptors
on another machine. Export on that machine using the resource export reference
under `../../../20261123-cann93-run09/reference/transport-resource-export-20261123-01/`,
adapting its container and paths. The existing connector setting
`ascend_local_comm_res_path` loads the descriptors; no new environment switch is needed.

## Start and request

Inside the prepared container, start P and all four D ranks concurrently in
separate terminals. P uses devices 0-3; D uses physical devices 4-7 through
visible-device indices 0-3.

```bash
bash /workspace/repro/runtime/start_prefill.sh
bash /workspace/repro/runtime/start_decode.sh 0 18200 4 0 127.0.0.1 13446 1 1
bash /workspace/repro/runtime/start_decode.sh 1 18201 4 1 127.0.0.1 13446 1 1
bash /workspace/repro/runtime/start_decode.sh 2 18202 4 2 127.0.0.1 13446 1 1
bash /workspace/repro/runtime/start_decode.sh 3 18203 4 3 127.0.0.1 13446 1 1
```

This smoke configuration uses an 8192-token context, eight sequences, FP8 KV,
DFLY5 and `hy4flash_dfly_dp_graph=true`. D graph request sizes are 1, 2, 4, 8,
with a 48-token capture limit. It is not the long-context performance setup.

Start the repository's existing PD proxy:

```bash
/workspace/repro/venv/bin/python /workspace/repro/ascend-src/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 127.0.0.1 --port 19385 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 18100 \
  --decoder-hosts 127.0.0.1 127.0.0.1 127.0.0.1 127.0.0.1 \
  --decoder-ports 18200 18201 18202 18203
```

Wait for `/health` to return HTTP 200 on P port 18100 and all four D ports.
Then send a chat request through the proxy using the model's conversation template:

```bash
curl --noproxy '*' http://127.0.0.1:19385/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy4-flash","messages":[{"role":"user","content":"Explain why the sky is blue."}],"max_tokens":256,"temperature":0,"chat_template_kwargs":{"reasoning_effort":"no_think"}}'
```

Success requires HTTP 200, nonempty generated text and positive completion token
count. This check does not measure model accuracy or performance and collects
no profiling data.

Raw `/v1/completions` does not apply the model's chat template. A later request-format
comparison reproduced repetition with a raw prompt and obtained a coherent,
normally terminated response using `messages` on the same service. See
[the diagnostic report](precision-debug/REPORT.md). The initial four raw completion
requests below remain execution-only checks; they do not validate response quality.

## Verified result

The isolated run completed four proxy requests, each with HTTP 200 and 32
completion tokens. All five service health checks returned 200. All four D
workers completed capture; target and DFly draft replay messages were recorded.
The focused Prefill/Draft regression suite passed 59 tests. This is a startup
and execution check; generated content was not evaluated for accuracy.

Resolved source commit: `9273c044deb48719ed7011bc2da2860699bff0f9`.
All 4931 tracked Ascend blob contents matched this source. The clean archive
omits the Catlass submodule gitlink; native operators are reused as described
above. The overlay also initializes the merged Decode field and forward context
in the layer-metadata unit-test fixture; it changes no runtime feature switch.

`evidence/` contains commands, dependency/native-library hashes, health checks,
four request/response pairs, source verification, graph log excerpts and test
output. The first installation log records the initial release version label;
before successful startup the generated, untracked `vllm/_version.py` was
corrected to `0.33.0.dev0+gba15685.empty`. `prepare.sh` sets this development
identity at installation time. The running service is retained for manual curl
testing at the user's request.
