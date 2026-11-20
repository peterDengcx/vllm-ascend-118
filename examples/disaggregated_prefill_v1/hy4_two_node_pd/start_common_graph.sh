#!/usr/bin/env bash
# One complete TP8 model per host; no cross-host DP group.
set -eo pipefail
ulimit -l unlimited
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
ASCEND_REPO=${ASCEND_REPO:-/workspace/code/vllm-ascend}
VLLM_REPO=${VLLM_REPO:-/workspace/code/vllm}
export PYTHONPATH="$ASCEND_REPO:$VLLM_REPO:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENGINE_READY_TIMEOUT_S=10800
export VLLM_HOST_IP="$NODE_IP" HCCL_IF_IP="$NODE_IP"
export HCCL_SOCKET_IFNAME=${NIC:-enp35s0f2} GLOO_SOCKET_IFNAME=${NIC:-enp35s0f2} TP_SOCKET_IFNAME=${NIC:-enp35s0f2}
export HCCL_BUFFSIZE=128
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=0
export NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
args=(python3 -m vllm.entrypoints.cli.main serve "${MODEL:-/mnt/share/g00887675/Hy4/Hy4-FP8-testing-3}"
 --host 0.0.0.0 --port "$API_PORT"
 --data-parallel-size 1 --data-parallel-size-local 1 --tensor-parallel-size 8
 --served-model-name hy4 --enable-expert-parallel --distributed-executor-backend mp
 --language-model-only --quantization "${QUANTIZATION:-compressed-tensors}"
 --safetensors-load-strategy prefetch
 --max-num-seqs "${MAX_NUM_SEQS:-4}" --max-model-len "${MAX_MODEL_LEN:-4096}"
 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.95
 --kv-cache-memory-bytes 536870912
 --offload-backend prefetch --offload-group-size 4 --offload-num-in-group 1 --offload-prefetch-step 1 --offload-params experts
 --enable-prefix-caching --no-async-scheduling --seed 0
 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'
 --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_load_config":{}}'
 --additional-config '{"weight_nz_mode":0}'
 --kv-transfer-config "{\"kv_connector\":\"MooncakeConnectorV1\",\"kv_buffer_device\":\"npu\",\"kv_role\":\"$KV_ROLE\",\"kv_rank\":$KV_RANK,\"kv_port\":$KV_PORT,\"kv_connector_extra_config\":{\"prefill\":{\"dp_size\":1,\"tp_size\":8},\"decode\":{\"dp_size\":1,\"tp_size\":8}}}")
printf '%q ' "${args[@]}"; printf '\n'
if [ "${1:-}" = --dry-run ]; then exit 0; fi
exec "${args[@]}" "$@"
