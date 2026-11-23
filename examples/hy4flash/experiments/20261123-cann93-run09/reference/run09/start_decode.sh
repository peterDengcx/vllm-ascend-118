#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.3.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export PYTHONPATH="/workspace/hy4-cann93/src/vllm-src:/workspace/hy4-cann93/src/vllm-ascend-hy4-flash-2c123f517504fa551379f7f5c7f3c4ac0e8b5686:/workspace/hy4-cann93/perf-prefix90-128k1k-20261123-09:${PYTHONPATH:-}"
export HY4_DFLY_PAGEABLE_ENGRAM=1
set -euo pipefail
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export VLLM_HOST_IP=141.61.81.118
export HCCL_BUFFSIZE=1024
export HCCL_IF_BASE_PORT=47600
export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL=INFO
export NO_PROXY="*" no_proxy="*"
export VLLM_USE_V2_MODEL_RUNNER=1
exec python /workspace/hy4-cann93/perf-prefix90-128k1k-20261123-09/launch.py decode
