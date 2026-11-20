#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="${ASCEND_REPO:-/workspace/code/vllm-ascend}:${VLLM_REPO:-/workspace/code/vllm}:${PYTHONPATH:-}"
export NO_PROXY='*' no_proxy='*'
exec python3 "${ASCEND_REPO:-/workspace/code/vllm-ascend}/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py" --host 0.0.0.0 --port "${PROXY_PORT:-18000}" --prefiller-hosts "${P_IP:-141.61.141.22}" --prefiller-ports 18010 --decoder-hosts "${D_IP:-141.61.141.26}" --decoder-ports 18020 "$@"
