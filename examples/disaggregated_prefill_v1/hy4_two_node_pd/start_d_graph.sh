#!/usr/bin/env bash
NODE_IP=${D_IP:-141.61.141.26} API_PORT=18020
KV_ROLE=kv_consumer KV_RANK=1 KV_PORT=24000
source "$(dirname "${BASH_SOURCE[0]}")/start_common_graph.sh"
