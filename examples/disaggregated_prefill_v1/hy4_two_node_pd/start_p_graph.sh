#!/usr/bin/env bash
NODE_IP=${P_IP:-141.61.141.22} API_PORT=18010
KV_ROLE=kv_producer KV_RANK=0 KV_PORT=23000
source "$(dirname "${BASH_SOURCE[0]}")/start_common_graph.sh"
