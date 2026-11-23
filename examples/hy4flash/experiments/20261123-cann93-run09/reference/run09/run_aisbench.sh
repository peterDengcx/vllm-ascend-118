#!/usr/bin/env bash
set -euo pipefail
arm="modified-run09"
R="/home/d00943518/hy4/runs/hy4-prefix90-128k1k-latest-20261123-02"
PY="/home/d00943518/hy4/runs/hy4-prefix90-128k1k-latest-20261123-02/aisdeps:/home/d00943518/qwen35b-c8-complete-20261121/deps:/home/d00943518/hy4/runs/hy4-prefix90-128k1k-latest-20261123-02/aisbench"
test "$(sha256sum "$R/prefix90_128k_frozen.jsonl" | cut -d' ' -f1)" = "8b46b164426cf173c2315bc20e270c15cb2cab42b12a493d65472cf2d420b9a2"
docker exec -e PYTHONPATH="$PY" hy4-118-cann93-ops-1123 python /workspace/hy4-cann93/perf-prefix90-128k1k-20261123-09/launch.py aisbench \
  --models vllm_api_general_stream \
  --datasets prefix90_frozen \
  --mode perf \
  --num-prompts 128 \
  --num-warmups 0 \
  --path /home/d00943518/weights/Hy4-flash-FP8 \
  --trust-remote-code \
  --model-name hy4-flash \
  --url http://127.0.0.1:19383 \
  --max-out-len 1024 \
  --batch-size 32 \
  --stream \
  --request-rate 0 \
  --retry 1 \
  --generation-kwargs '{"temperature":0,"ignore_eos":true}' \
  --work-dir "/workspace/hy4-cann93/perf-prefix90-128k1k-20261123-09/aisbench-results/$arm" \
  2>&1 | tee "/root/hy4-cann93-validate-20261123/perf-prefix90-128k1k-20261123-09/aisbench-$arm.log"
