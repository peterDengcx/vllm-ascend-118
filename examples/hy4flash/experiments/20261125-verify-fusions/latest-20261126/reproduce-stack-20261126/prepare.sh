#!/usr/bin/env bash
set -euo pipefail
# Run inside the recorded CANN 9.3 image. The archive checkout is read-only input.
ARCHIVE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd /workspace/repro
test ! -e ascend-src
test ! -e vllm-src
git clone https://github.com/vllm-project/vllm-ascend.git ascend-src
git -C ascend-src checkout edc7c505fd5be74a99fc68173ede69ce4318da9b
git -C ascend-src apply --check "$ARCHIVE/../hy4-decode-complete.patch"
git -C ascend-src apply "$ARCHIVE/../hy4-decode-complete.patch"
git -C ascend-src apply --check "$ARCHIVE/prefill-stack-resolved.patch"
git -C ascend-src apply "$ARCHIVE/prefill-stack-resolved.patch"
git clone https://github.com/vllm-project/vllm.git vllm-src
git -C vllm-src checkout ba15685ab7f4f5fbed8630bbb7441f8837e76fbe
git -C vllm-src apply --check /workspace/repro/ascend-src/patches/hy4flash-vllm-ba15685.patch
git -C vllm-src apply /workspace/repro/ascend-src/patches/hy4flash-vllm-ba15685.patch
python3 -m venv --system-site-packages venv
# ba15685 is a development revision. A release identity selects incompatible APIs.
(cd vllm-src && VLLM_TARGET_DEVICE=empty SETUPTOOLS_SCM_PRETEND_VERSION=0.33.0.dev0+gba15685 /workspace/repro/venv/bin/python -m pip install --no-deps --no-build-isolation -e .)
(cd ascend-src && COMPILE_CUSTOM_KERNELS=0 /workspace/repro/venv/bin/python -m pip install --no-deps --no-build-isolation -e .)
for lib in vllm_ascend_C.cpython-312-aarch64-linux-gnu.so libengram_hixl.so; do
  ln -s "/vllm-workspace/vllm-ascend/vllm_ascend/$lib" "ascend-src/vllm_ascend/$lib"
done
ln -s /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors ascend-src/vllm_ascend/_cann_ops_custom/vendors
g++ -O3 -std=c++17 -shared -fPIC -fvisibility=hidden ascend-src/vllm_ascend/models/hy_v4_flash/engram_row_lookup_cpu.cpp -o ascend-src/vllm_ascend/models/hy_v4_flash/engram_row_lookup_cpu.so
cp -a "$ARCHIVE/runtime" /workspace/repro/runtime
