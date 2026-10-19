# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import subprocess
import sys
from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import _load_memfabric_offload


def test_sparse_kv_module_import_does_not_provision_memfabric():
    script = """
import builtins
original_import = builtins.__import__
attempts = []
def guarded_import(name, *args, **kwargs):
    if name.startswith('memfabric_hybrid'):
        attempts.append(name)
        raise ImportError('Do not provision AICPU kernels in this test')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager
assert not attempts, attempts
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=90)


def test_memfabric_is_imported_when_offload_is_requested(monkeypatch):
    expected = object()
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", SimpleNamespace(offload=expected))
    assert _load_memfabric_offload() is expected
