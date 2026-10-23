# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily
from vllm_ascend.ops.triton import hy4_ihc


@pytest.mark.parametrize(
    "shape,dtype,device,contiguous,expected",
    [
        ((1, 4, 6144), torch.bfloat16, "npu", True, True),
        ((256, 6144), torch.bfloat16, "npu", True, True),
        ((0, 4, 6144), torch.bfloat16, "npu", True, False),
        ((257, 4, 6144), torch.bfloat16, "npu", True, False),
        ((8, 4, 4096), torch.bfloat16, "npu", True, False),
        ((8, 2, 6144), torch.bfloat16, "npu", True, False),
        ((8, 4, 6144), torch.float16, "npu", True, False),
        ((8, 4, 6144), torch.float32, "npu", True, False),
        ((8, 4, 6144), torch.bfloat16, "cpu", True, False),
        ((8, 4, 6144), torch.bfloat16, "npu", False, False),
        ((6144,), torch.bfloat16, "npu", True, False),
        ((8, 1, 4, 6144), torch.bfloat16, "npu", True, False),
    ],
)
def test_ihc_routes_only_verified_shapes(shape, dtype, device, contiguous, expected):
    value = SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=SimpleNamespace(type=device),
        is_contiguous=lambda: contiguous,
    )
    with (
        patch.object(hy4_ihc, "HAS_TRITON", True),
        patch.object(hy4_ihc, "current_platform", SimpleNamespace(device_type="npu")),
        patch.object(hy4_ihc.envs, "VLLM_BATCH_INVARIANT", False),
        patch.object(
            hy4_ihc,
            "get_current_hardware_profile",
            return_value=SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED),
        ),
    ):
        assert hy4_ihc.triton_ihc_supported(value) is expected


@pytest.mark.parametrize(
    "triton_available,batch_invariant,a5", [(False, False, True), (True, True, True), (True, False, False)]
)
def test_ihc_routes_unsupported_runtime_to_native(triton_available, batch_invariant, a5):
    value = SimpleNamespace(
        shape=(8, 4, 6144),
        ndim=3,
        dtype=torch.bfloat16,
        device=SimpleNamespace(type="npu"),
        is_contiguous=lambda: True,
    )
    with (
        patch.object(hy4_ihc, "HAS_TRITON", triton_available),
        patch.object(hy4_ihc, "current_platform", SimpleNamespace(device_type="npu")),
        patch.object(hy4_ihc.envs, "VLLM_BATCH_INVARIANT", batch_invariant),
        patch.object(
            hy4_ihc,
            "get_current_hardware_profile",
            return_value=SimpleNamespace(
                device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED if a5 else DeviceAdaptorFamily.STANDARD,
            ),
        ),
    ):
        assert not hy4_ihc.triton_ihc_supported(value)


@pytest.mark.parametrize("channels,gate_dtype", [(2, torch.float32), (4, torch.float16)])
def test_post_falls_back_for_unverified_channels_or_gates(channels, gate_dtype):
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    residual = torch.randn(2, channels, 8, dtype=torch.bfloat16)
    gates = torch.randn(2, channels, dtype=gate_dtype)
    actual = hy4_ihc.triton_ihc_post(x, residual, gates)
    expected = (gates.float().unsqueeze(-1) * x.float().unsqueeze(-2) + residual.float()).to(x.dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
