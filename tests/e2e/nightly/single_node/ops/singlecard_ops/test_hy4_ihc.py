# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile
from vllm_ascend.ops.triton.hy4_ihc import triton_ihc_post, triton_ihc_pre, triton_ihc_supported

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available()
    or not HAS_TRITON
    or get_current_hardware_profile().device_adaptor_family != DeviceAdaptorFamily.FP8_OPTIMIZED,
    reason="A5 NPU with Triton required",
)


def reference_pre(x, weight, scale, base):
    flattened = x.flatten(1).float()
    normalized_mixes = torch.nn.functional.linear(flattened, weight) * torch.rsqrt(
        flattened.square().mean(-1, keepdim=True) + 1e-5
    )
    pre = torch.sigmoid(normalized_mixes[:, :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * torch.sigmoid(normalized_mixes[:, 4:] * scale[1] + base[4:]) + 1e-6
    return (pre.unsqueeze(-1) * x).sum(1).to(x.dtype), post


def inputs(batch):
    torch.manual_seed(927)
    x = torch.randn(batch, 4, 6144, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(8, 24576, device="npu", dtype=torch.float32) * 0.05
    scale = torch.tensor([0.04, 0.03], device="npu")
    base = torch.cat([torch.full((4,), -1.1), torch.zeros(4)]).npu()
    return x, weight, scale, base


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 64, 256])
def test_ihc_pre_post_matches_native(batch):
    x, weight, scale, base = inputs(batch)
    actual, post = triton_ihc_pre(x, weight, scale, base, 2.0, 1e-6, 1e-5)
    expected, expected_post = reference_pre(x, weight, scale, base)
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.008)
    torch.testing.assert_close(post, expected_post, atol=2e-6, rtol=1e-5)
    difference = actual.float() - expected.float()
    relative_rmse = difference.square().mean().sqrt() / expected.float().square().mean().sqrt()
    assert relative_rmse.item() < 2e-4
    value = torch.randn(batch, 6144, device="npu", dtype=torch.bfloat16)
    actual_post = triton_ihc_post(value, x, post)
    expected_post = (post.unsqueeze(-1) * value.float().unsqueeze(-2) + x.float()).to(x.dtype)
    torch.testing.assert_close(actual_post, expected_post, atol=0, rtol=0)


def test_ihc_graph_replay_reads_updated_input():
    x, weight, scale, base = inputs(8)
    triton_ihc_pre(x, weight, scale, base, 2.0, 1e-6, 1e-5)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, post = triton_ihc_pre(x, weight, scale, base, 2.0, 1e-6, 1e-5)
        residual = triton_ihc_post(output, x, post)
    for amplitude in [0.1, 1.0, 4.0]:
        x.copy_(torch.randn_like(x) * amplitude)
        graph.replay()
        expected, expected_post = reference_pre(x, weight, scale, base)
        expected_residual = (expected_post.unsqueeze(-1) * expected.float().unsqueeze(-2) + x.float()).to(x.dtype)
        torch.testing.assert_close(output, expected, atol=0.015625 * amplitude, rtol=0.008)
        torch.testing.assert_close(post, expected_post, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(residual, expected_residual, atol=0.03125 * amplitude, rtol=0.008)


def test_ihc_support_guard_keeps_prefill_and_other_shapes_native():
    assert triton_ihc_supported(torch.empty(8, 4, 6144, device="npu", dtype=torch.bfloat16))
    assert not triton_ihc_supported(torch.empty(257, 4, 6144, device="npu", dtype=torch.bfloat16))
    assert not triton_ihc_supported(torch.empty(8, 4, 6144, device="npu", dtype=torch.float16))
    assert not triton_ihc_supported(torch.empty(8, 4, 4096, device="npu", dtype=torch.bfloat16))
    assert not triton_ihc_supported(torch.empty(8, 4, 6144, dtype=torch.bfloat16))
