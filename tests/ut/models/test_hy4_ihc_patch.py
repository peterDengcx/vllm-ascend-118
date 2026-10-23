# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.hy_v4.nvidia import hc

from vllm_ascend.patch.worker import patch_hy_v4


def test_hy4_installs_platform_ihc_dispatch():
    assert hc.HYV4HCPreLayer.forward is patch_hy_v4._hc_pre_forward
    assert hc.HYV4HCPostLayer.forward is patch_hy_v4._hc_post_forward


def test_pre_preserves_projection_hooks_when_parameters_are_offloaded():
    x = torch.empty(8, 4, 6144, device="meta", dtype=torch.bfloat16)
    projection = ReplicatedLinear.__new__(ReplicatedLinear)
    torch.nn.Module.__init__(projection)
    projection.weight = torch.nn.Parameter(torch.empty(8, 24576))
    layer = SimpleNamespace(
        hpc_op=None,
        hc_fn=projection,
        hc_scale=torch.empty(2),
        hc_base=torch.empty(8),
    )
    sentinel = object()
    with (
        patch.object(patch_hy_v4, "triton_ihc_supported", return_value=True),
        patch.object(patch_hy_v4, "triton_ihc_pre") as fused,
        patch.object(patch_hy_v4, "_HYV4_HC_PRE_FORWARD", return_value=sentinel) as native,
    ):
        assert hc.HYV4HCPreLayer.forward(layer, x) is sentinel
    native.assert_called_once_with(layer, x)
    fused.assert_not_called()


def test_pre_retains_native_projection_for_replaced_linear_layers():
    layer = SimpleNamespace(hpc_op=None, hc_fn=SimpleNamespace())
    value = torch.empty(8, 4, 6144, device="meta", dtype=torch.bfloat16)
    with (
        patch.object(patch_hy_v4, "_HYV4_HC_PRE_FORWARD", return_value="native") as native,
        patch.object(patch_hy_v4, "triton_ihc_supported") as guard,
    ):
        assert hc.HYV4HCPreLayer.forward(layer, value) == "native"
    native.assert_called_once_with(layer, value)
    guard.assert_not_called()


def test_pre_keeps_existing_hpc_priority():
    layer = SimpleNamespace(hpc_op=Mock())
    x = torch.zeros(1, 4, 8)
    with patch.object(patch_hy_v4, "triton_ihc_supported") as fused_guard:
        assert hc.HYV4HCPreLayer.forward(layer, x) is layer.hpc_op.return_value
    layer.hpc_op.assert_called_once_with(x)
    fused_guard.assert_not_called()


def test_pre_dispatches_resident_parameters_to_fused_kernel():
    projection = ReplicatedLinear.__new__(ReplicatedLinear)
    torch.nn.Module.__init__(projection)
    projection.weight = torch.nn.Parameter(torch.empty(8, 24576, device="meta"))
    layer = SimpleNamespace(
        hpc_op=None,
        hc_fn=projection,
        hc_scale=torch.empty(2, device="meta"),
        hc_base=torch.empty(8, device="meta"),
        magnitude=2.0,
        hc_eps=1e-6,
        layernorm_epsilon=1e-5,
    )
    x = torch.empty(8, 4, 6144, device="meta", dtype=torch.bfloat16)
    with (
        patch.object(patch_hy_v4, "triton_ihc_supported", return_value=True),
        patch.object(patch_hy_v4, "triton_ihc_pre", return_value="fused") as fused,
        patch.object(patch_hy_v4, "_HYV4_HC_PRE_FORWARD") as native,
    ):
        assert hc.HYV4HCPreLayer.forward(layer, x) == "fused"
    fused.assert_called_once_with(x, projection.weight, layer.hc_scale, layer.hc_base, 2.0, 1e-6, 1e-5)
    native.assert_not_called()


@pytest.mark.parametrize("pre_hook", [False, True])
def test_pre_preserves_projection_forward_hooks(pre_hook):
    projection = ReplicatedLinear.__new__(ReplicatedLinear)
    torch.nn.Module.__init__(projection)
    if pre_hook:
        projection.register_forward_pre_hook(lambda *_: None)
    else:
        projection.register_forward_hook(lambda *_: None)
    layer = SimpleNamespace(hpc_op=None, hc_fn=projection)
    x = torch.empty(8, 4, 6144, device="meta", dtype=torch.bfloat16)
    with (
        patch.object(patch_hy_v4, "triton_ihc_supported") as guard,
        patch.object(patch_hy_v4, "_HYV4_HC_PRE_FORWARD", return_value="native") as native,
    ):
        assert hc.HYV4HCPreLayer.forward(layer, x) == "native"
    guard.assert_not_called()
    native.assert_called_once_with(layer, x)


def test_unsupported_ihc_input_keeps_native_pre_post():
    torch.manual_seed(17)
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, 32)
    scale = torch.tensor([0.03, 0.04])
    base = torch.randn(8)
    layer = SimpleNamespace(
        hpc_op=None,
        hc_mult=4,
        hc_eps=1e-6,
        magnitude=2.0,
        layernorm_epsilon=1e-5,
        hc_fn=lambda value: (torch.nn.functional.linear(value, weight), None),
        hc_scale=scale,
        hc_base=base,
    )
    actual, post = hc.HYV4HCPreLayer.forward(layer, x)
    flat = x.flatten(1).float()
    mixes = torch.nn.functional.linear(flat, weight) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + 1e-6
    expected_post = 2 * torch.sigmoid(mixes[:, 4:] * scale[1] + base[4:]) + 1e-6
    expected = (pre.unsqueeze(-1) * x).sum(1).to(x.dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(post, expected_post, atol=0, rtol=0)
    result = hc.HYV4HCPostLayer.forward(SimpleNamespace(hpc_op=None), actual, x, post)
    expected = (post.unsqueeze(-1) * actual.float().unsqueeze(-2) + x.float()).to(x.dtype)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
