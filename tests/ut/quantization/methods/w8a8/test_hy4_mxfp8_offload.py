# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tests.ut.quantization.conftest_quantization import create_mock_ascend_config, create_mock_vllm_config
from vllm_ascend.quantization.methods.w8a8 import w8a8_mxfp8 as mxfp8


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_absorbed_kv_projection_is_dequantized_before_transpose(dtype):
    config = create_mock_vllm_config()
    config.model_config.dtype = dtype
    layer = nn.Module()
    layer.prefix = "model.layers.0.self_attn.kv_b_proj"
    source = torch.arange(128).reshape(2, 64).float().div(32).to(torch.float8_e4m3fn)
    scales = torch.tensor([[126, 127], [128, 129]], dtype=torch.uint8)
    layer.weight = nn.Parameter(source.clone(), requires_grad=False)
    layer.weight_scale = nn.Parameter(scales.clone(), requires_grad=False)
    expected = source.float() * torch.tensor([[0.5, 1.0], [2.0, 4.0]]).repeat_interleave(32, -1)
    with patch.object(mxfp8, "get_current_vllm_config", return_value=config):
        scheme = mxfp8.AscendW8A8MXFP8DynamicLinearMethod()
        scheme.process_weights_after_loading(layer)
        scheme.process_weights_after_loading(layer)
    assert layer.weight.dtype == dtype
    assert not hasattr(layer, "weight_scale")
    torch.testing.assert_close(layer.weight, expected.to(dtype))


def test_mxfp8_dequantization_rejects_invalid_scales():
    weight = torch.ones(2, 64).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="shape"):
        mxfp8.dequantize_mxfp8(weight, torch.ones(2, 1, dtype=torch.uint8), 32, torch.bfloat16)
    with pytest.raises(ValueError, match="NaN encoding"):
        mxfp8.dequantize_mxfp8(weight, torch.full((2, 2), 255, dtype=torch.uint8), 32, torch.bfloat16)


@pytest.mark.parametrize("prefetch", [False, True])
def test_linear_processed_aliases_follow_offload_ownership(prefetch):
    config = create_mock_vllm_config()
    config.offload_config = SimpleNamespace(offload_backend="prefetch" if prefetch else None)
    layer = nn.Module()
    source = torch.randn(64, 128).to(torch.float8_e4m3fn)
    layer.weight = nn.Parameter(source.clone(), requires_grad=False)
    layer.weight_scale = nn.Parameter(torch.full((64, 4), 127, dtype=torch.uint8), requires_grad=False)
    with (
        patch.object(mxfp8, "get_current_vllm_config", return_value=config),
        patch.object(mxfp8, "maybe_trans_nz", side_effect=lambda tensor, **kwargs: tensor),
    ):
        mxfp8.AscendW8A8MXFP8DynamicLinearMethod().process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight.float(), source.T.float())
    assert hasattr(layer, "_mxfp8_weight_buf") is not prefetch
    assert hasattr(layer, "_mxfp8_scale_buf") is not prefetch


@pytest.mark.parametrize("prefetch", [False, True])
def test_moe_processed_aliases_follow_offload_ownership(prefetch):
    config = create_mock_vllm_config()
    config.offload_config = SimpleNamespace(offload_backend="prefetch" if prefetch else None)
    layer = nn.Module()
    sources = {
        "w13_weight": torch.randn(2, 128, 128).to(torch.float8_e4m3fn),
        "w2_weight": torch.randn(2, 128, 64).to(torch.float8_e4m3fn),
        "w13_weight_scale": torch.full((2, 128, 4), 127, dtype=torch.uint8),
        "w2_weight_scale": torch.full((2, 128, 2), 127, dtype=torch.uint8),
    }
    for name, tensor in sources.items():
        layer.register_parameter(name, nn.Parameter(tensor.clone(), requires_grad=False))

    def transform(weight, scale, **kwargs):
        return weight.transpose(1, 2).contiguous(), scale.transpose(1, 2).contiguous()

    with (
        patch.object(mxfp8, "get_current_vllm_config", return_value=config),
        patch.object(mxfp8, "get_ascend_config", return_value=create_mock_ascend_config()),
        patch.object(mxfp8, "use_cann_megamoe", return_value=False),
        patch.object(mxfp8, "maybe_trans_nz_with_scale", side_effect=transform),
    ):
        scheme = mxfp8.AscendW8A8MXFP8DynamicFusedMoEMethod()
        scheme.use_expert_weight_list = False
        scheme.process_weights_after_loading(layer)
    assert hasattr(layer, "_mxfp8_moe_buffers") is not prefetch
    for name in ("w13_weight", "w2_weight"):
        torch.testing.assert_close(getattr(layer, name).float(), sources[name].transpose(1, 2).float())
