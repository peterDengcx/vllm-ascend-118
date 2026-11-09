# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase

from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.patch.hy4_vl.fp4 import (
    Hy4Fp4BlockLinearMethod,
    Hy4Fp4Config,
    Hy4Fp4MoEMethod,
    prepare_hy4_fp4_config,
)
from vllm_ascend.patch.worker import patch_hy_v4
from vllm_ascend.quantization.configs.fp8_config import AscendFp8Config
from vllm_ascend.quantization.configs.modelopt_mxfp8_config import AscendModelOptMxFp8Config
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig
from vllm_ascend.quantization.method_adapters import AscendLinearMethod


def make_config(store_dtype="mxfp4", scale_fmt="ue8m0", nested=False):
    metadata = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "expert_dtype": store_dtype,
        "scale_fmt": scale_fmt,
    }
    quant_config = AscendFp8Config.from_config(metadata)
    text = SimpleNamespace(quantization_config=metadata)
    hf_config = SimpleNamespace(text_config=text) if nested else text
    return SimpleNamespace(
        quant_config=quant_config, model_config=SimpleNamespace(hf_config=hf_config, dtype=torch.bfloat16)
    )


@pytest.mark.parametrize("store_dtype", [None, "fp8", "mxfp8"])
@pytest.mark.parametrize("scale_fmt", [None, "float32", "ue8m0"])
def test_fp8_constructor_keeps_original_config_and_context(store_dtype, scale_fmt):
    config = make_config(store_dtype, scale_fmt)
    original_state = vars(config.quant_config).copy()
    constructor = Mock(return_value="constructed")
    owner = object()
    with patch.object(patch_hy_v4, "set_current_vllm_config", side_effect=AssertionError("FP8 context changed")):
        result = patch_hy_v4._isolate_hy4_fp4_init(constructor)(owner, vllm_config=config, prefix="draft")
    assert result == "constructed"
    constructor.assert_called_once_with(owner, vllm_config=config, prefix="draft")
    assert prepare_hy4_fp4_config(config) is config
    assert vars(config.quant_config) == original_state


@pytest.mark.parametrize("backend", [None, "modelslim", "modelopt"])
def test_other_quantization_backends_keep_their_config(backend):
    config = make_config()
    if backend == "modelslim":
        config.quant_config = AscendModelSlimConfig({})
    elif backend == "modelopt":
        config.quant_config = AscendModelOptMxFp8Config.from_config(
            {
                "quant_method": "modelopt",
                "quantization": {"quant_algo": "MXFP8", "kv_cache_quant_algo": None, "exclude_modules": []},
            }
        )
    else:
        config.quant_config = None
    constructor = Mock()
    owner = object()
    with patch.object(patch_hy_v4, "set_current_vllm_config", side_effect=AssertionError("FP8 context changed")):
        patch_hy_v4._isolate_hy4_fp4_init(constructor)(owner, vllm_config=config)
    constructor.assert_called_once_with(owner, vllm_config=config, prefix="")
    assert prepare_hy4_fp4_config(config) is config


@pytest.mark.parametrize("store_dtype", ["mxfp4", "fp4"])
@pytest.mark.parametrize("nested", [False, True])
def test_fp4_constructor_uses_a_private_copy(store_dtype, nested):
    config = make_config(store_dtype, nested=nested)
    original = config.quant_config
    original_state = vars(original).copy()
    constructor = Mock()
    owner = object()
    with patch.object(patch_hy_v4, "set_current_vllm_config", return_value=nullcontext()) as context:
        patch_hy_v4._isolate_hy4_fp4_init(constructor)(owner, vllm_config=config, prefix="model")
    local = constructor.call_args.kwargs["vllm_config"]
    assert local is not config and local.model_config is config.model_config
    assert type(local.quant_config) is Hy4Fp4Config
    assert local.quant_config is not original and local.quant_config.is_scale_e8m0
    assert config.quant_config is original and vars(original) == original_state
    context.assert_called_once_with(local)


def test_explicit_fp8_store_dtype_takes_precedence_over_fp4_alias():
    config = make_config()
    metadata = config.model_config.hf_config.quantization_config
    metadata["store_dtype"] = "fp8"
    config.quant_config = AscendFp8Config.from_config(metadata)
    assert prepare_hy4_fp4_config(config) is config


@pytest.mark.parametrize("scale_fmt", [None, "float32"])
def test_fp4_rejects_incompatible_scale_metadata(scale_fmt):
    with pytest.raises(ValueError, match="scale_fmt=ue8m0"):
        prepare_hy4_fp4_config(make_config(scale_fmt=scale_fmt))


@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"])
def test_native_fp8_still_selects_the_public_linear_scheme(scale_fmt):
    config = make_config("fp8", scale_fmt)
    layer = Mock(spec=LinearBase)
    scheme = Mock()
    with (
        patch("vllm_ascend.quantization.configs.fp8_config.get_scheme_class", return_value=scheme) as registry,
        patch("vllm_ascend.quantization.method_adapters.AscendLinearMethod", return_value="original-fp8"),
    ):
        assert config.quant_config.get_quant_method(layer, "model.layers.1.self_attn.q_b_proj") == "original-fp8"
    registry.assert_called_once_with("fp8", "linear")
    scheme.assert_called_once_with([128, 128])


def test_fp4_linear_dispatch_is_private():
    config = prepare_hy4_fp4_config(make_config()).quant_config
    with patch("vllm_ascend.patch.hy4_vl.fp4.Hy4Fp4BlockLinearMethod") as scheme:
        method = config.get_quant_method(Mock(spec=LinearBase), "model.layers.78.self_attn.q_b_proj")
    assert isinstance(method, AscendLinearMethod)
    assert method.quant_method is scheme.return_value
    scheme.assert_called_once_with([128, 128])


@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"])
def test_native_fp8_still_selects_the_public_moe_scheme(scale_fmt):
    config = make_config("fp8", scale_fmt).quant_config
    layer = Mock(spec=RoutedExperts)
    layer.moe_config = Mock()
    scheme = Mock()
    with (
        patch("vllm_ascend.quantization.configs.fp8_config.get_scheme_class", return_value=scheme) as registry,
        patch("vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod", return_value="original-fp8-moe"),
    ):
        assert config.get_quant_method(layer, "model.layers.1.mlp.experts") == "original-fp8-moe"
    registry.assert_called_once_with("fp8", "moe")
    scheme.assert_called_once_with([128, 128], layer.moe_config)


def test_fp4_moe_dispatch_is_private():
    config = prepare_hy4_fp4_config(make_config()).quant_config
    layer = Mock(spec=RoutedExperts)
    layer.moe_config = Mock()
    tid2eid = {0: 1}
    with (
        patch("vllm_ascend.patch.hy4_vl.fp4.Hy4Fp4MoEMethod") as scheme,
        patch("vllm_ascend.quantization.method_adapters.AscendFusedMoEMethod", return_value="private-fp4") as adapter,
    ):
        assert config.get_quant_method(layer, "model.layers.1.mlp.experts", tid2eid) == "private-fp4"
    adapter.assert_called_once_with(scheme.return_value, layer.moe_config, tid2eid=tid2eid)


def test_fp4_ignored_linear_keeps_unquantized_weights():
    config = prepare_hy4_fp4_config(make_config()).quant_config
    prefix = "model.layers.0.self_attn.q_b_proj"
    config.ignored_layers = [prefix]
    with patch("vllm_ascend.patch.hy4_vl.fp4.Hy4Fp4BlockLinearMethod", side_effect=AssertionError("ignored layer")):
        assert isinstance(config.get_quant_method(Mock(spec=LinearBase), prefix), AscendUnquantizedLinearMethod)


def test_fp4_ignored_moe_keeps_unquantized_weights():
    config = prepare_hy4_fp4_config(make_config()).quant_config
    prefix = "model.layers.1.mlp.experts"
    config.ignored_layers = [prefix]
    layer = Mock(spec=RoutedExperts)
    layer.moe_config = Mock()
    with patch("vllm_ascend.ops.fused_moe.routed_experts.AscendUnquantizedFusedMoEMethod", return_value="unquantized"):
        assert config.get_quant_method(layer, prefix) == "unquantized"


@pytest.mark.parametrize("rows,cols", [(128, 128), (130, 192), (2624, 128)])
def test_fp4_block_scale_layout_preserves_bytes_and_partial_tiles(rows, cols):
    scheme = Hy4Fp4BlockLinearMethod.__new__(Hy4Fp4BlockLinearMethod)
    scheme.block_rows = scheme.block_cols = 128
    scheme.group_size = 32
    scheme.model_dtype = torch.bfloat16
    scheme.prefetch_offload = False
    raw = torch.arange(((rows + 127) // 128) * ((cols + 127) // 128), dtype=torch.uint8)
    raw = raw.reshape((rows + 127) // 128, (cols + 127) // 128) + 120
    layer = torch.nn.Module()
    layer.prefix = "model.layers.78.self_attn.q_b_proj"
    layer._fused_preprocess_managed = True
    weight = torch.ones(rows, cols, dtype=torch.float8_e4m3fn)
    layer.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
    layer.register_parameter(
        "weight_scale_inv", torch.nn.Parameter(raw.view(torch.float8_e8m0fnu), requires_grad=False)
    )
    scheme.process_weights_after_loading(layer)
    restored = layer.weight_scale.transpose(0, 1).reshape(rows, cols // 32)
    row_indices = torch.arange(rows) // 128
    group_indices = torch.arange(cols // 32) // 4
    torch.testing.assert_close(restored, raw[row_indices[:, None], group_indices[None, :]], rtol=0, atol=0)
    torch.testing.assert_close(layer.weight.float(), weight.transpose(0, 1).float(), rtol=0, atol=0)
    assert not hasattr(layer, "weight_scale_inv")


@pytest.mark.parametrize("exponent", [0, 126])
def test_fp4_kv_projection_dequantizes_to_model_dtype(exponent):
    scheme = Hy4Fp4BlockLinearMethod.__new__(Hy4Fp4BlockLinearMethod)
    scheme.block_rows = scheme.block_cols = 128
    scheme.group_size = 32
    scheme.model_dtype = torch.bfloat16
    layer = torch.nn.Module()
    layer.prefix = "model.layers.78.self_attn.kv_b_proj"
    layer.register_parameter("weight", torch.nn.Parameter(torch.ones(130, 128, dtype=torch.float8_e4m3fn), False))
    raw = torch.tensor([[exponent], [128]], dtype=torch.uint8)
    layer.register_parameter("weight_scale_inv", torch.nn.Parameter(raw.view(torch.float8_e8m0fnu), False))
    scheme.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.bfloat16
    torch.testing.assert_close(
        layer.weight[:128], torch.full((128, 128), 2.0 ** (exponent - 127), dtype=torch.bfloat16), rtol=0, atol=0
    )
    torch.testing.assert_close(layer.weight[128:], torch.full((2, 128), 2.0, dtype=torch.bfloat16))


def test_fp4_expert_scales_are_uint8():
    scheme = Hy4Fp4MoEMethod.__new__(Hy4Fp4MoEMethod)
    scheme.group_size = 32
    params = scheme.get_dynamic_quant_param(2, 128, 256, torch.bfloat16)
    assert params["w13_weight_scale"].shape == (2, 256, 8)
    assert params["w2_weight_scale"].shape == (2, 256, 4)
    assert all(tensor.dtype == torch.uint8 for tensor in params.values())
