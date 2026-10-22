# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

from tests.ut.quantization.conftest_quantization import create_mxfp_moe_layer
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.attention.utils import PreprocessType
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.prepare_finalize import MoEPrepareOutput
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicFusedMoEMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("scale_alg", [0, 1])
def test_mxfp8_prolog_requires_compatible_internal_quantization(scale_alg):
    scheme = AscendW8A8MXFP8DynamicLinearMethod.__new__(AscendW8A8MXFP8DynamicLinearMethod)
    scheme.dynamic_mx_quant_scale_alg = scale_alg
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.fused_qkv_a_proj = SimpleNamespace(quant_method=SimpleNamespace(quant_method=scheme))
    impl.q_proj = SimpleNamespace(_chunk_size=0)
    impl.q_a_layernorm = impl.kv_a_layernorm = Mock()
    impl.qk_rope_head_dim = 64
    impl.has_indexer = False
    impl.is_kv_consumer = False
    impl.enable_sparse_sfa_c8 = False
    impl.enable_sparse_sfa_turboquant = False
    impl.enable_mlapo = False

    assert impl._fused_preprocess_type() == (PreprocessType.PROLOG_V3 if scale_alg == 0 else None)
    if scale_alg == 1:
        assert impl._resolve_preprocess_type(torch.bfloat16) == PreprocessType.NATIVE


def test_mxfp8_ceil_scales_avoid_fused_activation_without_clipping():
    scheme = AscendW8A8MXFP8DynamicFusedMoEMethod.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
    scheme.dynamic_mx_quant_scale_alg = 1
    projected = torch.tensor([[1.875, -1.875, 7.5, -7.5]])
    scheme.apply_gmm1 = Mock(return_value=projected)
    scheme.apply_act_quant = Mock(side_effect=lambda _, x: (x, None))
    with (
        patch("torch_npu.npu_swiglu", side_effect=lambda x: F.silu(x[:, :2]) * x[:, 2:]),
        patch(
            "torch_npu.npu_grouped_matmul_swiglu_quant_v2", side_effect=AssertionError("ceil scales are unsupported")
        ),
    ):
        actual, _ = scheme.apply_gmm1_act_quant(SimpleNamespace(swiglu_limit=0.0))

    torch.testing.assert_close(actual, F.silu(projected[:, :2]) * projected[:, 2:])


def test_mxfp8_input_quantization_uses_requested_scale_rule():
    scheme = AscendW8A8MXFP8DynamicFusedMoEMethod.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
    scheme.dynamic_mx_quant_scale_alg = 1
    source = torch.full((2, 64), 1.875)
    quantized = source.to(torch.float8_e4m3fn)
    scales = torch.full((2, 1, 2), 120, dtype=torch.uint8)
    with (
        patch("torch_npu.npu_dynamic_mx_quant", return_value=(quantized, scales)) as quantize,
    ):
        scheme._quant_hidden_states(source)
    quantize.assert_called_once_with(source, dst_type=torch.float8_e4m3fn, scale_alg=1)


@pytest.mark.parametrize("mode,expected", [(None, True), (0, False), (4, True)])
def test_explicit_unquantized_dispatch_preserves_mxfp_activation_precision(mode, expected):
    assert MoEQuantParams(quant_type=QuantType.W8A8MXFP, comm_quant_mode=mode).dispatch_with_quant is expected


def test_hy4_mxfp8_moe_requests_unquantized_dispatch_for_ceil_scales():
    scheme = AscendW8A8MXFP8DynamicFusedMoEMethod.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
    scheme.dynamic_mx_quant_scale_alg = 1
    scheme.use_expert_weight_list = False
    layer = SimpleNamespace(
        ascend_expert_map=torch.tensor([0, 1]),
        global_redundant_expert_num=0,
        ascend_mc2_mask=None,
        apply_router_weight_on_input=False,
        ascend_pertoken_scale=None,
        activation="silu",
    )
    communication = Mock()
    with patch(
        "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8._EXTRA_CTX",
        SimpleNamespace(moe_comm_method=communication),
    ):
        scheme.apply(layer, torch.ones(2, 64), torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int32), None, None)

    payload = communication.fused_experts.call_args.kwargs["fused_experts_input"]
    assert payload.quant.comm_quant_mode == 0
    assert not payload.quant.dispatch_with_quant


@pytest.mark.parametrize("scale_alg", [0, 1])
def test_fused_mc2_falls_back_before_prepare_for_ceil_scales(scale_alg):
    source = torch.ones(2, 64)
    quant_method = Mock()
    quant_method.quant_method = SimpleNamespace(quant_type=QuantType.W8A8MXFP, dynamic_mx_quant_scale_alg=scale_alg)
    quant_method.apply.return_value = FusedExpertsResult(routed_out=source)
    layer = SimpleNamespace(
        enable_npugraph_ex_static_kernel=False,
        moe_config=SimpleNamespace(is_sequence_parallel=False),
        quant_type=QuantType.W8A8MXFP,
        quant_method=quant_method,
        router=object(),
        _select_experts=Mock(return_value=(torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int32))),
        _use_v2_model_runner=False,
        dynamic_eplb=False,
        return_with_event=False,
    )
    context = SimpleNamespace(moe_comm_type=MoECommType.FUSED_MC2, in_profile_run=False)
    comm = Mock()
    comm.prepare.return_value = MoEPrepareOutput(
        hidden_states=source, router_logits=source, mc2_mask=None, padded_hidden_states_shape=None
    )
    comm.finalize.return_value = source
    with (
        patch("vllm_ascend.ops.fused_moe.routed_experts.get_forward_context", return_value=SimpleNamespace()),
        patch("vllm_ascend.ops.fused_moe.routed_experts._EXTRA_CTX", context),
        patch("vllm_ascend.ops.fused_moe.routed_experts.get_moe_comm_method", return_value=comm) as select,
    ):
        result = AscendRoutedExperts.forward_impl(layer, hidden_states=source, router_logits=source)

    select.assert_called_once_with(MoECommType.ALLTOALL if scale_alg == 1 else MoECommType.FUSED_MC2, layer.moe_config)
    assert context.moe_comm_method is comm
    comm.prepare.assert_called_once()
    comm.finalize.assert_called_once()
    assert result is source


@pytest.mark.parametrize("expert_lists", [False, True])
def test_ceil_mxfp8_weights_keep_staged_layout_when_megamoe_is_enabled(expert_lists):
    scheme = AscendW8A8MXFP8DynamicFusedMoEMethod.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
    scheme.dynamic_mx_quant_scale_alg = 1
    scheme.use_expert_weight_list = expert_lists
    scheme.prefetch_offload = False
    layer = create_mxfp_moe_layer(num_experts=2, hidden_size=64, intermediate_size=64)
    with (
        patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.use_cann_megamoe", return_value=True),
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.maybe_trans_nz_with_scale",
            side_effect=lambda weight, scale, **_: (
                weight.transpose(1, 2).contiguous(),
                scale.transpose(1, 2).contiguous(),
            ),
        ),
    ):
        scheme.process_weights_after_loading(layer)

    assert not hasattr(layer, "cann_mega_moe_w13_weight_list")
    if expert_lists:
        assert layer.w13_weight_list[0].shape == (64, 128)
        assert layer.w13_weight_scale_list[0].shape == (1, 128, 2)
    else:
        assert layer.w13_weight.shape == (2, 64, 128)
        assert layer.w13_weight_scale.shape == (2, 1, 128, 2)
