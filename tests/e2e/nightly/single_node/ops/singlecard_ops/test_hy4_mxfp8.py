# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicFusedMoEMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.quantization.utils import get_dynamic_mx_quant_scale_alg


@pytest.mark.parametrize("limit", [0.0, 10.0])
@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("scale_alg", [0, 1])
@torch.inference_mode()
def test_mxfp8_moe_swiglu_matches_clamped_reference(limit, group_list_type, scale_alg):
    tokens, hidden, intermediate, experts = 32, 128, 128, 2
    scheme = AscendW8A8MXFP8DynamicFusedMoEMethod.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
    scheme.use_expert_weight_list = False
    scheme.dynamic_mx_quant_scale_alg = scale_alg
    x = torch.zeros(tokens, hidden, dtype=torch.bfloat16, device="npu")
    x[:, 0] = 1
    gate = torch.tensor([20, -20, 2, -2]).repeat(intermediate // 4)
    up = torch.tensor([20, 20, -20, -20]).repeat(intermediate // 4)
    weight = torch.zeros(experts, hidden, 2 * intermediate, dtype=torch.bfloat16)
    weight[:, 0] = torch.cat((gate, up))
    weight = weight.to(device="npu", dtype=torch.float8_e4m3fn)
    scale = torch.full((experts, hidden // 64, 2 * intermediate, 2), 127, dtype=torch.uint8, device="npu")
    layer = SimpleNamespace(w13_weight=weight, w13_weight_scale=scale)
    inputs = MoEMlpComputeInput(
        hidden_states=x,
        group_list=torch.tensor(
            [tokens // experts, tokens] if group_list_type == 0 else [tokens // experts] * experts,
            dtype=torch.int64,
            device="npu",
        ),
        group_list_type=group_list_type,
        dynamic_scale=None,
        topk_scales=None,
        weights=MoEWeights(weight, weight),
        quant=SimpleNamespace(),
        fusion=True,
        layer=layer,
        activation=MoEActivation.SILU,
        swiglu_limit=limit,
    )
    with (
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg", return_value=scale_alg
        ),
    ):
        quantized, scales = scheme.apply_gmm1_act_quant(inputs)

    exponents = scales.cpu().view(torch.uint8).reshape(tokens, -1).to(torch.int32) - 127
    scales = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents).repeat_interleave(32, dim=-1)
    actual = quantized.cpu().float() * scales
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    expected = (F.silu(gate.float()) * up.float()).expand(tokens, -1)
    torch.testing.assert_close(actual, expected, rtol=0.06, atol=0.03)


@torch.inference_mode()
def test_hy4_mxfp8_linear_preserves_gpu_ceil_scale_extrema():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=["HYV4ForCausalLM"], hf_text_config=SimpleNamespace(model_type="hy_v4")
        )
    )
    scheme = AscendW8A8MXFP8DynamicLinearMethod.__new__(AscendW8A8MXFP8DynamicLinearMethod)
    scheme.group_size = 32
    scheme.dynamic_mx_quant_scale_alg = get_dynamic_mx_quant_scale_alg(config)
    assert scheme.dynamic_mx_quant_scale_alg == 1
    # The old floor-exponent path clips 1.875 to 1.75 and 7.5 to 7.0.
    maximum = torch.tensor([1.875, 7.5, 15.0, 30.0], dtype=torch.bfloat16)
    x = maximum[:, None] * torch.linspace(-1, 1, 128).bfloat16()
    layer = SimpleNamespace(
        weight=torch.eye(128).to(dtype=torch.float8_e4m3fn, device="npu"),
        weight_scale=torch.full((2, 128, 2), 127, dtype=torch.uint8, device="npu"),
        mxfp8_tp_padding=(0, 0),
    )
    actual = scheme.apply(layer, x.npu()).cpu()
    blocks = x.float().reshape(4, 4, 32)
    scales = (blocks.abs().amax(-1) / torch.finfo(torch.float8_e4m3fn).max).log2().ceil().exp2()
    expected = ((blocks / scales[..., None]).to(torch.float8_e4m3fn).float() * scales[..., None]).reshape_as(x)
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(actual[:, -1], maximum, rtol=0, atol=0)


@torch.inference_mode()
def test_hy4_native_query_projection_preserves_ceil_scale_norm():
    scheme = AscendW8A8MXFP8DynamicLinearMethod.__new__(AscendW8A8MXFP8DynamicLinearMethod)
    scheme.group_size = 32
    scheme.dynamic_mx_quant_scale_alg = 1
    width, heads, latent, nope, rope = 2048, 8, 512, 192, 64
    layer = SimpleNamespace(
        weight=torch.eye(width).to(dtype=torch.float8_e4m3fn, device="npu"),
        weight_scale=torch.full((width // 64, width, 2), 127, dtype=torch.uint8, device="npu"),
        mxfp8_tp_padding=(0, 0),
    )
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.q_proj = lambda x: (scheme.apply(layer, x), None)
    impl.local_num_heads = heads
    impl.qk_head_dim = nope + rope
    impl.qk_nope_head_dim = nope
    impl.qk_rope_head_dim = rope
    impl.W_UK_T = torch.eye(nope, latent, dtype=torch.bfloat16, device="npu").unsqueeze(0).repeat(heads, 1, 1)
    query_norm = torch.full((1, width), 1.875, dtype=torch.bfloat16, device="npu")

    query, query_rope = impl._q_proj_and_k_up_proj(query_norm)

    expected = torch.zeros(1, heads, latent, dtype=torch.bfloat16)
    expected[..., :nope] = 1.875
    torch.testing.assert_close(query.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(query_rope.cpu(), torch.full((1, heads, rope), 1.875).bfloat16(), rtol=0, atol=0)
