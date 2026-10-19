# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from vllm.models.hy_v4.nvidia import attention as native_attention

from vllm_ascend.patch.worker.patch_hy_v4 import AscendHYV4MLAAttention, HeadwiseGate
from vllm_ascend.quantization.configs.fp8_config import AscendFp8Config
from vllm_ascend.quantization.configs.modelopt_mxfp8_config import AscendModelOptMxFp8Config
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("quantization", [None, "modelopt_mxfp8", "fp8", "ascend"])
def test_native_constructor_preserves_projections_and_quantization(dtype, quantization):
    """Keep the same projection owners for floating and quantized models."""
    if quantization == "modelopt_mxfp8":
        quant_config = AscendModelOptMxFp8Config.from_config(
            {
                "quant_method": "modelopt",
                "quantization": {"quant_algo": "MXFP8", "kv_cache_quant_algo": None, "exclude_modules": []},
            }
        )
    elif quantization == "fp8":
        quant_config = AscendFp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    elif quantization == "ascend":
        quant_config = AscendModelSlimConfig({"model.layers.1.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC"})
    else:
        quant_config = None

    class Projection(torch.nn.Module):
        def __init__(self, *args, quant_config=None, **kwargs):
            super().__init__()
            self.quant_config = quant_config

    class Wrapper(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.modules = args[8]
            self.quant_config = args[10]
            self.topk_indices_buffer = self.modules.topk_indices_buffer
            self.mla_attn = SimpleNamespace(impl=SimpleNamespace())

    config = SimpleNamespace(
        num_hidden_layers=2,
        layer_types=["sparse_attention", "sparse_attention"],
        indexer_types=["full", "shared"],
        index_topk=4,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        gated_mla=True,
        gating_type="elementwise",
        learnable_sink=True,
    )
    parallel = SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1)
    topk = torch.zeros(2, 4, dtype=torch.int32)
    with ExitStack() as stack:
        for name in (
            "ColumnParallelLinear",
            "MergedColumnParallelLinear",
            "ReplicatedLinear",
            "RowParallelLinear",
            "RMSNorm",
        ):
            stack.enter_context(patch.object(native_attention, name, Projection))
        stack.enter_context(patch.object(native_attention, "get_tensor_model_parallel_world_size", return_value=1))
        stack.enter_context(patch.object(native_attention, "get_attn_backend"))
        stack.enter_context(patch.object(native_attention, "get_rope", return_value=torch.nn.Identity()))
        stack.enter_context(patch.object(native_attention, "hpc_gated_mla_supported", return_value=False))
        stack.enter_context(
            patch(
                "vllm_ascend.patch.worker.patch_hy_v4.get_current_vllm_config",
                return_value=SimpleNamespace(parallel_config=parallel),
            )
        )
        stack.enter_context(patch("vllm_ascend.patch.worker.patch_hy_v4.MultiHeadLatentAttentionWrapper", Wrapper))
        stack.callback(torch.set_default_dtype, torch.get_default_dtype())
        torch.set_default_dtype(dtype)
        attention = AscendHYV4MLAAttention(
            vllm_config=SimpleNamespace(),
            config=config,
            hidden_size=8,
            num_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=8,
            q_lora_rank=4,
            kv_lora_rank=4,
            max_position_embeddings=32,
            cache_config=SimpleNamespace(cache_dtype="auto"),
            quant_config=quant_config,
            prefix="model.layers.1.self_attn",
            topk_indices_buffer=topk,
            layer_idx=1,
        )
    wrapper = attention.mla_attn
    assert wrapper.quant_config is quant_config
    assert wrapper.modules.fused_qkv_a_proj is attention.fused_qkv_a_proj
    assert wrapper.modules.o_proj is attention.o_proj
    assert wrapper.modules.g_proj is attention.linear_gate
    assert attention.o_proj.quant_config is quant_config
    assert attention.skip_topk and attention.indexer is None
    assert attention.topk_indices_buffer is topk
    assert wrapper.mla_attn.impl.learnable_sink_param is attention.learnable_sink_param


def test_headwise_gate_broadcasts_each_head_without_mixing_heads():
    projection = Mock(return_value=(torch.tensor([[1.0, -2.0]]), None))
    gate, bias = HeadwiseGate(projection, 3)(torch.zeros(1, 4))
    torch.testing.assert_close(gate, torch.tensor([[1.0, 1.0, 1.0, -2.0, -2.0, -2.0]]))
    assert bias is None
