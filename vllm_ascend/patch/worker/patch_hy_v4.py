# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Keep HY4's upstream projections and use Ascend's unified MLA execution.

The upstream constructor creates the checkpoint-visible modules. Defer its
CUDA attention allocation so exactly one live NPU KV cache is registered.
This patch can be removed when HY4 exposes a platform MLA factory upstream.
"""

from types import SimpleNamespace

import torch
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.models.hy_v4.nvidia import attention as hy_attention
from vllm.models.hy_v4.nvidia import model as hy_model

from vllm_ascend.attention.sfa_v1 import AscendSFABackend


class AscendHYV4Indexer(hy_attention.Indexer):
    # Checkpoints store [non-rotary, rotary] with interleaved rotary pairs.
    rope_dim_at_end = True
    is_rope_neox_style = False


class HeadwiseGate(nn.Module):
    def __init__(self, projection: nn.Module, head_dim: int):
        super().__init__()
        self.projection = projection
        self.head_dim = head_dim

    def forward(self, hidden_states):
        gates, bias = self.projection(hidden_states)
        return gates.repeat_interleave(self.head_dim, dim=-1), bias


class AscendHYV4MLAAttention(hy_attention.HYV4MLAAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.rotary_emb, "mrope_section", None):
            from vllm_ascend.ops.rotary_embedding import _record_cos_and_sin_cache

            if not hasattr(self.rotary_emb, "_ascend_mrope_cos"):
                cos, sin = self.rotary_emb.cos_sin_cache.chunk(2, dim=-1)
                self.rotary_emb.register_buffer("_ascend_mrope_cos", cos.repeat(1, 2), persistent=False)
                self.rotary_emb.register_buffer("_ascend_mrope_sin", sin.repeat(1, 2), persistent=False)
            _record_cos_and_sin_cache(self.rotary_emb._ascend_mrope_cos, self.rotary_emb._ascend_mrope_sin)
        deferred = self.mla_attn
        if not self.is_sparse or self.q_lora_rank is None:
            raise NotImplementedError("Ascend HY4 requires DSA layers with q_lora_rank.")
        parallel = get_current_vllm_config().parallel_config
        if parallel.decode_context_parallel_size > 1 or parallel.prefill_context_parallel_size > 1:
            raise NotImplementedError("HY4 attention sinks currently require PCP=1 and DCP=1.")
        gate = self.linear_gate
        if gate is not None and self.config.gating_type == "headwise":
            gate = HeadwiseGate(gate, self.v_head_dim)
        modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=self.q_proj,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=True,
            topk_indices_buffer=deferred.topk_indices_buffer,
            g_proj=gate,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            modules,
            deferred.cache_config,
            deferred.quant_config,
            self.prefix,
            skip_topk=self.skip_topk,
        )
        self.mla_attn.mla_attn.impl.learnable_sink_param = getattr(self, "learnable_sink_param", None)

    def _resolve_sink_backend(self, kv_cache_dtype):
        return AscendSFABackend

    def _force_sparse_mqa(self):
        # Ascend SFA uses its sparse entry point for both prefill and decode.
        pass

    @property
    def topk_indices_buffer(self):
        return self.mla_attn.topk_indices_buffer

    @topk_indices_buffer.setter
    def topk_indices_buffer(self, buffer):
        self.mla_attn.topk_indices_buffer = buffer

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        return self.mla_attn(positions, hidden_states)


def _defer_mla_initialization(**kwargs):
    return SimpleNamespace(**kwargs)


hy_attention.Indexer = AscendHYV4Indexer
hy_attention.MLAAttention = _defer_mla_initialization
hy_attention.HYV4MLAAttention = AscendHYV4MLAAttention
hy_model.HYV4MLAAttention = AscendHYV4MLAAttention
