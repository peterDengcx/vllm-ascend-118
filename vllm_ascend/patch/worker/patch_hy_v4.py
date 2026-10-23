# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Keep HY4's upstream projections and use Ascend's unified MLA execution.

The upstream constructor creates the checkpoint-visible modules. Defer its
CUDA attention allocation so exactly one live NPU KV cache is registered.
This patch can be removed when HY4 exposes a platform MLA factory upstream.
"""

from collections.abc import Iterable
from copy import copy
from functools import wraps
from types import SimpleNamespace

import torch
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.kv_cache import KVCacheScaleParameter
from vllm.model_executor.models.utils import get_pp_missing_layer_names
from vllm.models.hy_v4.nvidia import attention as hy_attention
from vllm.models.hy_v4.nvidia import hc as hy_hc
from vllm.models.hy_v4.nvidia import model as hy_model
from vllm.models.hy_v4.nvidia import mtp as hy_mtp

from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.ops.triton.hy4_ihc import triton_ihc_post, triton_ihc_pre, triton_ihc_supported
from vllm_ascend.quantization.configs.fp8_config import AscendFp8Config
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig

_HYV4_MTP_LOAD_WEIGHTS = hy_mtp.HYV4MTP.load_weights
_HYV4_CREATE_MTP_QUANT_CONFIG = hy_mtp._create_mtp_quant_config
_HYV4_MTP_FORWARD = hy_mtp.HYV4MultiTokenPredictor.forward
_HYV4_HC_PRE_FORWARD = getattr(hy_hc.HYV4HCPreLayer.forward, "__wrapped__", hy_hc.HYV4HCPreLayer.forward)
_HYV4_HC_POST_FORWARD = getattr(hy_hc.HYV4HCPostLayer.forward, "__wrapped__", hy_hc.HYV4HCPostLayer.forward)
_HYV4_IHC_CHANNELS = 4


@wraps(_HYV4_HC_PRE_FORWARD)
def _hc_pre_forward(self, x):
    if (
        self.hpc_op is None
        and isinstance(self.hc_fn, ReplicatedLinear)
        and not self.hc_fn._forward_pre_hooks
        and not self.hc_fn._forward_hooks
        and triton_ihc_supported(x)
    ):
        parameters = (self.hc_fn.weight, self.hc_scale, self.hc_base)
        if all(value.device == x.device and value.dtype == torch.float32 for value in parameters):
            return triton_ihc_pre(x, *parameters, self.magnitude, self.hc_eps, self.layernorm_epsilon)
    # Native projection hooks must still run when weights are offloaded.
    return _HYV4_HC_PRE_FORWARD(self, x)


@wraps(_HYV4_HC_POST_FORWARD)
def _hc_post_forward(self, x, residual, post):
    if (
        self.hpc_op is None
        and triton_ihc_supported(x)
        and residual.ndim == 3
        and residual.shape[1] == _HYV4_IHC_CHANNELS
        and residual.device == x.device
        and residual.dtype == x.dtype
        and post.device == x.device
        and post.dtype == torch.float32
    ):
        return triton_ihc_post(x, residual, post)
    return _HYV4_HC_POST_FORWARD(self, x, residual, post)


@wraps(_HYV4_MTP_FORWARD)
def _mtp_forward(self, input_ids, positions, previous_hidden_states, inputs_embeds=None):
    if positions.ndim != 2:
        return _HYV4_MTP_FORWARD(self, input_ids, positions, previous_hidden_states, inputs_embeds)
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # BOS masking uses the temporal axis; attention still needs all MRoPE axes.
    inputs_embeds = torch.where((positions[0] == 0).unsqueeze(-1), 0, inputs_embeds)
    current_step_idx = self.spec_step_idx % self.num_mtp_layers
    return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
        input_ids, positions, previous_hidden_states, inputs_embeds
    )


def _create_mtp_quant_config(hf_config, backbone_quant_config=None):
    quant_config = _HYV4_CREATE_MTP_QUANT_CONFIG(hf_config, backbone_quant_config)
    if type(quant_config) is Fp8Config:
        # Explicit mtp_quant_algo=FP8 creates a GPU config even on Ascend.
        ascend_config = AscendFp8Config(is_checkpoint_fp8_serialized=True)
        ascend_config.__dict__.update(vars(quant_config))
        return ascend_config
    if isinstance(quant_config, AscendModelSlimConfig):
        # ModelSlim keeps MTP under model.mtp_layers.N, while the draft
        # decoder uses model.layers.(num_hidden_layers + N). Adapt only the
        # draft config in memory so the original checkpoint remains usable.
        aliases = {}
        for name, scheme in quant_config.quant_description.items():
            if not name.startswith("model.mtp_layers."):
                continue
            index, separator, suffix = name.removeprefix("model.mtp_layers.").partition(".")
            if not separator or not index.isdigit():
                continue
            layer_prefix = f"model.layers.{hf_config.num_hidden_layers + int(index)}."
            for infix in ("", "mtp_block."):
                alias = layer_prefix + infix + suffix
                if alias not in quant_config.quant_description:
                    aliases[alias] = scheme
        if aliases:
            quant_config = copy(quant_config)
            quant_config.quant_description = {**quant_config.quant_description, **aliases}
    return quant_config


def _prepare_mtp_fp8_expert_scale(quant_config, name, loaded_weight):
    """Backport HY4 day0 scale decoding without changing packed expert bytes."""
    if isinstance(quant_config, Fp8Config) and quant_config.weight_block_size is not None and name.endswith(".scale"):
        name = name.removesuffix(".scale") + ".weight_scale_inv"
        if loaded_weight.dtype == torch.uint8:
            if getattr(quant_config, "is_scale_e8m0", False):
                loaded_weight = loaded_weight.view(torch.float8_e8m0fnu)
            elif ".mlp.experts." not in name:
                loaded_weight = loaded_weight.view(torch.float8_e8m0fnu).float()
    return name, loaded_weight


def _checked_mtp_weights(model, weights):
    """Reject an unpaired local FP8 indexer projection after streaming weights."""
    pending = {}
    missing_layers = get_pp_missing_layer_names(model)
    for name, weight in weights:
        if ".indexer.wk." in name:
            normalized = name
            if name.startswith("model.mtp_layers."):
                parts = name.split(".")
                spec_layer = None
                if parts[2].isdigit():
                    spec_layer = model.config.num_hidden_layers + int(parts[2])
                    normalized = name.replace(f"model.mtp_layers.{parts[2]}.", f"model.layers.{spec_layer}.")
            else:
                spec_layer = hy_mtp._get_spec_layer_idx_from_weight_name(model.config, name)
            if spec_layer is not None:
                normalized = model._rewrite_spec_layer_name(spec_layer, normalized)
                normalized, _ = _prepare_mtp_fp8_expert_scale(model.quant_config, normalized, weight)
                if not any(normalized.startswith(prefix) for prefix in missing_layers):
                    is_weight = normalized.endswith(".weight") and weight.dtype == torch.float8_e4m3fn
                    is_scale = "weight_scale" in normalized
                    if is_weight or is_scale:
                        prefix = normalized.rsplit(".wk.", 1)[0]
                        fields = pending.setdefault(prefix, set())
                        fields.add("weight" if is_weight else "scale")
                        if fields == {"weight", "scale"}:
                            del pending[prefix]
        yield name, weight
    if pending:
        raise ValueError("Incomplete MTP FP8 indexer weight/scale pairs: " + ", ".join(sorted(pending)))


@wraps(_HYV4_MTP_LOAD_WEIGHTS)
def _load_mtp_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    loaded = _HYV4_MTP_LOAD_WEIGHTS(self, _checked_mtp_weights(self, weights))
    missing_scales = sorted(
        name
        for name, param in self.named_parameters()
        if "weight_scale" in name and name not in loaded and not isinstance(param, KVCacheScaleParameter)
    )
    if missing_scales:
        raise ValueError("Missing MTP checkpoint weight scales: " + ", ".join(missing_scales))
    return loaded


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
hy_mtp._create_mtp_quant_config = _create_mtp_quant_config
hy_mtp._prepare_mtp_fp8_expert_scale = _prepare_mtp_fp8_expert_scale
hy_mtp.HYV4MTP.load_weights = _load_mtp_weights
hy_mtp.HYV4MultiTokenPredictor.forward = _mtp_forward


# Retain native dispatch, including projection hooks, for unsupported inputs.
hy_hc.HYV4HCPreLayer.forward = _hc_pre_forward
hy_hc.HYV4HCPostLayer.forward = _hc_post_forward
