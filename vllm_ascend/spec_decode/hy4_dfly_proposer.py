# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.patch.hy4_dfly.indices import anchor_kv_positions, build_reuse_indices
from vllm_ascend.patch.hy4_dfly.parallel import prepare_dfly_query_padding
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


class AscendHY4DFlyProposer(AscendDflashProposer):
    """HY4 block drafting with target sparse-index and writable MLA KV reuse."""

    def __init__(self, vllm_config, device, runner=None):
        super().__init__(vllm_config, device, runner)
        if vllm_config.lora_config is not None or vllm_config.kv_transfer_config is not None:
            raise ValueError("Ascend HY4 DFly does not support LoRA or PD disaggregation")
        if self.dynamic_spec is not None:
            raise ValueError("Ascend HY4 DFly does not support dynamic draft lengths")
        self.use_cuda_graph = False
        self.uses_hidden_correction = True
        self.num_query_per_req = self.num_speculative_tokens + 1
        if self.uses_mrope:
            self.mrope_positions = torch.zeros(
                (self.draft_model_config.mrope_num_dims, self.max_query_tokens),
                dtype=torch.int32, device=device,
            )
        self._target_snapshots = {}
        self._snapshot_hooks = []
        self._source_rows = None
        self._anchor_hidden = None

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None):
        # Shared layers have group membership but no separately allocated spec.
        # Resolve their specs from the target on a private metadata-only copy.
        draft_config = deepcopy(kv_cache_config)
        layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        for group in draft_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if not isinstance(spec, UniformTypeKVCacheSpecs):
                continue
            for name in self._draft_attn_layer_names & set(group.layer_names):
                if name not in spec.kv_cache_specs:
                    target = layers[name].kv_sharing_target_layer_name
                    spec.kv_cache_specs[name] = spec.kv_cache_specs[target]
        super().initialize_attn_backend(draft_config, kernel_block_sizes)

    @torch.inference_mode()
    def dummy_run(self, num_tokens, num_reqs=0, num_tokens_across_dp=None,
                  aclgraph_runtime_mode=None, batch_descriptor=None,
                  dummy_compute_logits=lambda hidden_states: None,
                  is_profile=False, **kwargs):
        # HY4 reuses target MLA KV rather than projecting DFlash context KV.
        # Use the generic runnable so profiling and warmup follow the same
        # build_model_inputs_first_pass path as real HY4 drafting.
        from vllm.config import CUDAGraphMode

        width = self.num_query_per_req
        # Draft buffers may be smaller than max_query_tokens (e.g. 1024
        # tokens for 256 requests with six query tokens per request).
        capacity = min(self.max_query_tokens, self.input_ids.shape[0], self.positions.shape[-1])
        aligned_capacity = capacity // width * width
        if aligned_capacity < width:
            raise ValueError("HY4 DFly dummy buffers cannot hold one query block")
        count = min(max(width, ((num_tokens + width - 1) // width) * width), aligned_capacity)
        batch = count // width
        # Match the reference DFly speculator's _reset_query_buffers: a
        # padded query block, no history indices, and zero target features.
        ids = self.input_ids[:count].view(batch, width)
        ids.fill_(self.model.config.mask_token_id)
        ids[:, 0].zero_()
        self.positions[:count].view(batch, width).copy_(
            torch.arange(width, device=self.device, dtype=self.positions.dtype)
        )
        if self.uses_mrope:
            self.mrope_positions[:, :count].zero_()
        self.model.model.topk_indices.fill_(-1)
        self._clear_dummy_slot_mappings()
        previous_anchor = self.model.anchor_hidden
        self.model.anchor_hidden = torch.zeros_like(self.hidden_states[:batch])
        self._dfly_dummy = True
        try:
            return AscendEagleProposer.dummy_run(
                self,
                num_tokens=count,
                num_reqs=batch,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=batch_descriptor,
                dummy_compute_logits=dummy_compute_logits,
                is_profile=is_profile,
            )
        finally:
            self._dfly_dummy = False
            self.model.anchor_hidden = previous_anchor

    def _raise_if_mrope(self):
        # set_inputs_first_pass expands every M-RoPE axis from the accepted anchor.
        pass

    def _get_model(self):
        from vllm.compilation.backends import set_model_tag
        from vllm_ascend.patch.hy4_dfly.model import load_dfly_model
        with set_model_tag("eagle_head"):
            return load_dfly_model(self._create_draft_vllm_config())

    def _create_draft_vllm_config(self):
        config = super()._create_draft_vllm_config()
        config.model_config = self.draft_model_config
        config.parallel_config = self.speculative_config.draft_parallel_config
        from vllm.config import VllmConfig
        config.quant_config = VllmConfig.get_quantization_config(
            self.draft_model_config, self.speculative_config.draft_load_config
        )
        return config

    def _maybe_share_embeddings(self, target_language_model):
        if self.vllm_config.parallel_config.pipeline_parallel_size > 1:
            # Target embedding normally belongs to the first PP stage. The
            # draft loader restores its own TP-sharded copy on the last stage.
            if not self.model.has_own_embed_tokens or self.model.model.embed_tokens is None:
                raise ValueError("HY4 DFly PP draft embedding was not loaded")
            return
        self.model.model.embed_tokens = target_language_model.model.embed_tokens

    def _maybe_share_topk_indices(self, target_language_model):
        # Every target index layer needs its own snapshot: its live scratch
        # buffer is overwritten by later target layers in the same forward.
        self._lim_topk_compactors = []
        layers = target_language_model.model.layers
        config = self.model.config
        if self.vllm_config.parallel_config.pipeline_parallel_size > 1:
            from vllm_ascend.patch.hy4_dfly.pipeline import require_local_target_layers

            require_local_target_layers(
                config.target_index_layer, config.target_kv_layers,
                target_language_model.model.start_layer, target_language_model.model.end_layer,
            )
        for index in set(config.target_index_layer):
            attention = layers[index].self_attn
            if attention.skip_topk or attention.indexer is None:
                raise ValueError(f"DFly target layer {index} must run its own indexer")
            snapshot = torch.full_like(attention.topk_indices_buffer, -1)
            self._target_snapshots[index] = snapshot
            def capture(module, args, kwargs, output, buffer=snapshot):
                hidden = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
                count = hidden.shape[0]
                buffer[:count].copy_(module.topk_indices_buffer[:count])
            self._snapshot_hooks.append(attention.register_forward_hook(capture, with_kwargs=True))
        for draft, index in zip(self.model.model.layers, config.target_kv_layers):
            source = layers[index].self_attn.mla_attn.mla_attn
            dest = draft.self_attn.mla_attn.mla_attn
            dest.kv_sharing_target_layer_name = source.layer_name
            dest.impl.kv_sharing_target_layer_name = source.layer_name

    def set_inputs_first_pass(self, target_token_ids, next_token_ids, target_positions,
                              target_hidden_states, token_indices_to_sample, cad,
                              num_rejected_tokens_gpu, *args, **kwargs):
        # Target graph metadata may contain padded request entries. DFlash
        # expands queries for cad.num_reqs only; anchors must use that same
        # active prefix rather than treating padding as local requests.
        rows = cad.query_start_loc[1:cad.num_reqs + 1] - 1
        if num_rejected_tokens_gpu is not None:
            rows = rows - num_rejected_tokens_gpu
        self._source_rows = rows.long()
        self._anchor_hidden = target_hidden_states[self._source_rows]
        # Cache indices and M-RoPE positions diverge after visual tokens.
        self._anchor_kv_positions = anchor_kv_positions(
            cad.seq_lens, num_rejected_tokens_gpu, cad.num_reqs
        )
        if self.uses_mrope:
            width = self.num_query_per_req
            offsets = torch.arange(width, device=self.device)
            positions = target_positions[:, self._source_rows] + 1
            self.mrope_positions[:, :cad.num_reqs * width] = (
                positions[:, :, None] + offsets
            ).flatten(1)
        result = super().set_inputs_first_pass(
            target_token_ids, next_token_ids,
            target_positions[0] if target_positions.ndim == 2 else target_positions,
            target_hidden_states,
            token_indices_to_sample, cad, num_rejected_tokens_gpu, *args, **kwargs
        )
        result[2].causal = True
        result[2].positions = self.positions[:result[0]]
        result[2].positions_cpu = None
        return result

    def build_model_inputs_first_pass(self, num_input_tokens, _context_slots):
        local_batch = prepare_dfly_query_padding(self, num_input_tokens)
        if not local_batch:
            return
        width = self.num_query_per_req
        anchor_positions = self._anchor_kv_positions[:local_batch]
        active = torch.ones(local_batch, dtype=torch.bool, device=self.device)
        for i, index in enumerate(self.model.config.target_index_layer):
            build_reuse_indices(
                self._target_snapshots[index][self._source_rows],
                anchor_positions, width, self.model.config.context_local_window or 0,
                active, self.model.model.topk_indices[i],
            )

    def compute_draft_token_ids(self, hidden_states, sampling_metadata=None):
        steps = self.num_speculative_tokens
        hidden = hidden_states.view(-1, steps, hidden_states.shape[-1])
        previous = self.input_ids[:hidden.shape[0] * (steps + 1):steps + 1].long()
        tokens = []
        draft_probs = []
        for step in range(steps):
            corrected = self.model.apply_hidden_correction(hidden[:, step], previous)
            logits = self.model.compute_logits(corrected)
            previous, probs = self._sample_draft_from_logits(logits, sampling_metadata)
            tokens.append(previous)
            if probs is not None:
                draft_probs.append(probs)
        # Export the distribution actually used at each conditional step.
        # Standard rejection sampling needs request-major [B * K, vocab].
        probabilities = None
        if draft_probs:
            if len(draft_probs) != steps:
                raise RuntimeError("DFly must export probabilities for every draft step")
            probabilities = torch.stack(draft_probs, dim=1).flatten(0, 1).contiguous()
        return torch.stack(tokens, dim=1).reshape(-1), probabilities
