# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Iterator
from copy import deepcopy

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig as CompressedTensorsConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.hy_v4 import HYV4Config
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict
from vllm.utils.math_utils import round_up

from vllm.models.hy_v4.nvidia.model import HYV4DecoderLayer


class HYV4DFlyQuantizationConfig(CompressedTensorsConfig):
    """Match draft linears against their checkpoint paths, without rewriting regexes."""

    model_prefix: str

    @classmethod
    def for_model(
        cls, source: CompressedTensorsConfig, model_prefix: str
    ) -> "HYV4DFlyQuantizationConfig":
        if source.quant_description is None:
            raise ValueError("DFly requires serialized compressed-tensors metadata")
        config = cls.from_config(deepcopy(source.quant_description))
        assert isinstance(config, cls)
        config.model_prefix = f"{model_prefix}."
        config.packed_modules_mapping = source.packed_modules_mapping.copy()
        return config

    def get_scheme_dict(self, layer: nn.Module, layer_name: str | None = None):
        if layer_name is not None and layer_name.startswith(self.model_prefix):
            local_name = layer_name.removeprefix(self.model_prefix)
            local_name = local_name.removeprefix("layers.")
            layer_name = f"model.dfly_layers.{local_name}"
        return super().get_scheme_dict(layer, layer_name)


class HYV4DFlyFusion(nn.Module):
    def __init__(
        self,
        config: HYV4Config,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        d = config.hidden_size
        self.enorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * d, d, bias=False, prefix=f"{prefix}.eh_proj", quant_config=quant_config
        )
        self.gate = nn.Parameter(torch.empty(d), requires_grad=False)

    def forward(
        self, anchors: torch.Tensor, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        fused, _ = self.eh_proj(
            torch.cat((self.enorm(anchors), self.hnorm(target_hidden)), -1)
        )
        return anchors + self.gate * fused


class HYV4DFlyHiddenCorrection(nn.Module):
    def __init__(
        self,
        config: HYV4Config,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        d = config.hidden_size
        intermediate = config.hidden_correction_intermediate_size
        self.hidden_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.embed_norm = RMSNorm(d, eps=config.rms_norm_eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            2 * d,
            [intermediate, intermediate],
            bias=False,
            prefix=f"{prefix}.gate_up_proj",
            quant_config=quant_config,
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(
            intermediate,
            d,
            bias=False,
            prefix=f"{prefix}.down_proj",
            quant_config=quant_config,
        )

    def forward(
        self, hidden: torch.Tensor, previous_embed: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat(
            (self.hidden_norm(hidden), self.embed_norm(previous_embed)), -1
        )
        gate_up, _ = self.gate_up_proj(combined)
        delta, _ = self.down_proj(self.act(gate_up))
        return hidden + delta


class HYV4DFlyModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        spec = vllm_config.speculative_config
        assert spec is not None
        config = deepcopy(spec.draft_model_config.hf_config)
        if not getattr(config, "ascend_dfly", False):
            raise ValueError("Expected a derived hy_v4_dfly configuration")
        self.config = config
        quant_config = vllm_config.quant_config
        if isinstance(quant_config, CompressedTensorsConfig):
            quant_config = HYV4DFlyQuantizationConfig.for_model(quant_config, prefix)
        self.quant_config = quant_config
        self.query_width = spec.num_speculative_tokens + 1
        config.index_topk = round_up(
            config.target_index_topk
            + (config.context_local_window or 0)
            + self.query_width,
            128,
        )
        config.dfly_external_topk = True
        self.embed_tokens: VocabParallelEmbedding | None = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        max_rows = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices",
            torch.full(
                (config.num_hidden_layers, max_rows, config.index_topk),
                -1,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        self.layers = nn.ModuleList(
            [
                HYV4DecoderLayer(
                    config,
                    vllm_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                    topk_indices_buffer=self.topk_indices[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        for layer in self.layers:
            attn = layer.self_attn
            attn.use_hpc_gated_mla = False
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_fusion = HYV4DFlyFusion(
            config, f"{prefix}.mtp_fusion", quant_config=quant_config
        )
        self.hidden_correction = HYV4DFlyHiddenCorrection(
            config, f"{prefix}.hidden_correction", quant_config=quant_config
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert self.embed_tokens is not None, "DFly embedding is not bound"
            inputs_embeds = self.embed_tokens(input_ids)
        hidden = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        return self.norm(hidden + residual)


class HYV4DFlyForCausalLM(nn.Module):
    checkpoint_prefix = "model.dfly_layers."
    fall_back_to_pt_during_load = False
    has_own_lm_head = False
    draft_id_to_target_id = None
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    }
    _weight_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj.": (".gate_up_proj.", 0),
            ".up_proj.": (".gate_up_proj.", 1),
            ".q_a_proj.": (".fused_qkv_a_proj.", 0),
            ".kv_a_proj_with_mqa.": (".fused_qkv_a_proj.", 1),
        }
    )
    _ignored_prefixes = ("confidence_head.",)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self._configure_weight_sources(vllm_config)
        self.model = HYV4DFlyModel(vllm_config, maybe_prefix(prefix, "model"))
        self.config = self.model.config
        self.anchor_hidden = None
        self.has_own_embed_tokens = False
        self.lm_head: ParallelLMHead | None = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.logits_processor.use_all_gather = True

    def _configure_weight_sources(self, vllm_config: VllmConfig) -> None:
        spec = vllm_config.speculative_config
        assert spec is not None
        draft = spec.draft_model_config
        index = get_hf_file_to_dict(
            "model.safetensors.index.json", draft.model, draft.revision
        )
        if not index or not isinstance(index.get("weight_map"), dict):
            raise ValueError("DFly requires the target safetensors index")
        shards = sorted(
            {
                shard
                for name, shard in index["weight_map"].items()
                if name.startswith(self.checkpoint_prefix)
            }
        )
        if not shards:
            raise ValueError("No model.dfly_layers weights in the target checkpoint")
        if any(
            not shard.endswith(".safetensors") or "/" in shard or "\\" in shard
            for shard in shards
        ):
            raise ValueError(
                "DFly weight shards must be safetensors in the model directory"
            )
        self.weight_shards = shards
        self.allow_patterns_overrides = shards[:1]
        self.secondary_weights = [
            DefaultModelLoader.Source(
                draft.model,
                draft.revision,
                fall_back_to_pt=False,
                allow_patterns_overrides=[shard],
            )
            for shard in shards[1:]
        ]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.model.embed_tokens is not None, "DFly embedding is not bound"
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        inputs_embeds = inputs_embeds.clone()
        anchors = inputs_embeds[::self.model.query_width]
        if self.anchor_hidden is not None:
            inputs_embeds[::self.model.query_width] = self.model.mtp_fusion(anchors, self.anchor_hidden[:anchors.shape[0]])
        return self.model(input_ids, positions, inputs_embeds)

    def apply_hidden_correction(
        self, hidden: torch.Tensor, previous_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.model.hidden_correction(hidden, self.embed_input_ids(previous_ids))

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.lm_head is None:
            raise RuntimeError(
                "DFly requires the target LM head before computing logits"
            )
        return self.logits_processor(self.lm_head, hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.mla_attn.mla_attn.layer_name for layer in self.model.layers]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        required: dict[str, torch.dtype] = {}
        for name, param in self.model.named_parameters():
            serialized_name = name.removesuffix("_inv") if name.endswith(".weight_scale_inv") else name
            dtype = (
                param.dtype
                if param.dtype == torch.float8_e4m3fn or name.endswith(".weight_scale_inv")
                else torch.bfloat16
            )
            if ".gate_up_proj." in name:
                required.update(
                    (serialized_name.replace(".gate_up_proj.", part), dtype)
                    for part in (".gate_proj.", ".up_proj.")
                )
            elif ".fused_qkv_a_proj." in name:
                required.update(
                    (serialized_name.replace(".fused_qkv_a_proj.", part), dtype)
                    for part in (".q_a_proj.", ".kv_a_proj_with_mqa.")
                )
            else:
                required[serialized_name] = dtype
        seen: set[str] = set()

        def checked_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if not name.startswith(self.checkpoint_prefix):
                    continue
                name = name.removeprefix(self.checkpoint_prefix)
                if name.startswith(self._ignored_prefixes):
                    continue
                if name.split(".", 1)[0].isdigit():
                    name = f"layers.{name}"
                elif name.startswith("final_norm."):
                    name = name.replace("final_norm.", "norm.", 1)
                if name.endswith(".learnable_sink_param"):
                    rank = get_tensor_model_parallel_rank()
                    size = get_tensor_model_parallel_world_size()
                    weight = shard_dfly_sink(weight, self.config.num_attention_heads, rank, size)
                if name not in required:
                    raise ValueError(
                        f"Unexpected DFly weight {name!r}, shape={tuple(weight.shape)}"
                    )
                if name in seen:
                    raise ValueError(f"Duplicate DFly weight {name!r}")
                if weight.dtype != required[name]:
                    raise ValueError(
                        f"DFly weight {name} requires {required[name]}, "
                        f"got {weight.dtype}"
                    )
                seen.add(name)
                yield f"model.{name}" + ("_inv" if name.endswith(".weight_scale") else ""), weight
            missing = required.keys() - seen - {"embed_tokens.weight"}
            if missing:
                raise ValueError(f"Missing required DFly weights: {sorted(missing)}")

        loaded = AutoWeightsLoader(self).load_weights(
            checked_weights(), mapper=self._weight_mapper
        )
        self.has_own_embed_tokens = "embed_tokens.weight" in seen
        if not self.has_own_embed_tokens:
            self.model.embed_tokens = None
        return loaded


def load_dfly_model(vllm_config):
    """Load only DFly shards; pinned vLLM validates the whole target index."""
    from pathlib import Path
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
    from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
    from vllm.utils.torch_utils import set_default_torch_dtype

    config = vllm_config.model_config
    directory = Path(config.model)
    if not directory.is_dir():
        raise ValueError("Ascend HY4 DFly requires a local target checkpoint directory")
    device = torch.device(vllm_config.device_config.device)
    with set_default_torch_dtype(config.dtype), set_current_vllm_config(vllm_config):
        with device:
            model = initialize_model(vllm_config=vllm_config, model_config=config, prefix="dfly")
        files = [str(directory / name) for name in model.weight_shards]
        allocated_embedding = model.model.embed_tokens
        model.load_weights(safetensors_weights_iterator(files, False))
        from vllm_ascend.patch.hy4_dfly.pipeline import restore_pipeline_embedding

        restore_pipeline_embedding(model, allocated_embedding, vllm_config, safetensors_weights_iterator)
        process_weights_after_loading(model, config, device)
    return model.eval()


def shard_dfly_sink(weight, num_heads, rank, size):
    """Checkpoint sinks are global; draft attention owns local TP heads."""
    if weight.shape != (num_heads,) or num_heads % size or not 0 <= rank < size:
        raise ValueError("Invalid DFly sink shape or tensor-parallel layout")
    local_heads = num_heads // size
    return weight.narrow(0, rank * local_heads, local_heads)
