# SPDX-License-Identifier: Apache-2.0
"""DFly on the final PP stage, with local target KV/index dependencies."""

import json
from pathlib import Path


def require_local_target_layers(index_layers, kv_layers, start_layer, end_layer):
    if isinstance(index_layers, int):
        index_layers = [index_layers]
    missing = sorted({i for i in [*index_layers, *kv_layers] if not start_layer <= i < end_layer})
    if missing:
        raise ValueError(
            f"HY4 DFly PP requires all target KV/index layers on the final stage "
            f"[{start_layer}, {end_layer}); nonlocal layers: {missing}. "
            "Adjust VLLM_PP_LAYER_PARTITION to keep these layers on the final stage. "
            "Cross-stage DFly KV transfer is not supported."
        )


def validate_dfly_pipeline_config(spec):
    """Use vLLM's real partitioner, including custom layer partitions."""
    from vllm.distributed.utils import get_pp_indices

    parallel = spec.target_parallel_config
    hf = spec.target_model_config.hf_config
    text = getattr(hf, "text_config", hf)
    start, end = get_pp_indices(
        text.num_hidden_layers, parallel.pipeline_parallel_size - 1, parallel.pipeline_parallel_size
    )
    require_local_target_layers(hf.dfly_target_index_layer, hf.dfly_target_kv_layers, start, end)


def restore_pipeline_embedding(model, allocated_embedding, vllm_config, weights_iterator):
    """Load a target embedding copy when the first PP stage owns the original.

    DFly's selected shards need not contain the target embedding. Read its
    indexed shard explicitly and invoke the existing TP-aware weight loader.
    This happens during normal model loading, before weight finalization.
    """
    if (
        vllm_config.speculative_config.target_parallel_config.pipeline_parallel_size == 1
        or model.has_own_embed_tokens
    ):
        return
    if allocated_embedding is None or not hasattr(allocated_embedding, "weight"):
        raise ValueError("HY4 DFly PP requires a local draft embedding allocation")
    directory = Path(vllm_config.model_config.model)
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    names = [name for name in ("model.embed_tokens.weight", "language_model.model.embed_tokens.weight") if name in index]
    if len(names) != 1:
        raise ValueError("HY4 DFly PP requires one indexed target embedding weight")
    name = names[0]
    shard = (directory / index[name]).resolve()
    if not shard.is_relative_to(directory.resolve()):
        raise ValueError("HY4 DFly embedding shard must be inside the checkpoint directory")
    iterator = weights_iterator([str(shard)], False)
    try:
        for tensor_name, tensor in iterator:
            if tensor_name == name:
                param = allocated_embedding.weight
                param.weight_loader(param, tensor)
                model.model.embed_tokens = allocated_embedding
                model.has_own_embed_tokens = True
                return
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    raise ValueError(f"HY4 DFly target embedding {name} is missing from {shard.name}")
