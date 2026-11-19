# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from transformers import PreTrainedConfig

def _validate_block_fp8_config(config: object) -> None:
    error = (
        "HYV4 DFly requires compressed-tensors W8A8 FP8 with 128x128 weight "
        "blocks, dynamic group-128 activations and no checkpoint kv_cache_scheme"
    )
    if not isinstance(config, dict) or (
        config.get("quant_method") != "compressed-tensors"
        or config.get("format") != "float-quantized"
        or config.get("kv_cache_scheme") is not None
        or config.get("transform_config") is not None
    ):
        raise ValueError(error)
    groups = config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(error)
    expected_weight = dict(
        num_bits=8,
        type="float",
        strategy="block",
        block_structure=[128, 128],
        symmetric=True,
        dynamic=False,
    )
    expected_activation = dict(
        num_bits=8,
        type="float",
        strategy="group",
        group_size=128,
        symmetric=True,
        dynamic=True,
    )
    for group in groups.values():
        if not isinstance(group, dict):
            raise ValueError(error)
        weights = group.get("weights")
        activations = group.get("input_activations")
        if (
            not isinstance(weights, dict)
            or not isinstance(activations, dict)
            or any(weights.get(k) != v for k, v in expected_weight.items())
            or any(activations.get(k) != v for k, v in expected_activation.items())
            or group.get("output_activations") is not None
            or group.get("format") not in (None, "float-quantized")
        ):
            raise ValueError(error)

def _derive_dfly_config(
    hf_config: PreTrainedConfig, *, target_config: PreTrainedConfig
) -> PreTrainedConfig:
    """Derive HYV4's text-only draft without changing the target configuration."""
    if hf_config.model_type.startswith("dummy_"):
        return hf_config
    config = deepcopy(target_config.get_text_config())
    if config.model_type != "hy_v4":
        raise ValueError("DFly requires a HYV4 target model")
    num_layers = getattr(target_config, "dfly_num_hidden_layers", None)
    if type(num_layers) is not int or num_layers < 1:
        raise ValueError("DFly requires a positive dfly_num_hidden_layers")
    config.target_num_hidden_layers = config.num_hidden_layers
    config.target_hidden_size = config.hidden_size
    for name, default in (
        ("enable_ihc", False),
        ("mtp_fusion", "slot0"),
        ("target_index_topk", config.index_topk),
        ("context_local_window", None),
        ("enable_hidden_correction", True),
        ("mlp_type", "dense"),
        ("mask_token_id", None),
        ("target_index_layer", None),
        ("target_kv_layers", None),
        ("hidden_correction_intermediate_size", config.hidden_size),
    ):
        setattr(config, name, deepcopy(getattr(target_config, f"dfly_{name}", default)))
    if config.mlp_type != "dense" or config.enable_ihc is not False:
        raise ValueError("DFly requires dense FFNs and dfly_enable_ihc=false")
    if getattr(target_config, "dfly_attn_causal", True) is not True:
        raise ValueError("DFly requires causal MLA attention")
    if config.mtp_fusion != "slot0" or config.enable_hidden_correction is not True:
        raise ValueError("DFly requires slot0 fusion and hidden correction")
    if type(config.mask_token_id) is not int or not (
        0 <= config.mask_token_id < config.vocab_size
    ):
        raise ValueError("DFly dfly_mask_token_id is outside the vocabulary")
    if (
        type(config.target_index_topk) is not int
        or config.target_index_topk <= 0
        or (
            config.context_local_window is not None
            and (
                type(config.context_local_window) is not int
                or config.context_local_window < 0
            )
        )
    ):
        raise ValueError("DFly needs positive target top-k and nonnegative window")
    if (
        type(config.hidden_correction_intermediate_size) is not int
        or config.hidden_correction_intermediate_size <= 0
    ):
        raise ValueError("DFly hidden correction intermediate size must be positive")
    config.model_type = "hy_v4_mtp"
    config.ascend_dfly = True
    config.architectures = ["HYV4DFlyModel"]
    config.num_hidden_layers = num_layers
    config.n_routed_experts = 0
    config.n_shared_experts = 0
    config.num_experts = 0
    config.num_nextn_predict_layers = 0
    config.n_predict = 1
    config.mlp_layer_types = ["dense"] * num_layers
    config.layer_types = ["deepseek_sparse_attention"] * num_layers
    config.indexer_types = ["shared"] * num_layers
    indices = config.target_index_layer
    if type(indices) is int:
        indices = [indices] * num_layers
    for name, selected in (("target_index_layer", indices), ("target_kv_layers", config.target_kv_layers)):
        if (not isinstance(selected, list) or len(selected) != num_layers
                or any(type(i) is not int or not 1 <= i < config.target_num_hidden_layers for i in selected)):
            raise ValueError(f"DFly {name} needs one valid target layer per draft layer")
    if len(set(config.target_kv_layers)) != num_layers:
        raise ValueError("DFly requires distinct target KV layers")
    config.target_index_layer = indices
    return config
