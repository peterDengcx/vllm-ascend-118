# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Private quantization methods for HY4's mixed FP4/FP8 checkpoint.

Only HY4 model constructors install this config. The public FP8 config and
scheme registry remain unchanged, including FP8 checkpoints with E8M0 scales.
"""

from copy import copy
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.utils.math_utils import cdiv

from vllm_ascend.quantization.configs import fp8_config
from vllm_ascend.quantization.configs.fp8_config import AscendFp8Config
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import AscendW4A8MXFPDSDynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
    dequantize_mxfp8,
)
from vllm_ascend.quantization.utils import is_fused_moe_layer


class Hy4Fp4MoEMethod(AscendW4A8MXFPDSDynamicFusedMoEMethod):
    """Keep the packed E2M1 experts' per-32 E8M0 scale bytes intact."""

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {
            "w13_weight_scale": torch.empty(
                num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.uint8
            ),
            "w2_weight_scale": torch.empty(
                num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.uint8
            ),
        }


class Hy4Fp4BlockLinearMethod(AscendW8A8MXFP8DynamicLinearMethod):
    """FP8 projections inside the FP4 export, with per-block E8M0 scales."""

    def __init__(self, weight_block_size):
        super().__init__()
        self.block_rows, self.block_cols = weight_block_size
        self.model_dtype = get_current_vllm_config().model_config.dtype

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        return {
            "weight_scale_inv": torch.empty(
                cdiv(output_size, self.block_rows), cdiv(input_size, self.block_cols), dtype=torch.float8_e8m0fnu
            ),
            "_block_quant_scale": (self.block_rows, self.block_cols),
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        scale = layer.weight_scale_inv.data.view(torch.uint8)
        weight = layer.weight.data
        rows, cols = weight.shape
        if self.block_cols % self.group_size:
            raise ValueError("HY4 FP4 linear blocks must contain complete MX quantization groups.")
        expanded = scale.repeat_interleave(self.block_rows, 0)
        expanded = expanded.repeat_interleave(self.block_cols // self.group_size, 1)
        # Fused Q/KV projections can end in a partial 128-row tile.
        expanded = expanded[:rows, : cdiv(cols, self.group_size)]
        if getattr(layer, "prefix", "").endswith("kv_b_proj"):
            # Match the shared MLA path's dense projection and E8M0 decoding,
            # including the smallest exponent and reserved NaN encoding.
            layer.weight.data = dequantize_mxfp8(weight.cpu(), expanded.cpu(), self.group_size, self.model_dtype).to(
                weight.device
            )
            del layer.weight_scale_inv
            return
        layer.register_parameter("weight_scale", torch.nn.Parameter(expanded, requires_grad=False))
        del layer.weight_scale_inv
        # Reuse the current kernel's matching weight/scale layout, NZ policy,
        # TP padding and persistent buffers rather than keeping an old layout.
        super().process_weights_after_loading(layer)


class Hy4Fp4Config(AscendFp8Config):
    """Model-owned FP4 config; never registered as a public quantization type."""

    def get_quant_method(self, layer: torch.nn.Module, prefix: str, tid2eid=None):
        is_linear = isinstance(layer, LinearBase)
        is_moe = is_fused_moe_layer(layer)
        if not is_linear and not is_moe:
            return super().get_quant_method(layer, prefix, tid2eid)
        skip_kwargs = {}
        if fp8_config._IS_LAYER_SKIPPED_SUPPORTS_MATCH_MODE:
            skip_kwargs["match_mode"] = getattr(self, "ignored_layers_match_mode", "exact")
        if fp8_config.is_layer_skipped(prefix, self.ignored_layers, self.packed_modules_mapping, **skip_kwargs):
            return super().get_quant_method(layer, prefix, tid2eid)
        self._verify_block_quantization()
        # The wrappers import the shared scheme package, so resolve them after
        # the private schemes and the config module have finished importing.
        from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod, AscendLinearMethod

        if is_linear:
            return AscendLinearMethod(Hy4Fp4BlockLinearMethod(self.weight_block_size))
        return AscendFusedMoEMethod(Hy4Fp4MoEMethod(), layer.moe_config, tid2eid=tid2eid)


def prepare_hy4_fp4_config(vllm_config):
    """Copy FP4 state for one HY4 constructor without changing its caller."""
    original = vllm_config.quant_config
    if type(original) is not AscendFp8Config or original.store_dtype not in ("mxfp4", "fp4"):
        return vllm_config
    hf_config = vllm_config.model_config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config)
    metadata = (
        getattr(text_config, "quantization_config", None) or getattr(hf_config, "quantization_config", None) or {}
    )
    if metadata.get("scale_fmt") != "ue8m0":
        raise ValueError("HY4 packed FP4 checkpoints require scale_fmt=ue8m0.")
    converted = Hy4Fp4Config(is_checkpoint_fp8_serialized=True)
    converted.__dict__.update(vars(original))
    converted.is_scale_e8m0 = True
    local_config = copy(vllm_config)
    local_config.quant_config = converted
    return local_config
