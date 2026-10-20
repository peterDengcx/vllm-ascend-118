# SPDX-License-Identifier: Apache-2.0
"""Explicit, deterministic initialization of missing HY4 vision parameters."""

import hashlib

import torch
from torch import nn
from vllm.logger import logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


@torch.no_grad()
def initialize_missing_vision_weights(visual: nn.Module, loaded: set[str], seed: int | None) -> set[str]:
    missing = {f"visual.{name}" for name, _ in visual.named_parameters()} - loaded
    if not missing:
        return set()
    if seed is None:
        raise ValueError(
            "HY4 visual weights are missing: "
            + ", ".join(sorted(missing))
            + ". Set ascend_random_vision_seed explicitly only for structural testing."
        )
    initialized = set()
    for module_name, module in visual.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            full_name = ".".join(filter(None, ("visual", module_name, name)))
            if full_name not in missing:
                continue
            shape = tuple(parameter.shape)
            if isinstance(module, LinearBase):
                shape = (module.output_size,) if name == "bias" else (module.output_size, module.input_size)
            generator = torch.Generator(device="cpu")
            digest = hashlib.sha256(f"{seed}:{full_name}".encode()).digest()
            generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
            source = torch.empty(shape, dtype=parameter.dtype, device="cpu")
            if name == "bias":
                source.zero_()
            elif isinstance(module, nn.LayerNorm) or "rms" in module_name or module_name.endswith(".norm"):
                source.fill_(1)
            else:
                source.normal_(mean=0.0, std=0.02, generator=generator)
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, source)
            initialized.add(full_name)
    if initialized != missing:
        raise RuntimeError(f"Uninitialized HY4 vision parameters: {missing - initialized}")
    logger.warning(
        "HY4_RANDOM_VISION seed=%d initialized=%d parameters; image/video semantics are untrained.",
        seed,
        len(initialized),
    )
    return initialized
