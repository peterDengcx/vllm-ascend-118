from functools import wraps

from vllm.config.model import ModelConfig
from vllm.logger import logger


def _is_compressed_fp8(config):
    if not config or config.get("quant_method") != "compressed-tensors":
        return False
    groups = config.get("config_groups", {})
    return bool(groups) and all(
        group.get("weights", {}).get("type") == "float"
        and group.get("weights", {}).get("num_bits") == 8
        for group in groups.values()
    )


def _wrap_verify_quantization(original):
    @wraps(original)
    def verify(self):
        config = self.model_arch_config.quantization_config
        if self.quantization == "fp8" and _is_compressed_fp8(config):
            logger.warning("Selecting compressed-tensors loader for compressed FP8 metadata.")
            self.quantization = "compressed-tensors"
        return original(self)
    return verify


ModelConfig._verify_quantization = _wrap_verify_quantization(ModelConfig._verify_quantization)
