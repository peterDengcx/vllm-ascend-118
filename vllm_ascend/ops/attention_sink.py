# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch


def apply_attention_sink(
    output: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Include a per-head zero-value sink in normalized TND attention."""
    num_tokens, num_heads = output.shape[:2]
    maximum = softmax_max.float().reshape(num_tokens, num_heads)
    denominator = softmax_sum.float().reshape(num_tokens, num_heads)
    logsumexp = maximum + denominator.clamp_min(torch.finfo(torch.float32).tiny).log()
    factor = torch.sigmoid(logsumexp - sink.float().view(1, num_heads))
    return torch.where((denominator > 0).unsqueeze(-1), output * factor.unsqueeze(-1).to(output.dtype), 0)
