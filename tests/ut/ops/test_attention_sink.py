# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch

from vllm_ascend.ops.attention_sink import apply_attention_sink


@pytest.mark.parametrize("sink_logit", [-1000.0, -2.0, 0.0, 3.0, 1000.0])
def test_sink_matches_softmax_with_an_extra_zero_value(sink_logit):
    generator = torch.Generator().manual_seed(17)
    scores = torch.randn(4, 3, 5, generator=generator) * 10
    values = torch.randn(4, 3, 5, 7, generator=generator)
    sink = torch.full((3,), sink_logit)
    maximum = scores.max(dim=-1).values
    denominator = (scores - maximum.unsqueeze(-1)).exp().sum(dim=-1)
    output = (scores.softmax(dim=-1).unsqueeze(-1) * values).sum(dim=-2)
    actual = apply_attention_sink(output, maximum.unsqueeze(0), denominator.unsqueeze(0), sink)
    padded_scores = torch.cat([scores, sink.view(1, 3, 1).expand(4, -1, -1)], dim=-1)
    expected = (padded_scores.softmax(dim=-1)[..., :-1].unsqueeze(-1) * values).sum(dim=-2)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_sink_clears_empty_rows_with_unwritten_output():
    output = torch.full((2, 3, 4), float("nan"), dtype=torch.bfloat16)
    maximum = torch.full((1, 2, 3), -float("inf"))
    denominator = torch.zeros_like(maximum)
    actual = apply_attention_sink(output, maximum, denominator, torch.zeros(3))
    assert actual.dtype == output.dtype
    torch.testing.assert_close(actual, torch.zeros_like(output))
