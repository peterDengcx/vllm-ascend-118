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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sink_keeps_float32_scale_until_final_output_cast(dtype):
    # One ordinary key has score 0 and these exactly representable values.
    # The extra sink has score 0.7 and zero value, so direct attention is an
    # independent reference without a pre-rounded normalization factor.
    values = torch.tensor([-2.453125, -2.40625, 2.359375, 3.125], dtype=dtype).view(1, 1, -1)
    sink = torch.tensor([0.7])
    scores = torch.tensor([0.0, 0.7])
    expected = (scores.softmax(0)[0] * values.float()).to(dtype)
    actual = apply_attention_sink(values, torch.zeros(1, 1), torch.ones(1, 1), sink)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
