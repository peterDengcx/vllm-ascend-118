# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicFusedMoEMethod


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_mxfp8_routed_swiglu_clamps_before_activation_and_quantization(dtype):
    gate = torch.tensor([[20, -20, 2, -2, 10, -10]], dtype=dtype)
    up = torch.tensor([[20, 20, -20, -20, 10, -10]], dtype=dtype)
    projected = torch.cat((gate, up), dim=-1)
    inputs = SimpleNamespace(swiglu_limit=10.0)
    scheme = AscendW8A8MXFP8DynamicFusedMoEMethod.__new__(AscendW8A8MXFP8DynamicFusedMoEMethod)
    scheme.apply_gmm1 = Mock(return_value=projected)
    scheme.apply_act_quant = Mock(side_effect=lambda _, states: (states, None))

    def swiglu(states):
        gate, up = states.float().chunk(2, dim=-1)
        return (F.silu(gate) * up).to(dtype)

    with (
        patch("torch_npu.npu_swiglu", side_effect=swiglu),
        patch("torch_npu.npu_grouped_matmul_swiglu_quant_v2", side_effect=AssertionError("clamp is unsupported")),
    ):
        actual, _ = scheme.apply_gmm1_act_quant(inputs)

    expected = F.silu(torch.tensor([[10, -20, 2, -2, 10, -10]]).float())
    expected *= torch.tensor([[10, 10, -10, -10, 10, -10]])
    torch.testing.assert_close(actual, expected.to(dtype))
    scheme.apply_gmm1.assert_called_once_with(inputs)
    scheme.apply_act_quant.assert_called_once()
    torch.testing.assert_close(scheme.apply_act_quant.call_args.args[1], actual)
