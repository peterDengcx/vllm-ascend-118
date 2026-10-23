# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compare native and fused HY4 iHC in NPU graphs on an idle A5 device."""

import argparse
import json
from functools import partial

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.hy4_ihc import triton_ihc_post, triton_ihc_pre, triton_ihc_supported

HIDDEN_SIZE = 6144
HC_MULT = 4
CALLS_PER_GRAPH = 10


def native_ihc(x, weight, scale, base):
    flat = x.flatten(1).float()
    mixes = torch.nn.functional.linear(flat, weight) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-5)
    pre = torch.sigmoid(mixes[:, :HC_MULT] * scale[0] + base[:HC_MULT]) + 1e-6
    post = 2 * torch.sigmoid(mixes[:, HC_MULT:] * scale[1] + base[HC_MULT:]) + 1e-6
    reduced = (pre.unsqueeze(-1) * x).sum(1).to(x.dtype)
    return (post.unsqueeze(-1) * reduced.float().unsqueeze(-2) + x.float()).to(x.dtype)


def fused_ihc(x, weight, scale, base):
    reduced, post = triton_ihc_pre(x, weight, scale, base, 2.0, 1e-6, 1e-5)
    return triton_ihc_post(reduced, x, post)


def graph_microseconds(function, repeats):
    for _ in range(3):
        function()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        for _ in range(CALLS_PER_GRAPH):
            function()
    begin = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    begin.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / (repeats * CALLS_PER_GRAPH)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.npu.set_device(args.device)
    torch.manual_seed(927)
    for batch in args.batch_sizes:
        if not 1 <= batch <= 256:
            parser.error("--batch-sizes must be between 1 and 256")
        x = torch.randn(batch, HC_MULT, HIDDEN_SIZE, device="npu", dtype=torch.bfloat16)
        if not triton_ihc_supported(x):
            raise RuntimeError("This device/configuration does not support the fused HY4 iHC path")
        weight = torch.randn(2 * HC_MULT, HC_MULT * HIDDEN_SIZE, device="npu", dtype=torch.float32) * 0.05
        scale = torch.tensor([0.04, 0.03], device="npu")
        base = torch.cat([torch.full((HC_MULT,), -1.1), torch.zeros(HC_MULT)]).npu()
        expected = native_ihc(x, weight, scale, base)
        actual = fused_ihc(x, weight, scale, base)
        torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.008)
        native_us = graph_microseconds(partial(native_ihc, x, weight, scale, base), args.repeats)
        fused_us = graph_microseconds(partial(fused_ihc, x, weight, scale, base), args.repeats)
        print(
            json.dumps({"batch": batch, "native_us": native_us, "fused_us": fused_us, "speedup": native_us / fused_us}),
            flush=True,
        )


if __name__ == "__main__":
    main()
