# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
import torch_npu
import vllm_ascend.vllm_ascend_C  # noqa: F401

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.model_executor.offloader.prefetch import (
    _ASCEND_PREFETCH_PARAMETER_ATTR,
    _ModuleOffloader,
    _prefetch_device_loading_context,
)
from vllm_ascend.ops.attention_sink import apply_attention_sink


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.uint8])
@pytest.mark.parametrize("contiguous", [False, True])
@torch.inference_mode()
def test_prefetch_loading_reuses_pinned_storage_after_transpose(dtype, contiguous):
    source = (torch.arange(4096).reshape(2, 32, 64) % 11).to(dtype)
    storage = torch.empty_like(source, pin_memory=True).copy_(source)
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(storage, requires_grad=False)
    setattr(module.weight, _ASCEND_PREFETCH_PARAMETER_ATTR, True)
    pointer = module.weight.data_ptr()

    def unexpected_fallback(*args):
        raise AssertionError("Marked dense weights should reuse their host storage")

    with _prefetch_device_loading_context(module, torch.device("npu:0"), fallback=unexpected_fallback):
        module.weight.data = module.weight.data.transpose(1, 2)
        if contiguous:
            module.weight.data = module.weight.data.contiguous()
    assert module.weight.data_ptr() == pointer
    assert module.weight.is_pinned()
    assert module.weight.device.type == "cpu"
    torch.testing.assert_close(module.weight.float(), source.transpose(1, 2).float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.uint8])
@torch.inference_mode()
def test_hy4_strided_pinned_weight_prefetch_capture_and_replay(dtype):
    shape, stride = (8, 1024, 512), (1024 * 512, 1, 1024)
    source = torch.empty_strided(shape, stride, dtype=dtype, pin_memory=True)
    source.fill_(1)
    target = torch.empty_strided(shape, stride, dtype=dtype, device="npu")
    offloader = _ModuleOffloader.__new__(_ModuleOffloader)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        offloader._copy_param_to_static(source, target)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        offloader._copy_param_to_static(source, target)
    for value in (2, 3):
        source.fill_(value)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(target.cpu().float(), source.float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rounding_boundary", [False, True])
@torch.inference_mode()
def test_hy4_sparse_sink_matches_dense_attention(dtype, rounding_boundary):
    generator = torch.Generator().manual_seed(73)
    tokens, heads, latent, rope_dim, block_size, topk = 4, 8, 512, 64, 128, 2048
    query = torch.randn(tokens, heads, latent, generator=generator).to(dtype) * 0.1
    query_rope = torch.randn(tokens, heads, rope_dim, generator=generator).to(dtype) * 0.1
    key = torch.randn(block_size, latent, generator=generator).to(dtype) * 0.1
    key_rope = torch.randn(block_size, rope_dim, generator=generator).to(dtype) * 0.1
    indices = torch.full((tokens, 1, topk), -1, dtype=torch.int32)
    for token in range(tokens):
        indices[token, 0, : token + 1] = torch.arange(token + 1)
    scale = 256**-0.5
    sink = torch.linspace(-3, 3, heads)
    if rounding_boundary:
        # One visible key makes the pre-sink attention output exactly equal
        # to its value, exposing rounding of the FP32 sink scale itself.
        tokens = 1
        query = torch.zeros(tokens, heads, latent, dtype=dtype)
        query_rope = torch.zeros(tokens, heads, rope_dim, dtype=dtype)
        key[:] = torch.tensor([-2.453125, -2.40625, 2.359375, 3.125], dtype=dtype).repeat(latent // 4)
        indices = torch.full((tokens, 1, topk), -1, dtype=torch.int32)
        indices[:, :, 0] = 0
        sink = torch.full((heads,), 0.7)
    output, maximum, denominator = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query.npu(),
        key=key.view(1, block_size, 1, latent).npu(),
        value=key.view(1, block_size, 1, latent).npu(),
        sparse_indices=indices.npu(),
        scale_value=scale,
        sparse_block_size=1,
        block_table=torch.tensor([[0]], dtype=torch.int32, device="npu"),
        actual_seq_lengths_query=torch.tensor([tokens], dtype=torch.int32, device="npu"),
        actual_seq_lengths_kv=torch.tensor([tokens], dtype=torch.int32, device="npu"),
        query_rope=query_rope.npu(),
        key_rope=key_rope.view(1, block_size, 1, rope_dim).npu(),
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=True,
    )
    actual = apply_attention_sink(output, maximum, denominator, sink.npu()).cpu()
    expected = []
    for token in range(tokens):
        scores = query[token].float() @ key[: token + 1].float().T
        scores += query_rope[token].float() @ key_rope[: token + 1].float().T
        scores = torch.cat([scores * scale, sink.unsqueeze(-1)], dim=-1)
        expected.append(scores.softmax(-1)[:, :-1] @ key[: token + 1].float())
    if rounding_boundary:
        torch.testing.assert_close(actual, torch.stack(expected).to(dtype), atol=0, rtol=0)
    else:
        torch.testing.assert_close(actual.float(), torch.stack(expected), atol=2e-3, rtol=2e-2)


@torch.inference_mode()
def test_hy4_partial_rope_and_sparse_selection_beyond_topk():
    tokens, heads, head_dim, rope_dim, length, topk = 3, 32, 128, 64, 4096, 2048
    generator = torch.Generator().manual_seed(29)
    query = torch.randn(tokens, heads, head_dim, generator=generator).bfloat16()
    key = torch.randn(length, 1, head_dim, generator=generator).bfloat16()
    weights = torch.rand(tokens, heads, generator=generator).bfloat16()
    angles = torch.arange(length).float().unsqueeze(1) * (
        10000000 ** (-torch.arange(0, rope_dim, 2).float() / rope_dim)
    )
    cosine, sine = angles.cos().repeat(1, 2), angles.sin().repeat(1, 2)
    backend = AscendSFAIndexerBackend.__new__(AscendSFAIndexerBackend)
    torch.nn.Module.__init__(backend)
    backend.qk_rope_head_dim = rope_dim
    backend.head_dim = head_dim
    backend.rope_dim_at_end = True
    backend.is_rope_neox_style = False
    rotated_query = backend._apply_rope(query.npu(), cosine[-tokens:].npu(), sine[-tokens:].npu())
    rotated_key = backend._apply_rope(key.npu(), cosine.npu(), sine.npu())

    def reference_rotation(x, cos, sin):
        nope, pe = x.split([head_dim - rope_dim, rope_dim], dim=-1)
        pairs = pe.float().reshape(*pe.shape[:-1], -1, 2)
        rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).flatten(-2)
        cos = cos[:, : rope_dim // 2].repeat_interleave(2, dim=-1).unsqueeze(1)
        sin = sin[:, : rope_dim // 2].repeat_interleave(2, dim=-1).unsqueeze(1)
        return torch.cat([nope, (pe.float() * cos + rotated * sin).to(x.dtype)], dim=-1)

    expected_query = reference_rotation(query, cosine[-tokens:], sine[-tokens:])
    expected_key = reference_rotation(key, cosine, sine)
    torch.testing.assert_close(rotated_query.cpu(), expected_query)
    torch.testing.assert_close(rotated_key.cpu(), expected_key)
    actual, _ = torch_npu.npu_lightning_indexer(
        query=rotated_query,
        key=rotated_key.view(-1, 128, 1, head_dim),
        weights=weights.npu(),
        actual_seq_lengths_query=torch.tensor([tokens], dtype=torch.int32, device="npu"),
        actual_seq_lengths_key=torch.tensor([length], dtype=torch.int32, device="npu"),
        block_table=torch.arange(length // 128, dtype=torch.int32, device="npu").unsqueeze(0),
        layout_query="TND",
        layout_key="PA_BSND",
        sparse_count=topk,
        sparse_mode=3,
    )
    scores = torch.einsum("thd,kd->thk", expected_query.float(), expected_key[:, 0].float()).relu()
    scores = (scores * weights.float().unsqueeze(-1)).sum(1)
    for token in range(tokens):
        visible = length - tokens + token + 1
        scores[token, visible:] = -float("inf")
        expected = scores[token].topk(topk).indices
        selected = actual[token].flatten().cpu().long()
        assert torch.all((selected >= 0) & (selected < visible))
        assert torch.isin(selected, expected).float().mean() > 0.99
