# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata, AscendSFAMetadataBuilder
from vllm_ascend.ops import rotary_embedding
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def _make_proposer(num_actual, num_input, mode, use_cuda_graph, supports_mm_inputs):
    """Keep real first-pass input construction and replace only runner services."""
    capacity = num_input + 8
    num_reqs = num_actual // 2 if mode == CUDAGraphMode.FULL else 1
    padded_reqs = num_input // 2 if mode == CUDAGraphMode.FULL else num_reqs
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.method = "mtp"
    proposer.needs_extra_input_slots = False
    proposer.uses_mrope = True
    proposer.pass_hidden_states_to_model = True
    proposer.supports_mm_inputs = supports_mm_inputs
    proposer.use_cuda_graph = use_cuda_graph
    proposer.parallel_drafting = False
    proposer.dcp_size = 1
    proposer.draft_model_config = SimpleNamespace(use_mla=True)
    proposer.draft_window_size = None
    proposer.input_ids = torch.full((capacity,), -99, dtype=torch.int32)
    proposer.hidden_states = torch.full((capacity, 4), -99.0)
    proposer.inputs_embeds = torch.full((capacity, 4), -99.0)
    # A real MRoPE proposer has no one-dimensional positions buffer.
    proposer.mrope_positions = torch.full((3, capacity + 1), -99, dtype=torch.int32)
    proposer.model = SimpleNamespace(embed_input_ids=Mock(side_effect=lambda ids: ids[:, None].expand(-1, 4).float()))
    proposer.slot_mapping_group = [torch.full((capacity,), -1, dtype=torch.int32)]
    proposer.seq_lens_group = [torch.zeros(capacity, dtype=torch.int32)]
    proposer.query_start_loc_group = [torch.zeros(capacity + 1, dtype=torch.int32)]
    proposer.query_start_loc = SimpleNamespace(
        gpu=torch.zeros(capacity + 1, dtype=torch.int32), cpu=torch.zeros(capacity + 1, dtype=torch.int32)
    )

    def pad_queries(buffer, *args):
        buffer.gpu[num_reqs + 1 : padded_reqs + 1] = torch.arange(num_reqs + 1, padded_reqs + 1) * 2
        buffer.cpu.copy_(buffer.gpu)
        return padded_reqs

    def dispatch(num_tokens, **kwargs):
        descriptor = SimpleNamespace(
            num_tokens=num_input if mode == CUDAGraphMode.FULL else num_tokens, num_reqs=padded_reqs
        )
        return mode, descriptor

    proposer.runner = SimpleNamespace(
        dcp_manager=None,
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        cudagraph_dispatcher=SimpleNamespace(dispatch=dispatch),
        _sync_metadata_across_dp=Mock(return_value=(num_input, torch.tensor([num_input, num_input]), mode)),
        _pad_query_start_loc_for_fia=pad_queries,
        seq_lens=torch.arange(capacity, dtype=torch.int32) + 100,
        optimistic_seq_lens_cpu=torch.arange(capacity, dtype=torch.int32) + 100,
    )
    query_start_loc = torch.arange(num_reqs + 1, dtype=torch.int32) * (num_actual // num_reqs)
    common = SimpleNamespace(
        batch_size=lambda: num_reqs,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual,
        num_input_tokens=num_actual,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.clone(),
        seq_lens=torch.arange(num_reqs, dtype=torch.int32) + 100,
        _seq_lens_cpu=torch.arange(num_reqs, dtype=torch.int32) + 100,
        num_computed_tokens_cpu=None,
        max_query_len=num_actual // num_reqs,
        max_seq_len=512,
        context_parallel_metadata=None,
        causal=True,
        attn_state=None,
        block_table_tensor=torch.zeros(num_reqs, 32, dtype=torch.int32),
        # Exercise a runner-owned slot buffer whose tail still contains old slots.
        slot_mapping=torch.arange(capacity, dtype=torch.int32),
        positions=torch.arange(capacity, dtype=torch.int32),
        rotary_positions=torch.zeros(3, capacity, dtype=torch.int32),
    )
    return proposer, common


@pytest.mark.parametrize(
    "num_actual,num_input,mode,use_cuda_graph",
    [
        pytest.param(293, 293, CUDAGraphMode.NONE, True, id="equal-dp-prefill"),
        pytest.param(293, 306, CUDAGraphMode.NONE, True, id="unequal-dp-prefill"),
        pytest.param(293, 306, CUDAGraphMode.NONE, False, id="unequal-dp-eager"),
        pytest.param(6, 8, CUDAGraphMode.FULL, True, id="padded-full-decode"),
    ],
)
@pytest.mark.parametrize("supports_mm_inputs", [False, True])
@pytest.mark.parametrize("async_metadata", [False, True])
def test_mrope_propose_aligns_inputs_rope_and_slots(
    monkeypatch, num_actual, num_input, mode, use_cuda_graph, supports_mm_inputs, async_metadata
):
    """DP/graph padding must preserve valid axes and mask every added KV slot."""
    proposer, common = _make_proposer(num_actual, num_input, mode, use_cuda_graph, supports_mm_inputs)
    common.seq_lens_cpu = None if async_metadata else common._seq_lens_cpu.clone()
    positions = torch.arange(num_actual, dtype=torch.int32).repeat(3, 1) + torch.tensor([[0], [400], [800]])
    hidden_states = torch.arange(num_actual * 4, dtype=torch.float32).reshape(num_actual, 4)
    rope_parameters = {"mrope_section": [12, 10, 10], "mrope_interleaved": True}
    # Distinct cache entries ensure that preserving shape alone cannot pass.
    cache = torch.arange(2048 * 32, dtype=torch.float32).reshape(2048, 32).repeat(1, 2)
    monkeypatch.setattr(rotary_embedding, "_cos_cache", cache)
    monkeypatch.setattr(rotary_embedding, "_sin_cache", -cache)
    monkeypatch.setattr(rotary_embedding, "_cos_mla", torch.zeros(num_input, 1, 1, 64))
    monkeypatch.setattr(rotary_embedding, "_sin_mla", torch.zeros(num_input, 1, 1, 64))
    builder = AscendSFAMetadataBuilder.__new__(AscendSFAMetadataBuilder)
    builder.kernel_block_size = 128
    builder.nope = False
    builder.use_pcp = False
    builder.decode_threshold = 2
    builder.mrope_parameters = rope_parameters
    builder.model_config = SimpleNamespace(get_head_size=lambda: 64)
    builder.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda *args: None)
    builder.metadata_cls = AscendSFAMetadata

    class MetadataChecked(Exception):
        pass

    def check_metadata(metadata, padded_tokens, actual_tokens):
        # Run the real SFA builder before any mocked model/NPU invocation.
        sfa = builder._build(metadata)
        assert padded_tokens == num_input and actual_tokens == num_actual
        assert metadata.num_actual_tokens == num_actual
        assert sfa.positions.shape == (3, num_input)
        assert sfa.cos.shape == sfa.sin.shape == (num_input, 1, 1, 64)
        assert sfa.cos.data_ptr() == rotary_embedding._cos_mla.data_ptr()
        assert metadata.rotary_positions.data_ptr() == proposer.mrope_positions.data_ptr()
        torch.testing.assert_close(sfa.positions[:, :num_actual], positions.long())
        expected_cos, expected_sin = rotary_embedding.get_cos_and_sin_mla(positions, rope_parameters=rope_parameters)
        torch.testing.assert_close(sfa.cos[:num_actual], expected_cos)
        torch.testing.assert_close(sfa.sin[:num_actual], expected_sin)
        torch.testing.assert_close(proposer.hidden_states[:num_actual], hidden_states)
        torch.testing.assert_close(sfa.slot_mapping[:num_actual], torch.arange(num_actual, dtype=torch.int32))
        assert torch.count_nonzero(proposer.mrope_positions[:, num_actual:num_input]) == 0
        assert torch.count_nonzero(proposer.input_ids[num_actual:num_input]) == 0
        assert torch.count_nonzero(proposer.hidden_states[num_actual:num_input]) == 0
        assert torch.all(sfa.slot_mapping[num_actual:num_input] == -1)
        if supports_mm_inputs:
            torch.testing.assert_close(
                proposer.inputs_embeds[:num_actual], proposer.input_ids[:num_actual, None].expand(-1, 4).float()
            )
            assert torch.count_nonzero(proposer.inputs_embeds[num_actual:num_input]) == 0
        # No buffers beyond this step's padded input should be overwritten.
        assert torch.all(proposer.mrope_positions[:, num_input:] == -99)
        raise MetadataChecked

    proposer.build_draft_attn_metadata = check_metadata
    with pytest.raises(MetadataChecked):
        proposer._propose(
            1,
            target_token_ids=torch.arange(num_actual, dtype=torch.int32),
            target_positions=positions,
            target_hidden_states=hidden_states,
            next_token_ids=torch.full((common.num_reqs,), 7, dtype=torch.int32),
            token_indices_to_sample=None,
            common_attn_metadata=common,
            target_model_batch_desc=SimpleNamespace(uniform=mode == CUDAGraphMode.FULL),
            sampling_metadata=None,
        )


@pytest.mark.parametrize(
    "num_input,mode",
    [
        pytest.param(2, CUDAGraphMode.NONE, id="unpadded-decode"),
        pytest.param(293, CUDAGraphMode.NONE, id="prefill"),
        pytest.param(306, CUDAGraphMode.NONE, id="dp-padded-prefill"),
        pytest.param(16, CUDAGraphMode.FULL, id="graph-padded-decode"),
    ],
)
@pytest.mark.parametrize("async_metadata", [False, True])
def test_later_mrope_draft_steps_align_rope_with_model_inputs(monkeypatch, num_input, mode, async_metadata):
    """All three merged MTP forwards need one RoPE row per model input row."""
    batch_size = 2
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.method = "mtp"
    proposer.uses_mrope = True
    proposer.max_model_len = 512
    proposer.block_size = 128
    proposer.has_gdn = False
    proposer.use_compress = False
    proposer.sliding_window = None
    proposer.arange = torch.arange(num_input + 1, dtype=torch.int32)
    proposer.token_arange_np = np.arange(num_input + 1, dtype=np.int32)
    proposer.slot_mapping_group = [torch.full((num_input,), -1, dtype=torch.int32) for _ in range(3)]
    proposer.seq_lens_group = [torch.zeros(num_input, dtype=torch.int32) for _ in range(3)]
    proposer.query_start_loc_group = [torch.zeros(num_input + 1, dtype=torch.int32) for _ in range(3)]
    proposer.runner = SimpleNamespace(dcp_manager=None)

    rope_parameters = {"mrope_section": [12, 10, 10], "mrope_interleaved": True}
    cache = torch.arange(1024 * 32, dtype=torch.float32).reshape(1024, 32).repeat(1, 2)
    monkeypatch.setattr(rotary_embedding, "_cos_cache", cache)
    monkeypatch.setattr(rotary_embedding, "_sin_cache", -cache)
    builder = AscendSFAMetadataBuilder.__new__(AscendSFAMetadataBuilder)
    builder.kernel_block_size = 128
    builder.nope = False
    builder.use_pcp = False
    builder.decode_threshold = 4
    builder.mrope_parameters = rope_parameters
    builder.model_config = SimpleNamespace(get_head_size=lambda: 64)
    builder.attn_mask_builder = SimpleNamespace(get_attention_mask=lambda *args: None)
    builder.metadata_cls = AscendSFAMetadata
    group = SimpleNamespace(get_metadata_builder=lambda: builder)

    initial_positions = torch.tensor([[10, 20], [110, 120], [210, 220]], dtype=torch.int32)
    positions = initial_positions.clone()
    common = SimpleNamespace(
        num_reqs=batch_size,
        num_actual_tokens=num_input,
        num_input_tokens=num_input,
        seq_lens=torch.tensor([11, 21], dtype=torch.int32),
        seq_lens_cpu=None if async_metadata else torch.tensor([11, 21], dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([11, 21], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([11, 21], dtype=torch.int32),
        num_computed_tokens_cpu=None,
        positions=torch.arange(num_input, dtype=torch.int32),
        rotary_positions=torch.full((3, num_input), 99, dtype=torch.int32),
        block_table_tensor=torch.tensor([[2], [7]], dtype=torch.int32),
        context_parallel_metadata=None,
        max_seq_len=21,
        causal=True,
    )
    original_common = common
    previous_rope = previous_rope_values = None
    for step in (1, 2):
        common, sfa = proposer.attn_update_stack_num_spec_norm(
            step, common, batch_size, num_input, positions, mode, attn_group=group
        )
        expected_positions = initial_positions + step
        assert sfa.positions.shape == (3, num_input)
        assert sfa.cos.shape == sfa.sin.shape == (num_input, 1, 1, 64)
        torch.testing.assert_close(sfa.positions[:, :batch_size], expected_positions.long())
        expected_cos, expected_sin = rotary_embedding.get_cos_and_sin_mla(
            expected_positions, rope_parameters=rope_parameters
        )
        torch.testing.assert_close(sfa.cos[:batch_size], expected_cos)
        torch.testing.assert_close(sfa.sin[:batch_size], expected_sin)
        assert torch.count_nonzero(sfa.positions[:, batch_size:]) == 0
        assert torch.all(sfa.slot_mapping[batch_size:] == -1)
        assert sfa.slot_mapping[:batch_size].tolist() == [256 + 10 + step, 896 + 20 + step]
        assert common.seq_lens[:batch_size].tolist() == [11 + step, 21 + step]
        assert common.max_seq_len == 21 + step
        assert sfa.max_seq_len == common.max_seq_len
        assert common.seq_lens_cpu_upper_bound.tolist() == [11 + step, 21 + step]
        if previous_rope is not None:
            assert sfa.cos.data_ptr() != previous_rope.data_ptr()
            torch.testing.assert_close(previous_rope, previous_rope_values)
        previous_rope, previous_rope_values = sfa.cos, sfa.cos.clone()

    assert original_common.seq_lens.tolist() == [11, 21]
    assert original_common.seq_lens_cpu_upper_bound.tolist() == [11, 21]
    assert torch.all(original_common.rotary_positions == 99)
