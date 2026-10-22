# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def _proposer_and_metadata():
    proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
    proposer.arange = torch.arange(8, dtype=torch.int32)
    proposer.token_arange_np = np.arange(8, dtype=np.int32)
    proposer.runner = SimpleNamespace(actual_seq_lengths_q=[4, 7], attn_state=None, decode_token_per_req=4)
    starts = torch.tensor([0, 4, 7], dtype=torch.int32)
    lengths = torch.tensor([100, 200], dtype=torch.int32)
    common = AscendCommonAttentionMetadata(
        query_start_loc=starts,
        query_start_loc_cpu=starts.clone(),
        seq_lens=lengths,
        seq_lens_cpu=lengths.clone(),
        _seq_lens_cpu=lengths.clone(),
        seq_lens_cpu_upper_bound=lengths.clone(),
        num_reqs=2,
        num_actual_tokens=7,
        num_input_tokens=7,
        max_query_len=4,
        max_seq_len=200,
        block_table_tensor=torch.zeros(2, 4, dtype=torch.int32),
        slot_mapping=torch.arange(7, dtype=torch.int32),
        positions=torch.arange(7, dtype=torch.int32),
        positions_cpu=torch.arange(7, dtype=torch.int32),
        causal=True,
    )
    return proposer, common


@pytest.mark.parametrize("async_metadata", [False, True])
def test_padded_draft_preserves_scheduler_sequence_bound(monkeypatch, async_metadata):
    monkeypatch.setattr("vllm_ascend.spec_decode.llm_base_proposer.HAS_TRITON", False)
    proposer, common = _proposer_and_metadata()
    if async_metadata:
        common.seq_lens_cpu = None
    spec = SimpleNamespace(cu_num_draft_tokens=torch.tensor([3, 5], dtype=torch.int32))

    draft, _, sample_indices, rejected = proposer.prepare_inputs_padded(common, spec, torch.tensor([2, 1]))

    assert draft.max_seq_len == 200
    torch.testing.assert_close(draft.seq_lens, torch.tensor([100, 200], dtype=torch.int32))
    torch.testing.assert_close(sample_indices, torch.tensor([1, 4]))
    torch.testing.assert_close(rejected, torch.tensor([2, 2]))
    assert common.max_seq_len == 200


def test_compacted_draft_uses_post_rejection_sequence_bound(monkeypatch):
    monkeypatch.setattr("vllm_ascend.spec_decode.llm_base_proposer.is_pin_memory_available", lambda: False)
    proposer, common = _proposer_and_metadata()

    draft, indices = proposer.prepare_inputs(common, [[11, 12], [13]], [3, 2])

    assert draft.max_seq_len == 198
    torch.testing.assert_close(draft.seq_lens, torch.tensor([98, 198], dtype=torch.int32))
    torch.testing.assert_close(indices, torch.tensor([0, 1, 4], dtype=torch.int32))
    assert common.max_seq_len == 200
