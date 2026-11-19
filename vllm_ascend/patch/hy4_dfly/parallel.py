# SPDX-License-Identifier: Apache-2.0
"""Parallel-layout checks and rank-local query padding for HY4 DFly."""

from copy import copy

import torch


def validate_dfly_parallel(spec):
    """DP replicas may differ in load; target and draft must share local TP."""
    parallel = spec.target_parallel_config
    for field in (
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, field) != 1:
            raise ValueError(f"HY4 DFly requires {field}=1")
    if parallel.pipeline_parallel_size < 1:
        raise ValueError("HY4 DFly requires pipeline_parallel_size >= 1")
    if parallel.pipeline_parallel_size > 1:
        from vllm_ascend.patch.hy4_dfly.pipeline import validate_dfly_pipeline_config

        validate_dfly_pipeline_config(spec)
    if parallel.data_parallel_size < 1:
        raise ValueError("HY4 DFly requires data_parallel_size >= 1")
    if spec.draft_tensor_parallel_size not in (None, parallel.tensor_parallel_size):
        raise ValueError("HY4 DFly draft and target must use the same TP size")


def inherit_dfly_parallel(spec):
    """Keep DP rank/local-rank/backend information lost by upstream's draft config.

    DFly runs dense layers in each target TP group and aliases that replica's
    target KV. Do not create a single global draft replica for several DP ranks.
    Copy the config so subsequent draft changes cannot modify the target.
    """
    if spec.target_parallel_config.pipeline_parallel_size > 1:
        from vllm_ascend.patch.worker.mtp_pipeline import local_mtp_parallel_config

        spec.draft_parallel_config = local_mtp_parallel_config(spec.target_parallel_config)
    else:
        spec.draft_parallel_config = copy(spec.target_parallel_config)


def prepare_dfly_query_padding(proposer, num_input_tokens):
    """Pad features/indices after the runner's existing DP token synchronization.

    No extra collectives: active, idle and skip-drafting ranks must participate
    in exactly the same synchronization sequence used by the base proposer.
    """
    width = proposer.num_query_per_req
    if num_input_tokens % width or not 0 < num_input_tokens <= proposer.max_query_tokens:
        raise ValueError("HY4 DFly padded tokens must be complete query blocks within capacity")
    batch = num_input_tokens // width
    dummy = getattr(proposer, "_dfly_dummy", False)
    rows = proposer._source_rows
    local_batch = 0 if dummy or rows is None else rows.numel()
    if local_batch > batch:
        raise ValueError("HY4 DFly padded batch is smaller than its local batch")

    # Never reuse anchors, query IDs, positions or indices left by a previous
    # larger batch on a less busy (or completely idle) DP rank.
    hidden = torch.zeros_like(proposer.hidden_states[:batch])
    if local_batch:
        hidden[:local_batch].copy_(proposer._anchor_hidden[:local_batch])
    proposer.model.anchor_hidden = hidden
    start = local_batch * width
    ids = proposer.input_ids[start:num_input_tokens].view(-1, width)
    ids.fill_(proposer.model.config.mask_token_id)
    ids[:, 0].zero_()
    proposer.positions[start:num_input_tokens].zero_()
    if proposer.uses_mrope:
        proposer.mrope_positions[:, start:num_input_tokens].zero_()
    proposer._slot_mapping_buffer[start:num_input_tokens].fill_(-1)
    for mapping in getattr(proposer, "slot_mapping_group", ()):
        mapping[start:num_input_tokens].fill_(-1)
    proposer.model.model.topk_indices[:, :num_input_tokens].fill_(-1)
    return local_batch
