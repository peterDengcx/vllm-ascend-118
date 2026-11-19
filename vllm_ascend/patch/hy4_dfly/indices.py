# SPDX-License-Identifier: Apache-2.0
import torch

def build_reuse_indices(
    target_rows: torch.Tensor,
    anchor_positions: torch.Tensor,
    query_width: int,
    local_window: int,
    active: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Build logical context and causal query indices, without a history scan."""
    batch, topk = target_rows.shape
    rows = out[: batch * query_width].view(batch, query_width, -1)
    rows.fill_(-1)
    starts = anchor_positions[:, None] - local_window
    history = torch.where((target_rows >= 0) & (target_rows < starts), target_rows, -1)
    rows[:, :, :topk] = history[:, None, :]
    if local_window:
        window = starts + torch.arange(local_window, device=out.device)
        rows[:, :, topk : topk + local_window] = window.clamp(min=-1)[:, None, :]
    offsets = torch.arange(query_width, device=out.device)
    block = anchor_positions[:, None] + offsets
    causal = offsets[None, :] <= offsets[:, None]
    rows[:, :, topk + local_window : topk + local_window + query_width] = torch.where(
        causal[None, :, :], block[:, None, :], -1
    )
    rows.masked_fill_(~active[:, None, None], -1)
    compact_sparse_indices(rows)


def compact_sparse_indices(rows: torch.Tensor) -> None:
    """Keep valid entries before padding for Ascend SFA compatibility.

    Target top-k padding can leave holes before valid query entries. Some
    SFA kernel paths terminate at padding and omit later query KV entries.
    Preserve the selected order while moving all padding to the tail.
    """
    valid = rows >= 0
    destinations = valid.cumsum(dim=-1) - 1
    packed = torch.zeros_like(rows)
    # Invalid entries add zero, so they cannot overwrite the last valid item.
    # Encode logical slot zero as one to distinguish it from padding.
    packed.scatter_add_(
        -1, destinations.clamp_min(0), torch.where(valid, rows + 1, 0)
    )
    rows.copy_(packed.sub_(1))


def anchor_kv_positions(seq_lens, num_rejected_tokens, num_reqs):
    """Return the bonus token's linear cache index, independently of M-RoPE.

    The target sequence length includes rejected speculative tokens. The
    bonus/anchor token is written just after its accepted context. Sparse
    attention indices address that linear cache, not the RoPE coordinates.
    """
    positions = seq_lens[:num_reqs]
    if num_rejected_tokens is not None:
        positions = positions - num_rejected_tokens[:num_reqs]
    return positions
