from collections.abc import Mapping, Sequence
from typing import cast

import torch

from miles.utils.sampling_mask import RolloutSamplingMask


def get_rollout_sampling_mask(batch: Mapping[str, object]) -> list[RolloutSamplingMask]:
    """Read the complete sampling mask required by an actor scoring pass."""
    sampling_mask = batch.get("rollout_sampling_mask")
    if sampling_mask is None:
        raise ValueError("truncated-sampling actor scoring requires rollout_sampling_mask")
    return [RolloutSamplingMask.from_mask_list(mask_list) for mask_list in cast(list[list[list[int]]], sampling_mask)]


def build_local_sampling_mask(
    logits: torch.Tensor,
    sampling_mask: RolloutSamplingMask,
    response_indices: Sequence[int],
    *,
    tp_rank: int,
) -> torch.Tensor:
    """Build the dense local-vocabulary mask consumed by the log-prob primitive.

    Args:
        logits: ``[local_rows, local_vocab_size]`` response-row logits this
            rank holds (TP vocab shard, CP row subset).
        sampling_mask: the sample's complete sampling mask.
        response_indices: ``[local_rows]`` global response position of each row.
        tp_rank: this rank's index in the TP group.

    Returns:
        Bool mask shaped like ``logits``; True marks ids inside the support.
    """
    if len(response_indices) != logits.size(0):
        raise ValueError(
            f"sampling-mask rows must align with logits: indices={len(response_indices)}, logits={logits.size(0)}"
        )

    # CP response rows form a small number of contiguous runs, so the CSR
    # gather is a handful of CPU slices before the GPU expansion.
    selected_ids, lengths = sampling_mask.select_masks(response_indices)
    selected_ids = selected_ids.to(logits.device)
    row_indices = torch.repeat_interleave(
        torch.arange(len(response_indices), dtype=torch.long, device=logits.device),
        lengths.to(device=logits.device, dtype=torch.long),
    )
    local_vocab_size = logits.size(-1)
    vocab_start = tp_rank * local_vocab_size
    is_local = (selected_ids >= vocab_start) & (selected_ids < vocab_start + local_vocab_size)
    flat_local_indices = row_indices[is_local] * local_vocab_size + selected_ids[is_local].to(torch.long) - vocab_start
    mask = torch.zeros(logits.numel(), dtype=torch.bool, device=logits.device)
    mask[flat_local_indices] = True
    return mask.view_as(logits)
