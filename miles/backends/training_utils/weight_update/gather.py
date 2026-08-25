"""Gather primitives for weight update.

Backend-neutral: callers pass the process group; nothing here queries training
parallel state.
"""

import math
from collections.abc import Sequence

import torch
import torch.distributed as dist


def broadcast_from_owners(
    named_tensors: Sequence[tuple[str, torch.Tensor]],
    group: dist.ProcessGroup,
) -> list[tuple[str, torch.Tensor]]:
    """Materialize the union of every rank's named tensors on all ranks of ``group``.

    For disjoint ownership with heterogeneous shapes and names (e.g. PP-local
    exports). Owner metadata is exchanged internally; same-dtype tensors from
    one owner are coalesced into one flat broadcast.
    """
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return list(named_tensors)
    rank = dist.get_rank(group=group)
    global_ranks = dist.get_process_group_ranks(group)
    device = torch.cuda.current_device()

    local_meta = [(n, tuple(t.shape), t.dtype) for n, t in named_tensors]
    all_meta: list = [None] * world_size
    dist.all_gather_object(all_meta, local_meta, group=group)

    local_by_name = {n: t for n, t in named_tensors}
    merged: dict[str, torch.Tensor] = {}
    for src, meta in enumerate(all_meta):
        by_dtype: dict = {}
        for n, shape, dtype in meta:
            by_dtype.setdefault(dtype, []).append((n, shape))
        for dtype, entries in by_dtype.items():
            numel = sum(math.prod(shape) for _, shape in entries)
            flat = torch.empty(numel, dtype=dtype, device=device)
            if src == rank:
                off = 0
                for n, shape in entries:
                    k = math.prod(shape)
                    flat[off : off + k].copy_(local_by_name[n].reshape(-1))
                    off += k
            dist.broadcast(flat, src=global_ranks[src], group=group)
            off = 0
            for n, shape in entries:
                k = math.prod(shape)
                merged[n] = flat[off : off + k].view(shape)
                off += k
    return sorted(merged.items())
