"""QSA sparse attention: attend only to the tokens the indexer selected.

Exact by construction rather than approximate. The indexer hands back, per query,
up to ``indexer_budget`` key positions (``-1`` padding where it selected fewer);
this turns those into an attention mask and runs ordinary scaled dot-product
attention under it. Restricting the mask is *precisely* the semantics of "attend
only to the selected tokens", so this matches the model spec exactly -- unlike
falling back to dense attention, which silently changes the logits.

A fused triton kernel is the eventual answer for speed and memory: the mask is
``[heads, T, T]`` booleans, so it costs ``T^2`` bits per head rather than the
``T * budget`` the selection actually implies. At the sequence lengths RL rollouts
use that is a few tens of MB and worth paying to get the numerics right first; the
kernel can drop in behind this same signature.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def selection_to_mask(topk_indices: Tensor, seq_len: int) -> Tensor:
    """``[T, budget]`` int32 selections (``-1`` = unused) -> ``[T, S]`` bool mask.

    The diagonal is always allowed: a query attends to itself regardless of what
    the indexer chose, matching the reference behaviour where the current token is
    never a candidate for selection but is always attended.
    """
    tokens = topk_indices.shape[0]
    mask = torch.zeros(tokens, seq_len, dtype=torch.bool, device=topk_indices.device)
    valid = topk_indices >= 0
    if valid.any():
        rows = torch.arange(tokens, device=topk_indices.device).unsqueeze(-1).expand_as(topk_indices)
        mask[rows[valid], topk_indices[valid].long()] = True
    # No forced diagonal: sglang's kernel attends exactly the listed indices
    # (top-k expansion + the query's partial-block tail, which contains the query
    # itself whenever its block is incomplete). Forcing the diagonal here made
    # this path diverge from both sglang and the triton kernel on saturated rows.
    return mask


def qsa_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    topk_indices: Tensor,
    scale: float | None = None,
    cu_seqlens: Tensor | None = None,
    compress_ratio: int = 4,
) -> Tensor:
    """``q`` ``[T, Hq, D]``, ``k``/``v`` ``[S, Hkv, D]`` -> ``[T, Hq, D]``.

    Grouped-query: ``Hq`` must be a multiple of ``Hkv``, and each group of query
    heads shares a kv head. The selection is per query token, not per head, so one
    mask serves every head.
    """
    tokens, hq, dim = q.shape
    seq, hkv, _ = k.shape
    assert hq % hkv == 0, f"query heads {hq} must be a multiple of kv heads {hkv}"
    repeat = hq // hkv

    mask = selection_to_mask(topk_indices, seq)          # [T, S]

    # The tail: each query also sees its own partial block. The indexer scores only
    # complete blocks, so without this a query can never attend the last
    # ``(pos+1) % compress_ratio`` tokens -- including itself. sglang appends
    # exactly these positions after the top-k expansion
    # (qsa/kernel.py expand_qsa_block_indices: tail_start = floor((pos+1)/r)*r),
    # and dropping them is a systematic modeling change in every full-attention
    # layer, not a rounding difference: it surfaced as Megatron being consistently
    # more confident than sglang on the same tokens.
    idx = torch.arange(seq, device=q.device)
    if cu_seqlens is not None:
        starts = cu_seqlens[:-1].long()
        seg = torch.zeros(seq, dtype=torch.long, device=q.device)
        seg[cu_seqlens[1:-1].long()] = 1
        seg = seg.cumsum(0)
        pos = idx - starts[seg]           # position within each sequence
        seq_start = idx - pos             # pack index of each sequence's first token
    else:
        pos = idx
        seq_start = torch.zeros_like(idx)
    tail_start = (pos + 1) // compress_ratio * compress_ratio  # in-sequence
    col_pos = pos.unsqueeze(0)                                  # [1, S]
    row_tail = (seq_start + tail_start)[:tokens].unsqueeze(1)   # [T, 1], pack index
    mask |= idx.unsqueeze(0) >= row_tail

    # Causality is already implied by the indexer's block-causal scoring and by the
    # tail construction, but assert it here so a selection bug cannot leak future
    # tokens into training.
    causal = torch.ones(tokens, seq, dtype=torch.bool, device=q.device).tril(
        diagonal=seq - tokens
    )
    mask &= causal
    if cu_seqlens is not None:
        # Packed (THD) batches put several sequences in one tensor. Attention must
        # not cross a boundary, and the indexer's selection is expressed in
        # positions within the pack, so confine each query to its own segment.
        mask &= seg[:tokens].unsqueeze(1) == seg.unsqueeze(0)

    qh = q.transpose(0, 1)                               # [Hq, T, D]
    kh = k.transpose(0, 1).repeat_interleave(repeat, dim=0)  # [Hq, S, D]
    vh = v.transpose(0, 1).repeat_interleave(repeat, dim=0)

    # Chunk over queries: an arbitrary boolean mask forces sdpa onto the math
    # path, which materializes [Hq, T, S] attention scores -- ~3.2 GB per layer at
    # 8k tokens, and with the backward that is what killed train-step workers.
    # 512-query chunks bound the peak at [Hq, 512, S] (~200 MB) with identical
    # numerics; activation recompute replays the same loop.
    chunk = 512
    eff_scale = scale if scale is not None else dim**-0.5
    outs = []
    for lo in range(0, tokens, chunk):
        hi = min(lo + chunk, tokens)
        o = F.scaled_dot_product_attention(
            qh[:, lo:hi].unsqueeze(0),
            kh.unsqueeze(0),
            vh.unsqueeze(0),
            attn_mask=mask[lo:hi].unsqueeze(0).unsqueeze(0),
            scale=eff_scale,
        )
        outs.append(o.squeeze(0))
    return torch.cat(outs, dim=1).transpose(0, 1).contiguous()
