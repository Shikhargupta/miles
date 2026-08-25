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
    mask.diagonal().fill_(True)
    return mask


def qsa_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    topk_indices: Tensor,
    scale: float | None = None,
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
    # Causality is already implied by the indexer's block-causal scoring, but assert
    # it here so a selection bug cannot leak future tokens into training.
    causal = torch.ones(tokens, seq, dtype=torch.bool, device=q.device).tril(
        diagonal=seq - tokens
    )
    mask &= causal

    qh = q.transpose(0, 1)                               # [Hq, T, D]
    kh = k.transpose(0, 1).repeat_interleave(repeat, dim=0)  # [Hq, S, D]
    vh = v.transpose(0, 1).repeat_interleave(repeat, dim=0)

    out = F.scaled_dot_product_attention(
        qh.unsqueeze(0),
        kh.unsqueeze(0),
        vh.unsqueeze(0),
        attn_mask=mask.unsqueeze(0).unsqueeze(0),
        scale=scale if scale is not None else dim**-0.5,
    )
    return out.squeeze(0).transpose(0, 1).contiguous()
