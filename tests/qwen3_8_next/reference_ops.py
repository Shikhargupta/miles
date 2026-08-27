"""Torch reference implementations for the triton kernel tests.

Moved verbatim out of the production tree (ops/hc.py, ops/sparse_attn.py,
ops/ple.py::causal_depthwise_conv) when training went triton-only: these are
the sglang-parity-verified oracles the tests compare against, and nothing else.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

def _acc_dtype(x: Tensor) -> torch.dtype:
    return x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32


def grouped_gemma_rmsnorm(x: Tensor, weight: Tensor, n: int, eps: float) -> Tensor:
    """Per-stream RMSNorm over ``[..., n*C]``.

    Traps: variance reduces WITHIN each stream (not over the whole n*C vector,
    which is what Megatron's learned_output_contract does); the scale enters as
    ``1 + weight`` (checkpoint stores a delta from unity — a plain RMSNorm
    multiplies by ~0). Returns fp32, not input dtype: callers feed the gate
    projections directly and an intermediate bf16 rounding is pure loss.
    """
    acc = _acc_dtype(x)
    x_grouped = x.to(acc).unflatten(-1, (n, x.shape[-1] // n))
    variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
    x_norm = (x_grouped * torch.rsqrt(variance + eps)).flatten(-2)
    return x_norm * (1.0 + weight.to(acc))


def hc_mix(
    normed: Tensor, w_down: Tensor, w_up: Tensor, n: int, hidden: int, out_dtype: torch.dtype
) -> Tensor:
    """Low-rank gated read ``[..., n*C] -> [..., C]``.

    Traps: the ``/ n`` sits between the down projection and the SiLU (moving it
    past the nonlinearity changes the result); the gate multiplies the NORMED
    streams; the reduction is a MEAN over streams (Megatron's mHC sums).
    """
    acc = normed.dtype
    gate = F.silu(F.linear(normed, w_down.to(acc)) / n)
    gate = torch.sigmoid(F.linear(gate, w_up.to(acc)))
    mixed = (gate.unflatten(-1, (n, hidden)) * normed.unflatten(-1, (n, hidden))).mean(dim=-2)
    return mixed.to(out_dtype)


def hc_inject_gate(normed: Tensor, w_inject: Tensor, n: int) -> Tensor:
    """Per-stream write gate ``a = 2 * sigmoid(W_inject N / n)`` -> ``[..., n]`` (fp32)."""
    acc = normed.dtype
    return 2 * torch.sigmoid(F.linear(normed, w_inject.to(acc)) / n)


def hc_combine(
    residual: Tensor, block_output: Tensor, h_post: Tensor, n: int, hidden: int
) -> Tensor:
    """``X'_c = X_c + a_c * y`` flattened to ``[..., n*C]``.

    No ``bmm(h_res^T, residual)``: Qwen3.8-Next's residual mixing is the
    identity, unlike Megatron's mHC.
    """
    out_dtype = residual.dtype
    acc = _acc_dtype(h_post) if h_post.dtype not in (torch.float32, torch.float64) else h_post.dtype
    R = residual.to(acc).unflatten(-1, (n, hidden))
    injection = block_output.to(acc).unsqueeze(-2) * h_post.to(acc).unsqueeze(-1)
    return (R + injection).flatten(-2).to(out_dtype)


def selection_to_mask(topk_indices: Tensor, seq_len: int) -> Tensor:
    """``[T, budget]`` int32 selections (``-1`` = unused) -> ``[T, S]`` bool mask."""
    tokens = topk_indices.shape[0]
    mask = torch.zeros(tokens, seq_len, dtype=torch.bool, device=topk_indices.device)
    valid = topk_indices >= 0
    if valid.any():
        rows = torch.arange(tokens, device=topk_indices.device).unsqueeze(-1).expand_as(topk_indices)
        mask[rows[valid], topk_indices[valid].long()] = True
    # Trap: NO forced diagonal -- sglang attends exactly the listed indices (the
    # query reaches itself via its partial-block tail); forcing it diverges on
    # saturated rows.
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


def causal_depthwise_conv(
    x: Tensor, weight: Tensor, dilation: int, cu_seqlens: Tensor | None = None
) -> Tensor:
    """Causal depthwise 1-D conv over the channel dim of ``[T, channels]``.

    ``weight`` is ``[channels, 1, kernel]``, one filter per channel, so this is
    ``groups=channels``. Left-padded by ``(kernel - 1) * dilation`` to stay causal.

    With ``cu_seqlens`` each sequence is convolved independently: the kernel
    must not reach across a document boundary.
    """
    channels, _, kernel = weight.shape
    pad = (kernel - 1) * dilation

    def _conv(seq: Tensor) -> Tensor:
        # [t, channels] -> [1, channels, t] for grouped conv1d
        h = seq.transpose(0, 1).unsqueeze(0)
        h = F.pad(h, (pad, 0))
        h = F.conv1d(h, weight, groups=channels, dilation=dilation)
        return h.squeeze(0).transpose(0, 1)

    if cu_seqlens is None:
        return _conv(x)

    out = torch.empty_like(x)
    bounds = cu_seqlens.tolist()
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi > lo:
            out[lo:hi] = _conv(x[lo:hi])
    return out
