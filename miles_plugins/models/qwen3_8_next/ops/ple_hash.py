"""Qwen3.8-Next PLE n-gram hashing -- pure torch, no Megatron.

Kept free of Megatron imports for the same reason as ``ops.py``: it makes these
functions unit-testable against sglang without pulling megatron.core (and with it
Transformer Engine).

The hash is fully determined by three tensors the checkpoint ships under
``ple.ple_embedding``, so nothing here has to reproduce sglang's PRNG:

    layer_multipliers      [ngram_size]  int64, odd
    ngram_heads_vocab_sizes[n_heads]     int64
    ngram_heads_offsets    [n_heads]     int64

For Qwen3.8-Flash-Next those are ``[23703573157769, 20109073645365,
8052911324071]`` and 16 heads whose vocab sizes are the 16 consecutive primes
after 2e7 (20000003 .. 20000171). Distinct primes per head is what decorrelates
collisions between heads; the offsets are the running sum of the sizes, laying
all 16 heads out in one flat 320,001,446-row space. sglang *derives* the
multipliers from splitmix64 seeded by ``config.seed + PRIME_1 * ple_layer_index``
-- reproducing that would mean matching its constants exactly, so we load the
shipped tensors instead.
"""

import torch
from torch import Tensor


def shift_right_ignore_eos(tokens: Tensor, n: int, eos_token_id: int) -> Tensor:
    """Shift right by ``n`` without letting context cross an EOS boundary.

    Positions fewer than ``n`` tokens into their segment get ``eos_token_id`` as
    filler rather than the previous document's tail, so n-grams never straddle a
    document break. Mirrors sglang's ``_shift_right_ignore_eos``.

    ``tokens``: ``[batch, seq]``.
    """
    if n == 0:
        return tokens
    batch_size, seq_len = tokens.shape
    idx = torch.arange(seq_len, device=tokens.device, dtype=torch.long)

    # Position of the most recent EOS at or before each column, exclusive of the
    # column itself -- hence the shift-by-one after the cummax.
    eos_pos = torch.where(tokens == eos_token_id, idx, torch.full_like(idx, -1))
    prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
    prev_eos = torch.cat(
        [eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]], dim=1
    )
    pos_in_segment = idx.unsqueeze(0) - (prev_eos + 1)

    src_idx = idx - n
    gathered = tokens.gather(
        dim=1, index=torch.clamp(src_idx, min=0).unsqueeze(0).expand(batch_size, -1)
    )
    valid = (pos_in_segment >= n) & (src_idx.unsqueeze(0) >= 0)
    return torch.where(valid, gathered, tokens.new_full((), eos_token_id))


def ngram_hash_ids(
    contexts: Tensor,
    layer_multipliers: Tensor,
    head_vocab_sizes: Tensor,
    head_offsets: Tensor,
    ngram_size: int,
    heads_per_ngram: int,
    eos_token_id: int,
) -> Tensor:
    """Row ids into the flat PLE table, one per hash head.

    ``contexts``: ``[num_tokens, ngram_size]`` -- each row is one token's sliding
    window ending at that token, which is what sglang's
    ``ngram_context.unfold(1, ngram_size, 1)`` produces. Returns
    ``[num_tokens, (ngram_size - 1) * heads_per_ngram]`` int64.

    Per n-gram order ``g`` in ``2..ngram_size``:

        mix = ctx_0 * m_0  XOR  ctx_1 * m_1  XOR ... XOR  ctx_{g-1} * m_{g-1}
        id  = (mix_at_current_token % vocab_size_h) + offset_h

    The multiply is int64 and deliberately allowed to be large -- sglang bounds
    the multipliers by ``(2**63 - 1) // vocab_size`` precisely so the products
    stay inside int64 -- and the XOR is what makes position order matter.
    """
    shifted = [contexts]
    for shift in range(1, ngram_size):
        shifted.append(shift_right_ignore_eos(contexts, shift, eos_token_id))

    blocks = []
    for ngram in range(2, ngram_size + 1):
        start = (ngram - 2) * heads_per_ngram
        end = start + heads_per_ngram
        mix = shifted[0] * layer_multipliers[0]
        for pos in range(1, ngram):
            mix = torch.bitwise_xor(mix, shifted[pos] * layer_multipliers[pos])
        # Only the window's last column carries the fully-formed n-gram for the
        # token this row belongs to.
        ids = torch.remainder(
            mix[:, -1:].unsqueeze(-1), head_vocab_sizes[start:end].view(1, 1, -1)
        )
        blocks.append((ids + head_offsets[start:end].view(1, 1, -1))[:, 0])
    return torch.cat(blocks, dim=-1)

def build_ngram_contexts(tokens: Tensor, ngram_size: int, eos_token_id: int) -> Tensor:
    """``[T]`` token ids -> ``[T, ngram_size]`` sliding windows, one row per token.

    Row ``t`` is ``tokens[t - ngram_size + 1 : t + 1]``, i.e. the window *ending* at
    token ``t``, which is the layout ``ngram_hash_ids`` expects and what sglang's
    ``ngram_context.unfold(1, ngram_size, 1)`` produces. The first few rows are
    padded on the left with ``eos_token_id`` -- the same filler
    ``shift_right_ignore_eos`` uses inside a segment -- so the start of the sequence
    is treated as a document start rather than borrowing whatever preceded it.
    """
    assert tokens.dim() == 1, f"expected a 1-D token sequence, got {tuple(tokens.shape)}"
    pad = tokens.new_full((ngram_size - 1,), eos_token_id)
    return torch.cat([pad, tokens]).unfold(0, ngram_size, 1)


def build_ngram_contexts_packed(
    tokens: Tensor, cu_seqlens: Tensor | None, ngram_size: int, eos_token_id: int
) -> Tensor:
    """``build_ngram_contexts`` for a packed (THD) batch.

    Each sequence gets its own EOS padding so a token's n-gram never reaches into
    the previous document -- the padding at the start of the pack alone would let
    sequence k's first tokens hash together with sequence k-1's last ones.
    """
    if cu_seqlens is None:
        return build_ngram_contexts(tokens, ngram_size, eos_token_id)
    bounds = cu_seqlens.tolist()
    parts = [
        build_ngram_contexts(tokens[lo:hi], ngram_size, eos_token_id)
        for lo, hi in zip(bounds[:-1], bounds[1:])
        if hi > lo
    ]
    return torch.cat(parts, dim=0)
