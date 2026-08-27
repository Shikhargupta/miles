"""Qwen3.8-Next QSA (Qwen Sparse Attention) indexer.

Runs on the full-attention layers (12 of 48) and picks, per query token, which
``indexer_budget`` key tokens the sparse attention will actually look at.

Reimplemented from sglang's ``QSAIndexer`` rather than imported -- miles takes no
"""

import math

import torch


def _indexer_acc_dtype(x):
    return x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32

from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor


def gemma_rmsnorm_last_dim(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm over the last dim with a Gemma-style ``1 + weight`` scale."""
    acc = _indexer_acc_dtype(x)
    xa = x.to(acc)
    var = xa.pow(2).mean(dim=-1, keepdim=True)
    return ((xa * torch.rsqrt(var + eps)) * (1.0 + weight.to(acc))).to(x.dtype)


def compress_keys_by_mean(token_k: Tensor, compress_ratio: int) -> Tensor:
    """``[T, head_dim] -> [ceil(T / r), head_dim]`` by averaging each run of ``r``."""
    tokens, dim = token_k.shape
    blocks = -(-tokens // compress_ratio)
    padded = blocks * compress_ratio
    if padded != tokens:
        pad = token_k.new_zeros(padded - tokens, dim)
        counts = token_k.new_ones(padded, 1)
        counts[tokens:] = 0
        summed = torch.cat([token_k, pad], dim=0).view(blocks, compress_ratio, dim).sum(1)
        denom = counts.view(blocks, compress_ratio, 1).sum(1).clamp_min(1)
        return summed / denom
    return token_k.view(blocks, compress_ratio, dim).mean(dim=1)


def block_causal_mask(query_positions: Tensor, num_blocks: int, compress_ratio: int) -> Tensor:
    """``[T, num_blocks]`` bool: which compressed blocks a query may attend to."""
    blocks = torch.arange(num_blocks, device=query_positions.device)
    first_invalid = (query_positions + 1) // compress_ratio
    return blocks.unsqueeze(0) < first_invalid.unsqueeze(1)


class Qwen38NextQSAIndexer(MegatronModule):
    """Selects the sparse-attention budget for one full-attention layer."""

    def __init__(self, config: TransformerConfig, layer_number: int, rotary_emb=None):
        super().__init__(config)
        self.layer_number = layer_number
        self.n_heads = config.qwen3_8_next_indexer_n_heads
        self.kv_heads = config.qwen3_8_next_indexer_kv_heads
        self.head_dim = config.qwen3_8_next_indexer_head_dim
        self.token_topk = config.qwen3_8_next_indexer_budget
        self.compress_ratio = config.qwen3_8_next_indexer_compress_ratio
        self.block_topk = self.token_topk // self.compress_ratio
        self.norm_eps = config.layernorm_epsilon
        self.rotary_emb = rotary_emb

        self.index_qk_proj = TELinear(
            config.hidden_size,
            (self.n_heads + self.kv_heads) * self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        dtype = config.params_dtype
        self.q_layernorm = torch.nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))
        self.k_layernorm = torch.nn.Parameter(torch.zeros(self.head_dim, dtype=dtype))
        for p in (self.q_layernorm, self.k_layernorm):
            setattr(p, "sequence_parallel", config.sequence_parallel)

    def project_qk(self, hidden_states: Tensor, positions: Tensor):
        """``[T, hidden] -> (q [T, n_heads, head_dim], block_k [B, head_dim])``."""
        qk, _ = self.index_qk_proj(hidden_states)
        split = self.n_heads * self.head_dim
        q_raw, token_k = qk[..., :split], qk[..., split:]

        q = gemma_rmsnorm_last_dim(
            q_raw.reshape(-1, self.head_dim), self.q_layernorm, self.norm_eps
        ).reshape(-1, self.n_heads, self.head_dim)

        block_k = compress_keys_by_mean(
            token_k.reshape(-1, self.head_dim), self.compress_ratio
        )
        block_k = gemma_rmsnorm_last_dim(block_k, self.k_layernorm, self.norm_eps)

        if self.rotary_emb is not None:
            block_positions = (
                torch.arange(block_k.shape[0], device=positions.device) * self.compress_ratio
            )
            q = self._apply_rope(positions, q)
            block_k = self._apply_rope(block_positions, block_k.unsqueeze(1)).squeeze(1)
        return q, block_k

    def _apply_rope(self, positions: Tensor, x: Tensor) -> Tensor:
        """Partial RoPE on the leading ``rotary_dim`` of each head."""
        rotary_dim = getattr(self.rotary_emb, "rotary_dim", x.shape[-1])
        if rotary_dim >= x.shape[-1]:
            return self.rotary_emb(positions, x)
        head = x[..., :rotary_dim]
        rest = x[..., rotary_dim:]
        return torch.cat([self.rotary_emb(positions, head), rest], dim=-1)

    def score_blocks(self, q: Tensor, block_k: Tensor, query_positions: Tensor) -> Tensor:
        """``[T, num_blocks]`` fp32 logits, invalid blocks at ``-inf``."""
        scores = torch.einsum("mhd,nd->mnh", q.float(), block_k.float())
        logits = torch.relu(scores).sum(dim=-1) / math.sqrt(self.head_dim)
        valid = block_causal_mask(query_positions, block_k.shape[0], self.compress_ratio)
        return logits.masked_fill(~valid, float("-inf"))

    def forward(self, hidden_states: Tensor, positions: Tensor) -> Tensor:
        """``[T, hidden] -> [T, token_topk]`` int32 token indices, ``-1`` where unused."""
        q, block_k = self.project_qk(hidden_states, positions)
        logits = self.score_blocks(q, block_k, positions)

        k = min(self.block_topk, logits.shape[-1])
        block_scores, block_idx = torch.topk(logits, k, dim=-1)
        block_idx = block_idx.masked_fill(block_scores == float("-inf"), -1)

        offsets = torch.arange(self.compress_ratio, device=block_idx.device)
        tokens = block_idx.unsqueeze(-1) * self.compress_ratio + offsets
        tokens = tokens.masked_fill(block_idx.unsqueeze(-1) < 0, -1).flatten(-2)
        tokens = tokens.masked_fill(tokens > positions.unsqueeze(-1), -1)
        return tokens[..., : self.token_topk].to(torch.int32)
