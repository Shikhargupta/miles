"""Frozen, host-resident, TP-sharded PLE n-gram table.

51.2 B parameters / 102.4 GB bf16 -- 28.4% of the whole model -- but every token
reads exactly 16 rows of it (one per hash head), i.e. 0.000005% of the table. That
ratio is what dictates the design:

  * **Frozen.** Training it densely would cost 102.4 GB of gradient plus 614 GB of
    Adam state and master weights, and an all-reduce of a 102 GB gradient every
    step would swamp the compute. It also comes free: a frozen table is bitwise
    identical between training and rollout, so it contributes exactly zero to the
    train/inference logprob gap. (Full-parameter training of it wants row-sharded
    sparse gradients -- DLRM-style embedding parallelism -- which is its own
    project.)
  * **Host-resident**, matching sglang's ``ple_offload_embedding=True`` and its
    ``Qwen4ExpPinnedHostEmbedding``: pinned CPU memory, read by a kernel through a
    raw host pointer. Never staged to HBM.
  * **TP-sharded by row**, like ``VocabParallelEmbedding``: each rank owns a
    contiguous row range, gathers only ids that fall inside it, zeros the rest, and
    an all-reduce sums the contributions. At TP4 that is 80,000,384 rows =
    25.6 GB of pinned host memory per rank.

The checkpoint stores the table as ``split_ngram_parts`` shards of equal height
(128 x [2,500,012, 160] for Qwen3.8-Flash-Next). Shards are assigned to ranks
whole, and the rank's shards are registered as parameter *views* into one
contiguous pinned allocation. That keeps the bridge mapping 1:1 with the
checkpoint -- no partial writes into a larger tensor -- while the gather still
sees a single flat base pointer.
"""

import torch
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from miles_plugins.models.qwen3_8_next.ops.kernel.ple_gather import gather_ple_rows
from miles_plugins.models.qwen3_8_next.ops.ple_hash import ngram_hash_ids


class Qwen38NextFrozenNGramEmbedding(MegatronModule):
    """Frozen host-resident n-gram table with row-wise tensor parallelism."""

    def __init__(self, config: TransformerConfig, tp_group=None):
        super().__init__(config)
        self.embedding_dim = config.qwen3_8_next_ple_embed_dim // self._num_heads(config)
        self.ngram_size = config.qwen3_8_next_ngram_size
        self._heads_per_ngram = config.qwen3_8_next_heads_per_ngram
        self.num_shards = config.qwen3_8_next_split_ngram_parts
        self.tp_group = tp_group
        tp_size = tp_group.size() if tp_group is not None else 1
        tp_rank = tp_group.rank() if tp_group is not None else 0

        if self.num_shards % tp_size:
            raise ValueError(
                f"split_ngram_parts={self.num_shards} must be divisible by "
                f"tensor-parallel size {tp_size} so shards can be assigned whole; "
                "partial shards would need the loader to write into the middle of a "
                "rank's allocation."
            )
        self.shards_per_rank = self.num_shards // tp_size
        self.shard_ids = list(
            range(tp_rank * self.shards_per_rank, (tp_rank + 1) * self.shards_per_rank)
        )

        # Shard height is not derivable from the config: the checkpoint rounds the
        # 320,001,446 hashed rows up to 128 x 2,500,012 = 320,001,536, so the last
        # shard carries unused padding. The bridge sets this from the safetensors
        # header; the fallback covers a from-scratch init.
        self.rows_per_shard = getattr(config, "qwen3_8_next_ngram_rows_per_shard", None)
        if self.rows_per_shard is None:
            total = self._total_hashed_rows(config)
            self.rows_per_shard = -(-total // self.num_shards)  # ceil
        self.row_start = self.shard_ids[0] * self.rows_per_shard
        self.row_end = (self.shard_ids[-1] + 1) * self.rows_per_shard

        # One contiguous pinned allocation; the per-shard parameters are views into
        # it so the gather can use a single base pointer.
        rows = self.row_end - self.row_start
        self._table = torch.empty(
            (rows, self.embedding_dim), dtype=torch.bfloat16, device="cpu", pin_memory=True
        )
        for i, shard_id in enumerate(self.shard_ids):
            lo = i * self.rows_per_shard
            view = self._table[lo : lo + self.rows_per_shard]
            self.register_parameter(
                f"shard_{shard_id}", torch.nn.Parameter(view, requires_grad=False)
            )

        # The hash is parameterised entirely by these three, and the checkpoint
        # ships them -- sglang derives the multipliers from splitmix64 seeded by
        # config.seed + PRIME_1 * ple_layer_index, so recomputing them would mean
        # reproducing its constants exactly. Buffers, not parameters: integer
        # constants that must travel with the checkpoint but never see a gradient.
        heads = self._num_heads(config)
        self.register_buffer(
            "layer_multipliers",
            torch.zeros(config.qwen3_8_next_ngram_size, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes", torch.zeros(heads, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "ngram_heads_offsets", torch.zeros(heads, dtype=torch.long), persistent=True
        )
        self.eos_token_id = getattr(config, "qwen3_8_next_eos_token_id", 0)

    @staticmethod
    def _num_heads(config: TransformerConfig) -> int:
        # n-gram orders 2..ngram_size, heads_per_ngram hash heads each.
        return (config.qwen3_8_next_ngram_size - 1) * config.qwen3_8_next_heads_per_ngram

    @staticmethod
    def _total_hashed_rows(config: TransformerConfig) -> int:
        heads = Qwen38NextFrozenNGramEmbedding._num_heads(config)
        return heads * config.qwen3_8_next_ngram_vocab_size_base

    def compute_ngram_ids(self, contexts: Tensor) -> Tensor:
        """``contexts`` ``[T, ngram_size]`` sliding windows -> ``[T, n_heads]`` row ids.

        Each row of ``contexts`` is one token's window ending at that token, which
        is what ``tokens.unfold(dim, ngram_size, 1)`` produces and what sglang feeds
        its hash.
        """
        return ngram_hash_ids(
            contexts,
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
            self.ngram_size,
            self._heads_per_ngram,
            self.eos_token_id,
        )

    def forward(self, ids: Tensor) -> Tensor:
        """``ids`` ``[T, n_heads]`` int64 -> ``[T, n_heads * embedding_dim]`` bf16.

        Concatenating the heads' rows reconstructs the full ``ple_embed_dim``
        vector: the table is a product decomposition, 16 narrow tables rather than
        one wide one.
        """
        rows = gather_ple_rows(self._table, ids, self.row_start, self.row_end)
        out = rows.flatten(start_dim=-2)
        if self.tp_group is not None and self.tp_group.size() > 1:
            # Each rank produced zeros for rows it does not own.
            torch.distributed.all_reduce(out, group=self.tp_group)
        return out
