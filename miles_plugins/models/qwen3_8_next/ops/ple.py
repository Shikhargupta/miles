"""Qwen3.8-Next PLE layer (Position-aware Local Embedding).

Runs on exactly one decoder layer (index 1 -- ``ple_layer_ids`` is 1-based) and
adds a contextual increment to the hyper-connection state *before* the attention
HC reads it.

Shape of the computation, transcribed from sglang's ``Qwen4ExpPLELayer.forward``
(reimplemented here rather than imported -- miles must not depend on sglang):

    E     = ngram_embedding(hash_ids)            [T, ple_embed_dim]
    key   = key_proj(E)   -> [T, n, C]           per-stream key
    value = value_proj(E) -> [T, C]              one value, shared across streams
    query = hc_state      -> [T, n, C]           the widened residual is the query

    s     = <norm_key(key), norm_query(query)> / sqrt(C)          [T, n, 1]
    g     = sigmoid( sign(s) * sqrt(max(|s|, 1e-6)) )             [T, n, 1]
    U     = g * value                                            [T, n, C]
    O     = U + SiLU(dwconv(norm_conv(U)))                       [T, n*C]

    hc_state += O

The ``sign(s) * sqrt(|s|)`` squashing before the sigmoid is what keeps the gate
responsive when the dot product is large; the ``clamp_min(1e-6)`` guards the
sqrt's gradient at zero.

Both ``norm_key``/``norm_query``/``norm_conv`` are per-stream Gemma-style RMSNorms
over the widened ``[.., n*C]`` vector -- the same operator the hyper-connections
use -- implemented in kernel/ple_triton.py (fused with the gate and conv).
"""


import torch
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from miles_plugins.models.qwen3_8_next.ops.kernel.ple_triton import ple_gate_conv_triton
from miles_plugins.models.qwen3_8_next.ops.ple_embedding import Qwen38NextFrozenNGramEmbedding


class Qwen38NextPLE(MegatronModule):
    """PLE increment for the hyper-connection state."""

    def __init__(self, config: TransformerConfig, layer_number: int, tp_group=None):
        super().__init__(config)
        self.layer_number = layer_number
        self.n = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = config.layernorm_epsilon
        self.ngram_size = config.qwen3_8_next_ngram_size
        self.heads_per_ngram = config.qwen3_8_next_heads_per_ngram
        self.embed_dim = config.qwen3_8_next_ple_embed_dim
        wide = self.n * self.hidden_size

        self.ple_embedding = Qwen38NextFrozenNGramEmbedding(
            config, layer_number=layer_number, tp_group=tp_group
        )

        # Replicated, not TP-sharded: avoids a second reduction on top of the table's.
        self.key_proj = TELinear(
            self.embed_dim, wide, config=config, init_method=config.init_method,
            bias=False, skip_bias_add=False, skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.value_proj = TELinear(
            self.embed_dim, self.hidden_size, config=config, init_method=config.init_method,
            bias=False, skip_bias_add=False, skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        dtype = config.params_dtype
        # Gemma-style: stored as a delta from unity, hence init to zeros.
        self.norm_key = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.norm_query = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.norm_conv = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))

        kernel = config.qwen3_8_next_ple_conv_kernel_size
        self.conv_dilation = getattr(config, "qwen3_8_next_ple_conv_dilation", 3)
        self.conv1d_weight = torch.nn.Parameter(torch.zeros(wide, 1, kernel, dtype=dtype))

        for p in (self.norm_key, self.norm_query, self.norm_conv, self.conv1d_weight):
            setattr(p, "sequence_parallel", config.sequence_parallel)

    def forward(
        self, hc_state: Tensor, ngram_ids: Tensor, cu_seqlens: Tensor | None = None
    ) -> Tensor:
        """``hc_state`` ``[T, n*C]``, ``ngram_ids`` ``[T, n_heads]`` -> increment ``[T, n*C]``.

        Returns the increment rather than the updated state so the caller keeps
        ownership of the residual, matching how the HC modules are structured.
        """
        # Trap: 3D input would reshape silently-wrong at batch 1; enforce 2D.
        if hc_state.dim() != 2 or ngram_ids.dim() != 2:
            raise RuntimeError(
                "PLE takes a flat token axis: expected hc_state [T, n*C] and ngram_ids "
                f"[T, heads], got {tuple(hc_state.shape)} and {tuple(ngram_ids.shape)}"
            )
        if hc_state.shape[0] != ngram_ids.shape[0]:
            raise RuntimeError(
                f"PLE token counts differ: state {hc_state.shape[0]} vs ids {ngram_ids.shape[0]}"
            )

        embeddings = self.ple_embedding(ngram_ids)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)

        tokens = hc_state.shape[0]
        if hc_state.shape[-1] != self.n * self.hidden_size:
            raise RuntimeError(
                "PLE hidden size does not match the hyper-connection layout: expected "
                f"{self.n * self.hidden_size}, got {hc_state.shape[-1]}"
            )

        return ple_gate_conv_triton(
            hc_state, key, value,
            self.norm_key, self.norm_query, self.norm_conv,
            self.conv1d_weight, self.n, self.norm_eps,
            self.conv_dilation, cu_seqlens,
        )
