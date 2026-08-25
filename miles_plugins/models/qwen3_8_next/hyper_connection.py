"""Qwen3.8-Next Hyper-Connection modules for Megatron.

Megatron's ``HyperConnectionModule`` (megatron/core/transformer/hyper_connection.py)
implements the mHC formulation DeepSeek-V4 uses:

    h_pre  : [s, b, n]     per-stream scalar read gate, from a single projection
    h_post : [s, b, n]     per-stream scalar write gate, 2*sigmoid
    h_res  : [s, b, n, n]  doubly-stochastic residual mixing matrix (Sinkhorn)
    aggregate: sum_j h_pre_j * x_j

Qwen3.8-Next differs in exactly two ways (see sglang's ``GatedResidual``, the
authority, which sglang instantiates for the attention HC, the MLP HC and the
model-level final mixer alike):

  * the read gate is **per-stream per-feature**, produced by a low-rank
    two-matrix MLP with a SiLU in the middle, and the aggregation is a **mean**
    over streams rather than a weighted sum;
  * there is **no residual mixing** -- ``h_res`` is the identity, so the
    write-back is just ``X_c += a_c * y``.

Everything else about Megatron's structure already matches: the block widens the
hidden state at the input, each layer reads one working vector and writes the
block output back to every stream, and MTP consumes the pre-contraction
``[s, b, n*C]`` state. So only the gating needs reimplementing, and it drops
into the existing ``TransformerLayerSubmodules.self_attention_hyper_connection``
/ ``mlp_hyper_connection`` ModuleSpec slots -- no Megatron core changes.

Contract that ``HyperConnectionTransformerLayer`` expects:

    hidden_states, h_res, h_post, residual = hc(hidden_states,
                                                mhc_recompute_manager=...)
    out = hc.fused_h_res_h_post_bda(h_res, residual, h_post,
                                    layer_output_with_bias, dropout_prob,
                                    training, fused, manager=...)

The layer never inspects ``h_res``/``h_post``, it only threads them from the
first call to the second, so returning ``None`` for ``h_res`` is safe and lets
``fused_h_res_h_post_bda`` skip the identity bmm entirely.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor

from miles_plugins.models.qwen3_8_next.ops import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_inject_gate,
    hc_mix,
)


class Qwen38NextHyperConnection(MegatronModule):
    """Per-layer hyper-connection: fills the attention and MLP HC spec slots.

    Parameters are replicated across tensor-parallel ranks -- the gating runs on
    the full hidden dimension, like a layernorm -- so they carry the
    ``sequence_parallel`` flag when sequence parallelism is on, which is what
    tells Megatron to reduce their gradients over the right group.

    Weights stay in ``config.params_dtype`` (bf16) instead of being promoted to
    fp32 the way Megatron's mHC keeps its ``hc_*`` params. That is deliberate:
    the goal is train/inference consistency against sglang, which holds these in
    bf16, and the fp32 work that actually matters (the norm reduction) happens
    inside ``grouped_gemma_rmsnorm`` regardless of storage dtype.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        hc_count: Optional[int] = None,
        use_combine: bool = True,
    ):
        super().__init__(config)
        self.layer_number = layer_number
        self.n = hc_count if hc_count is not None else config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.norm_eps = config.layernorm_epsilon
        self.use_combine = use_combine

        lowrank = config.qwen3_8_next_hc_lowrank
        wide = self.n * self.hidden_size
        dtype = config.params_dtype

        # Names mirror the checkpoint (model.language_model.layers.{i}.
        # {attn,mlp}_hyper_connection.*) so the bridge mapping stays obvious.
        self.hc_norm_weight = torch.nn.Parameter(torch.zeros(wide, dtype=dtype))
        self.input_mix_weight_down = torch.nn.Parameter(torch.empty(lowrank, wide, dtype=dtype))
        self.input_mix_weight_up = torch.nn.Parameter(torch.empty(wide, lowrank, dtype=dtype))
        params = [self.hc_norm_weight, self.input_mix_weight_down, self.input_mix_weight_up]

        if use_combine:
            self.block_inject_weight = torch.nn.Parameter(torch.empty(self.n, wide, dtype=dtype))
            params.append(self.block_inject_weight)
        else:
            self.block_inject_weight = None

        for p in params:
            setattr(p, "sequence_parallel", config.sequence_parallel)

        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.input_mix_weight_down)
            torch.nn.init.xavier_uniform_(self.input_mix_weight_up)
            if use_combine:
                torch.nn.init.xavier_uniform_(self.block_inject_weight)

    def mix(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        """``[s,b,n*C]`` -> ``(aggregated [s,b,C], normed [s,b,n*C])``."""
        # `normed` comes back in the fp32 accumulation dtype on purpose and is
        # threaded to the inject gate unrounded; only `aggregated` is cast back.
        normed = grouped_gemma_rmsnorm(hidden_states, self.hc_norm_weight, self.n, self.norm_eps)
        aggregated = hc_mix(
            normed,
            self.input_mix_weight_down,
            self.input_mix_weight_up,
            self.n,
            self.hidden_size,
            out_dtype=hidden_states.dtype,
        )
        return aggregated, normed

    def forward(
        self,
        hidden_states: Tensor,
        mhc_recompute_manager=None,
        output_slot=None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        """Returns ``(aggregated, h_res=None, h_post, residual)``.

        ``h_post`` is computed here rather than in the combine step because it
        depends on the same normed tensor as the read gate. Recomputing the norm
        later would cost an extra pass and risk drifting from sglang, which also
        reuses one normed tensor for both (its ``mix`` returns it alongside the
        raw residual).
        """
        if mhc_recompute_manager is not None or output_slot is not None:
            raise NotImplementedError(
                "Qwen38NextHyperConnection does not support the mHC recompute arena yet; "
                "run without 'mhc' in --recompute-modules."
            )
        assert self.use_combine, "per-layer HC needs the inject weight; use the Mixer for read-only"
        aggregated, normed = self.mix(hidden_states)
        h_post = hc_inject_gate(normed, self.block_inject_weight, self.n)
        return aggregated, None, h_post, hidden_states

    def fused_h_res_h_post_bda(
        self,
        h_res: Optional[Tensor],
        original_residual: Tensor,
        h_post: Tensor,
        layer_output_with_bias,
        dropout_prob: float,
        training: bool,
        fused: bool,
        manager=None,
    ) -> Tensor:
        """``X'_c = X_c + a_c * (y + bias)``, dropout applied to the injection."""
        assert h_res is None, "Qwen3.8-Next hyper-connection has no residual mixing matrix"
        if manager is not None:
            raise NotImplementedError("mHC recompute arena not supported yet")

        if isinstance(layer_output_with_bias, tuple):
            x, bias = layer_output_with_bias
        else:
            x, bias = layer_output_with_bias, None

        if bias is not None:
            x = x + bias.view(*([1] * (x.dim() - 1)), -1)
        if dropout_prob > 0.0 and training:
            x = F.dropout(x, p=dropout_prob)
        return hc_combine(original_residual, x, h_post, self.n, self.hidden_size)


class Qwen38NextHyperConnectionMixer(MegatronModule):
    """Model-level final contraction ``[s, b, n*C] -> [s, b, C]``.

    sglang builds this from the *same* ``GatedResidual`` class as the per-layer
    HC, only with ``use_combine=False``, so it reuses the read-gate path
    verbatim. It replaces Megatron's ``learned_output_contract``, which is a
    genuinely different function -- single projection to a per-stream scalar, a
    sum rather than a mean, and one RMS over the whole ``n*C`` vector -- and so
    is not a drop-in.
    """

    def __init__(self, config: TransformerConfig, hc_count: Optional[int] = None):
        super().__init__(config)
        self.hc = Qwen38NextHyperConnection(
            config, layer_number=-1, hc_count=hc_count, use_combine=False
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        aggregated, _ = self.hc.mix(hidden_states)
        return aggregated
