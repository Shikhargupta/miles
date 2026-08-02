"""Dimension resolvers shared by every declarative native-LoRA layout.

This is the only spec-side module allowed to read MCore/TE attribute names.
Each resolver has the signature ``(module, context) -> int`` where ``module``
is the attention/MLP block that owns the physical linears (not the linear
itself), so one table entry can relate several submodule shapes.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from miles_plugins.lora.spec.base import AttachContext

DimFn = Callable[[nn.Module, AttachContext], int]


def hidden(_module: nn.Module, context: AttachContext) -> int:
    return context.hidden


def cfg(field: str) -> DimFn:
    """A transformer-config field used verbatim (e.g. ``q_lora_rank``)."""

    def resolve(_module: nn.Module, context: AttachContext) -> int:
        value = getattr(context.transformer_config, field)
        assert value, f"transformer config field {field!r} must be a positive dimension, got {value!r}"
        return int(value)

    return resolve


def gqa_o_in_local(module: nn.Module, _context: AttachContext) -> int:
    """Row-parallel o_proj input: this rank's query heads times head dim."""
    return module.num_attention_heads_per_partition * module.hidden_size_per_attention_head


def mla_q_up_out_local(module: nn.Module, _context: AttachContext) -> int:
    """Column-parallel q_b output: local heads times the full MLA query head dim."""
    return module.num_attention_heads_per_partition * module.q_head_dim


def mla_kv_down_out(_module: nn.Module, context: AttachContext) -> int:
    """Replicated kv_a output: compressed KV rank plus the shared RoPE key slice."""
    config = context.transformer_config
    return int(config.kv_lora_rank + config.qk_pos_emb_head_dim)


def mla_kv_up_out_local(module: nn.Module, context: AttachContext) -> int:
    """Column-parallel kv_b output: local heads times (nope-K plus V) head dims."""
    config = context.transformer_config
    return module.num_attention_heads_per_partition * (config.qk_head_dim + config.v_head_dim)


def mla_o_in_local(module: nn.Module, context: AttachContext) -> int:
    """Row-parallel MLA o_proj input: local heads times the value head dim."""
    return module.num_attention_heads_per_partition * context.transformer_config.v_head_dim
