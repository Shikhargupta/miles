"""Unit tests for architecture-spec boundaries independent of Megatron."""

from types import SimpleNamespace

import pytest

from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.spec.base import AttachContext
from miles_plugins.lora.spec.moe import SHARED_EXPERT_ONLY_MOE_SPEC


def _context(*targets: str) -> AttachContext:
    return AttachContext(
        lora=LoRAConfig(
            rank=2,
            alpha=4,
            dropout=0.0,
            target_modules=frozenset(targets),
        ),
        transformer_config=SimpleNamespace(
            hidden_size=8,
            layernorm_epsilon=1e-6,
            sequence_parallel=False,
        ),
        tp_size=1,
        tp_rank=0,
        layer_prefix="model.layers.",
        shared_expert="mlp.shared_expert.",
    )


def test_moe_mlp_targets_preserve_existing_shared_expert_support():
    mlp = SimpleNamespace(
        experts=object(),
        shared_experts=SimpleNamespace(linear_fc1=object()),
    )
    SHARED_EXPERT_ONLY_MOE_SPEC.validate_layer(mlp, _context("gate_proj", "down_proj"))


def test_moe_mlp_targets_fail_when_only_routed_experts_could_match():
    mlp = SimpleNamespace(experts=object())
    with pytest.raises(AssertionError, match="routed/grouped expert"):
        SHARED_EXPERT_ONLY_MOE_SPEC.validate_layer(mlp, _context("gate_proj"))


def test_attention_only_moe_does_not_require_a_shared_expert():
    mlp = SimpleNamespace(experts=object())
    SHARED_EXPERT_ONLY_MOE_SPEC.validate_layer(mlp, _context("q_proj"))
