"""Unit tests for architecture-spec boundaries independent of Megatron."""

from types import SimpleNamespace

import pytest

from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.spec.base import AttachContext
from miles_plugins.lora.spec.moe import GENERAL_EXPERT_MOE_SPEC, SHARED_OUTER_EXPERT_MOE_SPEC


def _context(*targets: str, expanded_from_all_linear: bool = False) -> AttachContext:
    return AttachContext(
        lora=LoRAConfig(
            rank=2,
            alpha=4,
            dropout=0.0,
            target_modules=frozenset(targets),
            expanded_from_all_linear=expanded_from_all_linear,
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
    SHARED_OUTER_EXPERT_MOE_SPEC.validate_layer(mlp, _context("gate_proj", "down_proj"))


def test_moe_mlp_targets_fail_when_only_routed_experts_could_match():
    """A shared-outer-expert model with a routed-only layer delegates to the general spec."""
    mlp = SimpleNamespace(experts=object())
    with pytest.raises(AssertionError, match="routed/grouped expert"):
        SHARED_OUTER_EXPERT_MOE_SPEC.validate_layer(mlp, _context("gate_proj"))


def test_general_expert_spec_rejects_explicit_mlp_targets_directly():
    mlp = SimpleNamespace(experts=object())
    with pytest.raises(AssertionError, match="routed/grouped expert"):
        GENERAL_EXPERT_MOE_SPEC.validate_layer(mlp, _context("down_proj"))


def test_moe_parser_expanded_mlp_targets_skip_instead_of_failing():
    """all-linear on a shared-expert-less MoE (e.g. qwen3_moe) trains attention-only, like the base branch."""
    mlp = SimpleNamespace(experts=object())
    SHARED_OUTER_EXPERT_MOE_SPEC.validate_layer(
        mlp,
        _context("q_proj", "gate_proj", "up_proj", "down_proj", expanded_from_all_linear=True),
    )


def test_attention_only_moe_does_not_require_a_shared_expert():
    mlp = SimpleNamespace(experts=object())
    SHARED_OUTER_EXPERT_MOE_SPEC.validate_layer(mlp, _context("q_proj"))
