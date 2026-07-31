"""Unit tests for native-LoRA run configuration."""

from argparse import Namespace

import pytest

from miles_plugins.lora.config import LoRAConfig


def test_config_normalizes_megatron_targets_and_computes_scale():
    config = LoRAConfig.from_args(
        Namespace(
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1,
            lora_A_init_method="xavier",
            target_modules=["linear_qkv", "linear_fc2"],
        )
    )
    assert config.rank == 8
    assert config.scale == 2.0
    assert config.dropout == 0.1
    assert config.target_modules == {"q_proj", "k_proj", "v_proj", "down_proj"}


def test_config_rejects_non_positive_rank():
    with pytest.raises(AssertionError, match="lora-rank"):
        LoRAConfig.from_args(Namespace(lora_rank=0, lora_alpha=16, target_modules=["q_proj"]))


def test_config_preserves_all_linear_expansion_provenance():
    config = LoRAConfig.from_args(
        Namespace(
            lora_rank=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj"],
            _target_modules_expanded_from_all_linear=True,
        )
    )
    assert config.expanded_from_all_linear
