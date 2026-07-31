"""Unit tests for LoRA target-module argument parsing.

These exercise the production ``parse_lora_target_modules`` directly (the same
function ``miles_validate_args`` calls), so a change to the expansion or
provenance logic cannot pass the suite without being tested. Only the HF config
read is stubbed; everything else is the real code path.
"""

from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.utils.arguments import parse_lora_target_modules


def _args(**overrides) -> Namespace:
    defaults = dict(lora_rank=32, target_modules="all-linear", exclude_modules=None, hf_checkpoint=None)
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.fixture
def dense_hf_config(monkeypatch):
    """Stub the HF config read with a dense-model config (no MLA fields)."""
    monkeypatch.setattr("miles.utils.arguments.load_hf_config", lambda _checkpoint: SimpleNamespace())


@pytest.fixture
def mla_hf_config(monkeypatch):
    monkeypatch.setattr(
        "miles.utils.arguments.load_hf_config",
        lambda _checkpoint: SimpleNamespace(kv_lora_rank=512, q_lora_rank=1536),
    )


# ---------------------------------------------------------------------------
# Target modules expansion
# ---------------------------------------------------------------------------


class TestLoraTargetModuleParsing:
    def test_all_linear_expands_to_seven_modules(self, dense_hf_config):
        args = _args()
        parse_lora_target_modules(args)
        assert args.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert args._target_modules_expanded_from_all_linear

        parse_lora_target_modules(args)
        assert args._target_modules_expanded_from_all_linear
        assert args.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def test_all_linear_adds_mla_projections_when_the_config_declares_them(self, mla_hf_config):
        args = _args()
        parse_lora_target_modules(args)
        assert args.target_modules[-4:] == ["kv_a_proj_with_mqa", "kv_b_proj", "q_a_proj", "q_b_proj"]
        assert args._target_modules_expanded_from_all_linear

    def test_comma_separated_split(self):
        args = _args(lora_rank=16, target_modules="q_proj, k_proj, v_proj")
        parse_lora_target_modules(args)
        assert args.target_modules == ["q_proj", "k_proj", "v_proj"]
        assert not args._target_modules_expanded_from_all_linear

    def test_comma_separated_no_spaces(self):
        args = _args(lora_rank=16, target_modules="q_proj,k_proj")
        parse_lora_target_modules(args)
        assert args.target_modules == ["q_proj", "k_proj"]

    def test_single_module(self):
        args = _args(lora_rank=8, target_modules="q_proj")
        parse_lora_target_modules(args)
        assert args.target_modules == ["q_proj"]

    def test_lora_rank_zero_skips_parsing(self):
        args = _args(lora_rank=0)
        parse_lora_target_modules(args)
        assert args.target_modules == "all-linear"  # unchanged
        assert not args._target_modules_expanded_from_all_linear

    def test_missing_target_modules_asserts(self):
        args = _args(target_modules=None)
        with pytest.raises(AssertionError, match="--target-modules"):
            parse_lora_target_modules(args)


# ---------------------------------------------------------------------------
# Exclude modules filtering
# ---------------------------------------------------------------------------


class TestLoraExcludeModules:
    def test_single_exclude(self, dense_hf_config):
        args = _args(exclude_modules="o_proj")
        parse_lora_target_modules(args)
        assert "o_proj" not in args.target_modules
        assert len(args.target_modules) == 6

    def test_multiple_exclude_comma_separated(self, dense_hf_config):
        args = _args(exclude_modules="o_proj, down_proj")
        parse_lora_target_modules(args)
        assert "o_proj" not in args.target_modules
        assert "down_proj" not in args.target_modules
        assert len(args.target_modules) == 5

    def test_exclude_all_results_in_empty(self):
        args = _args(target_modules="q_proj,k_proj", exclude_modules="q_proj,k_proj")
        parse_lora_target_modules(args)
        assert args.target_modules == []

    def test_exclude_nonexistent_module_no_effect(self):
        args = _args(target_modules="q_proj,k_proj", exclude_modules="nonexistent")
        parse_lora_target_modules(args)
        assert args.target_modules == ["q_proj", "k_proj"]

    def test_no_exclude_modules(self):
        args = _args(target_modules="q_proj,k_proj")
        parse_lora_target_modules(args)
        assert args.target_modules == ["q_proj", "k_proj"]

    def test_empty_string_exclude(self):
        """Empty string is falsy; the target list must pass through unchanged."""
        args = _args(target_modules="q_proj,k_proj", exclude_modules="")
        parse_lora_target_modules(args)
        assert args.target_modules == ["q_proj", "k_proj"]

    @pytest.mark.parametrize("pattern", ["*.layers.0.linear_qkv", "o_pro?", "q_proj,*.experts.*"])
    def test_wildcard_exclude_is_rejected(self, pattern):
        """Subtracting exact names cannot honor a pattern, and neither LoRA path can:
        the native provider matches leaf names and bridge refuses excludes alongside targets."""
        args = _args(target_modules="q_proj,k_proj,o_proj", exclude_modules=pattern)
        with pytest.raises(AssertionError, match="wildcard"):
            parse_lora_target_modules(args)
