"""Unit tests for the native-LoRA architecture registry — no GPU, no megatron.

The registry gates raw-mode LoRA on the checkpoint's HF ``model_type``: a
verified architecture resolves to its spec family, an unregistered one fails at
startup naming the escape hatches, and a checkpoint with no config.json (the
numerical harnesses build bare mcore models) falls back to structural dispatch
with a warning.
"""

import json
from types import SimpleNamespace

import pytest

from miles_plugins.lora.codec.hf import _layer_prefix_from_mapping
from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.registry import (
    GQA,
    MLA,
    MODEL_SPECS,
    _model_type_candidates,
    resolve_model_spec,
    resolve_native_lora_config,
    resolve_registered_model_spec,
)
from miles_plugins.lora.spec.base import LoRAArchSpec


def _checkpoint(tmp_path, config: dict) -> str:
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def _args(hf_checkpoint) -> SimpleNamespace:
    return SimpleNamespace(hf_checkpoint=hf_checkpoint)


def _config(mla: bool = False) -> SimpleNamespace:
    return SimpleNamespace(multi_latent_attention=mla)


class TestModelTypeCandidates:
    def test_reads_the_top_level_model_type(self, tmp_path):
        path = _checkpoint(tmp_path, {"model_type": "qwen3"})
        assert _model_type_candidates(path) == ["qwen3"]

    def test_multimodal_configs_also_offer_the_text_config(self, tmp_path):
        path = _checkpoint(tmp_path, {"model_type": "some_vlm", "text_config": {"model_type": "qwen3"}})
        assert _model_type_candidates(path) == ["some_vlm", "qwen3"]

    def test_missing_checkpoint_or_config_yields_nothing(self, tmp_path):
        assert _model_type_candidates(None) == []
        assert _model_type_candidates(str(tmp_path)) == []


class TestResolveModelSpec:
    def test_registered_gqa_architecture_resolves(self, tmp_path):
        path = _checkpoint(tmp_path, {"model_type": "qwen3"})
        model_type, spec = resolve_model_spec(_args(path), _config())
        assert model_type == "qwen3"
        assert spec.name == GQA
        assert spec.attention.name == GQA

    def test_registered_mla_architecture_resolves(self, tmp_path):
        path = _checkpoint(tmp_path, {"model_type": "deepseek_v3"})
        model_type, spec = resolve_model_spec(_args(path), _config(mla=True))
        assert model_type == "deepseek_v3"
        assert spec.name == MLA
        assert spec.attention.name == MLA

    def test_nested_text_config_wins_when_the_wrapper_is_unregistered(self, tmp_path):
        path = _checkpoint(tmp_path, {"model_type": "some_vlm", "text_config": {"model_type": "qwen3_moe"}})
        model_type, spec = resolve_model_spec(_args(path), _config())
        assert model_type == "qwen3_moe"
        assert spec.name == GQA

    def test_unregistered_architecture_raises_naming_the_escape_hatches(self, tmp_path):
        path = _checkpoint(tmp_path, {"model_type": "gpt_oss"})
        with pytest.raises(AssertionError) as excinfo:
            resolve_model_spec(_args(path), _config())
        message = str(excinfo.value)
        assert "gpt_oss" in message
        assert "MODEL_SPECS" in message
        assert "--megatron-to-hf-mode bridge" in message
        assert "--lora-provider-path" in message

    def test_registry_and_built_model_must_agree_on_the_family(self, tmp_path):
        path = _checkpoint(tmp_path, {"model_type": "qwen3"})
        with pytest.raises(AssertionError, match="disagree"):
            resolve_model_spec(_args(path), _config(mla=True))

    def test_checkpoint_without_config_json_fails_closed(self, tmp_path):
        with pytest.raises(AssertionError, match="missing"):
            resolve_model_spec(_args(str(tmp_path)), _config())

    def test_config_json_without_model_type_names_the_actual_problem(self, tmp_path):
        """The error must not claim config.json is missing when it exists but lacks model_type."""
        path = _checkpoint(tmp_path, {"hidden_size": 64})
        with pytest.raises(AssertionError, match="neither") as excinfo:
            resolve_model_spec(_args(path), _config())
        assert "missing" not in str(excinfo.value)

    def test_no_checkpoint_falls_back_to_structural_dispatch(self):
        assert resolve_model_spec(_args(None), _config())[1].name == GQA
        assert resolve_model_spec(SimpleNamespace(), _config(mla=True))[1].name == MLA


class TestPreBuildResolution:
    def test_registered_spec_resolves_without_built_transformer_config(self, tmp_path):
        path = _checkpoint(tmp_path, {"model_type": "qwen3"})
        model_type, spec = resolve_registered_model_spec(path)
        assert model_type == "qwen3"
        assert spec.name == GQA

    def test_prebuild_resolution_has_no_structural_fallback(self):
        with pytest.raises(AssertionError, match="hf-checkpoint"):
            resolve_registered_model_spec(None)

    def test_effective_mla_targets_are_available_before_model_build(self, tmp_path):
        path = _checkpoint(tmp_path, {"model_type": "deepseek_v3"})
        args = SimpleNamespace(
            hf_checkpoint=path,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa",
                "kv_b_proj",
            ],
            _target_modules_expanded_from_all_linear=True,
        )
        effective = resolve_native_lora_config(args)
        assert not effective.target_modules.intersection({"q_proj", "k_proj", "v_proj"})
        assert {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"} <= effective.target_modules

    def test_prebuild_config_rejects_unsupported_targets(self, tmp_path):
        path = _checkpoint(tmp_path, {"model_type": "qwen3"})
        args = SimpleNamespace(
            hf_checkpoint=path,
            lora_rank=8,
            lora_alpha=16,
            target_modules=["not_a_projection"],
        )
        with pytest.raises(AssertionError, match="does not implement"):
            resolve_native_lora_config(args)


class TestRegistryTable:
    def test_every_entry_names_a_known_family(self):
        assert all(isinstance(spec, LoRAArchSpec) for spec in MODEL_SPECS.values())
        assert {spec.model_family for spec in MODEL_SPECS.values()} <= {GQA, MLA}

    def test_the_e2e_verified_architectures_are_registered(self):
        """Qwen3-0.6B and Qwen3-30B-A3B ran the full RL loop in #1792."""
        assert MODEL_SPECS["qwen3"].name == GQA
        assert MODEL_SPECS["qwen3_moe"].name == GQA

    def test_the_shipped_mla_families_are_registered_as_mla(self):
        for model_type in ("deepseek_v3", "deepseek_v32", "glm4_moe_lite", "glm_moe_dsa", "kimi_k25"):
            assert MODEL_SPECS[model_type].name == MLA, model_type

    def test_hybrid_gdn_family_warns_about_raw_mode_backward(self, tmp_path, caplog):
        """The adapter side is verified, but raw mode's frozen-base backward diverges there;
        the run must say so instead of starting clean and blowing up at step 1."""
        import logging

        path = _checkpoint(tmp_path, {"model_type": "qwen3_5_moe"})
        with caplog.at_level(logging.WARNING, logger="miles_plugins.lora.registry"):
            model_type, spec = resolve_model_spec(_args(path), _config())
        assert model_type == "qwen3_5_moe"
        assert spec.name == "gqa_gdn"
        assert "bridge" in caplog.text and "grad_norm" in caplog.text

    def test_stable_families_do_not_warn(self, tmp_path, caplog):
        import logging

        path = _checkpoint(tmp_path, {"model_type": "qwen3"})
        with caplog.at_level(logging.WARNING, logger="miles_plugins.lora.registry"):
            resolve_model_spec(_args(path), _config())
        assert caplog.text == ""

    def test_deepseek_v4_flash_is_not_registered(self):
        """DeepSeek-V4-Flash attention is wq_a/wq_b/wkv, not mcore MLA; docs declare it out of
        scope, so the registry must fail closed instead of crashing inside MLA attach."""
        assert "deepseek_v4" not in MODEL_SPECS

    def test_qwen_hybrids_use_per_layer_gqa_gdn_dispatch(self):
        for model_type in ("qwen3_5", "qwen3_6", "qwen3_next"):
            assert MODEL_SPECS[model_type].name == "gqa_gdn"
            assert MODEL_SPECS[model_type].attention.name == "gqa_gdn"

    def test_mla_all_linear_drops_only_parser_added_generic_qkv_targets(self):
        config = LoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.0,
            target_modules=frozenset(
                {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "q_a_proj",
                    "q_b_proj",
                    "kv_a_proj_with_mqa",
                    "kv_b_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                }
            ),
            expanded_from_all_linear=True,
        )
        normalized = MODEL_SPECS["deepseek_v3"].normalize_config(config)
        assert not normalized.target_modules.intersection({"q_proj", "k_proj", "v_proj"})
        assert {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"} <= normalized.target_modules

    def test_mla_explicit_mixed_targets_are_not_silently_rewritten(self):
        config = LoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.0,
            target_modules=frozenset({"q_proj", "q_a_proj"}),
        )
        normalized = MODEL_SPECS["deepseek_v3"].normalize_config(config)
        assert normalized is config
        assert "q_proj" not in MODEL_SPECS["deepseek_v3"].supported_targets

    def test_architecture_normalization_cannot_remove_every_target(self):
        config = LoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.0,
            target_modules=frozenset({"q_proj", "k_proj", "v_proj"}),
            expanded_from_all_linear=True,
        )
        with pytest.raises(AssertionError, match="no effective target modules"):
            MODEL_SPECS["deepseek_v3"].normalize_config(config)


class TestMbridgePrefixParsing:
    """The optional mbridge cross-check parses layer prefixes out of bridge mapping tables."""

    def test_qwen3_5_style_nested_prefix(self):
        mapping = {
            "self_attention.linear_proj.weight": ["model.language_model.layers.{layer_number}.self_attn.o_proj.weight"]
        }
        assert _layer_prefix_from_mapping(mapping) == "model.language_model.layers."

    def test_plain_prefix(self):
        mapping = {"self_attention.linear_proj.weight": ["model.layers.{layer_number}.self_attn.o_proj.weight"]}
        assert _layer_prefix_from_mapping(mapping) == "model.layers."

    def test_scalar_values_are_accepted(self):
        mapping = {"decoder.final_layernorm.weight": "model.norm.weight"}
        assert _layer_prefix_from_mapping(mapping) is None

    def test_no_layer_template_returns_none(self):
        assert _layer_prefix_from_mapping({}) is None
