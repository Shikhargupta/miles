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

from miles_plugins.lora.naming import _layer_prefix_from_mapping
from miles_plugins.lora.registry import GQA, MLA, MODEL_SPECS, _model_type_candidates, resolve_model_spec


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
        assert spec.attention == GQA

    def test_registered_mla_architecture_resolves(self, tmp_path):
        path = _checkpoint(tmp_path, {"model_type": "deepseek_v3"})
        model_type, spec = resolve_model_spec(_args(path), _config(mla=True))
        assert model_type == "deepseek_v3"
        assert spec.attention == MLA

    def test_nested_text_config_wins_when_the_wrapper_is_unregistered(self, tmp_path):
        path = _checkpoint(tmp_path, {"model_type": "some_vlm", "text_config": {"model_type": "qwen3_moe"}})
        model_type, spec = resolve_model_spec(_args(path), _config())
        assert model_type == "qwen3_moe"
        assert spec.attention == GQA

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

    def test_no_config_json_falls_back_to_structural_dispatch(self, tmp_path):
        assert resolve_model_spec(_args(str(tmp_path)), _config()) == (None, None)

    def test_no_checkpoint_falls_back_to_structural_dispatch(self):
        assert resolve_model_spec(_args(None), _config()) == (None, None)
        assert resolve_model_spec(SimpleNamespace(), _config()) == (None, None)


class TestRegistryTable:
    def test_every_entry_names_a_known_family(self):
        assert {spec.attention for spec in MODEL_SPECS.values()} <= {GQA, MLA}

    def test_the_e2e_verified_architectures_are_registered(self):
        """Qwen3-0.6B and Qwen3-30B-A3B ran the full RL loop in #1792."""
        assert MODEL_SPECS["qwen3"].attention == GQA
        assert MODEL_SPECS["qwen3_moe"].attention == GQA

    def test_the_shipped_mla_families_are_registered_as_mla(self):
        for model_type in ("deepseek_v3", "deepseek_v32", "glm4_moe_lite", "glm_moe_dsa", "kimi_k25"):
            assert MODEL_SPECS[model_type].attention == MLA, model_type


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
