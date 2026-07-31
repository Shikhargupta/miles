"""HF ``model_type`` to complete native-LoRA architecture-spec registry."""

from __future__ import annotations

import json
import logging
import os

from miles_plugins.lora.spec.attention import GQA_ATTENTION_SPEC, HYBRID_GQA_GDN_ATTENTION_SPEC, MLA_ATTENTION_SPEC
from miles_plugins.lora.spec.base import LoRAArchSpec
from miles_plugins.lora.spec.mlp import FUSED_GATED_MLP_SPEC
from miles_plugins.lora.spec.moe import SHARED_EXPERT_ONLY_MOE_SPEC

logger = logging.getLogger(__name__)

GQA = "gqa"
MLA = "mla"

_GQA_SPEC = LoRAArchSpec(
    name=GQA,
    model_family=GQA,
    attention=GQA_ATTENTION_SPEC,
    mlp=FUSED_GATED_MLP_SPEC,
    moe=SHARED_EXPERT_ONLY_MOE_SPEC,
)
_MLA_SPEC = LoRAArchSpec(
    name=MLA,
    model_family=MLA,
    attention=MLA_ATTENTION_SPEC,
    mlp=FUSED_GATED_MLP_SPEC,
    moe=SHARED_EXPERT_ONLY_MOE_SPEC,
)
_HYBRID_GQA_SPEC = LoRAArchSpec(
    name="gqa_gdn",
    model_family=GQA,
    attention=HYBRID_GQA_GDN_ATTENTION_SPEC,
    mlp=FUSED_GATED_MLP_SPEC,
    moe=SHARED_EXPERT_ONLY_MOE_SPEC,
)

# Every entry is explicitly mapped to a structurally covered spec. Variant
# dimensions and runtime validation remain inside the concrete specs; tests
# separately record which model types have completed end-to-end validation.
MODEL_SPECS: dict[str, LoRAArchSpec] = {
    "llama": _GQA_SPEC,
    "qwen2": _GQA_SPEC,
    "qwen2_moe": _GQA_SPEC,
    "qwen3": _GQA_SPEC,
    "qwen3_moe": _GQA_SPEC,
    "mimo": _GQA_SPEC,
    "glm4": _GQA_SPEC,
    "glm4_moe": _GQA_SPEC,
    "qwen3_5": _HYBRID_GQA_SPEC,
    "qwen3_5_moe": _HYBRID_GQA_SPEC,
    "qwen3_6": _HYBRID_GQA_SPEC,
    "qwen3_6_moe": _HYBRID_GQA_SPEC,
    "qwen3_next": _HYBRID_GQA_SPEC,
    "deepseek_v3": _MLA_SPEC,
    "deepseek_v32": _MLA_SPEC,
    # deepseek_v4 (DeepSeek-V4-Flash) stays unregistered: its wq_a/wq_b/wkv attention is not
    # mcore MLA, and docs/advanced/lora.md declares that layout out of scope for this provider.
    "glm4_moe_lite": _MLA_SPEC,
    "glm_moe_dsa": _MLA_SPEC,
    "kimi_k2": _MLA_SPEC,
    "kimi_k25": _MLA_SPEC,
    "joyai_llm_flash": _MLA_SPEC,
}


def _model_type_candidates(hf_checkpoint: str | None) -> list[str]:
    """Return outer and nested text ``model_type`` values from HF config.json."""
    if not hf_checkpoint:
        return []
    path = os.path.join(hf_checkpoint, "config.json")
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        config = json.load(handle)
    text_config = config.get("text_config") or {}
    return [model_type for model_type in (config.get("model_type"), text_config.get("model_type")) if model_type]


def _structural_spec(config) -> LoRAArchSpec:
    """Spec used only by bare numerical/unit harnesses without an HF checkpoint."""
    return _MLA_SPEC if bool(getattr(config, "multi_latent_attention", False)) else _GQA_SPEC


def resolve_model_spec(args, config) -> tuple[str | None, LoRAArchSpec]:
    """Resolve a complete architecture spec and verify it matches the built model.

    Production checkpoints fail closed when their model type is not registered.
    Bare test harnesses without config.json retain a warned structural fallback,
    but still receive a concrete spec that drives attachment.
    """
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    candidates = _model_type_candidates(hf_checkpoint)
    if not candidates:
        if hf_checkpoint:
            config_path = os.path.join(hf_checkpoint, "config.json")
            if os.path.exists(config_path):
                raise AssertionError(
                    f"native LoRA could not resolve a model_type: {config_path} declares neither "
                    "'model_type' nor 'text_config.model_type'. Fix the checkpoint config or use a "
                    "model-specific --lora-provider-path."
                )
            raise AssertionError(
                f"native LoRA could not load model_type because {hf_checkpoint!r}/config.json is missing. "
                "Provide a valid --hf-checkpoint or use a model-specific --lora-provider-path."
            )
        spec = _structural_spec(config)
        logger.warning(
            "[lora-native] no config.json under %r; using the %s architecture spec from the built "
            "model structure. Production checkpoints must register model_type explicitly.",
            hf_checkpoint,
            spec.name,
        )
        return None, spec

    model_type = next((candidate for candidate in candidates if candidate in MODEL_SPECS), None)
    assert model_type is not None, (
        f"native LoRA has no spec registered for model_type {candidates[0]!r} "
        f"(--hf-checkpoint {hf_checkpoint}). Registered architectures: {sorted(MODEL_SPECS)}. "
        "Verify the adapter math for this architecture and register it in "
        "miles_plugins.lora.registry.MODEL_SPECS, use --megatron-to-hf-mode bridge, or point "
        "--lora-provider-path at a model-specific provider."
    )
    spec = MODEL_SPECS[model_type]

    built = MLA if bool(getattr(config, "multi_latent_attention", False)) else GQA
    assert spec.model_family == built, (
        f"registry entry for model_type {model_type!r} says {spec.model_family} attention but the "
        f"built model uses {built}; the registry and the checkpoint disagree."
    )
    return model_type, spec
