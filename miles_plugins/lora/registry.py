"""Architecture registry for native LoRA: HF ``model_type`` -> spec family.

Native LoRA only runs on architectures someone has verified; a checkpoint whose
``model_type`` is not registered fails at startup instead of silently training
the generic math on a layout nobody checked. The key is the HF config's
``model_type`` -- the same key mbridge's ``register_model`` uses -- so this
table and the raw-mode weight-conversion support stay keyed alike.

An entry pins only the family (``spec/attention.py`` GQA fused-qkv vs MLA);
per-variant details -- query groups, latent ranks, the gated query slice,
hybrid GDN mixer layers, shared experts -- are still read off the built model's
config at attach time, and the family's own guards
(``_assert_supported_architecture``) still apply on top.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

GQA = "gqa"
MLA = "mla"


@dataclass(frozen=True)
class ModelLoRASpec:
    """What the registry pins for one architecture family."""

    attention: str  # GQA (fused linear_qkv) or MLA (latent down/up projections)


_GQA_SPEC = ModelLoRASpec(attention=GQA)
_MLA_SPEC = ModelLoRASpec(attention=MLA)

# GQA entries ride the fused linear_qkv / gated-MLP path (Qwen3-0.6B and
# Qwen3-30B-A3B ran e2e; the gated hybrids are covered by the qkv permutation
# and their GDN layers carry no attention adapter). MLA entries ride the
# latent projection path, verified numerically on MLATransformerConfig models;
# the q_lora_rank=None variant is rejected by the MLA guard regardless of the
# registry entry.
MODEL_SPECS: dict[str, ModelLoRASpec] = {
    "llama": _GQA_SPEC,
    "qwen2": _GQA_SPEC,
    "qwen2_moe": _GQA_SPEC,
    "qwen3": _GQA_SPEC,
    "qwen3_moe": _GQA_SPEC,
    "mimo": _GQA_SPEC,
    "glm4": _GQA_SPEC,
    "glm4_moe": _GQA_SPEC,
    "qwen3_5": _GQA_SPEC,
    "qwen3_5_moe": _GQA_SPEC,
    "qwen3_6": _GQA_SPEC,
    "qwen3_6_moe": _GQA_SPEC,
    "qwen3_next": _GQA_SPEC,
    "deepseek_v3": _MLA_SPEC,
    "deepseek_v32": _MLA_SPEC,
    "deepseek_v4": _MLA_SPEC,
    "glm4_moe_lite": _MLA_SPEC,
    "glm_moe_dsa": _MLA_SPEC,
    "kimi_k2": _MLA_SPEC,
    "kimi_k25": _MLA_SPEC,
    "joyai_llm_flash": _MLA_SPEC,
}


def _model_type_candidates(hf_checkpoint: str | None) -> list[str]:
    """``model_type`` strings named by the checkpoint's config.json, outermost first.

    Multimodal configs nest the decoder under ``text_config``, so both levels
    are candidates; an empty list means there was no config to read.
    """
    if not hf_checkpoint:
        return []
    path = os.path.join(hf_checkpoint, "config.json")
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        config = json.load(handle)
    text_config = config.get("text_config") or {}
    return [t for t in (config.get("model_type"), text_config.get("model_type")) if t]


def resolve_model_spec(args, config) -> tuple[str | None, ModelLoRASpec | None]:
    """Return ``(model_type, spec)`` for this run's checkpoint.

    Raises when the checkpoint names an architecture nobody registered. Returns
    ``(None, None)`` -- with a warning -- when there is no config.json to read
    (numerical harnesses and unit fixtures build bare mcore models): structural
    dispatch and the per-family guards are all that apply there.
    """
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    candidates = _model_type_candidates(hf_checkpoint)
    if not candidates:
        logger.warning(
            "[lora-native] no config.json under %r; skipping the architecture registry and "
            "dispatching on the built model's structure alone.",
            hf_checkpoint,
        )
        return None, None

    model_type = next((c for c in candidates if c in MODEL_SPECS), None)
    assert model_type is not None, (
        f"native LoRA has no spec registered for model_type {candidates[0]!r} "
        f"(--hf-checkpoint {hf_checkpoint}). Registered architectures: {sorted(MODEL_SPECS)}. "
        "Verify the adapter math for this architecture and register it in "
        "miles_plugins.lora.registry.MODEL_SPECS, use --megatron-to-hf-mode bridge, or point "
        "--lora-provider-path at a model-specific provider."
    )
    spec = MODEL_SPECS[model_type]

    built = MLA if bool(getattr(config, "multi_latent_attention", False)) else GQA
    assert spec.attention == built, (
        f"registry entry for model_type {model_type!r} says {spec.attention} attention but the "
        f"built model uses {built}; the registry and the checkpoint disagree."
    )
    return model_type, spec
