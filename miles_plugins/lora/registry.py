"""HF ``model_type`` to complete native-LoRA architecture-spec registry."""

from __future__ import annotations

import json
import logging
import os

from miles_plugins.lora.config import LoRAConfig
from miles_plugins.lora.spec.attention import GQA_ATTENTION_SPEC, HYBRID_GQA_GDN_ATTENTION_SPEC, MLA_ATTENTION_SPEC
from miles_plugins.lora.spec.base import LoRAArchSpec
from miles_plugins.lora.spec.mlp import FUSED_GATED_MLP_SPEC
from miles_plugins.lora.spec.moe import SHARED_OUTER_EXPERT_MOE_SPEC

logger = logging.getLogger(__name__)

GQA = "gqa"
MLA = "mla"

_GQA_SPEC = LoRAArchSpec(
    name=GQA,
    model_family=GQA,
    attention=GQA_ATTENTION_SPEC,
    mlp=FUSED_GATED_MLP_SPEC,
    moe=SHARED_OUTER_EXPERT_MOE_SPEC,
)
_MLA_SPEC = LoRAArchSpec(
    name=MLA,
    model_family=MLA,
    attention=MLA_ATTENTION_SPEC,
    mlp=FUSED_GATED_MLP_SPEC,
    moe=SHARED_OUTER_EXPERT_MOE_SPEC,
)
_HYBRID_GQA_SPEC = LoRAArchSpec(
    name="gqa_gdn",
    model_family=GQA,
    attention=HYBRID_GQA_GDN_ATTENTION_SPEC,
    mlp=FUSED_GATED_MLP_SPEC,
    moe=SHARED_OUTER_EXPERT_MOE_SPEC,
    # A PP/VPP chunk may contain only GDN mixer layers. Native GDN adapters are
    # intentionally absent, while GQA layers in another chunk still carry LoRA.
    allows_mixer_only_adapter_chunks=True,
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


# Registered (the adapter side is verified: attach, export and SGLang ingest all check out) but
# raw mode's own frozen-base backward blows up on these; see scripts/run_lora_native.py's header.
_RAW_MODE_BACKWARD_UNSTABLE = frozenset({"qwen3_5", "qwen3_5_moe", "qwen3_6", "qwen3_6_moe", "qwen3_next"})


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


def resolve_registered_model_spec(hf_checkpoint: str | None) -> tuple[str, LoRAArchSpec]:
    """Resolve a registered spec from checkpoint metadata without a built model.

    Unlike :func:`resolve_model_spec`, this helper has no structural fallback:
    serving/configuration callers run before model construction and must fail
    closed when checkpoint metadata is absent or unsupported.
    """
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
        raise AssertionError(
            "native LoRA requires --hf-checkpoint/config.json to resolve an architecture spec before "
            "model construction; provide a checkpoint or a model-specific --lora-provider-path."
        )

    model_type = next((candidate for candidate in candidates if candidate in MODEL_SPECS), None)
    assert model_type is not None, (
        f"native LoRA has no spec registered for model_type {candidates[0]!r} "
        f"(--hf-checkpoint {hf_checkpoint}). Registered architectures: {sorted(MODEL_SPECS)}. "
        "Verify the adapter math for this architecture and register it in "
        "miles_plugins.lora.registry.MODEL_SPECS, use --megatron-to-hf-mode bridge, or point "
        "--lora-provider-path at a model-specific provider."
    )
    return model_type, MODEL_SPECS[model_type]


def resolve_native_lora_config(args) -> LoRAConfig:
    """Return the architecture-normalized config before native model build.

    Rollout setup can consume ``.target_modules`` from this result so SGLang
    allocates buffers for the same effective projection set the native model
    will attach (notably MLA ``all-linear`` normalization).
    """
    _model_type, spec = resolve_registered_model_spec(getattr(args, "hf_checkpoint", None))
    config = spec.normalize_config(LoRAConfig.from_args(args))
    spec.validate_targets(config.target_modules)
    return config


def resolve_model_spec(args, config) -> tuple[str | None, LoRAArchSpec]:
    """Resolve a complete architecture spec and verify it matches the built model.

    Production checkpoints fail closed when their model type is not registered.
    Bare test harnesses without config.json retain a warned structural fallback,
    but still receive a concrete spec that drives attachment.
    """
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        spec = _structural_spec(config)
        logger.warning(
            "[lora-native] no config.json under %r; using the %s architecture spec from the built "
            "model structure. Production checkpoints must register model_type explicitly.",
            hf_checkpoint,
            spec.name,
        )
        return None, spec

    model_type, spec = resolve_registered_model_spec(hf_checkpoint)
    if model_type in _RAW_MODE_BACKWARD_UNSTABLE:
        logger.warning(
            "[lora-native] %s adapter attachment is verified, but raw mode's own backward is known to "
            "diverge on this family once the base is frozen (grad_norm 1e7-1e10 with recompute, NaN "
            "without it, NaN at CP=2), while bridge mode stays stable on the identical batch. This is a "
            "model-path issue rather than a LoRA one; prefer --megatron-to-hf-mode bridge until it is "
            "fixed, and watch grad_norm from step 1 if you continue.",
            model_type,
        )

    built = MLA if bool(getattr(config, "multi_latent_attention", False)) else GQA
    assert spec.model_family == built, (
        f"registry entry for model_type {model_type!r} says {spec.model_family} attention but the "
        f"built model uses {built}; the registry and the checkpoint disagree."
    )
    return model_type, spec
