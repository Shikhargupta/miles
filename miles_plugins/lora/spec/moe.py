"""Native-LoRA routed/grouped-expert architecture boundary.

Two per-layer MoE cases exist today, split by whether the layer carries a
DeepSeek-style always-on shared expert:

- ``SharedOuterExpertMoESpec`` — layers with a shared expert: LoRA attaches to
  the shared expert's fused MLP (the outer, always-active expert). Routed
  expert projections remain out of native scope.
- ``GeneralExpertMoESpec`` — layers with only routed/grouped experts: native
  LoRA has no MLP module to adapt, so parser-expanded ``all-linear`` targets
  skip and explicit MLP targets fail closed.

Neither implements the bridge-only ``--experts-shared-outer-loras`` adapter
layout (SGLang PR #21466, routed experts sharing outer A/B matrices); when
routed-expert LoRA lands natively, it plugs in behind these same seams.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch.nn as nn

from miles_plugins.lora.spec.base import AttachContext
from miles_plugins.lora.spec.mlp import MLP_TARGETS

logger = logging.getLogger(__name__)

_warned_dropped_parser_mlp_targets = False


@dataclass(frozen=True)
class GeneralExpertMoESpec:
    """MoE layers without a shared expert: routed/grouped experts only.

    Routed-expert LoRA is not natively supported, so MLP targets cannot attach
    anywhere on such a layer.
    """

    supported_targets: frozenset[str] = frozenset()

    def validate_layer(self, mlp: nn.Module, context: AttachContext) -> None:
        if not hasattr(mlp, "experts") or not context.targets.intersection(MLP_TARGETS):
            return
        if context.lora.expanded_from_all_linear:
            # Parser-added all-linear names mirror the MLA generic-qkv normalization:
            # skip what this architecture cannot attach instead of failing the run.
            global _warned_dropped_parser_mlp_targets
            if not _warned_dropped_parser_mlp_targets:
                _warned_dropped_parser_mlp_targets = True
                logger.info(
                    "[lora-native] all-linear MLP targets %s skipped on MoE layers without an attachable "
                    "shared expert; routed/grouped expert LoRA needs --megatron-to-hf-mode bridge or a "
                    "model-specific --lora-provider-path.",
                    sorted(context.targets.intersection(MLP_TARGETS)),
                )
            return
        raise AssertionError(
            "Miles-native LoRA does not yet support routed/grouped expert projections, and this MoE "
            "layer has no attachable shared expert. Attention-only LoRA is supported for this model; "
            "for expert gate/up/down LoRA, use --megatron-to-hf-mode bridge or a model-specific "
            "--lora-provider-path."
        )


@dataclass(frozen=True)
class SharedOuterExpertMoESpec:
    """MoE layers with a shared (outer) expert: LoRA adapts the shared expert's MLP.

    Layers without a shared expert delegate to ``GeneralExpertMoESpec``, so one
    registry entry covers models that mix both layer kinds.
    """

    supported_targets: frozenset[str] = frozenset()

    def validate_layer(self, mlp: nn.Module, context: AttachContext) -> None:
        if not hasattr(mlp, "experts") or not context.targets.intersection(MLP_TARGETS):
            return
        shared = getattr(mlp, "shared_experts", None)
        if shared is not None and hasattr(shared, "linear_fc1"):
            return
        GENERAL_EXPERT_MOE_SPEC.validate_layer(mlp, context)


GENERAL_EXPERT_MOE_SPEC = GeneralExpertMoESpec()
SHARED_OUTER_EXPERT_MOE_SPEC = SharedOuterExpertMoESpec()
