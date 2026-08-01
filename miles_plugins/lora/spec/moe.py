"""MoE architecture specifications for Miles-native LoRA.

``SharedOuterExpertMoESpec`` supports always-on shared-expert MLPs.
``GeneralExpertMoESpec`` validates routed-only layers and rejects unsupported
explicit MLP targets while allowing ``all-linear`` to skip them.

Unsupported:

- EP-shared and shared-outer routed/grouped-expert LoRA.
- Sequential per-expert and router LoRA.
- Expert-TP/EP coordination and expert-axis HF/SGLang codecs.

TODO:

- Add MoE attachment support and expert-TP/EP context.
- Implement expert adapters, synchronization, HF codec support, and SGLang
  packing.
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
