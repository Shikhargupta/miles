"""Native-LoRA spec for fused gated MLPs and plain shared experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from miles_plugins.lora.modules.linear import LoRASplitFC1
from miles_plugins.lora.spec import dims
from miles_plugins.lora.spec.attach import FusedAttach, ModuleLayout, ProjectionBinding, attach_layout
from miles_plugins.lora.spec.base import COLUMN, ROW, AttachContext, ProjectionSpec

FC1_PROJECTIONS = (
    ProjectionSpec("gate_proj", "gate", COLUMN),
    ProjectionSpec("up_proj", "up", COLUMN),
)
DOWN_PROJECTION = ProjectionSpec("down_proj", "down", ROW)
MLP_TARGETS = frozenset(projection.hf for projection in (*FC1_PROJECTIONS, DOWN_PROJECTION))


def _fc1_inter_local(mlp: nn.Module, _context: AttachContext) -> int:
    """Local intermediate width of the fused ``[gate; up]`` FC1."""
    return mlp.linear_fc1.weight.shape[0] // 2


def _build_split_fc1(
    mlp: nn.Module,
    hf_prefix: str,
    context: AttachContext,
    active: tuple[ProjectionSpec, ...],
) -> LoRASplitFC1:
    return LoRASplitFC1(
        hf_prefix=hf_prefix,
        reference=mlp.linear_fc1.weight,
        context=context,
        projections=active,
        inter_local=_fc1_inter_local(mlp, context),
    )


FUSED_GATED_MLP_LAYOUT = ModuleLayout(
    name="fused_gated_mlp",
    present_when_attr="linear_fc1",
    fused=(
        FusedAttach(
            module_attr="linear_fc1",
            projections=FC1_PROJECTIONS,
            adapter_attr="lora_fc1_adapter",
            build=_build_split_fc1,
        ),
    ),
    singles=(
        ProjectionBinding(
            projection=DOWN_PROJECTION,
            module_attr="linear_fc2",
            in_dim=_fc1_inter_local,
            out_dim=dims.hidden,
            adapter_attr="lora_fc2_adapter",
        ),
    ),
)


@dataclass(frozen=True)
class FusedGatedMLPSpec:
    """Fused ``[gate; up]`` FC1 plus row-parallel down projection."""

    name: str = "fused_gated_mlp"
    supported_targets: frozenset[str] = MLP_TARGETS

    def attach(self, mlp: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        return attach_layout(mlp, FUSED_GATED_MLP_LAYOUT, hf_prefix, context)


FUSED_GATED_MLP_SPEC = FusedGatedMLPSpec()
