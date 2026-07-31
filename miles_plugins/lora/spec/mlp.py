"""Native-LoRA spec for fused gated MLPs and plain shared experts."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from miles_plugins.lora.modules.linear import LoRALinear, SplitFC1, attach_adapter_forward
from miles_plugins.lora.spec.base import COLUMN, ROW, AttachContext, ProjectionSpec

FC1_PROJECTIONS = (
    ProjectionSpec("gate_proj", "gate", COLUMN),
    ProjectionSpec("up_proj", "up", COLUMN),
)
DOWN_PROJECTION = ProjectionSpec("down_proj", "down", ROW)
MLP_TARGETS = frozenset(projection.hf for projection in (*FC1_PROJECTIONS, DOWN_PROJECTION))


@dataclass(frozen=True)
class FusedGatedMLPSpec:
    """Fused ``[gate; up]`` FC1 plus row-parallel down projection."""

    name: str = "fused_gated_mlp"
    supported_targets: frozenset[str] = MLP_TARGETS

    def attach(self, mlp: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        count = 0
        inter_local = mlp.linear_fc1.weight.shape[0] // 2

        fc1_projections = tuple(projection for projection in FC1_PROJECTIONS if projection.hf in context.targets)
        if fc1_projections:
            adapter = SplitFC1(
                hf_prefix=hf_prefix,
                reference=mlp.linear_fc1.weight,
                context=context,
                projections=fc1_projections,
                inter_local=inter_local,
            )
            mlp.lora_fc1_adapter = adapter
            attach_adapter_forward(mlp.linear_fc1, adapter, context.scale)
            count += 1

        if context.wants("down_proj"):
            adapter = LoRALinear(
                hf_prefix=hf_prefix,
                projection=DOWN_PROJECTION,
                reference=mlp.linear_fc2.weight,
                context=context,
                in_features=inter_local,
                out_features=context.hidden,
            )
            mlp.lora_fc2_adapter = adapter
            attach_adapter_forward(mlp.linear_fc2, adapter, context.scale)
            count += 1
        return count


FUSED_GATED_MLP_SPEC = FusedGatedMLPSpec()


def _attach_mlp(mlp: nn.Module, hf_prefix: str, context: AttachContext) -> int:
    return FUSED_GATED_MLP_SPEC.attach(mlp, hf_prefix, context)
