"""Callable linear adapter branches used by the Miles-native LoRA specs.

The adapter remains a sibling of the physical MCore/TE linear so base parameter
names such as ``linear_qkv.weight`` do not change. A single attachment seam
patches the physical linear's forward and calls one of the modules below for the
LoRA delta. This gives LoRA parameters a real module call path without changing
base checkpoint, weight-sync, or quantizer naming contracts.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from miles_plugins.lora.distributed import apply_lora_dropout, branch_input, reduce_row_parallel
from miles_plugins.lora.spec.base import COLUMN, REPLICATED, ROW, AttachContext, ProjectionSpec


def new_lora_parameter(
    reference: torch.Tensor,
    shape,
    *,
    init: str,
    grad_sum_group: str | None = None,
    partition_dim: int | None = None,
) -> nn.Parameter:
    """Create an adapter parameter matching the base weight's dtype and device."""
    tensor = torch.empty(*shape, dtype=reference.dtype, device=reference.device)
    if init == "zero":
        tensor.zero_()
    else:
        nn.init.xavier_uniform_(tensor)
    parameter = nn.Parameter(tensor)
    parameter.tensor_model_parallel = partition_dim is not None
    parameter.partition_dim = partition_dim if partition_dim is not None else -1
    parameter.partition_stride = 1
    if grad_sum_group is not None:
        parameter._lora_grad_sum_group = grad_sum_group
    return parameter


class NativeLoRAAdapter(nn.Module):
    """Base class for a self-describing native-LoRA delta module."""

    def __init__(
        self,
        hf_prefix: str,
        projection_specs: Sequence[ProjectionSpec],
        tp_rank: int,
    ):
        super().__init__()
        projection_specs = tuple(projection_specs)
        assert projection_specs, "a native LoRA adapter requires at least one projection"
        assert len({projection.hf for projection in projection_specs}) == len(
            projection_specs
        ), "native LoRA projection HF names must be unique"
        assert len({projection.attr for projection in projection_specs}) == len(
            projection_specs
        ), "native LoRA projection parameter attributes must be unique"
        assert all(
            projection.layout in (COLUMN, ROW, REPLICATED) for projection in projection_specs
        ), "native LoRA projection has an unknown parallel layout"
        self.hf_prefix = hf_prefix
        self.tp_rank = tp_rank
        self._projection_specs = projection_specs

    @property
    def projection_specs(self) -> tuple[ProjectionSpec, ...]:
        return self._projection_specs

    def _validate_projection_parameters(self) -> None:
        for projection in self._projection_specs:
            assert hasattr(self, f"{projection.attr}_A") and hasattr(
                self, f"{projection.attr}_B"
            ), f"native LoRA projection {projection.hf!r} has no complete A/B parameter pair"

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Keep adapter params out of MCore distributed checkpoints for now."""
        return {}


class LoRALinear(NativeLoRAAdapter):
    """One logical column-parallel, row-parallel, or replicated LoRA projection."""

    def __init__(
        self,
        *,
        hf_prefix: str,
        projection: ProjectionSpec,
        reference: torch.Tensor,
        context: AttachContext,
        in_features: int,
        out_features: int,
    ):
        super().__init__(hf_prefix, (projection,), context.tp_rank)
        self.context = context
        self.attr = projection.attr
        self.layout = projection.layout

        a_grad_group = "tp" if self.layout == COLUMN else None
        b_grad_group = "tp" if self.layout in (ROW, REPLICATED) and context.sequence_parallel else None
        if self.layout == REPLICATED and context.sequence_parallel:
            a_grad_group = "tp"
        self.register_parameter(
            f"{self.attr}_A",
            new_lora_parameter(
                reference,
                (context.rank, in_features),
                init=context.a_init,
                grad_sum_group=a_grad_group,
                partition_dim=1 if self.layout == ROW else None,
            ),
        )
        self.register_parameter(
            f"{self.attr}_B",
            new_lora_parameter(
                reference,
                (out_features, context.rank),
                init="zero",
                grad_sum_group=b_grad_group,
                partition_dim=0 if self.layout == COLUMN else None,
            ),
        )
        self._validate_projection_parameters()

    def forward(self, x: torch.Tensor, base_module: nn.Module) -> torch.Tensor:
        a = getattr(self, f"{self.attr}_A")
        b = getattr(self, f"{self.attr}_B")
        if self.layout == COLUMN:
            x = branch_input(x, base_module, self.context)
            return F.linear(F.linear(x, a), b)
        if self.layout == ROW:
            x = apply_lora_dropout(x, self.context, base_module.training)
            partial = F.linear(x, a)
            return F.linear(reduce_row_parallel(partial, self.context), b)
        assert self.layout == REPLICATED, f"unknown LoRA linear layout {self.layout}"
        x = apply_lora_dropout(x, self.context, base_module.training)
        return F.linear(F.linear(x, a), b)


class LoRASplitQKV(NativeLoRAAdapter):
    """Independent Q/K/V adapters whose delta is packed into one fused QKV output."""

    def __init__(
        self,
        *,
        hf_prefix: str,
        reference: torch.Tensor,
        context: AttachContext,
        projections: Sequence[ProjectionSpec],
        num_q: int,
        num_kv: int,
        head_dim: int,
    ):
        q_rows = num_q * head_dim * (2 if context.output_gate else 1)
        projections = tuple(projections)
        attrs = [projection.attr for projection in projections]
        assert len(set(attrs)) == len(attrs), "LoRASplitQKV projection attributes must be unique"
        assert set(attrs) <= {"q", "k", "v"}, "LoRASplitQKV requires q/k/v projections"
        assert all(
            projection.layout == COLUMN for projection in projections
        ), "LoRASplitQKV projections must be column parallel"
        by_attr = {projection.attr: projection for projection in projections}
        projections = tuple(by_attr[name] for name in ("q", "k", "v") if name in by_attr)
        super().__init__(
            hf_prefix,
            projections,
            context.tp_rank,
        )
        self.context = context
        self._rows = {"q": q_rows, "k": num_kv * head_dim, "v": num_kv * head_dim}
        self._active = tuple(projection.attr for projection in projections)
        for name in self._active:
            self.register_parameter(
                f"{name}_A",
                new_lora_parameter(
                    reference,
                    (context.rank, context.hidden),
                    init=context.a_init,
                    grad_sum_group="tp",
                ),
            )
            self.register_parameter(
                f"{name}_B",
                new_lora_parameter(
                    reference,
                    (self._rows[name], context.rank),
                    init="zero",
                    partition_dim=0,
                ),
            )
        self.register_buffer(
            "out_perm",
            build_qkv_permutation(num_q, num_kv, head_dim, reference.device, context.output_gate),
            persistent=False,
        )
        self._validate_projection_parameters()

    def forward(self, x: torch.Tensor, base_module: nn.Module) -> torch.Tensor:
        x = branch_input(x, base_module, self.context)
        rank = self.context.rank
        down = F.linear(x, torch.cat([getattr(self, f"{name}_A") for name in self._active], dim=0))
        active_delta = {
            name: F.linear(down[..., index * rank : (index + 1) * rank], getattr(self, f"{name}_B"))
            for index, name in enumerate(self._active)
        }
        full_delta = [
            active_delta[name] if name in active_delta else x.new_zeros(*x.shape[:-1], rows)
            for name, rows in self._rows.items()
        ]
        return torch.cat(full_delta, dim=-1).index_select(-1, self.out_perm)


class LoRASplitFC1(NativeLoRAAdapter):
    """Independent gate/up adapters whose delta is packed into one fused FC1 output."""

    def __init__(
        self,
        *,
        hf_prefix: str,
        reference: torch.Tensor,
        context: AttachContext,
        projections: Sequence[ProjectionSpec],
        inter_local: int,
    ):
        projections = tuple(projections)
        attrs = [projection.attr for projection in projections]
        assert len(set(attrs)) == len(attrs), "LoRASplitFC1 projection attributes must be unique"
        assert set(attrs) <= {"gate", "up"}, "LoRASplitFC1 requires gate/up projections"
        assert all(
            projection.layout == COLUMN for projection in projections
        ), "LoRASplitFC1 projections must be column parallel"
        by_attr = {projection.attr: projection for projection in projections}
        projections = tuple(by_attr[name] for name in ("gate", "up") if name in by_attr)
        super().__init__(
            hf_prefix,
            projections,
            context.tp_rank,
        )
        self.context = context
        self.inter_local = inter_local
        self._active = tuple(projection.attr for projection in projections)
        for name in self._active:
            self.register_parameter(
                f"{name}_A",
                new_lora_parameter(
                    reference,
                    (context.rank, context.hidden),
                    init=context.a_init,
                    grad_sum_group="tp",
                ),
            )
            self.register_parameter(
                f"{name}_B",
                new_lora_parameter(
                    reference,
                    (inter_local, context.rank),
                    init="zero",
                    partition_dim=0,
                ),
            )
        self._validate_projection_parameters()

    def forward(self, x: torch.Tensor, base_module: nn.Module) -> torch.Tensor:
        x = branch_input(x, base_module, self.context)
        rank = self.context.rank
        down = F.linear(x, torch.cat([getattr(self, f"{name}_A") for name in self._active], dim=0))
        active_delta = {
            name: F.linear(down[..., index * rank : (index + 1) * rank], getattr(self, f"{name}_B"))
            for index, name in enumerate(self._active)
        }
        return torch.cat(
            [
                active_delta[name] if name in active_delta else x.new_zeros(*x.shape[:-1], self.inter_local)
                for name in ("gate", "up")
            ],
            dim=-1,
        )


def attach_adapter_forward(module: nn.Module, adapter: NativeLoRAAdapter, scale: float) -> None:
    """Add a callable adapter module's delta while preserving ``(out, bias)``."""
    original = module.forward

    def forward(x, *args, **kwargs):
        out, bias = original(x, *args, **kwargs)
        return torch.add(out, adapter(x, module), alpha=scale), bias

    module.forward = forward


def build_qkv_permutation(
    num_q_heads: int,
    num_groups: int,
    head_dim: int,
    device,
    output_gate: bool = False,
) -> torch.Tensor:
    """Map plain ``[q; k; v]`` rows into MCore's per-query-group layout."""
    q_per_group = num_q_heads // num_groups
    q_slices = 2 if output_gate else 1
    k_base = num_q_heads * q_slices * head_dim
    v_base = k_base + num_groups * head_dim
    index: list[int] = []
    for group in range(num_groups):
        for slice_index in range(q_slices):
            for head in range(q_per_group):
                start = ((group * q_per_group + head) * q_slices + slice_index) * head_dim
                index.extend(range(start, start + head_dim))
        index.extend(range(k_base + group * head_dim, k_base + (group + 1) * head_dim))
        index.extend(range(v_base + group * head_dim, v_base + (group + 1) * head_dim))
    return torch.tensor(index, dtype=torch.long, device=device)


def iter_adapters(model_chunks: Sequence[nn.Module]):
    for chunk in model_chunks:
        module = chunk
        while hasattr(module, "module"):
            module = module.module
        for child in module.modules():
            if isinstance(child, NativeLoRAAdapter):
                yield child
