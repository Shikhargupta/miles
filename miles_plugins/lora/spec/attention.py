"""Native-LoRA attention specs for fused GQA, gated GQA, MLA, and future GDN.

Each family is a declarative :class:`ModuleLayout` table — the projection
names, the physical linears they live on, and their shard geometry — walked by
the shared :func:`attach_layout` mechanism. Only genuinely per-family logic
(fused-QKV construction, replicated-layout guards, hybrid dispatch) remains as
code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from miles_plugins.lora.modules.linear import LoRASplitQKV
from miles_plugins.lora.spec import dims
from miles_plugins.lora.spec.attach import FusedAttach, ModuleLayout, ProjectionBinding, ServingGroup, attach_layout
from miles_plugins.lora.spec.base import COLUMN, REPLICATED, ROW, AttachContext, ProjectionSpec

QKV_PROJECTIONS = (
    ProjectionSpec("q_proj", "q", COLUMN),
    ProjectionSpec("k_proj", "k", COLUMN),
    ProjectionSpec("v_proj", "v", COLUMN),
)
O_PROJECTION = ProjectionSpec("o_proj", "o", ROW)
MLA_Q_A_PROJECTION = ProjectionSpec("q_a_proj", "a", REPLICATED)
MLA_Q_B_PROJECTION = ProjectionSpec("q_b_proj", "b", COLUMN)
MLA_KV_A_PROJECTION = ProjectionSpec("kv_a_proj_with_mqa", "a", REPLICATED)
MLA_KV_B_PROJECTION = ProjectionSpec("kv_b_proj", "b", COLUMN)

GQA_TARGETS = frozenset(projection.hf for projection in (*QKV_PROJECTIONS, O_PROJECTION))
MLA_TARGETS = frozenset(
    projection.hf
    for projection in (MLA_Q_A_PROJECTION, MLA_Q_B_PROJECTION, MLA_KV_A_PROJECTION, MLA_KV_B_PROJECTION, O_PROJECTION)
)
GDN_TARGETS = frozenset({"in_proj_qkvz", "in_proj_ba"})
_GENERIC_QKV_TARGETS = frozenset({"q_proj", "k_proj", "v_proj"})


def _build_split_qkv(
    attention: nn.Module,
    hf_prefix: str,
    context: AttachContext,
    active: tuple[ProjectionSpec, ...],
) -> LoRASplitQKV:
    return LoRASplitQKV(
        hf_prefix=hf_prefix,
        reference=attention.linear_qkv.weight,
        context=context,
        projections=active,
        num_q=attention.num_attention_heads_per_partition,
        num_kv=attention.num_query_groups_per_partition,
        head_dim=attention.hidden_size_per_attention_head,
    )


def _replicated_guard(host: nn.Module, _context: AttachContext, projection: ProjectionSpec, full_out: int) -> None:
    assert _is_replicated_linear(host, full_out), (
        f"native MLA LoRA expects a replicated {projection.hf} (TELinear parallel_mode='duplicated'); "
        f"this build shards it ({tuple(host.weight.shape)} vs full out {full_out}). "
        "Use --lora-provider-path for this variant."
    )


GQA_LAYOUT = ModuleLayout(
    name="gqa",
    present_when_attr="linear_qkv",
    fused=(
        FusedAttach(
            module_attr="linear_qkv",
            projections=QKV_PROJECTIONS,
            adapter_attr="lora_qkv_adapter",
            build=_build_split_qkv,
        ),
    ),
    singles=(
        ProjectionBinding(
            projection=O_PROJECTION,
            module_attr="linear_proj",
            in_dim=dims.gqa_o_in_local,
            out_dim=dims.hidden,
            adapter_attr="lora_o_adapter",
        ),
    ),
)

# SGLang packs the two replicated MLA down projections into one
# fused_qkv_a_proj_with_mqa buffer; each member's true output width comes from
# the architecture config, not from its sibling's shape.
_MLA_A_SERVING_GROUP = ServingGroup(
    name="mla_a",
    member_rows=(
        ("q_a_proj", dims.cfg("q_lora_rank")),
        ("kv_a_proj_with_mqa", dims.mla_kv_down_out),
    ),
)

MLA_LAYOUT = ModuleLayout(
    name="mla",
    singles=(
        ProjectionBinding(
            projection=MLA_Q_A_PROJECTION,
            module_attr="linear_q_down_proj",
            in_dim=dims.hidden,
            out_dim=dims.cfg("q_lora_rank"),
            adapter_attr="lora_mla_q_a_adapter",
            guard=_replicated_guard,
            serving_group=_MLA_A_SERVING_GROUP,
        ),
        ProjectionBinding(
            projection=MLA_Q_B_PROJECTION,
            module_attr="linear_q_up_proj",
            in_dim=dims.cfg("q_lora_rank"),
            out_dim=dims.mla_q_up_out_local,
            adapter_attr="lora_mla_q_b_adapter",
        ),
        ProjectionBinding(
            projection=MLA_KV_A_PROJECTION,
            module_attr="linear_kv_down_proj",
            in_dim=dims.hidden,
            out_dim=dims.mla_kv_down_out,
            adapter_attr="lora_mla_kv_a_adapter",
            guard=_replicated_guard,
            serving_group=_MLA_A_SERVING_GROUP,
        ),
        ProjectionBinding(
            projection=MLA_KV_B_PROJECTION,
            module_attr="linear_kv_up_proj",
            in_dim=dims.cfg("kv_lora_rank"),
            out_dim=dims.mla_kv_up_out_local,
            adapter_attr="lora_mla_kv_b_adapter",
        ),
        ProjectionBinding(
            projection=O_PROJECTION,
            module_attr="linear_proj",
            in_dim=dims.mla_o_in_local,
            out_dim=dims.hidden,
            adapter_attr="lora_o_adapter",
        ),
    ),
)


@dataclass(frozen=True)
class GQAAttentionSpec:
    """Fused MCore QKV, including the gated-query layout used by Qwen hybrids."""

    name: str = "gqa"
    supported_targets: frozenset[str] = GQA_TARGETS

    def normalize_targets(
        self,
        targets: frozenset[str],
        *,
        expanded_from_all_linear: bool,
    ) -> frozenset[str]:
        del expanded_from_all_linear
        return targets

    def validate(self, config, *, tp_size: int) -> None:
        num_query_groups = getattr(config, "num_query_groups", None)
        assert num_query_groups is None or num_query_groups >= tp_size, (
            "native LoRA (--megatron-to-hf-mode raw) does not support this architecture: "
            f"num_query_groups ({num_query_groups}) < tensor parallel size ({tp_size}), so mcore splits a "
            "single query group across ranks and the local qkv rows are not a per-group slice. "
            "Use --megatron-to-hf-mode bridge, or point --lora-provider-path at a model-specific provider."
        )

    def attach(self, attention: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        # Hybrid GDN/linear-attention layers (no linear_qkv) are reported by the orchestrator.
        return attach_layout(attention, GQA_LAYOUT, hf_prefix, context)


@dataclass(frozen=True)
class MLAAttentionSpec:
    """Compressed query and key/value projection layout used by DeepSeek/GLM/Kimi.

    Unsupported:

    - MLA without ``q_lora_rank``; SGLang expects the fused qkv_a layout.

    TODO:

    - Add a COLUMN ``linear_q_proj`` -> ``q_proj`` branch.
    """

    name: str = "mla"
    supported_targets: frozenset[str] = MLA_TARGETS

    def normalize_targets(
        self,
        targets: frozenset[str],
        *,
        expanded_from_all_linear: bool,
    ) -> frozenset[str]:
        """Drop generic Q/K/V names added by Miles' architecture-neutral all-linear expansion.

        MLA checkpoints with ``q_lora_rank`` have q_a/q_b and kv_a/kv_b
        projections instead. The argument parser records whether it expanded
        the ``all-linear`` shorthand, so explicit mixed requests retain exact
        semantics and fail validation rather than being silently rewritten.
        """
        if expanded_from_all_linear:
            return targets - _GENERIC_QKV_TARGETS
        return targets

    def validate(self, config, *, tp_size: int) -> None:
        del tp_size
        assert getattr(config, "q_lora_rank", None), (
            "native LoRA does not support multi-latent attention without q_lora_rank "
            "(DeepSeek-V2-Lite, Moonlight): the query path is uncompressed, so the adapter exports "
            "an unfused q_proj alongside kv_a_proj_with_mqa, and SGLang's loader expects the fused "
            "qkv_a layout. Use --megatron-to-hf-mode bridge, or point --lora-provider-path at a "
            "model-specific provider."
        )

    def attach(self, attention: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        return attach_layout(attention, MLA_LAYOUT, hf_prefix, context)


@dataclass(frozen=True)
class GDNAttentionSpec:
    """Explicit future boundary for GDN/linear-attention LoRA projections.

    TODO:

    - Split the fused ``in_proj`` four ways in ``codec/hf.py``.
    """

    name: str = "gdn"
    supported_targets: frozenset[str] = GDN_TARGETS

    def normalize_targets(
        self,
        targets: frozenset[str],
        *,
        expanded_from_all_linear: bool,
    ) -> frozenset[str]:
        del expanded_from_all_linear
        return targets

    def validate(self, config, *, tp_size: int) -> None:
        del config, tp_size
        raise AssertionError(
            "Miles-native GDN LoRA is not implemented yet; use --megatron-to-hf-mode bridge "
            "or point --lora-provider-path at a model-specific provider."
        )

    def attach(self, attention: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        del attention, hf_prefix
        if context.targets.intersection(self.supported_targets):
            self.validate(None, tp_size=context.tp_size)
        return 0


@dataclass(frozen=True)
class HybridGQAGDNAttentionSpec:
    """Per-layer dispatch for Qwen hybrids containing both GQA and GDN mixers."""

    name: str = "gqa_gdn"
    supported_targets: frozenset[str] = GQA_TARGETS

    def normalize_targets(
        self,
        targets: frozenset[str],
        *,
        expanded_from_all_linear: bool,
    ) -> frozenset[str]:
        del expanded_from_all_linear
        return targets

    def validate(self, config, *, tp_size: int) -> None:
        GQA_ATTENTION_SPEC.validate(config, tp_size=tp_size)

    def attach(self, attention: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        if hasattr(attention, "linear_qkv"):
            return GQA_ATTENTION_SPEC.attach(attention, hf_prefix, context)
        return GDN_ATTENTION_SPEC.attach(attention, hf_prefix, context)


def _is_replicated_linear(module: nn.Module, full_out: int) -> bool:
    if getattr(module, "parallel_mode", None) == "duplicated":
        return True
    return module.weight.shape[0] == full_out


GQA_ATTENTION_SPEC = GQAAttentionSpec()
MLA_ATTENTION_SPEC = MLAAttentionSpec()
GDN_ATTENTION_SPEC = GDNAttentionSpec()
HYBRID_GQA_GDN_ATTENTION_SPEC = HybridGQAGDNAttentionSpec()
