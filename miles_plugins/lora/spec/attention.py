"""Native-LoRA attention specs for fused GQA, gated GQA, MLA, and future GDN."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from miles_plugins.lora.modules.linear import LoRALinear, LoRASplitQKV, attach_adapter_forward
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
        """Attach fused-QKV and output-projection adapter modules."""
        if not hasattr(attention, "linear_qkv"):
            # Hybrid GDN/linear-attention layers are reported by the orchestrator.
            return 0

        count = 0
        num_q = attention.num_attention_heads_per_partition
        num_kv = attention.num_query_groups_per_partition
        head_dim = attention.hidden_size_per_attention_head

        qkv_projections = tuple(projection for projection in QKV_PROJECTIONS if projection.hf in context.targets)
        if qkv_projections:
            adapter = LoRASplitQKV(
                hf_prefix=hf_prefix,
                reference=attention.linear_qkv.weight,
                context=context,
                projections=qkv_projections,
                num_q=num_q,
                num_kv=num_kv,
                head_dim=head_dim,
            )
            attention.lora_qkv_adapter = adapter
            attach_adapter_forward(attention.linear_qkv, adapter, context.scale)
            count += 1

        if context.wants("o_proj"):
            in_local = num_q * head_dim
            adapter = LoRALinear(
                hf_prefix=hf_prefix,
                projection=O_PROJECTION,
                reference=attention.linear_proj.weight,
                context=context,
                in_features=in_local,
                out_features=context.hidden,
            )
            attention.lora_o_adapter = adapter
            attach_adapter_forward(attention.linear_proj, adapter, context.scale)
            count += 1
        return count


@dataclass(frozen=True)
class MLAAttentionSpec:
    """Compressed query and key/value projection layout used by DeepSeek/GLM/Kimi."""

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
        config = context.transformer_config
        count = 0
        heads_local = attention.num_attention_heads_per_partition
        q_head_dim = attention.q_head_dim
        v_head_dim = config.v_head_dim
        kv_lora_rank = config.kv_lora_rank
        kv_down_out = kv_lora_rank + config.qk_pos_emb_head_dim

        def add_replicated(module, projection: ProjectionSpec, adapter_name: str, full_out: int) -> int:
            assert _is_replicated_linear(module, full_out), (
                f"native MLA LoRA expects a replicated {projection.hf} (TELinear parallel_mode='duplicated'); "
                f"this build shards it ({tuple(module.weight.shape)} vs full out {full_out}). "
                "Use --lora-provider-path for this variant."
            )
            adapter = LoRALinear(
                hf_prefix=hf_prefix,
                projection=projection,
                reference=module.weight,
                context=context,
                in_features=context.hidden,
                out_features=full_out,
            )
            setattr(attention, adapter_name, adapter)
            attach_adapter_forward(module, adapter, context.scale)
            return 1

        def add_column_parallel(
            module,
            projection: ProjectionSpec,
            adapter_name: str,
            in_features: int,
            out_local: int,
        ) -> int:
            adapter = LoRALinear(
                hf_prefix=hf_prefix,
                projection=projection,
                reference=module.weight,
                context=context,
                in_features=in_features,
                out_features=out_local,
            )
            setattr(attention, adapter_name, adapter)
            attach_adapter_forward(module, adapter, context.scale)
            return 1

        if hasattr(attention, "linear_q_down_proj"):
            if context.wants("q_a_proj"):
                count += add_replicated(
                    attention.linear_q_down_proj,
                    MLA_Q_A_PROJECTION,
                    "lora_mla_q_a_adapter",
                    config.q_lora_rank,
                )
            if context.wants("q_b_proj"):
                count += add_column_parallel(
                    attention.linear_q_up_proj,
                    MLA_Q_B_PROJECTION,
                    "lora_mla_q_b_adapter",
                    config.q_lora_rank,
                    heads_local * q_head_dim,
                )
        if context.wants("kv_a_proj_with_mqa"):
            count += add_replicated(
                attention.linear_kv_down_proj,
                MLA_KV_A_PROJECTION,
                "lora_mla_kv_a_adapter",
                kv_down_out,
            )
        if context.wants("kv_b_proj"):
            count += add_column_parallel(
                attention.linear_kv_up_proj,
                MLA_KV_B_PROJECTION,
                "lora_mla_kv_b_adapter",
                kv_lora_rank,
                heads_local * (config.qk_head_dim + v_head_dim),
            )
        if context.wants("o_proj"):
            in_local = heads_local * v_head_dim
            adapter = LoRALinear(
                hf_prefix=hf_prefix,
                projection=O_PROJECTION,
                reference=attention.linear_proj.weight,
                context=context,
                in_features=in_local,
                out_features=context.hidden,
            )
            attention.lora_o_adapter = adapter
            attach_adapter_forward(attention.linear_proj, adapter, context.scale)
            count += 1
        return count


@dataclass(frozen=True)
class GDNAttentionSpec:
    """Explicit future boundary for GDN/linear-attention LoRA projections."""

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
