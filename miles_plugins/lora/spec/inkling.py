"""Native-LoRA architecture spec for the Inkling TITO family.

Inkling diverges from plain mcore in every block, so this spec pairs the
declarative layout for what fits the tables (the 4-way fused q/k/v/r split and
the fused dense MLP) with method-level attachment for what does not (grouped
routed experts, per-sub-expert shared experts, the lm_head), all built on the
plugin's shared modules and IO.

TML export naming: ``language_model.layers.{i}.attn.{wq_du,wk_dv,wv_dv,wr_du,
wo_ud}``, ``...mlp.{gate_up_proj,down_proj}`` on dense layers,
``...mlp.experts.{w1,w2,w3}`` / ``...mlp.shared_experts.{w1,w2,w3}`` on MoE
layers, and ``language_model.lm_head``. SGLang auto-detects these names, so
serving passes ``lora_target_modules=["all"]``.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from miles_plugins.lora.modules.linear import LoRALinear, LoRASplitAdapter, attach_adapter_forward
from miles_plugins.lora.modules.moe import LoRAGroupedFC1, LoRAGroupedFC2, LoRAOutputHead, LoRASharedExpertsAdapter
from miles_plugins.lora.spec.base import AttachContext, AttentionFamily, ProjectionSpec, ShardLayout
from miles_plugins.lora.spec.layout import AttentionSpecBase, FusedAttach, LayoutSpec, ModuleLayout, ProjectionBinding

MODEL_TYPES = ("inkling_mm_model",)

_NO_TARGETS: frozenset[str] = frozenset()


def _hidden(_module: nn.Module, context: AttachContext) -> int:
    return context.hidden


def _o_in_local(attention: nn.Module, _context: AttachContext) -> int:
    return attention.nh_l * attention.hd


class _InklingSplitQKVR(LoRASplitAdapter):
    """Four independent adapters over Inkling's plain-concat fused [q;k;v;r]."""

    _group_name = "qkvr"


def _build_split_qkvr(
    attention: nn.Module,
    hf_prefix: str,
    context: AttachContext,
    active: tuple[ProjectionSpec, ...],
    members: tuple[ProjectionSpec, ...],
) -> _InklingSplitQKVR:
    return _InklingSplitQKVR(
        hf_prefix=hf_prefix,
        reference=attention.linear_qkv.weight,
        context=context,
        projections=active,
        member_projections=members,
        rows={
            "q": attention.nh_l * attention.hd,
            "k": attention.nkv_l * attention.hd,
            "v": attention.nkv_l * attention.hd,
            "r": attention.nh_l * attention.d_rel,
        },
    )


class InklingAttentionSpec(AttentionSpecBase):
    """GQA-with-relative-projection: fused [q;k;v;r], plain concat (no group permute)."""

    name = "inkling"
    family = AttentionFamily.GQA
    layout = ModuleLayout(
        name="inkling_attention",
        present_when_attr="linear_qkv",
        hf_block_prefix="attn.",
        fused=(
            FusedAttach(
                module_attr="linear_qkv",
                projections=(
                    ProjectionSpec("wq_du", "q", ShardLayout.COLUMN),
                    ProjectionSpec("wk_dv", "k", ShardLayout.COLUMN),
                    ProjectionSpec("wv_dv", "v", ShardLayout.COLUMN),
                    ProjectionSpec("wr_du", "r", ShardLayout.COLUMN),
                ),
                adapter_attr="lora_qkv_adapter",
                build=_build_split_qkvr,
            ),
        ),
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("wo_ud", "o", ShardLayout.ROW),
                module_attr="linear_proj",
                in_dim=_o_in_local,
                out_dim=_hidden,
                adapter_attr="lora_o_adapter",
            ),
        ),
    )

    def normalize_targets(
        self,
        targets: frozenset[str],
        *,
        expanded_from_all_linear: bool,
    ) -> frozenset[str]:
        """Every request maps to the full native set.

        Inkling's TML projection names are disjoint from the HF families the
        parser knows, so a passthrough would fail validation on all-linear and
        on plain q_proj-style requests alike. LoRA covers every projection,
        matching the model-specific provider this spec replaces.
        """
        del targets, expanded_from_all_linear
        return self.layout.targets | InklingDenseMLPSpec.layout.targets


class _InklingFusedFC1(LoRALinear):
    """Dense fused gate/up as ONE projection: local B stacks [gate_loc; up_loc].

    A plain dim-0 TP gather of B would interleave the ranks' halves, so export
    gathers each half separately and load slices them back per rank.
    """

    def export_plan(self, gather) -> list:
        b = getattr(self, f"{self.attr}_B")
        i_loc = b.shape[0] // 2
        gate = gather.request(b[:i_loc], 0)
        up = gather.request(b[i_loc:], 0)
        return [
            (f"{self.hf_prefix}gate_up_proj.lora_A.weight", getattr(self, f"{self.attr}_A")),
            (f"{self.hf_prefix}gate_up_proj.lora_B.weight", lambda: torch.cat([gate(), up()], dim=0)),
        ]

    def load_plan_custom(self, take) -> list:
        a = getattr(self, f"{self.attr}_A")
        b = getattr(self, f"{self.attr}_B")
        i_loc = b.shape[0] // 2
        full_b = take(f"{self.hf_prefix}gate_up_proj.lora_B.weight")
        full_i = full_b.shape[0] // 2
        lo = self.tp_rank * i_loc
        return [
            (a, take(f"{self.hf_prefix}gate_up_proj.lora_A.weight")),
            (b[:i_loc], full_b[lo : lo + i_loc]),
            (b[i_loc:], full_b[full_i + lo : full_i + lo + i_loc]),
        ]


def _fc1_out_local(mlp: nn.Module, _context: AttachContext) -> int:
    return mlp.linear_fc1.weight.shape[0]


def _fc2_in_local(mlp: nn.Module, _context: AttachContext) -> int:
    return mlp.linear_fc1.weight.shape[0] // 2


class InklingDenseMLPSpec(LayoutSpec):
    """Fused gate/up as a single TML-named projection, plus the down projection."""

    name = "inkling_dense_mlp"
    layout = ModuleLayout(
        name="inkling_dense_mlp",
        present_when_attr="linear_fc1",
        hf_block_prefix="mlp.",
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("gate_up_proj", "fc1", ShardLayout.COLUMN),
                module_attr="linear_fc1",
                in_dim=_hidden,
                out_dim=_fc1_out_local,
                adapter_attr="lora_fc1_adapter",
                adapter_class=_InklingFusedFC1,
            ),
            ProjectionBinding(
                projection=ProjectionSpec("down_proj", "fc2", ShardLayout.ROW),
                module_attr="linear_fc2",
                in_dim=_fc2_in_local,
                out_dim=_hidden,
                adapter_attr="lora_fc2_adapter",
            ),
        ),
    )


class InklingMoESpec:
    """Routed grouped experts (shared-A / per-expert-B) plus shared sub-experts."""

    supported_targets: frozenset[str] = frozenset()

    def validate_layer(self, mlp: nn.Module, context: AttachContext) -> frozenset[str]:
        del mlp, context
        return _NO_TARGETS

    def attach(self, mlp: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        from megatron.core import parallel_state

        if not hasattr(mlp, "experts"):
            return 0
        config = mlp.config
        assert (getattr(config, "expert_tensor_parallel_size", 1) or 1) == 1, "Inkling LoRA assumes ETP=1"
        experts = mlp.experts
        is_ep = parallel_state.get_expert_model_parallel_world_size() > 1
        count = 0

        fc1_adapter = LoRAGroupedFC1(
            hf_prefix=hf_prefix + "mlp.experts.",
            reference=experts.linear_fc1.weight0,
            context=context,
            num_local_experts=experts.num_local_experts,
            moe_intermediate=config.moe_ffn_hidden_size,
            is_ep=is_ep,
        )
        experts.lora_fc1_adapter = fc1_adapter
        attach_adapter_forward(experts.linear_fc1, fc1_adapter, context.scale)
        count += 1

        fc2_adapter = LoRAGroupedFC2(
            hf_prefix=hf_prefix + "mlp.experts.",
            reference=experts.linear_fc2.weight0,
            context=context,
            num_local_experts=experts.num_local_experts,
            moe_intermediate=config.moe_ffn_hidden_size,
            is_ep=is_ep,
        )
        experts.lora_fc2_adapter = fc2_adapter
        attach_adapter_forward(experts.linear_fc2, fc2_adapter, context.scale)
        count += 1

        shared = getattr(mlp, "shared_experts", None)
        if shared is not None:
            count += self._attach_shared(shared, hf_prefix, context)
        return count

    @staticmethod
    def _attach_shared(shared: nn.Module, hf_prefix: str, context: AttachContext) -> int:
        subs = list(shared.experts)
        local_intermediate = shared.experts[0].linear_fc1.weight.shape[0] // 2
        adapter = LoRASharedExpertsAdapter(
            hf_prefix=hf_prefix + "mlp.shared_experts.",
            fc1_reference=subs[0].linear_fc1.weight,
            fc2_reference=subs[0].linear_fc2.weight,
            context=context,
            num_shared=len(subs),
            local_intermediate=local_intermediate,
        )
        shared.lora_adapter = adapter

        for index, sub in enumerate(subs):
            for host_attr, delta in (("linear_fc1", adapter.fc1_delta), ("linear_fc2", adapter.fc2_delta)):
                host = getattr(sub, host_attr)
                original = host.forward

                def forward(x, *args, _original=original, _host=host, _delta=delta, _index=index, **kwargs):
                    out, bias = _original(x, *args, **kwargs)
                    return torch.add(out, _delta(x, _host, _index), alpha=context.scale), bias

                host.forward = forward
        return 1


class InklingExtrasSpec:
    """Model-level adapters: the lm_head projection (muP-scaled, pad-trimmed)."""

    def attach(self, model: nn.Module, args, context: AttachContext) -> int:
        from megatron.core import parallel_state

        if not getattr(model, "post_process", False) or getattr(model, "output_layer", None) is None:
            return 0
        output_layer = model.output_layer
        mup = getattr(model.config.inkling, "logits_mup_width_multiplier", None)
        adapter = LoRAOutputHead(
            hf_prefix="language_model.lm_head.",
            reference=output_layer.weight,
            context=context,
            vocab_local=output_layer.weight.shape[0],
            mup_width_multiplier=float(mup) if mup else None,
            unpadded_vocab_size=_unpadded_vocab_size(getattr(args, "hf_checkpoint", None)),
        )
        del parallel_state
        model.lora_lm_head_adapter = adapter
        attach_adapter_forward(output_layer, adapter, context.scale)
        return 1


def _unpadded_vocab_size(hf_checkpoint: str | None) -> int | None:
    """True (unpadded) vocab size from the HF config, or None if absent."""
    if not hf_checkpoint:
        return None
    try:
        with open(os.path.join(hf_checkpoint, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        return (config.get("text_config") or config).get("unpadded_vocab_size")
    except Exception:
        return None


__all__ = [
    "MODEL_TYPES",
    "InklingAttentionSpec",
    "InklingDenseMLPSpec",
    "InklingExtrasSpec",
    "InklingMoESpec",
]
