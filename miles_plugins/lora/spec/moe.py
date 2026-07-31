"""Native-LoRA routed/grouped-expert architecture boundary.

Two per-layer MoE cases exist today, split by whether the layer carries a
DeepSeek-style always-on shared expert:

- ``SharedOuterExpertMoESpec`` — layers with a shared expert: LoRA attaches to
  the shared expert's fused MLP (the outer, always-active expert). Routed
  expert projections remain out of native scope.
- ``GeneralExpertMoESpec`` — layers with only routed/grouped experts: native
  LoRA has no MLP module to adapt, so parser-expanded ``all-linear`` targets
  skip and explicit MLP targets fail closed.

Roadmap: what Megatron-Bridge PEFT already does here and native does not
-----------------------------------------------------------------------
Bridge's structural advantage is that its adapters *are* real MCore
Column/RowParallelLinear submodules inside the wrapped module tree
(``megatron/bridge/peft/utils.py:983-1032``), so the expert-TP group, DDP
bucketing and distributed-optimizer sharding come for free. Native's
sibling-attach design trades that away to keep base parameter names stable, so
each expert layout below needs explicit work. Ordered by value; every item is
reachable today via ``--megatron-to-hf-mode bridge``.

1. **EP-shared routed-expert LoRA** (bridge's default, and what miles actually
   runs in bridge mode): one ``ParallelLinearAdapter(is_expert=True)`` shared by
   all of an EP rank's local experts, built on the ``expt_tp`` group, with a
   backward hook SUM-all-reducing adapter grads across EP before MCore's
   expert-DDP reduces over ``expt_dp``
   (``peft/utils.py:966-1032``, ``:1242-1266``; ``peft/lora.py:171-237``).
   Native seams to change: give ``MoELoRASpec`` an ``attach()`` (spec/base.py),
   call it from the layer loop (``lora.py``), extend ``AttachContext`` with
   ep/etp rank+size and ``num_local_experts``, implement the adapter in the
   empty ``modules/moe.py``, and tag its grads ``_lora_grad_sum_group="ep"`` —
   ``distributed.py``'s "ep" reduce branch is already wired and unused.
2. **Shared-outer routed-expert LoRA** (``--experts-shared-outer-loras``,
   SGLang PR #21466): fc1 = shared ``lora_A`` + per-expert ``lora_B``, fc2
   mirrored, per-expert side packed as one 3D ``[N_local, out, in]`` parameter
   driven by ``torch._grouped_mm``, shared side kept bit-identical across EP by
   broadcast + SUM all-reduce (``peft/utils.py:2231-2517``). Needs everything in
   (1) plus a mixed-arity adapter (2D shared side + 3D per-expert side), a
   cross-EP replication primitive, and an export that emits the shared side once
   under an expert-index-stripped HF name. Must stay byte-compatible with
   SGLang's ``expert_dim=1`` buffers, which miles already asserts on.
3. **SequentialMLP per-expert LoRA** (``mlp.experts.local_experts.N.linear_fc*``)
   is the cheapest routed layout to port: each is a plain Column/RowParallelLinear,
   so today's 2D ``LoRALinear``/``LoRASplitFC1`` work unchanged and only the
   expert-TP group and per-expert HF naming are new (``peft/utils.py:645-651``,
   ``:1455-1480``).
4. **Router LoRA** on the ``TopKRouter`` gating logits
   (``peft/lora_layers.py:96-109``). This one cannot use
   ``attach_adapter_forward``: the router's forward returns
   ``(probs, routing_map)`` and interposes jitter / float32 expert bias /
   load balancing between gating and routing, so the delta must be injected
   mid-forward. Note SGLang has no router-LoRA buffer type, so it would train
   but not serve.

Related roadmap notes live next to the code that would change: the expert-TP
group in ``miles_plugins/lora/distributed.py``, the expert-axis export in
``miles_plugins/lora/codec/hf.py``, and GDN / uncompressed-MLA attention in
``miles_plugins/lora/spec/attention.py``.
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
