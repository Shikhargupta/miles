"""Native (raw-mode) LoRA for standard Megatron-core GPT models.

Attaches LoRA adapters directly to the mcore model built by miles' own model
provider (``--megatron-to-hf-mode raw``), without going through Megatron-Bridge.
Adapters are registered per HF projection, so the HF/PEFT export and the SGLang
adapter sync are plain name joins.

Covered modules, selected by ``--target-modules`` (HF leaf names):

===========================  ==========================================  ===============
target                       megatron module                             kind
===========================  ==========================================  ===============
q_proj / k_proj / v_proj     ``self_attention.linear_qkv`` (fused)       column-parallel
o_proj                       ``self_attention.linear_proj``              row-parallel
gate_proj / up_proj          ``mlp.linear_fc1`` (fused ``[gate; up]``)   column-parallel
down_proj                    ``mlp.linear_fc2``                          row-parallel
===========================  ==========================================  ===============

Multi-latent attention (DeepSeek / GLM / Kimi) is covered too, with its own
projection set -- see ``_attach_mla_attention``:

===========================  ==========================================  ===============
target                       megatron module                             kind
===========================  ==========================================  ===============
q_a_proj                     ``self_attention.linear_q_down_proj``       replicated
q_b_proj                     ``self_attention.linear_q_up_proj``         column-parallel
q_proj (no q_lora_rank)      ``self_attention.linear_q_proj``            column-parallel
kv_a_proj_with_mqa           ``self_attention.linear_kv_down_proj``      replicated
kv_b_proj                    ``self_attention.linear_kv_up_proj``        column-parallel
===========================  ==========================================  ===============

Adapters export under the HF names the checkpoint itself uses: the decoder-layer
prefix and the shared-expert segment are read off its weight index (see
``_hf_naming``), because Qwen3.5 nests the decoder under
``model.language_model.layers.`` and DeepSeek / GLM / Kimi spell the shared expert
``mlp.shared_experts.``.

``mlp.shared_experts`` (a plain MLP) follows the same MLP targets. Routed MoE
experts are deliberately out of scope here: their adapters need a serving-side
layout contract of their own, so a MoE model supplies them through its own
provider (see ``--lora-provider-path``).

Parallelism contract, mirroring the module each adapter wraps:

* column-parallel: ``A`` is replicated and each rank computes a partial product,
  so its grads are summed over TP (tagged ``_lora_grad_sum_group``); ``B`` is
  row-sharded to this rank's output slice.
* row-parallel: ``A`` is column-sharded to this rank's input slice and the
  partial products are TP-reduced (reduce-scatter under sequence parallelism);
  ``B`` is replicated, and its grads are TP-summed when sequence parallel.
* replicated (MLA down-projections): both ``A`` and ``B`` are replicated, so their
  grads only diverge per rank -- and therefore only need summing -- under sequence
  parallelism, where each rank feeds a different sequence shard.

Models whose modules diverge from plain mcore plug in their own implementation
via ``--lora-provider-path``. A provider module must expose the three entry
points this module does: ``wrap_model_provider_with_lora``,
``load_lora_adapter_hf`` and ``export_lora_hf_named``.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_hf

logger = logging.getLogger(__name__)

_QKV, _O, _FC1, _FC2 = "qkv", "o", "fc1", "fc2"
_MLA_Q_A, _MLA_Q_B, _MLA_KV_A, _MLA_KV_B, _MLA_Q = "mla_q_a", "mla_q_b", "mla_kv_a", "mla_kv_b", "mla_q"

COLUMN, ROW, REPLICATED = "column", "row", "replicated"


@dataclass(frozen=True)
class _Proj:
    """One HF projection carried by an adapter kind.

    ``attr`` names the parameter pair on the adapter (``<attr>_A`` / ``<attr>_B``).
    ``layout`` fixes both the export gather and the load slice, which is why export
    and load can share this table instead of restating the contract separately:

    ==========  ===================  ==============================
    layout      A                    B
    ==========  ===================  ==============================
    column      replicated           sharded over rows (dim 0)
    row         sharded over columns replicated
    replicated  replicated           replicated
    ==========  ===================  ==============================

    ``width`` returns this rank's extent along the sharded axis, read off the
    adapter's ``shard_meta``; it is unused for ``replicated``.
    """

    hf: str
    attr: str
    layout: str
    width: object = None


_ADAPTER_LAYOUT: dict[str, tuple[_Proj, ...]] = {
    _QKV: (
        _Proj("q_proj", "q", COLUMN, lambda m: m["q_rows"]),
        _Proj("k_proj", "k", COLUMN, lambda m: m["num_kv"] * m["head_dim"]),
        _Proj("v_proj", "v", COLUMN, lambda m: m["num_kv"] * m["head_dim"]),
    ),
    _O: (_Proj("o_proj", "o", ROW, lambda m: m["in_local"]),),
    _FC1: (
        _Proj("gate_proj", "gate", COLUMN, lambda m: m["inter_local"]),
        _Proj("up_proj", "up", COLUMN, lambda m: m["inter_local"]),
    ),
    _FC2: (_Proj("down_proj", "down", ROW, lambda m: m["in_local"]),),
    _MLA_Q_A: (_Proj("q_a_proj", "a", REPLICATED),),
    _MLA_KV_A: (_Proj("kv_a_proj_with_mqa", "a", REPLICATED),),
    _MLA_Q_B: (_Proj("q_b_proj", "b", COLUMN, lambda m: m["out_local"]),),
    _MLA_KV_B: (_Proj("kv_b_proj", "b", COLUMN, lambda m: m["out_local"]),),
    _MLA_Q: (_Proj("q_proj", "b", COLUMN, lambda m: m["out_local"]),),
}

SUPPORTED_TARGETS = frozenset(proj.hf for projs in _ADAPTER_LAYOUT.values() for proj in projs)


_DEFAULT_LAYER_PREFIX = "model.layers."
_DEFAULT_SHARED_EXPERT = "mlp.shared_expert."


def _hf_naming(hf_checkpoint: str | None) -> tuple[str, str]:
    """Read the decoder-layer prefix and shared-expert segment off the checkpoint itself.

    Both vary by family and both have to match exactly, because the exported adapter
    names are what SGLang looks up against its own module paths: Qwen3.5 nests the
    decoder under ``model.language_model.layers.`` and spells the shared expert
    ``mlp.shared_expert.``, while DeepSeek / GLM / Kimi use ``model.layers.`` and
    ``mlp.shared_experts.``. Reading the weight index avoids a per-model table.
    """
    index_path = os.path.join(hf_checkpoint or "", "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return _DEFAULT_LAYER_PREFIX, _DEFAULT_SHARED_EXPERT
    with open(index_path) as handle:
        names = json.load(handle).get("weight_map", {})

    prefixes: collections.Counter[str] = collections.Counter()
    for name in names:
        if name.startswith("mtp.") or "vision" in name:
            continue
        match = re.match(r"^((?:[\w.]+\.)?layers\.)\d+\.", name)
        if match:
            prefixes[match.group(1)] += 1
    layer_prefix = prefixes.most_common(1)[0][0] if prefixes else _DEFAULT_LAYER_PREFIX
    shared = "mlp.shared_experts." if any(".mlp.shared_experts." in n for n in names) else _DEFAULT_SHARED_EXPERT
    return layer_prefix, shared


class NativeLoRAAdapter(nn.Module):
    """Holds one module's adapter params. Invisible to the dist-checkpoint.

    ``hf_prefix`` is the HF module path the params export under (e.g.
    ``model.layers.3.self_attn.``); ``shard_meta`` records what this rank owns,
    so the loader can slice a full HF adapter down to it.
    """

    def __init__(self, kind: str, hf_prefix: str, **shard_meta):
        super().__init__()
        self.kind = kind
        self.hf_prefix = hf_prefix
        self.shard_meta = shard_meta

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return {}


@dataclass(frozen=True)
class _Spec:
    """Everything the wrappers need that is constant across a model chunk."""

    rank: int
    scale: float
    dropout: float
    a_init: str
    eps: float
    hidden: int
    sequence_parallel: bool
    zero_centered_gamma: bool
    tp_size: int
    targets: frozenset[str]
    output_gate: bool
    layer_prefix: str
    shared_expert: str

    @classmethod
    def from_args(cls, args, config) -> _Spec:
        from megatron.core import parallel_state as ps

        rank = int(args.lora_rank)
        assert rank > 0, "native LoRA requires --lora-rank > 0"
        targets = frozenset(convert_target_modules_to_hf(list(args.target_modules or ())))
        unsupported = sorted(targets - SUPPORTED_TARGETS)
        assert not unsupported, (
            f"native LoRA (--megatron-to-hf-mode raw) does not implement {unsupported}. "
            f"Supported targets are {sorted(SUPPORTED_TARGETS)}; Megatron-style names are accepted "
            "and normalised. Use --megatron-to-hf-mode bridge, or point --lora-provider-path at a "
            "model-specific provider."
        )
        layer_prefix, shared_expert = _hf_naming(getattr(args, "hf_checkpoint", None))
        return cls(
            rank=rank,
            scale=float(args.lora_alpha) / rank,
            dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
            a_init=getattr(args, "lora_A_init_method", "xavier") or "xavier",
            eps=config.layernorm_epsilon,
            hidden=config.hidden_size,
            sequence_parallel=bool(config.sequence_parallel),
            zero_centered_gamma=bool(getattr(config, "layernorm_zero_centered_gamma", False)),
            tp_size=ps.get_tensor_model_parallel_world_size(),
            targets=targets,
            output_gate=bool(getattr(config, "attention_output_gate", False)),
            layer_prefix=layer_prefix,
            shared_expert=shared_expert,
        )

    def wants(self, *names: str) -> bool:
        return bool(self.targets.intersection(names))


def _new_param(reference: torch.Tensor, shape, *, init: str, grad_sum_group: str | None = None) -> nn.Parameter:
    """Adapter param matching ``reference``'s dtype/device.

    ``B`` matrices are zero-init so a fresh adapter is an exact no-op. Params are
    marked non-tensor-parallel: the sharding is expressed by their shape here, not
    by megatron's partitioning. ``grad_sum_group`` tags a replicated param whose
    grads must be summed over that group (see ``reduce_marked_lora_grads``).
    """
    t = torch.empty(*shape, dtype=reference.dtype, device=reference.device)
    if init == "zero":
        t.zero_()
    else:
        nn.init.xavier_uniform_(t)
    param = nn.Parameter(t)
    param.tensor_model_parallel = False
    param.partition_dim = -1
    param.partition_stride = 1
    if grad_sum_group is not None:
        param._lora_grad_sum_group = grad_sum_group
    return param


def _rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float, zero_centered_gamma: bool = False) -> torch.Tensor:
    """Recompute the RMSNorm fused into TELayerNormColumnParallelLinear (fp32 internals).

    Under ``--apply-layernorm-1p`` (Qwen3.5 / Qwen3-Next) the stored weight is
    ``gamma - 1``, so the branch has to add the 1 back or it sees a differently
    scaled input than the base GEMM does.
    """
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    weight = gamma.float() + 1.0 if zero_centered_gamma else gamma.float()
    return (normed * weight).to(x.dtype)


def _branch_input(x: torch.Tensor, module: nn.Module, spec: _Spec) -> torch.Tensor:
    """Input to a column-parallel adapter branch.

    The wrapped module may fuse its layernorm (``layer_norm_weight``), in which
    case the branch has to recompute it to see the same input the base GEMM does.
    Under sequence parallelism the module's input is sequence-sharded, so gather
    it back to the full sequence the adapter's replicated ``A`` expects.
    """
    gamma = getattr(module, "layer_norm_weight", None)
    if gamma is not None:
        x = _rmsnorm(x, gamma, spec.eps, spec.zero_centered_gamma)
    if spec.sequence_parallel:
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        x = gather_from_sequence_parallel_region(x)
    return _dropout(x, spec, module.training)


def _dropout(x: torch.Tensor, spec: _Spec, training: bool) -> torch.Tensor:
    if spec.dropout and training:
        return F.dropout(x, p=spec.dropout, training=True)
    return x


def _reduce_row_parallel(partial: torch.Tensor, spec: _Spec) -> torch.Tensor:
    """Complete a row-parallel adapter branch: each rank holds a partial sum."""
    if spec.tp_size <= 1:
        return partial
    from megatron.core.tensor_parallel.mappings import (
        reduce_from_tensor_model_parallel_region,
        reduce_scatter_to_sequence_parallel_region,
    )

    if spec.sequence_parallel:
        return reduce_scatter_to_sequence_parallel_region(partial)
    return reduce_from_tensor_model_parallel_region(partial)


def _build_qkv_perm(
    num_q_heads: int, num_groups: int, head_dim: int, device, output_gate: bool = False
) -> torch.Tensor:
    """Row permutation from plain ``[q; k; v]`` order into mcore's fused qkv layout.

    mcore emits qkv grouped per query group -- ``q1 q2 k1 v1 | q3 q4 k2 v2 | ...``
    (see ``SelfAttention.get_query_key_value_tensors``) -- while the adapter keeps
    one ``B`` per projection. Permuting the assembled delta is cheaper and easier
    to reason about than storing ``B`` interleaved.

    With ``attention_output_gate`` (Qwen3.5 / Qwen3-Next) the query side carries a
    second slice per head, and the two orders differ in more than the grouping: mcore
    holds a group as ``[q heads][gate heads]`` while HF's ``q_proj`` interleaves them
    per head as ``[q h0][gate h0][q h1][gate h1] ...``. The same permutation expresses
    both, so the adapter still keeps one plain ``B`` per HF projection.
    """
    q_per_group = num_q_heads // num_groups
    q_slices = 2 if output_gate else 1
    k_base = num_q_heads * q_slices * head_dim
    v_base = k_base + num_groups * head_dim
    index: list[int] = []
    for g in range(num_groups):
        for slice_idx in range(q_slices):
            for head in range(q_per_group):
                start = ((g * q_per_group + head) * q_slices + slice_idx) * head_dim
                index.extend(range(start, start + head_dim))
        index.extend(range(k_base + g * head_dim, k_base + (g + 1) * head_dim))
        index.extend(range(v_base + g * head_dim, v_base + (g + 1) * head_dim))
    return torch.tensor(index, dtype=torch.long, device=device)


def _wrap_forward(module: nn.Module, delta_fn, scale: float) -> None:
    """Add ``scale * delta_fn(x)`` to ``module``'s output, keeping its (out, bias) contract."""
    original = module.forward

    def forward(x, *args, **kwargs):
        out, bias = original(x, *args, **kwargs)
        return torch.add(out, delta_fn(x), alpha=scale), bias

    module.forward = forward


def _attach_row_parallel(
    module: nn.Module,
    owner: nn.Module,
    kind: str,
    hf_prefix: str,
    spec: _Spec,
    *,
    in_local: int,
    tp_rank: int,
    attr: str,
) -> int:
    """Adapter on a row-parallel linear: ``A`` col-sharded, ``B`` replicated.

    Shared by ``o_proj`` on both attention flavours and by the MLP's ``down_proj``:
    the three differ only in this rank's input width and the parameter names.
    ``B`` is replicated, so its gradient only diverges per rank -- and therefore only
    needs TP summing -- under sequence parallelism.
    """
    adapter = NativeLoRAAdapter(kind, hf_prefix, in_local=in_local, tp_rank=tp_rank)
    reference = module.weight
    adapter.register_parameter(f"{attr}_A", _new_param(reference, (spec.rank, in_local), init=spec.a_init))
    adapter.register_parameter(
        f"{attr}_B",
        _new_param(
            reference,
            (spec.hidden, spec.rank),
            init="zero",
            grad_sum_group="tp" if spec.sequence_parallel else None,
        ),
    )
    setattr(owner, f"lora_{kind}_adapter", adapter)

    def delta(x, _m=module, _ad=adapter):
        partial = F.linear(_dropout(x, spec, _m.training), getattr(_ad, f"{attr}_A"))
        return F.linear(_reduce_row_parallel(partial, spec), getattr(_ad, f"{attr}_B"))

    _wrap_forward(module, delta, spec.scale)
    return 1


def _attach_attention(attn: nn.Module, hf_prefix: str, spec: _Spec) -> int:
    """Adapters on the fused qkv (column-parallel) and the output proj (row-parallel)."""
    from megatron.core import parallel_state as ps

    n = 0
    tp_rank = ps.get_tensor_model_parallel_rank()
    num_q = attn.num_attention_heads_per_partition
    num_kv = attn.num_query_groups_per_partition
    head_dim = attn.hidden_size_per_attention_head

    if spec.wants("q_proj", "k_proj", "v_proj"):
        qkv = attn.linear_qkv
        q_rows = num_q * head_dim * (2 if spec.output_gate else 1)
        ad = NativeLoRAAdapter(
            _QKV, hf_prefix, num_q=num_q, num_kv=num_kv, head_dim=head_dim, q_rows=q_rows, tp_rank=tp_rank
        )
        ref = qkv.weight
        for name, rows in (("q", q_rows), ("k", num_kv * head_dim), ("v", num_kv * head_dim)):
            ad.register_parameter(
                f"{name}_A", _new_param(ref, (spec.rank, spec.hidden), init=spec.a_init, grad_sum_group="tp")
            )
            ad.register_parameter(f"{name}_B", _new_param(ref, (rows, spec.rank), init="zero"))
        ad.register_buffer(
            "out_perm", _build_qkv_perm(num_q, num_kv, head_dim, ref.device, spec.output_gate), persistent=False
        )
        attn.lora_qkv_adapter = ad

        def qkv_delta(x, _m=qkv, _ad=ad):
            xn = _branch_input(x, _m, spec)
            r = spec.rank
            s = F.linear(xn, torch.cat([_ad.q_A, _ad.k_A, _ad.v_A], 0))
            delta = torch.cat(
                [
                    F.linear(s[..., 0:r], _ad.q_B),
                    F.linear(s[..., r : 2 * r], _ad.k_B),
                    F.linear(s[..., 2 * r : 3 * r], _ad.v_B),
                ],
                dim=-1,
            )
            return delta.index_select(-1, _ad.out_perm)

        _wrap_forward(qkv, qkv_delta, spec.scale)
        n += 1

    if spec.wants("o_proj"):
        n += _attach_row_parallel(
            attn.linear_proj, attn, _O, hf_prefix, spec, in_local=num_q * head_dim, tp_rank=tp_rank, attr="o"
        )
    return n


def _is_replicated_linear(module: nn.Module, full_out: int) -> bool:
    """True when this linear holds the whole output dim (TELinear parallel_mode='duplicated')."""
    if getattr(module, "parallel_mode", None) == "duplicated":
        return True
    return module.weight.shape[0] == full_out


def _attach_mla_attention(attn: nn.Module, hf_prefix: str, spec: _Spec, config) -> int:
    """Adapters on multi-latent attention (DeepSeek / GLM / Kimi style).

    MLA has no fused qkv. The query and key/value paths each compress to a latent and
    then expand, so the adapter surface is four projections plus the output one:

    ==========================  ==============================  =================
    target                      megatron module                 sharding
    ==========================  ==============================  =================
    q_a_proj                    ``linear_q_down_proj``          replicated
    q_b_proj                    ``linear_q_up_proj``            column-parallel
    kv_a_proj_with_mqa          ``linear_kv_down_proj``         replicated
    kv_b_proj                   ``linear_kv_up_proj``           column-parallel
    o_proj                      ``linear_proj``                 row-parallel
    ==========================  ==============================  =================

    When ``q_lora_rank`` is unset there is no compression on the query path and
    ``linear_q_proj`` (column-parallel, straight from hidden) takes ``q_proj`` instead.

    The latent layernorms (``q_layernorm`` / ``kv_layernorm``) are separate modules
    applied *before* the up-projections, so an up-projection's adapter sees an
    already-normed input and does not recompute anything -- unlike the fused-layernorm
    linears on the GQA path.
    """
    from megatron.core import parallel_state as ps

    n = 0
    tp_rank = ps.get_tensor_model_parallel_rank()
    heads_local = attn.num_attention_heads_per_partition
    q_head_dim = attn.q_head_dim
    v_head_dim = config.v_head_dim
    kv_lora_rank = config.kv_lora_rank
    kv_down_out = kv_lora_rank + config.qk_pos_emb_head_dim

    def add_replicated(module, kind, hf_name, full_out):
        """Down-projection: weight, A and B are all replicated across TP."""
        assert _is_replicated_linear(module, full_out), (
            f"native MLA LoRA expects a replicated {hf_name} (TELinear parallel_mode='duplicated'); "
            f"this build shards it ({tuple(module.weight.shape)} vs full out {full_out}). "
            "Use --lora-provider-path for this variant."
        )
        tag = "tp" if spec.sequence_parallel else None
        ad = NativeLoRAAdapter(kind, hf_prefix, tp_rank=tp_rank)
        ad.register_parameter(
            "a_A", _new_param(module.weight, (spec.rank, spec.hidden), init=spec.a_init, grad_sum_group=tag)
        )
        ad.register_parameter("a_B", _new_param(module.weight, (full_out, spec.rank), init="zero", grad_sum_group=tag))
        setattr(attn, f"lora_{kind}_adapter", ad)

        def delta(x, _m=module, _ad=ad):
            return F.linear(F.linear(_dropout(x, spec, _m.training), _ad.a_A), _ad.a_B)

        _wrap_forward(module, delta, spec.scale)
        return 1

    def add_column_parallel(module, kind, in_dim, out_local):
        """Up-projection (or plain q_proj): A replicated, B sharded to this rank's heads."""
        ad = NativeLoRAAdapter(kind, hf_prefix, out_local=out_local, tp_rank=tp_rank)
        ad.register_parameter(
            "b_A", _new_param(module.weight, (spec.rank, in_dim), init=spec.a_init, grad_sum_group="tp")
        )
        ad.register_parameter("b_B", _new_param(module.weight, (out_local, spec.rank), init="zero"))
        setattr(attn, f"lora_{kind}_adapter", ad)

        def delta(x, _m=module, _ad=ad):
            return F.linear(F.linear(_branch_input(x, _m, spec), _ad.b_A), _ad.b_B)

        _wrap_forward(module, delta, spec.scale)
        return 1

    if hasattr(attn, "linear_q_down_proj"):
        if spec.wants("q_a_proj"):
            n += add_replicated(attn.linear_q_down_proj, _MLA_Q_A, "q_a_proj", config.q_lora_rank)
        if spec.wants("q_b_proj"):
            n += add_column_parallel(attn.linear_q_up_proj, _MLA_Q_B, config.q_lora_rank, heads_local * q_head_dim)
    elif spec.wants("q_proj") and hasattr(attn, "linear_q_proj"):
        n += add_column_parallel(attn.linear_q_proj, _MLA_Q, spec.hidden, heads_local * q_head_dim)

    if spec.wants("kv_a_proj_with_mqa"):
        n += add_replicated(attn.linear_kv_down_proj, _MLA_KV_A, "kv_a_proj_with_mqa", kv_down_out)
    if spec.wants("kv_b_proj"):
        n += add_column_parallel(
            attn.linear_kv_up_proj, _MLA_KV_B, kv_lora_rank, heads_local * (config.qk_head_dim + v_head_dim)
        )

    if spec.wants("o_proj"):
        n += _attach_row_parallel(
            attn.linear_proj, attn, _O, hf_prefix, spec, in_local=heads_local * v_head_dim, tp_rank=tp_rank, attr="o"
        )
    return n


def _attach_mlp(mlp: nn.Module, hf_prefix: str, spec: _Spec) -> int:
    """Adapters on a gated MLP: fused ``[gate; up]`` fc1 and row-parallel fc2."""
    from megatron.core import parallel_state as ps

    n = 0
    tp_rank = ps.get_tensor_model_parallel_rank()
    inter_local = mlp.linear_fc1.weight.shape[0] // 2

    if spec.wants("gate_proj", "up_proj"):
        fc1 = mlp.linear_fc1
        ad = NativeLoRAAdapter(_FC1, hf_prefix, inter_local=inter_local, tp_rank=tp_rank)
        ref = fc1.weight
        for name in ("gate", "up"):
            ad.register_parameter(
                f"{name}_A", _new_param(ref, (spec.rank, spec.hidden), init=spec.a_init, grad_sum_group="tp")
            )
            ad.register_parameter(f"{name}_B", _new_param(ref, (inter_local, spec.rank), init="zero"))
        mlp.lora_fc1_adapter = ad

        def fc1_delta(x, _m=fc1, _ad=ad):
            xn = _branch_input(x, _m, spec)
            r = spec.rank
            s = F.linear(xn, torch.cat([_ad.gate_A, _ad.up_A], 0))
            return torch.cat([F.linear(s[..., :r], _ad.gate_B), F.linear(s[..., r:], _ad.up_B)], dim=-1)

        _wrap_forward(fc1, fc1_delta, spec.scale)
        n += 1

    if spec.wants("down_proj"):
        n += _attach_row_parallel(
            mlp.linear_fc2, mlp, _FC2, hf_prefix, spec, in_local=inter_local, tp_rank=tp_rank, attr="down"
        )
    return n


def _assert_supported_architecture(config, tp_size: int = 1) -> None:
    """Reject layouts this generic implementation would silently get wrong.

    Each of these needs a different fused-qkv slicing or a down/up projection pair,
    so they belong in a model-specific provider. Layers whose mixer is not a fused
    qkv at all (linear-attention / GDN) are not an error: they simply carry no
    attention adapter, which ``apply_native_lora`` reports.
    """
    if bool(getattr(config, "multi_latent_attention", False)):
        return
    num_query_groups = getattr(config, "num_query_groups", None)
    assert num_query_groups is None or num_query_groups >= tp_size, (
        "native LoRA (--megatron-to-hf-mode raw) does not support this architecture: "
        f"num_query_groups ({num_query_groups}) < tensor parallel size ({tp_size}), so mcore splits a "
        "single query group across ranks and the local qkv rows are not a per-group slice. "
        "Use --megatron-to-hf-mode bridge, or point --lora-provider-path at a model-specific provider."
    )


def _require_grad_on_first_activation(model) -> nn.Module | None:
    """Force the embedding output to require grad, so recomputation still trains.

    Under activation recomputation mcore runs each transformer block's forward
    inside ``torch.no_grad()`` and recomputes it during backward. Every adapter
    param lives *inside* that block, so with the base frozen the checkpointed
    region has no grad-requiring input at all: autograd never enters it, the
    recompute never runs, and every adapter gradient comes back exactly zero
    (``grad_norm`` 0.0, ``B`` stuck at its zero init, so the LoRA delta is a
    permanent no-op that still passes every sync check).

    Making the first activation a grad-requiring leaf reconnects the chain. This
    is what PEFT does as ``enable_input_require_grads`` for the same reason, and
    it is a no-op cost when recomputation is off.
    """
    embedding = getattr(model, "embedding", None)
    if embedding is None:
        return None

    def hook(_module, _inputs, output):
        return output if output.requires_grad else output.requires_grad_(True)

    embedding.register_forward_hook(hook)
    return embedding


def apply_native_lora(model, args):
    """Attach LoRA to ONE built model chunk, before the Float16Module / DDP wrap.

    Wrapping here (rather than after) means DDP sees an already-frozen base and
    only builds grad buffers for the adapter params.
    """
    config = model.config
    spec = _Spec.from_args(args, config)
    _assert_supported_architecture(config, tp_size=spec.tp_size)

    for param in model.parameters():
        param.requires_grad = False
    hooked_embedding = _require_grad_on_first_activation(model)

    wrapped = 0
    mixer_only_layers = []
    for layer in model.decoder.layers:
        hf_layer = f"{spec.layer_prefix}{layer.layer_number - 1}."
        attn = getattr(layer, "self_attention", None)
        if attn is not None:
            if getattr(config, "multi_latent_attention", False):
                wrapped += _attach_mla_attention(attn, hf_layer + "self_attn.", spec, config)
            elif hasattr(attn, "linear_qkv"):
                wrapped += _attach_attention(attn, hf_layer + "self_attn.", spec)
            else:
                mixer_only_layers.append(layer.layer_number - 1)

        mlp = layer.mlp
        if hasattr(mlp, "linear_fc1"):
            assert getattr(mlp.config, "gated_linear_unit", True), "native LoRA assumes a gated (SwiGLU) MLP"
            wrapped += _attach_mlp(mlp, hf_layer + "mlp.", spec)
        shared = getattr(mlp, "shared_experts", None)
        if shared is not None and hasattr(shared, "linear_fc1"):
            wrapped += _attach_mlp(shared, hf_layer + spec.shared_expert, spec)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "[lora-native] rank=%d alpha=%s scale=%.3f dropout=%s targets=%s | %d modules wrapped, "
        "trainable %s / %s params (%.4f%%), input-grad hook=%s",
        spec.rank,
        args.lora_alpha,
        spec.scale,
        spec.dropout,
        sorted(spec.targets),
        wrapped,
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
        hooked_embedding is not None,
    )
    if mixer_only_layers:
        shown = f"{mixer_only_layers[:4]}{'...' if len(mixer_only_layers) > 4 else ''}"
        logger.info(
            "[lora-native] %d of %d layers have no linear_qkv (linear-attention / GDN mixer) and carry no "
            "attention adapter: %s. Their mixer projections need a model-specific --lora-provider-path.",
            len(mixer_only_layers),
            len(model.decoder.layers),
            shown,
        )
    assert wrapped > 0, (
        f"native LoRA matched no modules for --target-modules {sorted(spec.targets)}; "
        "expected some of q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj"
    )
    return model


def wrap_model_provider_with_lora(provider_func, args):
    """Wrap a miles model provider so every chunk it builds gets LoRA."""

    def wrapped(*args_, **kwargs):
        return apply_native_lora(provider_func(*args_, **kwargs), args)

    return wrapped


class _TpGather:
    """Batches per-tensor TP all-gathers into one flat all-gather.

    Requesting returns a handle; ``flush()`` performs the single collective and the
    handles resolve afterwards. Every rank must request in the same order.
    """

    def __init__(self):
        self._requests: list[tuple[torch.Tensor, int]] = []
        self._resolved: list[torch.Tensor] | None = None

    def request(self, local: torch.Tensor, cat_dim: int):
        index = len(self._requests)
        self._requests.append((local, cat_dim))
        return lambda: self._resolved[index]

    def flush(self) -> None:
        from megatron.core import parallel_state as ps

        world = ps.get_tensor_model_parallel_world_size()
        if world == 1 or not self._requests:
            self._resolved = [local for local, _ in self._requests]
            return
        assert len({local.dtype for local, _ in self._requests}) == 1, "mixed adapter dtypes"
        flats = [local.detach().contiguous().reshape(-1) for local, _ in self._requests]
        sizes = [f.numel() for f in flats]
        local_flat = torch.cat(flats)
        gathered = local_flat.new_empty(world * local_flat.numel())
        dist.all_gather_into_tensor(gathered, local_flat, group=ps.get_tensor_model_parallel_group())
        per_rank = gathered.view(world, -1)

        self._resolved = []
        offset = 0
        for (local, cat_dim), size in zip(self._requests, sizes, strict=True):
            shards = [per_rank[r, offset : offset + size].view(local.shape) for r in range(world)]
            self._resolved.append(torch.cat(shards, dim=cat_dim))
            offset += size


def _projections(kind: str) -> tuple[_Proj, ...]:
    assert kind in _ADAPTER_LAYOUT, f"unknown adapter kind {kind}"
    return _ADAPTER_LAYOUT[kind]


def _iter_adapters(model_chunks):
    for chunk in model_chunks:
        module = chunk
        while hasattr(module, "module"):
            module = module.module
        for child in module.modules():
            if isinstance(child, NativeLoRAAdapter):
                yield child


def export_lora_hf_named(model_chunks) -> list[tuple[str, torch.Tensor]]:
    """Return ``(hf_name, full_tensor)`` for every adapter param, in HF/PEFT layout.

    TP shards are gathered, so every rank returns the same complete set (e.g.
    ``model.layers.3.self_attn.q_proj.lora_A.weight``). PP is not gathered here;
    callers that need the whole model's adapter assemble across PP stages.

    The logged ``max|lora_B|`` is 0 exactly while the adapter is still at its zero
    init, which separates "the sync works" from "the sync carries anything".
    """
    started = time.perf_counter()
    gather = _TpGather()
    plan: list[tuple[str, object]] = []

    for adapter in _iter_adapters(model_chunks):
        prefix = adapter.hf_prefix
        for proj in _projections(adapter.kind):
            a = getattr(adapter, f"{proj.attr}_A")
            b = getattr(adapter, f"{proj.attr}_B")
            if proj.layout == COLUMN:
                a, b = a, gather.request(b, 0)
            elif proj.layout == ROW:
                a, b = gather.request(a, 1), b
            plan.append((f"{prefix}{proj.hf}.lora_A.weight", a))
            plan.append((f"{prefix}{proj.hf}.lora_B.weight", b))

    gather.flush()
    exported = [
        (name, (source() if callable(source) else source).detach().to(torch.bfloat16).contiguous())
        for name, source in plan
    ]
    if not dist.is_initialized() or dist.get_rank() == 0:
        peak_b = max(
            (t.abs().max().item() for name, t in exported if name.endswith("lora_B.weight")),
            default=0.0,
        )
        logger.info(
            "[lora-native] exported %d adapter tensors in %.1f ms (max|lora_B|=%.3e)",
            len(exported),
            (time.perf_counter() - started) * 1e3,
            peak_b,
        )
    return exported


def load_lora_adapter_hf(model_chunks, adapter_path: str) -> int:
    """Load an HF/PEFT adapter into the attached params, slicing each to this rank.

    Call after the base checkpoint load: adapter params live outside the
    dist-checkpoint, so a base load would not touch (or would clobber) them.
    """
    from safetensors import safe_open

    path = os.path.join(adapter_path, "adapter_model.safetensors")
    assert os.path.exists(path), (
        f"[lora-native] no adapter_model.safetensors under {adapter_path}; "
        "checkpoints written by save_lora_checkpoint use that name"
    )
    loaded = 0
    with safe_open(path, framework="pt") as f:
        keys = {re.sub(r"^base_model\.model\.", "", k): k for k in f.keys()}

        def take(name: str) -> torch.Tensor:
            assert name in keys, f"[lora-native] adapter tensor missing: {name}"
            return f.get_tensor(keys[name])

        def copy_into(param: torch.Tensor, tensor: torch.Tensor) -> None:
            nonlocal loaded
            assert (
                param.shape == tensor.shape
            ), f"[lora-native] shape mismatch: param {tuple(param.shape)} vs adapter slice {tuple(tensor.shape)}"
            with torch.no_grad():
                param.copy_(tensor.to(dtype=param.dtype, device=param.device))
            loaded += 1

        for adapter in _iter_adapters(model_chunks):
            prefix, meta = adapter.hf_prefix, adapter.shard_meta
            tp_rank = meta["tp_rank"]
            for proj in _projections(adapter.kind):
                a_full = take(f"{prefix}{proj.hf}.lora_A.weight")
                b_full = take(f"{prefix}{proj.hf}.lora_B.weight")
                if proj.layout != REPLICATED:
                    width = proj.width(meta)
                    span = slice(tp_rank * width, (tp_rank + 1) * width)
                    if proj.layout == COLUMN:
                        b_full = b_full[span]
                    else:
                        a_full = a_full[:, span]
                copy_into(getattr(adapter, f"{proj.attr}_A"), a_full)
                copy_into(getattr(adapter, f"{proj.attr}_B"), b_full)
    logger.info("[lora-native] loaded %d adapter tensors from %s", loaded, adapter_path)
    return loaded


def resolve_lora_provider(args):
    """Return the module implementing the native-LoRA provider protocol.

    ``--lora-provider-path`` selects a model-specific implementation (a dotted
    module path); the default is this module.
    """
    path = getattr(args, "lora_provider_path", None)
    if not path:
        return sys.modules[__name__]
    import importlib

    module = importlib.import_module(path)
    for entry_point in ("wrap_model_provider_with_lora", "load_lora_adapter_hf", "export_lora_hf_named"):
        assert hasattr(module, entry_point), f"--lora-provider-path {path} must define {entry_point}()"
    return module
