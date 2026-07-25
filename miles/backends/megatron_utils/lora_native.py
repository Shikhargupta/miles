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

Models whose modules diverge from plain mcore plug in their own implementation
via ``--lora-provider-path``. A provider module must expose the three entry
points this module does: ``wrap_model_provider_with_lora``,
``load_lora_adapter_hf`` and ``export_lora_hf_named``.
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)

# Adapter kinds. The wrapped module determines the sharding, so kinds map 1:1 to
# the four rows of the table above.
_QKV, _O, _FC1, _FC2 = "qkv", "o", "fc1", "fc2"


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
    tp_size: int
    targets: frozenset[str]

    @classmethod
    def from_args(cls, args, config) -> _Spec:
        from megatron.core import parallel_state as ps

        rank = int(args.lora_rank)
        assert rank > 0, "native LoRA requires --lora-rank > 0"
        return cls(
            rank=rank,
            scale=float(args.lora_alpha) / rank,
            dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
            a_init=getattr(args, "lora_A_init_method", "xavier") or "xavier",
            eps=config.layernorm_epsilon,
            hidden=config.hidden_size,
            sequence_parallel=bool(config.sequence_parallel),
            tp_size=ps.get_tensor_model_parallel_world_size(),
            targets=frozenset(args.target_modules or ()),
        )

    def wants(self, *names: str) -> bool:
        return bool(self.targets.intersection(names))


# ---------------------------------------------------------------------------
# Adapter parameters and the shared forward pieces
# ---------------------------------------------------------------------------


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


def _rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Recompute the RMSNorm fused into TELayerNormColumnParallelLinear (fp32 internals)."""
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * gamma.float()).to(x.dtype)


def _branch_input(x: torch.Tensor, module: nn.Module, spec: _Spec) -> torch.Tensor:
    """Input to a column-parallel adapter branch.

    The wrapped module may fuse its layernorm (``layer_norm_weight``), in which
    case the branch has to recompute it to see the same input the base GEMM does.
    Under sequence parallelism the module's input is sequence-sharded, so gather
    it back to the full sequence the adapter's replicated ``A`` expects.
    """
    gamma = getattr(module, "layer_norm_weight", None)
    if gamma is not None:
        x = _rmsnorm(x, gamma, spec.eps)
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


def _build_qkv_perm(num_q_heads: int, num_groups: int, head_dim: int, device) -> torch.Tensor:
    """Row permutation from plain ``[q; k; v]`` order into mcore's fused qkv layout.

    mcore emits qkv grouped per query group -- ``q1 q2 k1 v1 | q3 q4 k2 v2 | ...``
    (see ``SelfAttention.get_query_key_value_tensors``) -- while the adapter keeps
    one ``B`` per projection. Permuting the assembled delta is cheaper and easier
    to reason about than storing ``B`` interleaved.
    """
    q_per_group = num_q_heads // num_groups
    k_base = num_q_heads * head_dim
    v_base = (num_q_heads + num_groups) * head_dim
    index: list[int] = []
    for g in range(num_groups):
        index.extend(range(g * q_per_group * head_dim, (g + 1) * q_per_group * head_dim))
        index.extend(range(k_base + g * head_dim, k_base + (g + 1) * head_dim))
        index.extend(range(v_base + g * head_dim, v_base + (g + 1) * head_dim))
    return torch.tensor(index, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Attaching adapters
# ---------------------------------------------------------------------------


def _wrap_forward(module: nn.Module, delta_fn, scale: float) -> None:
    """Add ``scale * delta_fn(x)`` to ``module``'s output, keeping its (out, bias) contract."""
    original = module.forward

    def forward(x, *args, **kwargs):
        out, bias = original(x, *args, **kwargs)
        return torch.add(out, delta_fn(x), alpha=scale), bias

    module.forward = forward


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
        ad = NativeLoRAAdapter(_QKV, hf_prefix, num_q=num_q, num_kv=num_kv, head_dim=head_dim, tp_rank=tp_rank)
        ref = qkv.weight
        for name, rows in (("q", num_q * head_dim), ("k", num_kv * head_dim), ("v", num_kv * head_dim)):
            ad.register_parameter(
                f"{name}_A", _new_param(ref, (spec.rank, spec.hidden), init=spec.a_init, grad_sum_group="tp")
            )
            ad.register_parameter(f"{name}_B", _new_param(ref, (rows, spec.rank), init="zero"))
        ad.register_buffer("out_perm", _build_qkv_perm(num_q, num_kv, head_dim, ref.device), persistent=False)
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
        proj = attn.linear_proj
        in_local = num_q * head_dim
        ad = NativeLoRAAdapter(_O, hf_prefix, in_local=in_local, tp_rank=tp_rank)
        ref = proj.weight
        ad.register_parameter("o_A", _new_param(ref, (spec.rank, in_local), init=spec.a_init))
        ad.register_parameter(
            "o_B",
            _new_param(
                ref, (spec.hidden, spec.rank), init="zero", grad_sum_group="tp" if spec.sequence_parallel else None
            ),
        )
        attn.lora_o_adapter = ad

        def o_delta(x, _m=proj, _ad=ad):
            partial = F.linear(_dropout(x, spec, _m.training), _ad.o_A)
            return F.linear(_reduce_row_parallel(partial, spec), _ad.o_B)

        _wrap_forward(proj, o_delta, spec.scale)
        n += 1
    return n


def _attach_mlp(mlp: nn.Module, hf_prefix: str, spec: _Spec) -> int:
    """Adapters on a gated MLP: fused ``[gate; up]`` fc1 and row-parallel fc2."""
    from megatron.core import parallel_state as ps

    n = 0
    tp_rank = ps.get_tensor_model_parallel_rank()
    # fc1 is the fused gated projection, so its local rows are [gate_local; up_local].
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
        fc2 = mlp.linear_fc2
        ad = NativeLoRAAdapter(_FC2, hf_prefix, inter_local=inter_local, tp_rank=tp_rank)
        ref = fc2.weight
        ad.register_parameter("down_A", _new_param(ref, (spec.rank, inter_local), init=spec.a_init))
        ad.register_parameter(
            "down_B",
            _new_param(
                ref, (spec.hidden, spec.rank), init="zero", grad_sum_group="tp" if spec.sequence_parallel else None
            ),
        )
        mlp.lora_fc2_adapter = ad

        def fc2_delta(x, _m=fc2, _ad=ad):
            partial = F.linear(_dropout(x, spec, _m.training), _ad.down_A)
            return F.linear(_reduce_row_parallel(partial, spec), _ad.down_B)

        _wrap_forward(fc2, fc2_delta, spec.scale)
        n += 1
    return n


def _assert_supported_architecture(config, model, tp_size: int = 1) -> None:
    """Reject layouts this generic implementation would silently get wrong.

    Each of these needs a different fused-qkv slicing, a down/up projection pair,
    or a non-GEMM mixer, so they belong in a model-specific provider.
    """
    unsupported = []
    if getattr(config, "attention_output_gate", False):
        unsupported.append(
            "attention_output_gate=True (Qwen3.5 / Qwen3-Next): linear_qkv emits a 4th gate "
            "slice, so the per-projection row split here does not hold"
        )
    if getattr(config, "multi_latent_attention", False):
        unsupported.append("multi_latent_attention=True (MLA): q/kv down+up projections, not a fused qkv")
    num_query_groups = getattr(config, "num_query_groups", None)
    if num_query_groups is not None and num_query_groups < tp_size:
        # mcore re-gathers the full qkv and re-slices per group in this regime, so this
        # rank's weight rows are not a per-group slice of its own output.
        unsupported.append(
            f"num_query_groups ({num_query_groups}) < tensor parallel size ({tp_size}): mcore splits "
            "a single query group across ranks, so the local qkv rows are not a per-group slice"
        )
    missing_qkv = [
        layer.layer_number - 1
        for layer in model.decoder.layers
        if getattr(layer, "self_attention", None) is not None and not hasattr(layer.self_attention, "linear_qkv")
    ]
    if missing_qkv:
        shown = f"{missing_qkv[:4]}{'...' if len(missing_qkv) > 4 else ''}"
        unsupported.append(f"layers {shown} have no linear_qkv (linear-attention / GDN mixer)")
    assert not unsupported, (
        "native LoRA (--megatron-to-hf-mode raw) does not support this architecture: "
        + "; ".join(unsupported)
        + ". Use --megatron-to-hf-mode bridge, or point --lora-provider-path at a model-specific provider."
    )


def apply_native_lora(model, args):
    """Attach LoRA to ONE built model chunk, before the Float16Module / DDP wrap.

    Wrapping here (rather than after) means DDP sees an already-frozen base and
    only builds grad buffers for the adapter params.
    """
    config = model.config
    spec = _Spec.from_args(args, config)
    _assert_supported_architecture(config, model, tp_size=spec.tp_size)

    for param in model.parameters():
        param.requires_grad = False

    wrapped = 0
    for layer in model.decoder.layers:
        hf_layer = f"model.layers.{layer.layer_number - 1}."
        attn = getattr(layer, "self_attention", None)
        if attn is not None:
            wrapped += _attach_attention(attn, hf_layer + "self_attn.", spec)

        mlp = layer.mlp
        # A dense MLP, or the shared expert of an MoE layer: both are plain gated MLPs.
        if hasattr(mlp, "linear_fc1"):
            assert getattr(mlp.config, "gated_linear_unit", True), "native LoRA assumes a gated (SwiGLU) MLP"
            wrapped += _attach_mlp(mlp, hf_layer + "mlp.", spec)
        shared = getattr(mlp, "shared_experts", None)
        if shared is not None and hasattr(shared, "linear_fc1"):
            wrapped += _attach_mlp(shared, hf_layer + "mlp.shared_expert.", spec)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "[lora-native] rank=%d alpha=%s scale=%.3f dropout=%s targets=%s | %d modules wrapped, "
        "trainable %s / %s params (%.4f%%)",
        spec.rank,
        args.lora_alpha,
        spec.scale,
        spec.dropout,
        sorted(spec.targets),
        wrapped,
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
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


# ---------------------------------------------------------------------------
# Export to HF names (adapter weight sync + checkpointing)
# ---------------------------------------------------------------------------


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
    """
    started = time.perf_counter()
    gather = _TpGather()
    plan: list[tuple[str, object]] = []  # (hf_name, tensor or callable)

    for adapter in _iter_adapters(model_chunks):
        prefix = adapter.hf_prefix
        if adapter.kind == _QKV:
            # B rows stay in plain per-projection order (the interleave is applied to
            # the delta at forward time), so this is a plain row gather.
            for proj, a, b in (
                ("q_proj", adapter.q_A, adapter.q_B),
                ("k_proj", adapter.k_A, adapter.k_B),
                ("v_proj", adapter.v_A, adapter.v_B),
            ):
                plan.append((f"{prefix}{proj}.lora_A.weight", a))  # replicated
                plan.append((f"{prefix}{proj}.lora_B.weight", gather.request(b, 0)))
        elif adapter.kind == _O:
            plan.append((f"{prefix}o_proj.lora_A.weight", gather.request(adapter.o_A, 1)))
            plan.append((f"{prefix}o_proj.lora_B.weight", adapter.o_B))  # replicated
        elif adapter.kind == _FC1:
            plan.append((f"{prefix}gate_proj.lora_A.weight", adapter.gate_A))
            plan.append((f"{prefix}gate_proj.lora_B.weight", gather.request(adapter.gate_B, 0)))
            plan.append((f"{prefix}up_proj.lora_A.weight", adapter.up_A))
            plan.append((f"{prefix}up_proj.lora_B.weight", gather.request(adapter.up_B, 0)))
        elif adapter.kind == _FC2:
            plan.append((f"{prefix}down_proj.lora_A.weight", gather.request(adapter.down_A, 1)))
            plan.append((f"{prefix}down_proj.lora_B.weight", adapter.down_B))  # replicated
        else:
            raise ValueError(f"unknown adapter kind {adapter.kind}")

    gather.flush()
    exported = [
        (name, (source() if callable(source) else source).detach().to(torch.bfloat16).contiguous())
        for name, source in plan
    ]
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(
            "[lora-native] exported %d adapter tensors in %.1f ms",
            len(exported),
            (time.perf_counter() - started) * 1e3,
        )
    return exported


# ---------------------------------------------------------------------------
# Loading an HF/PEFT adapter
# ---------------------------------------------------------------------------


def load_lora_adapter_hf(model_chunks, adapter_path: str) -> int:
    """Load an HF/PEFT adapter into the attached params, slicing each to this rank.

    Call after the base checkpoint load: adapter params live outside the
    dist-checkpoint, so a base load would not touch (or would clobber) them.
    """
    from safetensors import safe_open

    path = os.path.join(adapter_path, "adapter_model.safetensors")
    loaded = 0
    with safe_open(path, framework="pt") as f:
        # PEFT checkpoints prefix names with base_model.model.; index both spellings.
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
            if adapter.kind == _QKV:
                head_dim, num_q, num_kv = meta["head_dim"], meta["num_q"], meta["num_kv"]
                for proj, name, rows in (
                    ("q_proj", "q", num_q * head_dim),
                    ("k_proj", "k", num_kv * head_dim),
                    ("v_proj", "v", num_kv * head_dim),
                ):
                    copy_into(getattr(adapter, f"{name}_A"), take(f"{prefix}{proj}.lora_A.weight"))
                    rows_full = take(f"{prefix}{proj}.lora_B.weight")
                    copy_into(getattr(adapter, f"{name}_B"), rows_full[tp_rank * rows : (tp_rank + 1) * rows])
            elif adapter.kind == _O:
                in_local = meta["in_local"]
                cols = take(f"{prefix}o_proj.lora_A.weight")
                copy_into(adapter.o_A, cols[:, tp_rank * in_local : (tp_rank + 1) * in_local])
                copy_into(adapter.o_B, take(f"{prefix}o_proj.lora_B.weight"))
            elif adapter.kind == _FC1:
                inter = meta["inter_local"]
                for proj, name in (("gate_proj", "gate"), ("up_proj", "up")):
                    copy_into(getattr(adapter, f"{name}_A"), take(f"{prefix}{proj}.lora_A.weight"))
                    rows_full = take(f"{prefix}{proj}.lora_B.weight")
                    copy_into(getattr(adapter, f"{name}_B"), rows_full[tp_rank * inter : (tp_rank + 1) * inter])
            elif adapter.kind == _FC2:
                inter = meta["inter_local"]
                cols = take(f"{prefix}down_proj.lora_A.weight")
                copy_into(adapter.down_A, cols[:, tp_rank * inter : (tp_rank + 1) * inter])
                copy_into(adapter.down_B, take(f"{prefix}down_proj.lora_B.weight"))
            else:
                raise ValueError(f"unknown adapter kind {adapter.kind}")
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
