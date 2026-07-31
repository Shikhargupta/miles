"""Spec-shared runtime: per-run constants and the attach primitives.

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
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_hf

from ..adapter import SUPPORTED_TARGETS, NativeLoRAAdapter, _new_param
from ..naming import _hf_naming


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

    Without sequence parallelism the input is replicated across TP instead, and each
    rank's branch produces only its own output slice, so each computes a partial
    ``dL/dx``. ``copy_to_tensor_model_parallel_region`` is identity forward and
    all-reduce backward, which sums those partials the way the base GEMM's own copy
    does; leaving it out sends every upstream layer a fraction of the adapter's
    gradient.
    """
    gamma = getattr(module, "layer_norm_weight", None)
    if gamma is not None:
        x = _rmsnorm(x, gamma, spec.eps, spec.zero_centered_gamma)
    if spec.sequence_parallel:
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        x = gather_from_sequence_parallel_region(x)
    elif spec.tp_size > 1:
        from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region

        x = copy_to_tensor_model_parallel_region(x)
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
