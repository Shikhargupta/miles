"""MLP spec: gated (SwiGLU) MLP, also used for a MoE layer's shared expert.

===========================  ==========================================  ===============
target                       megatron module                             kind
===========================  ==========================================  ===============
gate_proj / up_proj          ``mlp.linear_fc1`` (fused ``[gate; up]``)   column-parallel
down_proj                    ``mlp.linear_fc2``                          row-parallel
===========================  ==========================================  ===============

Routed MoE experts are deliberately not handled here: their adapters need a
serving-side layout contract of their own, so a MoE model supplies them through
its own provider (see ``--lora-provider-path``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adapter import _FC1, _FC2, NativeLoRAAdapter, _new_param
from .base import _attach_row_parallel, _branch_input, _Spec, _wrap_forward


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
