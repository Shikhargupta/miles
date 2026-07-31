"""Adapter taxonomy for native LoRA: self-describing param holders and their HF projections.

``NativeLoRAAdapter`` carries one wrapped module's adapter params together with
everything needed to name and shard them (``kind`` + ``hf_prefix`` +
``shard_meta``), so export and load in ``io.py`` are plain name joins driven by
``_ADAPTER_LAYOUT`` and never need to know which model -- or which spec --
attached the adapter. A new spec only has to register the right adapter to get
export, load, TP gathering and grad summing for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

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
