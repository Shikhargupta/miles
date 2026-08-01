"""SGLang-specific adapter export compatibility.

Native LoRA keeps the exact projection set requested by the user.  SGLang,
however, stores Q/K/V and gate/up adapters in fused buffers.  Its current
normalizer only accepts some partial combinations, so weight sync expands a
split adapter with zero-valued siblings while the ordinary HF checkpoint
export remains exact.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn

from miles_plugins.lora.codec.hf import export_lora_hf_named
from miles_plugins.lora.modules.linear import LoRASplitFC1, LoRASplitQKV, iter_adapters

_QKV_NAMES = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}
_FC1_NAMES = {"gate": "gate_proj", "up": "up_proj"}
_MLA_A_NAMES = ("q_a_proj", "kv_a_proj_with_mqa")


def expand_sglang_target_modules(target_modules: Iterable[str]) -> list[str]:
    """Expand logical split targets to the fused-buffer projection families.

    SGLang normalizes every Q/K/V target to ``qkv_proj`` and every gate/up
    target to ``gate_up_proj``.  Advertising all logical siblings keeps its
    adapter config consistent with the zero-padded serving export.
    """

    targets = list(dict.fromkeys(target_modules))
    target_set = set(targets)
    if target_set.intersection(_QKV_NAMES.values()):
        targets.extend(name for name in _QKV_NAMES.values() if name not in target_set)
        target_set.update(_QKV_NAMES.values())
    if target_set.intersection(_FC1_NAMES.values()):
        targets.extend(name for name in _FC1_NAMES.values() if name not in target_set)
        target_set.update(_FC1_NAMES.values())
    if target_set.intersection(_MLA_A_NAMES):
        targets.extend(name for name in _MLA_A_NAMES if name not in target_set)
    return targets


def _add_zero_pair(
    exported: dict[str, torch.Tensor],
    *,
    prefix: str,
    hf_name: str,
    a_like: torch.Tensor,
    b_rows: int,
) -> None:
    a_name = f"{prefix}{hf_name}.lora_A.weight"
    b_name = f"{prefix}{hf_name}.lora_B.weight"
    assert a_name not in exported and b_name not in exported, f"duplicate synthetic SGLang LoRA key {hf_name}"
    exported[a_name] = torch.zeros_like(a_like)
    exported[b_name] = a_like.new_zeros((b_rows, a_like.shape[0]))


def export_lora_sglang_named(model_chunks: Sequence[nn.Module]) -> list[tuple[str, torch.Tensor]]:
    """Export native adapter weights in a form every fused SGLang path accepts.

    Only serving sync uses this entry point.  ``export_lora_hf_named`` remains
    the lossless, exact-target checkpoint representation.
    """

    exact = export_lora_hf_named(model_chunks)
    exported = dict(exact)
    assert len(exported) == len(exact), "native LoRA export produced duplicate HF names across model chunks"

    mla_a_by_prefix = {}
    for adapter in iter_adapters(model_chunks):
        for projection in adapter.projection_specs:
            if projection.hf in _MLA_A_NAMES:
                mla_a_by_prefix.setdefault(adapter.hf_prefix, adapter.context)
        if isinstance(adapter, LoRASplitQKV):
            active = set(adapter._active)
            exemplar = next(iter(active))
            a_like = exported[f"{adapter.hf_prefix}{_QKV_NAMES[exemplar]}.lora_A.weight"]
            for attr, hf_name in _QKV_NAMES.items():
                if attr not in active:
                    _add_zero_pair(
                        exported,
                        prefix=adapter.hf_prefix,
                        hf_name=hf_name,
                        a_like=a_like,
                        b_rows=adapter._rows[attr] * adapter.context.tp_size,
                    )
        elif isinstance(adapter, LoRASplitFC1):
            active = set(adapter._active)
            exemplar = next(iter(active))
            a_like = exported[f"{adapter.hf_prefix}{_FC1_NAMES[exemplar]}.lora_A.weight"]
            for attr, hf_name in _FC1_NAMES.items():
                if attr not in active:
                    _add_zero_pair(
                        exported,
                        prefix=adapter.hf_prefix,
                        hf_name=hf_name,
                        a_like=a_like,
                        b_rows=adapter.inter_local * adapter.context.tp_size,
                    )

    # SGLang packs the two replicated MLA down projections into one
    # fused_qkv_a_proj_with_mqa buffer.  Its normalizer can zero-fill a missing
    # kv_a only by copying q_a's shape, which is wrong whenever q_lora_rank and
    # kv_lora_rank + qk_pos_emb_head_dim differ; it cannot start from kv_a at
    # all.  Materialize the absent pair with the architecture's true output
    # width in either direction.
    for prefix, context in mla_a_by_prefix.items():
        present = {hf_name for hf_name in _MLA_A_NAMES if f"{prefix}{hf_name}.lora_A.weight" in exported}
        if len(present) != 1:
            continue
        exemplar = next(iter(present))
        a_like = exported[f"{prefix}{exemplar}.lora_A.weight"]
        config = context.transformer_config
        rows = {
            "q_a_proj": int(config.q_lora_rank),
            "kv_a_proj_with_mqa": int(config.kv_lora_rank + config.qk_pos_emb_head_dim),
        }
        missing = next(name for name in _MLA_A_NAMES if name not in present)
        _add_zero_pair(
            exported,
            prefix=prefix,
            hf_name=missing,
            a_like=a_like,
            b_rows=rows[missing],
        )

    return list(exported.items())


__all__ = ["expand_sglang_target_modules", "export_lora_sglang_named"]
