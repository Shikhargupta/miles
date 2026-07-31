"""Attention specs: GQA fused-qkv and multi-latent attention.

GQA targets, selected by ``--target-modules`` (HF leaf names):

===========================  ==========================================  ===============
target                       megatron module                             kind
===========================  ==========================================  ===============
q_proj / k_proj / v_proj     ``self_attention.linear_qkv`` (fused)       column-parallel
o_proj                       ``self_attention.linear_proj``              row-parallel
===========================  ==========================================  ===============

MLA (DeepSeek / GLM / Kimi) has no fused qkv and carries its own projection set:

===========================  ==========================================  ===============
target                       megatron module                             kind
===========================  ==========================================  ===============
q_a_proj                     ``self_attention.linear_q_down_proj``       replicated
q_b_proj                     ``self_attention.linear_q_up_proj``         column-parallel
q_proj (no q_lora_rank)      ``self_attention.linear_q_proj``            column-parallel
kv_a_proj_with_mqa           ``self_attention.linear_kv_down_proj``      replicated
kv_b_proj                    ``self_attention.linear_kv_up_proj``        column-parallel
o_proj                       ``self_attention.linear_proj``              row-parallel
===========================  ==========================================  ===============
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adapter import _MLA_KV_A, _MLA_KV_B, _MLA_Q, _MLA_Q_A, _MLA_Q_B, _O, _QKV, NativeLoRAAdapter, _new_param
from .base import _attach_row_parallel, _branch_input, _dropout, _Spec, _wrap_forward


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

    The query and key/value paths each compress to a latent and then expand, so
    the adapter surface is four projections plus the output one (see the module
    docstring's table). When ``q_lora_rank`` is unset there is no compression on
    the query path and ``linear_q_proj`` (column-parallel, straight from hidden)
    takes ``q_proj`` instead.

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


def _assert_supported_architecture(config, tp_size: int = 1) -> None:
    """Reject layouts these specs would silently get wrong.

    These are config-times-parallelism checks the registry's per-``model_type``
    gate cannot express: each needs a different fused-qkv slicing or a down/up
    projection pair, so they belong in a model-specific provider. Layers whose
    mixer is not a fused qkv at all (linear-attention / GDN) are not an error:
    they simply carry no attention adapter, which ``apply_native_lora`` reports.
    """
    if bool(getattr(config, "multi_latent_attention", False)):
        assert getattr(config, "q_lora_rank", None), (
            "native LoRA does not support multi-latent attention without q_lora_rank "
            "(DeepSeek-V2-Lite, Moonlight): the query path is uncompressed, so the adapter exports "
            "an unfused q_proj alongside kv_a_proj_with_mqa, and SGLang's loader expects the fused "
            "qkv_a layout. Use --megatron-to-hf-mode bridge, or point --lora-provider-path at a "
            "model-specific provider."
        )
        return
    num_query_groups = getattr(config, "num_query_groups", None)
    assert num_query_groups is None or num_query_groups >= tp_size, (
        "native LoRA (--megatron-to-hf-mode raw) does not support this architecture: "
        f"num_query_groups ({num_query_groups}) < tensor parallel size ({tp_size}), so mcore splits a "
        "single query group across ranks and the local qkv rows are not a per-group slice. "
        "Use --megatron-to-hf-mode bridge, or point --lora-provider-path at a model-specific provider."
    )
