"""Qwen3.8-Next full-attention layer: QSA indexer + sparse attention.

12 of the 48 layers are full attention. Each projects its own indexer queries and
compressed keys, scores them, and keeps a budget of ``indexer_budget`` key tokens
per query; attention then reads only those.

Wiring note. Megatron's ``Attention.forward`` runs qkv projection ->
``self.core_attention`` -> output projection, and ``core_attention`` never sees
``hidden_states`` -- but the indexer projects from ``hidden_states``. So the
selection is computed in ``forward`` before delegating to the base class, and
handed to a replacement ``core_attention`` through the module instance. Same shape
of solution as the PLE side channel, and for the same reason: the value is needed
at a point in the call stack that does not carry it.

The selection is exact, not approximate: ``qsa_sparse_attention`` masks attention
to precisely the selected positions, so this matches the model spec rather than
falling back to dense and quietly changing the logits.
"""

import torch
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.module import MegatronModule
from torch import Tensor

from miles_plugins.models.qwen3_8_next.ops.qsa_indexer import Qwen38NextQSAIndexer
from miles_plugins.models.qwen3_8_next.ops.sparse_attn import qsa_sparse_attention


class Qwen38NextQSACoreAttention(MegatronModule):
    """Core attention restricted to the indexer's selection.

    Reads the selection off the owning attention module rather than taking it as an
    argument, because Megatron fixes ``core_attention``'s signature.
    """

    def __init__(self, config, layer_number: int, owner):
        super().__init__(config)
        self.layer_number = layer_number
        # Plain attribute, not a submodule: registering the parent here would make
        # the module graph cyclic and duplicate every parameter in the state dict.
        object.__setattr__(self, "_owner", owner)
        self.softmax_scale = config.kv_channels**-0.5
        self.compress_ratio = config.qwen3_8_next_indexer_compress_ratio

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_mask=None, **kwargs):
        selection = getattr(self._owner, "_qsa_selection", None)
        if selection is None:
            raise RuntimeError(
                "QSA core attention ran with no selection published. "
                "Qwen38NextAttention.forward sets it before delegating; reaching here "
                "means core_attention was called out of band."
            )
        cu_seqlens = getattr(self._owner, "_qsa_cu_seqlens", None)

        # Two layouts reach here. With packed_seq_params.qkv_format == 'thd' -- which
        # is what miles always feeds, since the linear-attention layers require
        # cu_seqlens -- Megatron has already squeezed the dummy batch dim away and
        # hands [t, np, hn], then reshapes whatever comes back to (t, 1, -1) itself.
        # Without packing it is [s, b, np, hn]. Handle both explicitly instead of
        # unpacking four names and failing on the layout that actually occurs.
        if query.dim() == 3:
            from miles_plugins.models.qwen3_8_next.ops.backend import use_triton

            if use_triton("QSA"):
                from miles_plugins.models.qwen3_8_next.ops.kernel.qsa_sparse_attn import (
                    qsa_sparse_attention_triton,
                )

                return qsa_sparse_attention_triton(
                    query, key, value, selection, self.softmax_scale
                ).reshape(query.shape[0], -1)
            return qsa_sparse_attention(
                query, key, value, selection,
                scale=self.softmax_scale,
                cu_seqlens=cu_seqlens,
                compress_ratio=self.compress_ratio,
            )

        if query.dim() != 4:
            raise RuntimeError(
                f"QSA core attention expected a 3D (thd) or 4D (sbhd) query, got "
                f"{tuple(query.shape)}"
            )

        s, b, hq, d = query.shape
        out = [
            qsa_sparse_attention(
                query[:, i], key[:, i], value[:, i], selection,
                scale=self.softmax_scale,
                cu_seqlens=cu_seqlens,
                compress_ratio=self.compress_ratio,
            )
            for i in range(b)
        ]
        return torch.stack(out, dim=1).reshape(s, b, hq * d)


class Qwen38NextAttention(SelfAttention):
    """Megatron self-attention whose key set is chosen by a QSA indexer."""

    def __init__(self, config, submodules, layer_number=1, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        self.indexer = Qwen38NextQSAIndexer(config, layer_number=layer_number)
        self.compress_ratio = config.qwen3_8_next_indexer_compress_ratio
        self.core_attention = Qwen38NextQSACoreAttention(config, layer_number, owner=self)
        self._qsa_selection = None

    @staticmethod
    def _packed_positions(cu_seqlens: Tensor, total: int) -> Tensor:
        """Positions restarting at 0 for each sequence in a packed batch.

        A single ``arange`` over the pack would place every sequence after the first
        at the wrong offsets, which changes both the RoPE the indexer applies and
        which compressed blocks count as causally visible -- silently, since the
        shapes are unaffected.
        """
        idx = torch.arange(total, device=cu_seqlens.device)
        starts = cu_seqlens[:-1].long()
        seg = torch.zeros(total, dtype=torch.long, device=cu_seqlens.device)
        seg[starts[1:]] = 1
        seg = seg.cumsum(0)
        return idx - starts[seg]

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        packed = kwargs.get("packed_seq_params")

        # Under sequence parallelism this wrapper sees only its [T/tp] shard, but
        # the indexer scores the whole pack (cu_seqlens are full-sequence
        # positions; the q/k/v the core attention receives are also full-length,
        # gathered inside the SP-aware linear). Gather for the indexer only.
        # no_grad: the selection is integer indices, so no gradient path exists
        # through it -- the indexer is effectively frozen in RL training, same as
        # on the inference side, and skipping autograd avoids holding the gathered
        # activations.
        indexer_states = hidden_states
        if getattr(self.config, "sequence_parallel", False):
            from megatron.core import parallel_state as _ps
            from megatron.core import tensor_parallel as _tp

            if _ps.get_tensor_model_parallel_world_size() > 1:
                with torch.no_grad():
                    indexer_states = _tp.gather_from_sequence_parallel_region(
                        hidden_states,
                        tensor_parallel_output_grad=False,
                        group=_ps.get_tensor_model_parallel_group(),
                    )
        seq = indexer_states.shape[0]
        if packed is not None:
            cu = getattr(packed, "cu_seqlens_q", None)
            if cu is None:
                raise NotImplementedError(
                    "packed_seq_params without cu_seqlens_q: QSA needs the sequence "
                    "boundaries to place positions and to keep attention inside a "
                    "document."
                )
            self._qsa_cu_seqlens = cu
            positions = self._packed_positions(cu, seq)
        else:
            self._qsa_cu_seqlens = None
            positions = torch.arange(seq, device=hidden_states.device)
        # The indexer works on one sequence's [T, hidden]; batch is folded per item
        # inside the core attention, so score on the first batch element's states.
        with torch.no_grad():
            selection = self.indexer(indexer_states[:, 0], positions)
            # Append the query's own partial-block tail as explicit indices. The
            # triton kernel is list-semantics (it attends exactly the listed
            # tokens), so the tail must be in the list; the torch mask path
            # dedupes, so the extra entries are harmless there. Indexer output
            # covers only complete blocks strictly before the query, so the tail
            # never duplicates a selected token.
            seq_start = torch.arange(seq, device=positions.device) - positions
            r = self.compress_ratio
            tail_in_seq = (positions + 1) // r * r
            offs = torch.arange(r, device=positions.device)
            tail_pos = tail_in_seq.unsqueeze(1) + offs.unsqueeze(0)          # in-seq
            tail_idx = seq_start.unsqueeze(1) + tail_pos                     # pack index
            tail_idx = torch.where(
                tail_pos <= positions.unsqueeze(1), tail_idx,
                torch.full_like(tail_idx, -1),
            )
            merged = torch.cat([selection, tail_idx.to(selection.dtype)], dim=1)
            # Enforce causality and segment confinement on the index list itself:
            # the triton kernel attends exactly what is listed (the torch path's
            # mask re-derives these constraints, the kernel must not rely on it).
            # Leaks show up as impossibly good logprobs on repeated text, which is
            # how this surfaced.
            pack_pos = torch.arange(seq, device=positions.device).unsqueeze(1)
            seg_lo = seq_start.unsqueeze(1)
            ok = (merged >= seg_lo) & (merged <= pack_pos) & (merged >= 0)
            self._qsa_selection = torch.where(
                ok, merged, torch.full_like(merged, -1)
            )
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._qsa_selection = None
            self._qsa_cu_seqlens = None
