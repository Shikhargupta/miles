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

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_mask=None, **kwargs):
        selection = getattr(self._owner, "_qsa_selection", None)
        if selection is None:
            raise RuntimeError(
                "QSA core attention ran with no selection published. "
                "Qwen38NextAttention.forward sets it before delegating; reaching here "
                "means core_attention was called out of band."
            )
        # Megatron hands these in [s, b, h, d]; the sparse path is per sequence.
        s, b, hq, d = query.shape
        out = []
        for i in range(b):
            out.append(
                qsa_sparse_attention(
                    query[:, i], key[:, i], value[:, i], selection, scale=self.softmax_scale
                )
            )
        return torch.stack(out, dim=1).reshape(s, b, hq * d)


class Qwen38NextAttention(SelfAttention):
    """Megatron self-attention whose key set is chosen by a QSA indexer."""

    def __init__(self, config, submodules, layer_number=1, *args, **kwargs):
        super().__init__(config, submodules, layer_number, *args, **kwargs)
        self.indexer = Qwen38NextQSAIndexer(config, layer_number=layer_number)
        self.core_attention = Qwen38NextQSACoreAttention(config, layer_number, owner=self)
        self._qsa_selection = None

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        # [s, b, h] -> positions along the sequence. Packed (THD) batches carry their
        # own offsets in packed_seq_params; handling those is still open, so refuse
        # rather than silently score against the wrong positions.
        packed = kwargs.get("packed_seq_params")
        if packed is not None:
            raise NotImplementedError(
                "QSA with packed sequences needs per-sequence positions from "
                "packed_seq_params; scoring against a single arange would place every "
                "sequence after the first at the wrong offsets."
            )
        seq = hidden_states.shape[0]
        positions = torch.arange(seq, device=hidden_states.device)
        # The indexer works on one sequence's [T, hidden]; batch is folded per item
        # inside the core attention, so score on the first batch element's states.
        self._qsa_selection = self.indexer(hidden_states[:, 0], positions)
        try:
            return super().forward(hidden_states, *args, **kwargs)
        finally:
            self._qsa_selection = None
