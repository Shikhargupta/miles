"""Qwen3.8-Next ops: the map.

Every hand-written op family has a parity-verified torch reference and a triton
kernel set validated against it; ``ops/backend.py`` is the single switch
(``{QSA,HC,PLE}_BACKEND`` env, defaults listed there).

    family  torch reference          triton kernels               test
    ------  -----------------------  ---------------------------  -------------------------------
    QSA     sparse_attn.py           kernel/qsa_sparse_attn.py    tests/.../test_qsa_triton.py
    HC      hc.py                    kernel/hc_triton.py          tests/.../test_hc_triton.py
    PLE     ple.py (gate+conv path)  kernel/ple_triton.py         tests/.../test_ple_triton.py

Supporting modules (no triton counterpart by design):
    attention.py      QSA wrapper: SP gather for the indexer, selection + tail
                      merge, backend dispatch into the kernels above.
    qsa_indexer.py    indexer projections + topk -- cuBLAS GEMMs and cub topk,
                      nothing to hand-write.
    ple_embedding.py  frozen n-gram table (TP-sharded gather;
                      kernel/ple_gather.py holds its triton gather).
    ple_hash.py       n-gram hashing, pure int ops.
    ple_context.py    side channel handing token ids to the PLE layer.

Parity debugging is fully out of the production tree: dump points are
injected on demand via the dumper's source patcher (assets under
scripts/qwen3_8_next/debug/), never inline here.

``hc`` and ``ple_hash`` are deliberately free of Megatron imports so they can be
unit-tested against sglang without pulling megatron.core (and with it Transformer
Engine, whose prebuilt extension is sensitive to the torch build). The modules that
do subclass ``MegatronModule`` -- ``ple``, ``ple_embedding``, ``qsa_indexer`` -- are
imported on demand, not here, so importing this package stays cheap and side-effect
free.
"""

from miles_plugins.models.qwen3_8_next.ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_inject_gate,
    hc_mix,
)
from miles_plugins.models.qwen3_8_next.ops.ple_hash import (
    build_ngram_contexts,
    ngram_hash_ids,
    shift_right_ignore_eos,
)

__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_inject_gate",
    "hc_mix",
    "build_ngram_contexts",
    "ngram_hash_ids",
    "shift_right_ignore_eos",
]
