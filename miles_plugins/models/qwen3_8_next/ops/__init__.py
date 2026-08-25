"""Qwen3.8-Next ops.

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
    ngram_hash_ids,
    shift_right_ignore_eos,
)

__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_inject_gate",
    "hc_mix",
    "ngram_hash_ids",
    "shift_right_ignore_eos",
]
