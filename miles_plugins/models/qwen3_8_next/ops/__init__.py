"""Qwen3.8-Next ops: the map. Training is triton-only.

    family  kernels                       test (oracle lives with the test)
    ------  ----------------------------  ------------------------------------
    QSA     kernel/qsa_sparse_attn.py     tests/.../test_qsa_triton.py
    HC      kernel/hc_triton.py           tests/.../test_hc_triton.py
    PLE     kernel/ple_triton.py          tests/.../test_ple_triton.py

Torch reference implementations moved to tests/qwen3_8_next/reference_ops.py
(sglang-parity-verified oracles, compared against by the tests above).
``backend.py`` holds the kernel warmup registry the model provider drives.

Supporting modules:
    attention.py      QSA wrapper: SP gather for the indexer, selection + tail
                      merge, kernel dispatch.
    qsa_indexer.py    indexer projections + topk (cuBLAS / cub).
    ple.py            PLE module: projections + the fused gate/conv kernel.
    ple_embedding.py  frozen n-gram table (kernel/ple_gather.py gathers rows).
    ple_hash.py       n-gram hashing, pure int ops.
    ple_context.py    side channel handing token ids to the PLE layer.

Parity debugging is fully out of the production tree: dump points are injected
on demand via the sglang dumper's source patcher when needed.

``ple_hash`` stays Megatron-import-free so it unit-tests against sglang without
pulling megatron.core; MegatronModule subclasses import on demand, keeping this
package import cheap and side-effect free.
"""

from miles_plugins.models.qwen3_8_next.ops.ple_hash import (
    build_ngram_contexts,
    ngram_hash_ids,
    shift_right_ignore_eos,
)

__all__ = [
    "build_ngram_contexts",
    "ngram_hash_ids",
    "shift_right_ignore_eos",
]
