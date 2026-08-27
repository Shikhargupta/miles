"""Triton kernel warmup registry (training is triton-only).

Triton JITs a kernel at first call; left to nature, bwd kernels compile during
the first BACKWARD -- inside the 1F1B pipeline, where stages hit their first
backward one after another and everyone else waits, turning a 2-3 min JIT into
a 25-40 min first step. Each kernel module registers a warmup closure next to
its kernels (the only place that knows their signatures); the model provider
calls ``warm_kernels(**model_config_values)`` once at init, all ranks in
parallel. Closures use tiny token counts: T is do_not_specialize, so one
compile at T=8 serves every later shape. The node-local compile cache seeded by
e2e_node.sh makes this near-instant after any prior run of a kernel version.
"""

import importlib

_WARMUPS: dict[str, list] = {}
_KERNEL_MODULES = ("hc_triton", "ple_triton", "qsa_sparse_attn")


def register_warmup(family: str):
    """Decorator: register ``fn(**spec)`` to pre-compile a family's kernels."""

    def deco(fn):
        _WARMUPS.setdefault(family, []).append(fn)
        return fn

    return deco


def warm_kernels(**spec) -> None:
    """Compile every kernel family now. Idempotent."""
    for module in _KERNEL_MODULES:
        importlib.import_module(f"miles_plugins.models.qwen3_8_next.ops.kernel.{module}")
    for fns in _WARMUPS.values():
        for fn in fns:
            fn(**spec)
