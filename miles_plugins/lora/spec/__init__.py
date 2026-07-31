"""Per-architecture native LoRA specs.

``base`` holds the per-run constants (``_Spec``) and the attach primitives every
spec shares; ``attention`` holds the GQA fused-qkv and MLA attach paths with
their guards; ``mlp`` holds the gated MLP (and shared-expert) attach path.
Routed-MoE experts are deliberately absent: their adapters need a serving-side
layout contract of their own (a future ``moe`` spec).
"""

from .attention import _assert_supported_architecture, _attach_attention, _attach_mla_attention, _build_qkv_perm
from .base import _rmsnorm, _Spec
from .mlp import _attach_mlp

__all__ = [
    "_Spec",
    "_assert_supported_architecture",
    "_attach_attention",
    "_attach_mla_attention",
    "_attach_mlp",
    "_build_qkv_perm",
    "_rmsnorm",
]
