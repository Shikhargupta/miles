"""HF naming for exported adapters.

The exported names are what SGLang looks up against its own module paths, so the
decoder-layer prefix and the shared-expert segment have to match the served
checkpoint exactly: Qwen3.5 nests the decoder under
``model.language_model.layers.`` and spells the shared expert
``mlp.shared_expert.``, while DeepSeek / GLM / Kimi use ``model.layers.`` and
``mlp.shared_experts.``.

``_hf_naming`` reads both off the checkpoint's own weight index -- the file the
engine serves is the ground truth, so it cannot drift. When mbridge carries a
bridge for this ``model_type``, ``_mbridge_cross_check`` compares its mapping
table against the index-derived prefix and warns on disagreement; it never
overrides and never raises, and is a no-op when mbridge is not importable.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

_DEFAULT_LAYER_PREFIX = "model.layers."
_DEFAULT_SHARED_EXPERT = "mlp.shared_expert."


def _hf_naming(hf_checkpoint: str | None) -> tuple[str, str]:
    """Read the decoder-layer prefix and shared-expert segment off the checkpoint itself."""
    index_path = os.path.join(hf_checkpoint or "", "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return _DEFAULT_LAYER_PREFIX, _DEFAULT_SHARED_EXPERT
    with open(index_path) as handle:
        names = json.load(handle).get("weight_map", {})

    prefixes: collections.Counter[str] = collections.Counter()
    for name in names:
        if name.startswith("mtp.") or "vision" in name:
            continue
        match = re.match(r"^((?:[\w.]+\.)?layers\.)\d+\.", name)
        if match:
            prefixes[match.group(1)] += 1
    layer_prefix = prefixes.most_common(1)[0][0] if prefixes else _DEFAULT_LAYER_PREFIX
    shared = "mlp.shared_experts." if any(".mlp.shared_experts." in n for n in names) else _DEFAULT_SHARED_EXPERT
    return layer_prefix, shared


def _layer_prefix_from_mapping(mapping: dict) -> str | None:
    """Decoder-layer prefix declared by an mbridge weight-mapping table, or None.

    Bridge classes template their HF names as e.g.
    ``model.language_model.layers.{layer_number}.self_attn.o_proj.weight``.
    """
    for hf_names in mapping.values():
        names = hf_names if isinstance(hf_names, (list, tuple)) else [hf_names]
        for name in names:
            match = re.match(r"^((?:[\w.]+\.)?layers\.)\{layer_number\}", str(name))
            if match:
                return match.group(1)
    return None


def _mbridge_cross_check(model_type: str | None, layer_prefix: str) -> None:
    """Warn when mbridge's mapping for this model_type disagrees with the checkpoint index.

    mbridge's per-``model_type`` bridge classes carry the megatron -> HF weight
    tables raw mode converts checkpoints with, so a disagreement means the
    checkpoint and its bridge are out of step. The checkpoint wins either way.
    """
    if not model_type:
        return
    try:
        import miles_plugins.mbridge  # noqa: F401  (registers the miles bridge subclasses)
        from mbridge.core.bridge import _MODEL_REGISTRY
    except Exception:
        return
    bridge_cls = _MODEL_REGISTRY.get(model_type)
    if bridge_cls is None:
        return
    expected = _layer_prefix_from_mapping(getattr(bridge_cls, "_ATTENTION_MAPPING", None) or {})
    if expected is not None and expected != layer_prefix:
        logger.warning(
            "[lora-native] adapter layer prefix %r (from the checkpoint weight index) disagrees with "
            "mbridge's %s mapping (%r); trusting the checkpoint.",
            layer_prefix,
            model_type,
            expected,
        )
