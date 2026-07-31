"""HF naming, load/export, and SGLang-facing packing for native LoRA."""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import time
from collections.abc import Iterable

import torch
import torch.distributed as dist

from miles_plugins.lora.distributed import TensorParallelGather
from miles_plugins.lora.modules.linear import NativeLoRAAdapter, iter_adapters
from miles_plugins.lora.spec.base import COLUMN, REPLICATED, ROW

logger = logging.getLogger(__name__)

_DEFAULT_LAYER_PREFIX = "model.layers."
_DEFAULT_SHARED_EXPERT = "mlp.shared_expert."


def target_modules_from_hf_names(names: Iterable[str]) -> list[str]:
    """Return the exact logical projection leaves represented by HF LoRA tensors."""
    targets = set()
    for name in names:
        match = re.search(r"(?:^|\.)([^.]+)\.lora_[AB]\.weight$", name)
        if match:
            targets.add(match.group(1))
    return sorted(targets)


def resolve_hf_naming(hf_checkpoint: str | None) -> tuple[str, str]:
    """Read decoder-layer and shared-expert prefixes from the served checkpoint."""
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
    shared = "mlp.shared_experts." if any(".mlp.shared_experts." in name for name in names) else _DEFAULT_SHARED_EXPERT
    return layer_prefix, shared


def _layer_prefix_from_mapping(mapping: dict) -> str | None:
    """Return the decoder-layer prefix declared by an mbridge mapping table."""
    for hf_names in mapping.values():
        names = hf_names if isinstance(hf_names, (list, tuple)) else [hf_names]
        for name in names:
            match = re.match(r"^((?:[\w.]+\.)?layers\.)\{layer_number\}", str(name))
            if match:
                return match.group(1)
    return None


def mbridge_cross_check(model_type: str | None, layer_prefix: str) -> None:
    """Warn if optional mbridge conversion metadata disagrees with HF naming."""
    if not model_type:
        return
    try:
        import miles_plugins.mbridge  # noqa: F401  (registers Miles bridge subclasses)
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


def export_lora_hf_named(model_chunks) -> list[tuple[str, torch.Tensor]]:
    """Materialize full HF/PEFT adapter tensors on every TP rank.

    The resulting names and tensors are also the native provider's SGLang
    packing contract. PP assembly remains with the shared Miles checkpoint
    orchestrator until the bridge path is split in a later refactor.
    """
    started = time.perf_counter()
    gather = TensorParallelGather()
    plan: list[tuple[str, object]] = []

    for adapter in iter_adapters(model_chunks):
        for projection in adapter.projection_specs:
            a = getattr(adapter, f"{projection.attr}_A")
            b = getattr(adapter, f"{projection.attr}_B")
            if projection.layout == COLUMN:
                b = gather.request(b, 0)
            elif projection.layout == ROW:
                a = gather.request(a, 1)
            plan.append((f"{adapter.hf_prefix}{projection.hf}.lora_A.weight", a))
            plan.append((f"{adapter.hf_prefix}{projection.hf}.lora_B.weight", b))

    gather.flush()
    exported = [
        (name, (source() if callable(source) else source).detach().to(torch.bfloat16).contiguous())
        for name, source in plan
    ]
    if not dist.is_initialized() or dist.get_rank() == 0:
        peak_b = max(
            (tensor.abs().max().item() for name, tensor in exported if name.endswith("lora_B.weight")),
            default=0.0,
        )
        logger.info(
            "[lora-native] exported %d adapter tensors in %.1f ms (max|lora_B|=%.3e)",
            len(exported),
            (time.perf_counter() - started) * 1e3,
            peak_b,
        )
    return exported


def load_lora_adapter_hf(model_chunks, adapter_path: str) -> int:
    """Load and slice an HF/PEFT adapter into attached native modules."""
    from safetensors import safe_open

    path = os.path.join(adapter_path, "adapter_model.safetensors")
    assert os.path.exists(path), (
        f"[lora-native] no adapter_model.safetensors under {adapter_path}; "
        "checkpoints written by save_lora_checkpoint use that name"
    )
    loaded = 0
    with safe_open(path, framework="pt") as adapter_file:
        keys = {re.sub(r"^base_model\.model\.", "", key): key for key in adapter_file.keys()}

        def take(name: str) -> torch.Tensor:
            assert name in keys, f"[lora-native] adapter tensor missing: {name}"
            return adapter_file.get_tensor(keys[name])

        def copy_into(parameter: torch.Tensor, tensor: torch.Tensor) -> None:
            nonlocal loaded
            assert parameter.shape == tensor.shape, (
                f"[lora-native] shape mismatch: param {tuple(parameter.shape)} "
                f"vs adapter slice {tuple(tensor.shape)}"
            )
            with torch.no_grad():
                parameter.copy_(tensor.to(dtype=parameter.dtype, device=parameter.device))
            loaded += 1

        for adapter in iter_adapters(model_chunks):
            _load_adapter(adapter, take, copy_into)
    logger.info("[lora-native] loaded %d adapter tensors from %s", loaded, adapter_path)
    return loaded


def _load_adapter(adapter: NativeLoRAAdapter, take, copy_into) -> None:
    for projection in adapter.projection_specs:
        a_parameter = getattr(adapter, f"{projection.attr}_A")
        b_parameter = getattr(adapter, f"{projection.attr}_B")
        a_full = take(f"{adapter.hf_prefix}{projection.hf}.lora_A.weight")
        b_full = take(f"{adapter.hf_prefix}{projection.hf}.lora_B.weight")
        if projection.layout != REPLICATED:
            if projection.layout == COLUMN:
                width = b_parameter.shape[0]
                span = slice(adapter.tp_rank * width, (adapter.tp_rank + 1) * width)
                b_full = b_full[span]
            else:
                width = a_parameter.shape[1]
                span = slice(adapter.tp_rank * width, (adapter.tp_rank + 1) * width)
                a_full = a_full[:, span]
        copy_into(a_parameter, a_full)
        copy_into(b_parameter, b_full)


# Compatibility names retained for the PR's numerical and registry tests.
_hf_naming = resolve_hf_naming
_mbridge_cross_check = mbridge_cross_check
