"""Native (raw-mode) LoRA for standard Megatron-core GPT models -- provider entrypoint.

Attaches LoRA adapters directly to the mcore model built by miles' own model
provider (``--megatron-to-hf-mode raw``), without going through Megatron-Bridge.
Adapters are registered per HF projection, so the HF/PEFT export and the SGLang
adapter sync are plain name joins. Bridge-mode LoRA is a separate path and is
not affected by this plugin.

This module is the default ``--lora-provider-path`` provider. A provider must
expose the three entry points this module does: ``wrap_model_provider_with_lora``,
``load_lora_adapter_hf`` and ``export_lora_hf_named``; models whose modules
diverge from plain mcore plug in their own implementation that way.

Package layout:

* ``registry.py`` -- HF ``model_type`` -> spec family; unregistered
  architectures fail at startup.
* ``spec/`` -- per-architecture attach paths: ``attention.py`` (GQA fused-qkv,
  MLA) and ``mlp.py`` (gated MLP, shared experts) over the shared primitives in
  ``base.py``. Target/module tables live in their docstrings.
* ``adapter.py`` -- the self-describing ``NativeLoRAAdapter`` and the
  projection/layout table export and load are driven by.
* ``io.py`` -- HF/PEFT export (TP-gathered) and load (sliced per rank).
* ``naming.py`` -- HF naming read off the checkpoint's weight index, with an
  optional mbridge cross-check.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from miles.backends.megatron_utils.lora_utils import convert_target_modules_to_hf  # noqa: F401  (re-export)

from .adapter import _ADAPTER_LAYOUT, SUPPORTED_TARGETS, NativeLoRAAdapter  # noqa: F401  (re-export)
from .io import export_lora_hf_named, load_lora_adapter_hf  # noqa: F401  (provider protocol)
from .naming import _hf_naming, _mbridge_cross_check  # noqa: F401  (re-export)
from .registry import MODEL_SPECS, resolve_model_spec  # noqa: F401  (re-export)
from .spec import (  # noqa: F401  (re-export)
    _assert_supported_architecture,
    _attach_attention,
    _attach_mla_attention,
    _attach_mlp,
    _build_qkv_perm,
    _rmsnorm,
    _Spec,
)

logger = logging.getLogger(__name__)


def _require_grad_on_first_activation(model) -> nn.Module | None:
    """Force the embedding output to require grad, so recomputation still trains.

    Under activation recomputation mcore runs each transformer block's forward
    inside ``torch.no_grad()`` and recomputes it during backward. Every adapter
    param lives *inside* that block, so with the base frozen the checkpointed
    region has no grad-requiring input at all: autograd never enters it, the
    recompute never runs, and every adapter gradient comes back exactly zero
    (``grad_norm`` 0.0, ``B`` stuck at its zero init, so the LoRA delta is a
    permanent no-op that still passes every sync check).

    Making the first activation a grad-requiring leaf reconnects the chain. This
    is what PEFT does as ``enable_input_require_grads`` for the same reason, and
    it is a no-op cost when recomputation is off.
    """
    embedding = getattr(model, "embedding", None)
    if embedding is None:
        return None

    def hook(_module, _inputs, output):
        return output if output.requires_grad else output.requires_grad_(True)

    embedding.register_forward_hook(hook)
    return embedding


def _assert_supported_run(args, config, spec: _Spec) -> None:
    """Reject flag combinations this implementation is known to get wrong.

    Each is a fail-fast rather than a fix: the interaction is understood but not
    handled, and silently producing wrong weights is the worse outcome.
    """
    assert not getattr(args, "overlap_param_gather", False), (
        "native LoRA does not support --overlap-param-gather: the adapter is never called as a "
        "module (its params are read inside the wrapped module's closure), so no forward pre-hook "
        "dispatches its bucket's all-gather and step 1 onward would run on stale shards. "
        "Drop the flag, or use --megatron-to-hf-mode bridge."
    )
    assert not getattr(args, "moe_shared_expert_overlap", False), (
        "native LoRA does not support --moe-shared-expert-overlap: the dispatcher owns the "
        "shared-expert communication, so the adapter's gather/reduce derived from the global "
        "sequence-parallel flag no longer matches the module's effective parallel mode. "
        "Drop the flag, or use --megatron-to-hf-mode bridge."
    )
    if getattr(args, "colocate", False) and spec.targets:
        assert getattr(args, "enable_weights_backuper", True), (
            "native LoRA under --colocate needs the weights backuper: the adapter pages are "
            "memory-saver-paused while the export runs, so the sync would read released memory. "
            "Keep the backuper enabled, or drop --colocate."
        )


def apply_native_lora(model, args):
    """Attach LoRA to ONE built model chunk, before the Float16Module / DDP wrap.

    Wrapping here (rather than after) means DDP sees an already-frozen base and
    only builds grad buffers for the adapter params.
    """
    config = model.config
    model_type, _model_spec = resolve_model_spec(args, config)
    spec = _Spec.from_args(args, config)
    _assert_supported_architecture(config, tp_size=spec.tp_size)
    _assert_supported_run(args, config, spec)
    _mbridge_cross_check(model_type, spec.layer_prefix)

    for param in model.parameters():
        param.requires_grad = False
    hooked_embedding = _require_grad_on_first_activation(model)

    wrapped = 0
    mixer_only_layers = []
    for layer in model.decoder.layers:
        hf_layer = f"{spec.layer_prefix}{layer.layer_number - 1}."
        attn = getattr(layer, "self_attention", None)
        if attn is not None:
            if getattr(config, "multi_latent_attention", False):
                wrapped += _attach_mla_attention(attn, hf_layer + "self_attn.", spec, config)
            elif hasattr(attn, "linear_qkv"):
                wrapped += _attach_attention(attn, hf_layer + "self_attn.", spec)
            else:
                mixer_only_layers.append(layer.layer_number - 1)

        mlp = layer.mlp
        if hasattr(mlp, "linear_fc1"):
            assert getattr(mlp.config, "gated_linear_unit", True), "native LoRA assumes a gated (SwiGLU) MLP"
            wrapped += _attach_mlp(mlp, hf_layer + "mlp.", spec)
        shared = getattr(mlp, "shared_experts", None)
        if shared is not None and hasattr(shared, "linear_fc1"):
            wrapped += _attach_mlp(shared, hf_layer + spec.shared_expert, spec)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "[lora-native] arch=%s rank=%d alpha=%s scale=%.3f dropout=%s targets=%s | %d modules wrapped, "
        "trainable %s / %s params (%.4f%%), input-grad hook=%s",
        model_type or "unregistered",
        spec.rank,
        args.lora_alpha,
        spec.scale,
        spec.dropout,
        sorted(spec.targets),
        wrapped,
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
        hooked_embedding is not None,
    )
    if mixer_only_layers:
        shown = f"{mixer_only_layers[:4]}{'...' if len(mixer_only_layers) > 4 else ''}"
        logger.info(
            "[lora-native] %d of %d layers have no linear_qkv (linear-attention / GDN mixer) and carry no "
            "attention adapter: %s. Their mixer projections need a model-specific --lora-provider-path.",
            len(mixer_only_layers),
            len(model.decoder.layers),
            shown,
        )
    assert wrapped > 0, (
        f"native LoRA matched no modules for --target-modules {sorted(spec.targets)}; "
        "expected some of q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj"
    )
    return model


def wrap_model_provider_with_lora(provider_func, args):
    """Wrap a miles model provider so every chunk it builds gets LoRA."""

    def wrapped(*args_, **kwargs):
        return apply_native_lora(provider_func(*args_, **kwargs), args)

    return wrapped
