"""Native adapter checkpoint helpers.

Shared save/load orchestration and PP assembly still serve both native and
Megatron-Bridge paths in ``miles.backends.megatron_utils.lora_utils``. They stay
there until the bridge refactor; this module owns only native-specific pieces.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from miles_plugins.lora.modules.linear import iter_adapters


def native_adapter_shard_name(tp_rank: int, pp_rank: int, ep_rank: int) -> str:
    """Return the per-rank native adapter filename, including EP when sharded."""
    suffix = f"_ep{ep_rank}" if ep_rank > 0 else ""
    return f"adapter_megatron_tp{tp_rank}_pp{pp_rank}{suffix}.pt"


def native_adapter_state_dict(model_chunks: Sequence[nn.Module]) -> dict[str, torch.Tensor]:
    """Collect local native-LoRA parameters with their current model-tree names."""
    native_parameter_ids = {
        id(parameter) for adapter in iter_adapters(model_chunks) for parameter in adapter.parameters(recurse=False)
    }
    state: dict[str, torch.Tensor] = {}
    for chunk in model_chunks:
        for name, parameter in chunk.named_parameters():
            if id(parameter) in native_parameter_ids:
                state[name] = parameter.detach().cpu()
    return state


def load_native_adapter_state_dict(
    model_chunks: Sequence[nn.Module],
    state_dict: dict[str, torch.Tensor],
) -> tuple[int, list[str], list[str]]:
    """Load a local native shard and report both target-set mismatch directions.

    Returns ``(loaded, unexpected, missing)``: *unexpected* are checkpoint
    tensors absent from the current exact target set, *missing* are current
    adapter parameters the checkpoint has no tensor for (they keep their fresh
    initialization). Either direction means the shard was saved for a different
    ``--target-modules`` set.
    """
    native_parameter_ids = {
        id(parameter) for adapter in iter_adapters(model_chunks) for parameter in adapter.parameters(recurse=False)
    }
    current_names = set()
    loaded = 0
    for chunk in model_chunks:
        for name, parameter in chunk.named_parameters():
            if id(parameter) not in native_parameter_ids:
                continue
            current_names.add(name)
            if name not in state_dict:
                continue
            parameter.data.copy_(state_dict[name].to(device=parameter.device, dtype=parameter.dtype))
            loaded += 1
    return loaded, sorted(set(state_dict) - current_names), sorted(current_names - set(state_dict))


def has_native_adapters(model_chunks: Sequence[nn.Module]) -> bool:
    return next(iter(iter_adapters(model_chunks)), None) is not None
