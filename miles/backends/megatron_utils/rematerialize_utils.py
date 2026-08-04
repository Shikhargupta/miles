"""Wiring for --rematerialize-param-from-master-weight: rebuild the low-precision
weights from the distributed optimizer's fp32 main params instead of a pinned CPU copy."""

import logging
from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix
from miles.utils.tensor_backper import MainCastContext

logger = logging.getLogger(__name__)


def named_restore_extras(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    """Tensors not rematerializable from fp32 master weights: expert_bias buffers and
    fp32-dtype params (their optimizer "main" is a view of the param itself)."""
    for vp_stage, model_module in enumerate(model):
        for name, buffer in model_module.named_buffers():
            if "expert_bias" in name:
                yield f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}", buffer
        for name, param in model_module.named_parameters():
            if param.dtype == torch.float32:
                yield f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}", param


def build_main_cast_context(args: Namespace, model: Sequence[torch.nn.Module], optimizer) -> MainCastContext:
    extras = list(named_restore_extras(model))
    extras_bytes = sum(t.numel() * t.element_size() for _, t in extras)
    logger.info(
        f"rematerialize-param-from-master-weight: {len(extras)} extra tensors "
        f"({extras_bytes / 2**20:.1f} MiB) kept in pinned backup: "
        f"{[name for name, _ in extras[:20]]}"
    )
    return MainCastContext(
        optimizer=optimizer,
        model_chunks=model,
        extras_getter=lambda: named_restore_extras(model),
        rematerializable_ids=_assert_rematerialize_coverage(model, extras),
        check=args.check_rematerialize_param_from_master_weight,
    )


def _assert_rematerialize_coverage(model: Sequence[torch.nn.Module], extras: list[tuple[str, torch.Tensor]]) -> set:
    """A param outside the DDP buffers (restored via cast + all-gather) and
    the extras backup would silently come back as garbage. Optimizer-side
    structures only cover this rank's owned shard under DP>1."""
    restorable = {id(t) for _, t in extras}
    for model_module in model:
        for buffer in model_module.buffers + model_module.expert_parallel_buffers:
            restorable.update(id(p) for p in buffer.params)
    uncovered = []
    for model_module in model:
        for name, param in model_module.named_parameters():
            if id(param) not in restorable:
                uncovered.append(name)
    assert not uncovered, (
        f"--rematerialize-param-from-master-weight cannot restore {len(uncovered)} params "
        f"(not in the DDP param buffers nor in the extras backup): {uncovered[:10]}"
    )
    return restorable
