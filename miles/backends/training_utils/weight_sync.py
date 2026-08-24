"""The rollout-engine side of a weight push, shared by every training backend.

What a weight push looks like from sglang's perspective is a property of sglang,
not of the trainer: pause generation, drop the prefix cache, open an update
session, receive weights, close the session, resume. Only the middle -- how a
backend turns its sharded parameters into HF-named tensors on the wire -- is
backend-specific.

Keeping the handshake here means a protocol change lands once. There were three
copies before this module, and they had already drifted:

* the FSDP path called ``pause_generation()`` with no arguments, silently
  ignoring ``--pause-generation-mode``;
* only Megatron's distributed path skipped the cache flush under ``in_place``
  pausing -- the other two flushed unconditionally, discarding the KV cache that
  ``in_place`` exists to preserve.

This module adopts the correct behavior in both cases. The flush change is a
no-op for the ``abort`` and ``retract`` modes (including the default), and a
fix for ``in_place``.
"""

import logging
import random
from argparse import Namespace
from collections.abc import Sequence
from contextlib import contextmanager

import ray
import torch.distributed as dist
from ray.actor import ActorHandle

from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)


def begin_weight_update(rollout_engines: Sequence[ActorHandle], selector: str = "all") -> None:
    """Open a weight-update session on the selected engines (restores packed weights)."""
    ray.get([engine.begin_weight_update.remote(selector=selector) for engine in rollout_engines])


def end_weight_update(rollout_engines: Sequence[ActorHandle]) -> None:
    """Close the session (post-load + quantization post-process on the full model)."""
    ray.get([engine.end_weight_update.remote() for engine in rollout_engines])


def weight_update_selector(args: Namespace) -> str:
    """Exclude the draft model only when the trainer provably has no MTP block to send."""
    if (
        getattr(args, "sglang_speculative_algorithm", None)
        and not getattr(args, "mtp_num_layers", None)
        and getattr(args, "megatron_to_hf_mode", "raw") != "bridge"
    ):
        return "target"
    return "all"


@contextmanager
def weight_push_session(
    args: Namespace,
    rollout_engines: Sequence[ActorHandle],
    *,
    announce: bool = True,
):
    """Hold the rollout engines quiesced for the duration of a weight push.

    Rank 0 drives the engine RPCs; the surrounding barriers keep every trainer
    rank inside the window so no rank streams weights into an engine that is
    still generating. ``announce=False`` skips the begin/end session markers for
    a push that sends no fresh base weights (the LoRA-only path), where the
    engine must not run its post-load step.
    """
    is_driver = dist.get_rank() == 0
    mode = args.pause_generation_mode

    if is_driver:
        ray.get([engine.pause_generation.remote(mode=mode) for engine in rollout_engines])
        # in_place freezes requests and resumes them against their existing KV
        # cache, so flushing here would discard exactly what that mode preserves.
        if mode != "in_place":
            ray.get([engine.flush_cache.remote() for engine in rollout_engines])
        if announce:
            begin_weight_update(rollout_engines, weight_update_selector(args))
    dist.barrier(group=get_gloo_group())

    try:
        yield
    finally:
        dist.barrier(group=get_gloo_group())
        if is_driver:
            if announce:
                end_weight_update(rollout_engines)
            ray.get([engine.continue_generation.remote() for engine in rollout_engines])
        dist.barrier(group=get_gloo_group())


def connect_engines_if_stale(
    weight_updater,
    rollout_manager,
    info,
) -> None:
    """(Re)establish the transport's engine connection when the fleet changed.

    ``info`` is the ``EnginesAndLock`` snapshot the driver passed down. The
    has-new-engines flag is cleared on the manager only after every rank has
    connected, so a rank that has not yet reconnected cannot observe it as clear.
    """
    if not info.has_new_engines and _is_fresh(weight_updater):
        return

    weight_updater.connect_rollout_engines(
        info.rollout_engines,
        info.rollout_engine_lock,
        engine_gpu_counts=info.engine_gpu_counts,
        engine_gpu_offsets=info.engine_gpu_offsets,
    )
    dist.barrier(group=get_gloo_group())
    if dist.get_rank() == 0:
        ray.get(rollout_manager.clear_updatable_has_new_engines.remote())


def _is_fresh(weight_updater) -> bool:
    """Transports that can go stale without the fleet changing expose this; the
    others are always fresh once connected."""
    checker = getattr(weight_updater, "is_rollout_engines_fresh", None)
    return checker() if checker is not None else True


def verify_engine_weight_version(weight_updater, rollout_engines: Sequence[ActorHandle]) -> None:
    """CI guard: confirm an engine actually took the version we just pushed."""
    if not rollout_engines:
        return

    engine = random.choice(list(rollout_engines))
    engine_version = ray.get(engine.get_weight_version.remote())
    if str(engine_version) != str(weight_updater.weight_version):
        raise RuntimeError(
            f"Weight version mismatch! Engine: {engine_version}, Updater: {weight_updater.weight_version}"
        )
