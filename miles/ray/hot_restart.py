from __future__ import annotations

import asyncio
import logging
import time
from argparse import Namespace
from collections.abc import Coroutine
from typing import Any, TypeVar

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.client.misc import RETRYABLE_ERRORS, ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError

logger = logging.getLogger(__name__)

TAKE_OVER_GATE_TIMEOUT_SECONDS = 600.0
EXECUTOR_FREE_TIMEOUT_SECONDS = 1800.0

_EXECUTOR_POLL_INTERVAL_SECONDS = 5.0
_TRAINERS_GATE = "the trainers"
_TRAINER_STATE_GATE = "the trainer state"
_INFERENCE_CONTROLLER_GATE = "the inference controller"

_T = TypeVar("_T")


async def _within_gate(call: Coroutine[Any, Any, _T], *, remaining: float, gate: str, action: str) -> _T:
    try:
        return await asyncio.wait_for(call, timeout=remaining)
    except (TimeoutError, asyncio.TimeoutError) as e:
        raise TimeoutError(
            f"Taking over {gate} ran out of its {TAKE_OVER_GATE_TIMEOUT_SECONDS}s budget while trying to {action} "
            f"({e!r}); a take-over is a bounded gate, so this run stops instead of driving a system it does not own"
        ) from e


async def quiesce_and_claim_trainers(handles: dict[str, BaseWorkerHandle]) -> None:
    """Gate 1: let every surviving trainer finish the call it is running, then take ownership of it."""
    assert handles, "a run drives at least one trainer, so a take-over with no trainer to claim is a wiring bug"

    expires_at = time.monotonic() + TAKE_OVER_GATE_TIMEOUT_SECONDS

    resumed = [
        await _quiesce_and_claim_trainer(handle, trainer_id=trainer_id, expires_at=expires_at)
        for trainer_id, handle in handles.items()
    ]

    assert len(set(resumed)) == 1, (
        f"the trainers of this run disagree about whether a previous orchestration script already initialized them "
        f"({resumed}); a take-over drives all of them or none of them, so this run stops before gate 2 aborts a "
        f"fleet and resets a broadcast lock on behalf of a run it cannot resume"
    )


async def _quiesce_and_claim_trainer(trainer: BaseWorkerHandle, *, trainer_id: str, expires_at: float) -> bool:
    await trainer.claim_driver_epoch()

    if resumed := await trainer.is_initialized():
        logger.info(
            f"Trainer {trainer_id!r} is already initialized, so a previous orchestration script built it; "
            f"waiting until it finished whatever it is still running before taking it over"
        )
        remaining = max(expires_at - time.monotonic(), 0.0)
        await _within_gate(
            trainer.wait_idle(timeout=remaining),
            remaining=remaining,
            gate=_TRAINERS_GATE,
            action=(
                f"wait until trainer {trainer_id!r} finished the call of the previous orchestration script, "
                f"because reloading a checkpoint into a model a train step is still writing would corrupt it"
            ),
        )

    return resumed


async def init_or_load_trainer(
    trainer: BaseWorkerHandle, model_args: Namespace, *, trainer_id: str, resumed: bool
) -> list[Any]:
    if not resumed:
        return await trainer.init(model_args)

    start_rollout_ids = await _within_gate(
        trainer.load_state(),
        remaining=TAKE_OVER_GATE_TIMEOUT_SECONDS,
        gate=_TRAINER_STATE_GATE,
        action=f"roll trainer {trainer_id!r} back to its checkpoint",
    )
    logger.info(f"Resumed the already-initialized trainer {trainer_id!r} at rollout ids {start_rollout_ids}")
    return start_rollout_ids


async def wait_until_rollout_executor_is_free(
    handle: BaseWorkerHandle, *, timeout: float = EXECUTOR_FREE_TIMEOUT_SECONDS
) -> None:
    """Wait until the rollout executor answering us is a fresh one, not the process the previous script drove."""

    expires_at = time.monotonic() + timeout

    while True:
        remaining = max(expires_at - time.monotonic(), 0.0)
        cause: BaseException | None = None
        try:
            try:
                if not await handle.is_initialized():
                    return
                reason = "the rollout executor still belongs to the previous orchestration script"
            except ServerRestartedError as e:
                cause = e
                reason = "the rollout executor is being replaced right now"
                await handle.wait_ready(timeout=min(remaining, _EXECUTOR_POLL_INTERVAL_SECONDS))
        except (WorkerUnreachableError, *RETRYABLE_ERRORS) as e:
            cause = e
            reason = repr(e)

        log_structured(
            logger.info,
            tag="hot_restart",
            op="wait_rollout_executor_free",
            reason=reason,
            remaining_s=round(max(expires_at - time.monotonic(), 0.0), 1),
        )

        if time.monotonic() >= expires_at:
            raise TimeoutError(
                f"The rollout executor answering us was still not a fresh process after {timeout}s ({reason}); a hot "
                f"restart replaces its pod, and everything up to and including that replacement has to fit in this "
                f"budget, so either the pod is not being replaced or the previous script's executor never went away"
            ) from cause
        await asyncio.sleep(_EXECUTOR_POLL_INTERVAL_SECONDS)


async def init_or_reset_inference_controller(inference_controller: BaseWorkerHandle) -> None:
    """Gate 2: initialize the inference side, or reset the one a previous orchestration script left running."""
    if not await inference_controller.is_initialized():
        await inference_controller.init()
        await inference_controller.claim_driver_epoch()
        return

    logger.info("The inference controller outlived a previous orchestration script; taking it over as it is")
    expires_at = time.monotonic() + TAKE_OVER_GATE_TIMEOUT_SECONDS

    remaining = max(expires_at - time.monotonic(), 0.0)
    await _within_gate(
        inference_controller.wait_idle(timeout=remaining),
        remaining=remaining,
        gate=_INFERENCE_CONTROLLER_GATE,
        action="wait for the calls of the previous orchestration script to end",
    )
    await _reset_broadcast_lock(inference_controller, expires_at=expires_at)

    remaining = max(expires_at - time.monotonic(), 0.0)
    await _within_gate(
        inference_controller.wait_expected_num_cells(timeout=remaining),
        remaining=remaining,
        gate=_INFERENCE_CONTROLLER_GATE,
        action="wait for every engine this run expects to be in the fleet it takes over",
    )
    await _abort_inflight_rollouts(inference_controller, expires_at=expires_at)
    await inference_controller.claim_driver_epoch()


async def _reset_broadcast_lock(inference_controller: BaseWorkerHandle, *, expires_at: float) -> None:
    """A script that died between start_update_weights and end_update_weights left the lock detached and held."""
    reset = await _within_gate(
        inference_controller.reset_broadcast_lock(),
        remaining=max(expires_at - time.monotonic(), 0.0),
        gate=_INFERENCE_CONTROLLER_GATE,
        action="reset the broadcast lock the previous orchestration script left held",
    )
    if reset:
        logger.warning(
            "The previous orchestration script stopped inside a weight update, so the inference controller was "
            "still holding the broadcast lock that update took; it is released and health checking is resumed, and "
            "the full update-weights pass this script starts with repairs whatever that broadcast half-wrote"
        )


async def _abort_inflight_rollouts(inference_controller: BaseWorkerHandle, *, expires_at: float) -> None:
    """Stop every generation a previous orchestration script left behind; this run owns the fleet now."""
    await _within_gate(
        inference_controller.abort_all(),
        remaining=max(expires_at - time.monotonic(), 0.0),
        gate=_INFERENCE_CONTROLLER_GATE,
        action="abort the generations the previous orchestration script left in flight",
    )

    logger.info("Aborted every in-flight generation, so this run starts from a quiet inference fleet")
