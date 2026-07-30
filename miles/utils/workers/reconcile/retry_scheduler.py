# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import logging
import math

from miles.utils.test_utils.clock import Clock
from miles.utils.workers.reconcile.source_event import ParentKey
from miles.utils.workers.reconcile.work_queue import WorkQueue

logger = logging.getLogger(__name__)


class RetryScheduler:
    def __init__(self, *, queue: WorkQueue, failure_base_delay: float, failure_max_delay: float, clock: Clock) -> None:
        assert failure_base_delay > 0, f"{failure_base_delay=} must be positive"
        assert failure_max_delay >= failure_base_delay, f"{failure_max_delay=} must be >= {failure_base_delay=}"

        self._queue = queue
        self._failure_base_delay = failure_base_delay
        self._failure_max_delay = failure_max_delay
        self._max_backoff_exponent = max(0, math.ceil(math.log2(failure_max_delay / failure_base_delay)))
        self._clock = clock

        self._failures: dict[ParentKey, int] = {}
        self._timers: dict[ParentKey, asyncio.Task[None]] = {}
        self._shutdown = False

    def note_failure(self, parent_key: ParentKey) -> None:
        if self._shutdown:
            return
        failures = self._failures.get(parent_key, 0) + 1
        self._failures[parent_key] = failures
        exponent = min(failures - 1, self._max_backoff_exponent)
        delay = min(self._failure_base_delay * 2**exponent, self._failure_max_delay)

        self._cancel_timer(parent_key)
        self._timers[parent_key] = asyncio.create_task(self._fire_after(parent_key=parent_key, delay=delay))

    def note_success(self, parent_key: ParentKey) -> None:
        self._failures.pop(parent_key, None)
        self._cancel_timer(parent_key)

    async def shutdown(self) -> None:
        self._shutdown = True

        timers = list(self._timers.values())
        self._timers = {}
        for timer in timers:
            timer.cancel()
        await asyncio.gather(*timers, return_exceptions=True)

    def _cancel_timer(self, parent_key: ParentKey) -> None:
        pending = self._timers.pop(parent_key, None)
        if pending is not None:
            pending.cancel()

    async def _fire_after(self, *, parent_key: ParentKey, delay: float) -> None:
        await self._clock.sleep(delay)
        if self._timers.get(parent_key) is asyncio.current_task():
            del self._timers[parent_key]
        self._queue.add(parent_key)
