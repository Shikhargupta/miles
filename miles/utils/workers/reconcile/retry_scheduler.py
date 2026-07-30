# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

from miles.utils.test_utils.clock import Clock

KeyT = TypeVar("KeyT", bound=Hashable)


class RetryScheduler(Generic[KeyT]):
    def __init__(
        self,
        *,
        on_retry: Callable[[KeyT], None],
        failure_base_delay: float,
        failure_max_delay: float,
        clock: Clock,
    ) -> None:
        assert failure_base_delay > 0, f"{failure_base_delay=} must be positive"
        assert failure_max_delay >= failure_base_delay, f"{failure_max_delay=} must be >= {failure_base_delay=}"

        self._on_retry = on_retry
        self._failure_base_delay = failure_base_delay
        self._failure_max_delay = failure_max_delay
        self._max_backoff_exponent = max(0, math.ceil(math.log2(failure_max_delay / failure_base_delay)))
        self._clock = clock

        self._failures: dict[KeyT, int] = {}
        self._timers: dict[KeyT, asyncio.Task[None]] = {}
        self._shutdown = False

    def note_failure(self, key: KeyT) -> None:
        if self._shutdown:
            return
        failures = self._failures.get(key, 0) + 1
        self._failures[key] = failures
        exponent = min(failures - 1, self._max_backoff_exponent)
        delay = min(self._failure_base_delay * 2**exponent, self._failure_max_delay)

        self._cancel_timer(key)
        self._timers[key] = asyncio.create_task(self._fire_after(key=key, delay=delay))

    def note_success(self, key: KeyT) -> None:
        self._failures.pop(key, None)
        self._cancel_timer(key)

    async def shutdown(self) -> None:
        self._shutdown = True

        timers = list(self._timers.values())
        self._timers = {}
        for timer in timers:
            timer.cancel()
        await asyncio.gather(*timers, return_exceptions=True)

    def _cancel_timer(self, key: KeyT) -> None:
        pending = self._timers.pop(key, None)
        if pending is not None:
            pending.cancel()

    async def _fire_after(self, *, key: KeyT, delay: float) -> None:
        await self._clock.sleep(delay)
        if self._timers.get(key) is asyncio.current_task():
            del self._timers[key]
        self._on_retry(key)
