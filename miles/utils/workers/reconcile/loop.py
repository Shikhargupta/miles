# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from miles.utils.test_utils.clock import Clock, RealClock
from miles.utils.workers.reconcile.object_store import KeyMapFn, ObjectStore
from miles.utils.workers.reconcile.retry_scheduler import RetryScheduler
from miles.utils.workers.reconcile.source_event import ParentKey, SourceWatchFn
from miles.utils.workers.reconcile.source_stream_driver import SourceStreamDriver
from miles.utils.workers.reconcile.work_queue import WorkQueue

logger = logging.getLogger(__name__)

ReconcileFn = Callable[[ParentKey], Awaitable[None]]


class ReconcileLoop:
    """A source stream feeds a store; every changed parent key is reconciled once, level-triggered.

    - `source` returns an async iterator of `SourceEvent`.
    - A stream opens with `ReplaceEvent`, a whole-store replace that synthesizes deletions.
    - Later `UpsertEvent` and `DeleteEvent` apply immediately; a relist sends another `ReplaceEvent`.
    - `reconcile` receives a key only and re-derives everything from `get_by_parent()`.
    - `reconcile` must not block on I/O: one worker serves every parent key.
    - `reconcile` must be idempotent: delivery is at-least-once.
    - Objects handed out are the source's own, so treat them as read-only.
    """

    def __init__(
        self,
        *,
        source: SourceWatchFn,
        reconcile: ReconcileFn,
        key_map: KeyMapFn | None = None,
        resync_period: float | None = None,
        failure_base_delay: float = 1.0,
        failure_max_delay: float = 60.0,
        source_retry_delay: float = 1.0,
        clock: Clock | None = None,
    ) -> None:
        assert resync_period is None or resync_period > 0, f"{resync_period=} must be positive or None"
        assert source_retry_delay > 0, f"{source_retry_delay=} must be positive"

        self._reconcile = reconcile
        self._resync_period = resync_period
        self._clock = clock or RealClock()

        self._store = ObjectStore(key_map=key_map)
        self._queue: WorkQueue[ParentKey] = WorkQueue()
        self._retry: RetryScheduler[ParentKey] = RetryScheduler(
            on_retry=self._queue.add,
            failure_base_delay=failure_base_delay,
            failure_max_delay=failure_max_delay,
            clock=self._clock,
        )
        self._driver = SourceStreamDriver(
            source=source,
            store=self._store,
            on_affected=self._enqueue_all,
            retry_delay=source_retry_delay,
            clock=self._clock,
        )

        self._start_called = False
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        assert not self._start_called, "ReconcileLoop.start() must be called exactly once"
        self._start_called = True

        driver_task = asyncio.create_task(self._driver.run())
        try:
            await self._driver.wait_for_sync()
        except asyncio.CancelledError:
            driver_task.cancel()
            await asyncio.gather(driver_task, return_exceptions=True)
            raise

        self._tasks = [asyncio.create_task(self._worker_loop()), driver_task]
        if self._resync_period is not None:
            self._tasks.append(asyncio.create_task(self._resync_loop()))

    async def stop(self) -> None:
        assert self._tasks, "ReconcileLoop.stop() must come after start(); abort a hung start() by cancelling its task"
        assert asyncio.current_task() not in self._tasks, (
            "ReconcileLoop.stop() waits for the worker, so it cannot be awaited from inside reconcile; "
            "call asyncio.create_task(loop.stop()) instead"
        )

        self._queue.shutdown()
        await self._retry.shutdown()

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        await self._driver.aclose()
        self._tasks = []

    def get_by_parent(self, parent_key: ParentKey) -> list[Any]:
        return self._store.get_by_parent(parent_key)

    def _enqueue_all(self, parent_keys: set[ParentKey]) -> None:
        for parent_key in sorted(parent_keys):
            self._queue.add(parent_key)

    async def _worker_loop(self) -> None:
        while True:
            parent_key = await self._queue.get()
            if parent_key is None:
                return
            try:
                await self._reconcile(parent_key)
                self._retry.note_success(parent_key)
            except Exception:
                logger.error(f"ReconcileLoop reconcile failed {parent_key=}", exc_info=True)
                self._retry.note_failure(parent_key)

    async def _resync_loop(self) -> None:
        assert self._resync_period is not None
        while True:
            await self._clock.sleep(self._resync_period)
            self._enqueue_all(self._store.parent_keys())
