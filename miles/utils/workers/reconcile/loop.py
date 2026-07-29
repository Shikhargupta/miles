# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from miles.utils.test_utils.clock import Clock, RealClock
from miles.utils.workers.reconcile.object_store import KeyMapFn, ObjectStore
from miles.utils.workers.reconcile.retry_scheduler import RetryScheduler
from miles.utils.workers.reconcile.source_event import SourceEvent, SourceWatchFn
from miles.utils.workers.reconcile.source_stream_driver import SourceStreamDriver
from miles.utils.workers.reconcile.work_queue import WorkQueue

logger = logging.getLogger(__name__)

ReconcileFn = Callable[[str], Awaitable[None]]


class ReconcileLoop:
    """A source stream feeds a store; every changed parent key is reconciled once, level-triggered.

    `source` returns an async iterator of `SourceEvent`. A stream opens with `SyncStart`, closed by `SyncDone`:
    events between them are buffered and applied as a whole-store replace that synthesizes deletions, while
    events outside a segment apply immediately. Reconcile receives a key only and re-derives everything from
    `get_by_parent()`.
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
        self._queue = WorkQueue()
        self._retry = RetryScheduler(
            queue=self._queue,
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

        self._started = False
        self._stopped = False
        self._sync_task: asyncio.Task[AsyncGenerator[SourceEvent, None]] | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        assert not self._started, "ReconcileLoop.start() must be called exactly once"
        self._started = True
        if self._stopped:
            return

        self._sync_task = asyncio.create_task(self._driver.open_synced_stream())
        try:
            stream = await self._sync_task
        except asyncio.CancelledError:
            self._sync_task.cancel()
            if self._stopped:
                return
            raise
        if self._stopped:
            return

        self._tasks = [asyncio.create_task(self._worker_loop()), asyncio.create_task(self._driver.run(stream))]
        if self._resync_period is not None:
            self._tasks.append(asyncio.create_task(self._resync_loop()))

    async def stop(self) -> None:
        assert asyncio.current_task() not in self._tasks, (
            "ReconcileLoop.stop() waits for the worker, so it cannot be awaited from inside reconcile; "
            "call asyncio.create_task(loop.stop()) instead"
        )

        self._stopped = True
        self._queue.shutdown()
        self._retry.shutdown()
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._teardown(self._running_tasks()))
        await asyncio.shield(self._stop_task)

    def _running_tasks(self) -> list[asyncio.Task[Any]]:
        candidates = [*self._tasks, *self._retry.pending_timers(), self._sync_task]
        return [task for task in candidates if task is not None]

    async def _teardown(self, tasks: list[asyncio.Task[Any]]) -> None:
        for task in tasks:
            task.cancel()
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self._driver.aclose()
            self._tasks = []
            self._retry.drop_timers()
            self._sync_task = None

    def get_by_parent(self, parent_key: str) -> list[Any]:
        return self._store.get_by_parent(parent_key)

    def _enqueue_all(self, keys: set[str]) -> None:
        for key in sorted(keys):
            self._queue.add(key)

    async def _worker_loop(self) -> None:
        while True:
            key = await self._queue.get()
            if key is None:
                return
            try:
                await self._reconcile(key)
                self._retry.note_success(key)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.error(f"ReconcileLoop reconcile failed {key=}", exc_info=True)
                self._retry.note_failure(key)

    async def _resync_loop(self) -> None:
        assert self._resync_period is not None
        while True:
            await self._clock.sleep(self._resync_period)
            self._enqueue_all(self._store.parent_keys())
