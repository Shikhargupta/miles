# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable

from miles.utils.test_utils.clock import Clock
from miles.utils.workers.reconcile.object_store import ObjectStore
from miles.utils.workers.reconcile.source_event import ParentKey, SourceEvent, SourceWatchFn, SyncDone, SyncStart

logger = logging.getLogger(__name__)


class SourceStreamDriver:
    def __init__(
        self,
        *,
        source: SourceWatchFn,
        store: ObjectStore,
        on_affected: Callable[[set[ParentKey]], None],
        retry_delay: float,
        clock: Clock,
    ) -> None:
        self._source = source
        self._store = store
        self._on_affected = on_affected
        self._retry_delay = retry_delay
        self._clock = clock
        self._stream: AsyncGenerator[SourceEvent, None] | None = None

    async def open_synced_stream(self) -> AsyncGenerator[SourceEvent, None]:
        while True:
            self._store.reset_segment()
            stream: AsyncGenerator[SourceEvent, None] | None = None
            try:
                stream = self._source()
                await self._consume_until_synced(stream)
                self._stream = stream
                return stream
            except asyncio.CancelledError:
                await _aclose(stream)
                raise
            except Exception:
                logger.error("SourceStreamDriver initial sync failed, retrying", exc_info=True)
                await _aclose(stream)
                await self._clock.sleep(self._retry_delay)

    async def run(self, stream: AsyncGenerator[SourceEvent, None]) -> None:
        while True:
            try:
                async for event in stream:
                    self._apply(event)
            except asyncio.CancelledError:
                await _aclose(stream)
                raise
            except Exception:
                logger.error("SourceStreamDriver source stream failed, reopening", exc_info=True)
            else:
                logger.warning("SourceStreamDriver source stream ended, reopening")
            await _aclose(stream)
            self._stream = None
            await self._clock.sleep(self._retry_delay)
            stream = await self.open_synced_stream()

    async def aclose(self) -> None:
        await _aclose(self._stream)
        self._stream = None

    async def _consume_until_synced(self, stream: AsyncGenerator[SourceEvent, None]) -> None:
        first = True
        async for event in stream:
            if first and not isinstance(event, SyncStart):
                raise RuntimeError(f"A source stream must open with SyncStart, got {event=}")
            first = False
            if self._apply(event):
                return
        raise RuntimeError("Source stream ended before the initial sync completed")

    def _apply(self, event: SourceEvent) -> bool:
        update = self._store.handle_event(event)
        self._on_affected(update.affected_parents)
        return isinstance(event, SyncDone)


async def _aclose(stream: AsyncGenerator[SourceEvent, None] | None) -> None:
    if stream is None:
        return
    try:
        await stream.aclose()
    except Exception:
        logger.error("SourceStreamDriver failed to close a source stream", exc_info=True)
