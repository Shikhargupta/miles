# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
from collections import deque

from miles.utils.workers.reconcile.source_event import ParentKey


class WorkQueue:
    def __init__(self) -> None:
        self._parent_keys: deque[ParentKey] = deque()
        self._wakeup = asyncio.Event()
        self._shutdown = False

    def add(self, parent_key: ParentKey) -> None:
        if self._shutdown:
            return
        if parent_key not in self._parent_keys:
            self._parent_keys.append(parent_key)
        self._wakeup.set()

    async def get(self) -> ParentKey | None:
        while not self._shutdown:
            if self._parent_keys:
                return self._parent_keys.popleft()
            self._wakeup.clear()
            await self._wakeup.wait()
        return None

    def shutdown(self) -> None:
        self._shutdown = True
        self._wakeup.set()
