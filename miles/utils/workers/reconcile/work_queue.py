# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
from collections import deque


class WorkQueue:
    def __init__(self) -> None:
        self._keys: deque[str] = deque()
        self._wakeup = asyncio.Event()
        self._shutdown = False

    def add(self, key: str) -> None:
        if self._shutdown:
            return
        if key not in self._keys:
            self._keys.append(key)
        self._wakeup.set()

    async def get(self) -> str | None:
        while not self._shutdown:
            if self._keys:
                return self._keys.popleft()
            self._wakeup.clear()
            if not self._keys:
                await self._wakeup.wait()
        return None

    def shutdown(self) -> None:
        self._shutdown = True
        self._wakeup.set()
