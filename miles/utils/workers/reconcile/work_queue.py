# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio


class WorkQueue:
    def __init__(self) -> None:
        self._keys: dict[str, None] = {}
        self._wakeup = asyncio.Event()
        self._shutdown = False

    def add(self, key: str) -> None:
        if self._shutdown:
            return
        self._keys[key] = None
        self._wakeup.set()

    async def get(self) -> str | None:
        while not self._shutdown:
            if self._keys:
                key = next(iter(self._keys))
                del self._keys[key]
                return key
            self._wakeup.clear()
            if not self._keys:
                await self._wakeup.wait()
        return None

    def shutdown(self) -> None:
        self._shutdown = True
        self._wakeup.set()
