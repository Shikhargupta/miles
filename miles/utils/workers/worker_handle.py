from __future__ import annotations

import abc
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

_WAIT_DEAD_PROBE_INTERVAL_SECONDS = 1.0


class WorkerUnreachableError(Exception):
    pass


class BaseWorkerHandle(abc.ABC):
    @abc.abstractmethod
    async def wait_ready(self, *, timeout: float) -> None: ...

    async def wait_dead(self, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if await self._probe_is_dead_within(deadline=deadline):
                return True
            if time.monotonic() >= deadline:
                logger.error(
                    "Timed out after %.0fs waiting for %r to die, so its death stays unconfirmed and its caller has "
                    "to keep treating it as a process that may still be running",
                    timeout,
                    self,
                )
                return False
            await asyncio.sleep(_WAIT_DEAD_PROBE_INTERVAL_SECONDS)

    async def _probe_is_dead_within(self, *, deadline: float) -> bool:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False
        try:
            return await asyncio.wait_for(self._probe_is_dead(), timeout=remaining)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(
                "Probing whether %r is dead did not answer within the %.1fs its caller had left, so its death stays "
                "unconfirmed rather than the probe outrunning the budget it was given",
                self,
                remaining,
            )
            return False

    @abc.abstractmethod
    async def _probe_is_dead(self) -> bool: ...
