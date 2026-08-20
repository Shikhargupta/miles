import asyncio
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager

PARK_TIMEOUT_SECONDS = 3600.0
_POLL_SECONDS = 1.0


class Parker:
    """Parks every follower at its round boundary while the leader holds `with_all_parked()`."""

    def __init__(self, *, num_followers: int) -> None:
        self._num_running_followers = num_followers
        self._event = asyncio.Event()
        self._event.set()
        self._num_ready = 0

    @contextmanager
    def running_follower(self) -> Iterator[None]:
        try:
            yield
        finally:
            self._num_running_followers -= 1

    @asynccontextmanager
    async def with_all_parked(self) -> AsyncIterator[None]:
        await self._wait_num_ready(0)
        self._event.clear()
        try:
            await self._wait_all_parked()
            yield
        finally:
            self._event.set()

    async def maybe_park_follower(self) -> None:
        self._num_ready += 1
        await self._event.wait()
        self._num_ready -= 1

    async def _wait_all_parked(self) -> None:
        await self._wait_until(
            lambda: self._num_ready == self._num_running_followers, want=lambda: self._num_running_followers
        )

    async def _wait_num_ready(self, target: int) -> None:
        await self._wait_until(lambda: self._num_ready == target, want=lambda: target)

    async def _wait_until(self, reached: Callable[[], bool], *, want: Callable[[], int]) -> None:
        deadline = time.monotonic() + PARK_TIMEOUT_SECONDS
        while not reached():
            assert (
                time.monotonic() < deadline
            ), f"waited {PARK_TIMEOUT_SECONDS}s for the followers to park; ready {self._num_ready}, want {want()}"
            await asyncio.sleep(_POLL_SECONDS)
