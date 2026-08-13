from __future__ import annotations

import contextlib
import functools
import inspect
import logging
from collections.abc import Callable, Iterator
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


def init_once_guarded(fn: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            with self._init_once.guard():
                return await fn(self, *args, **kwargs)

    else:

        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            with self._init_once.guard():
                return fn(self, *args, **kwargs)

    return wrapper


class _InitState(Enum):
    NOT_STARTED = "not started"
    INITIALIZING = "initializing"
    COMPLETE = "complete"
    FAILED = "failed"


class InitOnce:
    def __init__(self, *, component: str) -> None:
        self._component = component
        self._state = _InitState.NOT_STARTED

    @property
    def is_initialized(self) -> bool:
        return self._state is _InitState.COMPLETE

    @contextlib.contextmanager
    def guard(self) -> Iterator[None]:
        assert self._state is _InitState.NOT_STARTED, (
            f"{self._component} is {self._state.value} in this process, and initializing it again would "
            f"re-initialize a live system behind the back of whoever is already driving it; a restarted "
            f"orchestration script has to resume a complete component instead of initializing it again, and a "
            f"component whose init failed or never finished has to be replaced before anything drives it"
        )
        self._state = _InitState.INITIALIZING
        try:
            yield
        except BaseException:
            self._state = _InitState.FAILED
            logger.error(f"Initializing {self._component} failed, so nothing may drive it", exc_info=True)
            raise
        self._state = _InitState.COMPLETE
        logger.info(f"{self._component} is now initialized")

    def assert_initialized(self) -> None:
        assert self.is_initialized, f"{self._component} is {self._state.value}, not initialized"
