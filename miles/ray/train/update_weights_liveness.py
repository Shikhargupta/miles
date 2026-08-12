from __future__ import annotations

import functools
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

UPDATE_WEIGHTS_LIVENESS_CONCURRENCY_GROUP = "update_weights_liveness"
UPDATE_WEIGHTS_LIVENESS_DEADLINE_SECONDS = 600.0
_UPDATE_WEIGHTS_STOP_CONFIRMATION_GRACE_SECONDS = 60.0
UPDATE_WEIGHTS_STOP_CONFIRMATION_TIMEOUT_SECONDS = (
    UPDATE_WEIGHTS_LIVENESS_DEADLINE_SECONDS + _UPDATE_WEIGHTS_STOP_CONFIRMATION_GRACE_SECONDS
)

_T = TypeVar("_T")


class UpdateWeightsLiveness:
    def __init__(self) -> None:
        self._in_flight = threading.Event()

    def is_in_flight(self) -> bool:
        return self._in_flight.is_set()

    @contextmanager
    def marked(self) -> Iterator[None]:
        self._in_flight.set()
        try:
            yield
        finally:
            self._in_flight.clear()


def marks_update_weights_in_flight(fn: Callable[..., _T]) -> Callable[..., _T]:
    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> _T:
        with self._update_weights_liveness.marked():
            return fn(self, *args, **kwargs)

    return wrapper
