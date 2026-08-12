from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from miles.utils.retry_utils import NonRetryableError
from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.metadata import RpcMethodSpec
from miles.utils.workers.rpc.common.protocol import CallStatusResponse

logger = logging.getLogger(__name__)


class CallAbandonedError(NonRetryableError):
    pass


class RpcCallExecutor:
    def __init__(self, *, worker: object, specs: dict[str, RpcMethodSpec]) -> None:
        self._worker = worker
        self._executors = {
            group: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rpc-{group}")
            for group in sorted({spec.concurrency_group for spec in specs.values() if not spec.is_async})
        }
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._in_flight_lock = threading.Lock()
        self._in_flight: dict[str, _InFlightCall] = {}

    @property
    def concurrency_groups(self) -> list[str]:
        return sorted(self._executors)

    def start(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any], call_id: str, finish: Callable[..., None]) -> None:
        in_flight = _InFlightCall(method_name=spec.name, is_async=spec.is_async)
        with self._in_flight_lock:
            self._in_flight[call_id] = in_flight

        task = asyncio.create_task(
            self._run(spec=spec, kwargs=kwargs, call_id=call_id, in_flight=in_flight, finish=finish)
        )
        in_flight.task = task
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def abandon(self, *, call_id: str) -> bool:
        with self._in_flight_lock:
            in_flight = self._in_flight.get(call_id)
            if in_flight is None:
                return False
            in_flight.abandoned = True
            task = in_flight.task if in_flight.is_async else None

        if task is not None:
            task.cancel()
        return True

    def has_abandoned_call(self, *, method_name: str) -> bool:
        with self._in_flight_lock:
            return any(
                in_flight.abandoned for in_flight in self._in_flight.values() if in_flight.method_name == method_name
            )

    async def _run(
        self,
        *,
        spec: RpcMethodSpec,
        kwargs: dict[str, Any],
        call_id: str,
        in_flight: _InFlightCall,
        finish: Callable[..., None],
    ) -> None:
        started_at = time.monotonic()
        log_fields = {"tag": "rpc", "op": "execute", "method": spec.name, "call": call_id}
        log_structured(logger.debug, phase="start", **log_fields, group=spec.concurrency_group)

        try:
            try:
                result = await self._call_worker(spec=spec, kwargs=kwargs, in_flight=in_flight)
            except asyncio.CancelledError as e:
                log_structured(logger.warning, phase="end", ok=False, cancelled=True, **log_fields)
                finish(outcome=CallStatusResponse(status="failed", error=repr(e), non_retryable=True))
                raise
            except Exception as e:
                log_structured(logger.error, phase="end", ok=False, **log_fields, exc_info=True)
                finish(outcome=_failure_of(e, non_retryable=isinstance(e, NonRetryableError)))
                return

            try:
                encoded = spec.serializer.encode_result(result)
            except Exception as e:
                log_structured(logger.error, phase="end", ok=False, encoded=False, **log_fields, exc_info=True)
                finish(outcome=_failure_of(e, non_retryable=True))
                return

            log_structured(
                logger.debug, phase="end", ok=True, **log_fields, elapsed_s=round(time.monotonic() - started_at, 3)
            )
            finish(outcome=CallStatusResponse(status="success", result=encoded))
        finally:
            with self._in_flight_lock:
                self._in_flight.pop(call_id, None)

    async def _call_worker(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any], in_flight: _InFlightCall) -> Any:
        method = getattr(self._worker, spec.name)
        if spec.is_async:
            return await method(**kwargs)
        return await asyncio.get_running_loop().run_in_executor(
            self._executors[spec.concurrency_group],
            functools.partial(self._invoke_unless_abandoned, method=method, kwargs=kwargs, in_flight=in_flight),
        )

    def _invoke_unless_abandoned(
        self, *, method: Callable[..., Any], kwargs: dict[str, Any], in_flight: _InFlightCall
    ) -> Any:
        with self._in_flight_lock:
            if in_flight.abandoned:
                raise CallAbandonedError(f"{in_flight.method_name} was abandoned by its caller before it started")

        return method(**kwargs)


@dataclasses.dataclass
class _InFlightCall:
    method_name: str
    is_async: bool
    task: asyncio.Task[None] | None = None
    abandoned: bool = False


def _failure_of(error: Exception, *, non_retryable: bool) -> CallStatusResponse:
    return CallStatusResponse(
        status="failed", error="".join(traceback.format_exception(error)), non_retryable=non_retryable
    )
