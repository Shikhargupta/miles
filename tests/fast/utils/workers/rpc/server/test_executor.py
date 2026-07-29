import asyncio
import logging
from typing import Any

import pytest

from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.rpc.common.protocol import CallStatusResponse
from miles.utils.workers.rpc.server import executor as executor_module
from miles.utils.workers.rpc.server.executor import RpcCallExecutor


class _BlockingWorker:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self, value: int) -> int:
        self.started.set()
        await self.release.wait()
        return value


class _CancelResistantWorker:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def run(self) -> str:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.release.wait()
        return "done"


def _make_executor(*, worker: object) -> RpcCallExecutor:
    return RpcCallExecutor(worker=worker, specs=collect_rpc_method_specs(type(worker)))


class TestRpcCallExecutorShutdown:
    async def test_shutdown_cancels_an_in_flight_call(self) -> None:
        """Shutdown cancels an in-flight call instead of waiting for it."""
        worker = _BlockingWorker()
        executor = _make_executor(worker=worker)
        outcomes: list[CallStatusResponse] = []
        executor.start(
            spec=collect_rpc_method_specs(_BlockingWorker)["run"],
            kwargs={"value": 7},
            call_id="c1",
            finish=lambda **kwargs: outcomes.append(kwargs["outcome"]),
        )
        await asyncio.wait_for(worker.started.wait(), timeout=1.0)

        await asyncio.wait_for(executor.shutdown(), timeout=1.0)

        assert worker.release.is_set() is False
        assert [outcome.status for outcome in outcomes] == ["failed"]

    async def test_shutdown_returns_after_cancel_grace_for_unresponsive_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Shutdown abandons a task that ignores cancellation past its grace."""
        monkeypatch.setattr(executor_module, "_CANCEL_GRACE_SECONDS", 0.01)
        worker = _CancelResistantWorker()
        executor = _make_executor(worker=worker)
        outcomes: list[CallStatusResponse] = []

        def finish(**kwargs: Any) -> None:
            outcomes.append(kwargs["outcome"])

        executor.start(
            spec=collect_rpc_method_specs(_CancelResistantWorker)["run"],
            kwargs={},
            call_id="c1",
            finish=finish,
        )
        await asyncio.wait_for(worker.started.wait(), timeout=1.0)

        with caplog.at_level(logging.ERROR, logger="miles.utils.workers.rpc.server.executor"):
            await asyncio.wait_for(executor.shutdown(), timeout=1.0)

        assert worker.cancelled.is_set()
        assert outcomes == []
        assert any("phase=abandoned_call" in record.message for record in caplog.records)

        worker.release.set()
        for _ in range(10):
            if outcomes:
                break
            await asyncio.sleep(0)
        assert outcomes == [CallStatusResponse(status="success", result="done")]
