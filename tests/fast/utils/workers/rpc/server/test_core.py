from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable

import pytest
from fastapi import HTTPException

from miles.utils.retry_utils import NonRetryableError
from miles.utils.workers.rpc.common.protocol import CallStatusResponse, SubmitRequest
from miles.utils.workers.rpc.server.core import RpcServer

GATE_TIMEOUT_SECONDS = 10.0
CONDITION_TIMEOUT_SECONDS = 10.0
POLL_INTERVAL_SECONDS = 0.01


class _Worker:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.ran: list[str] = []

    def block(self, tag: str) -> str:
        self.started.set()
        assert self.release.wait(timeout=GATE_TIMEOUT_SECONDS)
        self.ran.append(tag)
        return tag

    def instant(self, tag: str) -> str:
        self.ran.append(tag)
        return tag

    def refuse(self, message: str) -> None:
        raise NonRetryableError(message)

    async def block_async(self, tag: str) -> str:
        self.started.set()
        await asyncio.Event().wait()
        return tag


def _submit(server: RpcServer, *, method_name: str, call_id: str, tag: str = "x") -> None:
    query = {"message": tag} if method_name == "refuse" else {"tag": tag}
    server.submit_call(method_name=method_name, request=SubmitRequest(call_id=call_id, query=query))


async def _wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + CONDITION_TIMEOUT_SECONDS
    while not condition():
        assert time.monotonic() < deadline, "the condition never became true"
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def _outcome(server: RpcServer, *, call_id: str) -> CallStatusResponse:
    outcome = await server.query_call(call_id=call_id, timeout=CONDITION_TIMEOUT_SECONDS)
    assert outcome.status != "pending"
    return outcome


class TestTheOutcomeTheServerHandsBack:
    async def test_a_refusal_keeps_saying_it_must_not_be_retried(self):
        """Fault tolerance stops the run on this flag; without it an unrecoverable failure is retried forever."""
        server = RpcServer(worker=_Worker())
        _submit(server, method_name="refuse", call_id="c1", tag="stop")

        outcome = await _outcome(server, call_id="c1")

        assert (outcome.status, outcome.non_retryable) == ("failed", True)

    async def test_the_stored_outcome_is_handed_back_rather_than_rebuilt(self):
        """Rebuilding it field by field is what dropped non_retryable, and would drop the next field too."""
        server = RpcServer(worker=_Worker())
        stored = CallStatusResponse(status="failed", result={"partial": 1}, error="boom", non_retryable=True)
        server._store.begin(call_id="c1")
        server._store.finish(call_id="c1", outcome=stored)

        assert await server.query_call(call_id="c1", timeout=0.0) is stored


class TestACallTheCallerGaveUpOn:
    async def test_a_replacement_call_is_refused_while_the_abandoned_one_runs(self):
        """Two train steps in one worker process interleave their optimizer steps, silently."""
        worker = _Worker()
        server = RpcServer(worker=worker)
        _submit(server, method_name="block", call_id="c1", tag="first")
        await _wait_until(worker.started.is_set)

        server.abandon_call(call_id="c1")

        with pytest.raises(HTTPException) as exc:
            _submit(server, method_name="block", call_id="c2", tag="second")
        assert exc.value.status_code == 409
        worker.release.set()
        await _outcome(server, call_id="c1")

    async def test_a_replacement_call_is_accepted_once_the_abandoned_one_is_over(self):
        """The refusal lasts exactly as long as the risk it exists for."""
        worker = _Worker()
        server = RpcServer(worker=worker)
        _submit(server, method_name="block", call_id="c1", tag="first")
        await _wait_until(worker.started.is_set)
        server.abandon_call(call_id="c1")
        worker.release.set()
        await _outcome(server, call_id="c1")

        _submit(server, method_name="block", call_id="c2", tag="second")

        assert (await _outcome(server, call_id="c2")).status == "success"

    async def test_a_queued_call_that_was_abandoned_never_reaches_the_worker(self):
        """The client is gone, so running its call can only produce side effects nobody asked for."""
        worker = _Worker()
        server = RpcServer(worker=worker)
        _submit(server, method_name="block", call_id="c1", tag="blocking")
        await _wait_until(worker.started.is_set)
        _submit(server, method_name="instant", call_id="c2", tag="queued")

        server.abandon_call(call_id="c2")
        worker.release.set()
        await _outcome(server, call_id="c1")

        assert (await _outcome(server, call_id="c2")).non_retryable is True
        assert worker.ran == ["blocking"]

    async def test_an_abandoned_async_call_is_stopped(self):
        """An async method really can be cancelled, so abandoning one frees the loop it was holding."""
        worker = _Worker()
        server = RpcServer(worker=worker)
        _submit(server, method_name="block_async", call_id="c1", tag="forever")
        await _wait_until(worker.started.is_set)

        server.abandon_call(call_id="c1")

        assert (await _outcome(server, call_id="c1")).non_retryable is True

    async def test_abandoning_a_call_nobody_submitted_is_refused(self):
        """A call id the server never saw is a client bug, not something to answer ok to."""
        server = RpcServer(worker=_Worker())

        with pytest.raises(HTTPException) as exc:
            server.abandon_call(call_id="never-submitted")

        assert exc.value.status_code == 404

    async def test_another_method_stays_callable(self):
        """Only the method that may still be running twice is refused; the worker is not fenced off."""
        worker = _Worker()
        server = RpcServer(worker=worker)
        _submit(server, method_name="block_async", call_id="c1", tag="forever")
        await _wait_until(worker.started.is_set)
        server.abandon_call(call_id="c1")

        _submit(server, method_name="instant", call_id="c2", tag="other")

        assert (await _outcome(server, call_id="c2")).status == "success"
