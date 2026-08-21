import asyncio
import contextlib
import importlib
import socket
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h11
import pytest
import uvicorn
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from starlette.types import Receive, Scope, Send

from miles.utils.workers.rpc.common.metadata import rpc
from miles.utils.workers.rpc.server.app import _RequestBodyLimitMiddleware, create_rpc_app


@dataclass
class _RunningServer:
    server: uvicorn.Server
    task: asyncio.Task[None]
    port: int


class _AdmissionProbe:
    def __init__(self) -> None:
        self.state = SimpleNamespace(rpc_control_paths=frozenset({"/v1/dynamic-control"}))
        self.data_started = asyncio.Event()
        self.data_calls = 0
        self.control_calls = 0
        self.paths: list[str] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.paths.append(scope["path"])
        if scope["path"] in {
            "/v1/health",
            "/v1/in-flight",
            "/v1/calls/c1",
            "/v1/calls/c1/ack",
            "/v1/dynamic-control",
        }:
            self.control_calls += 1
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})
            return

        self.data_calls += 1
        self.data_started.set()
        while (message := await receive())["type"] != "http.disconnect":
            if message["type"] == "http.request" and not message.get("more_body", False):
                break


class _ImmediateProbe:
    def __init__(self) -> None:
        self.state = SimpleNamespace(rpc_control_paths=frozenset())
        self.paths: list[str] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.paths.append(scope["path"])
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})


class _SequencedProbe:
    def __init__(self) -> None:
        self.state = SimpleNamespace(rpc_control_paths=frozenset())
        self.requests: asyncio.Queue[tuple[str, asyncio.Event]] = asyncio.Queue()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        release = asyncio.Event()
        await self.requests.put((scope["path"], release))
        await release.wait()
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})


class _DisconnectProbe:
    def __init__(self) -> None:
        self.state = SimpleNamespace(rpc_control_paths=frozenset())
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.stopped: asyncio.Queue[str] = asyncio.Queue()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope["path"]
        await self.started.put(path)
        while (await receive())["type"] != "http.disconnect":
            pass
        await self.stopped.put(path)


class _BlockingControlProbe:
    def __init__(self) -> None:
        self.state = SimpleNamespace(rpc_control_paths=frozenset())
        self.control_started = asyncio.Event()
        self.release_control = asyncio.Event()
        self.control_calls = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.control_calls += 1
        self.control_started.set()
        await self.release_control.wait()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})


class _BodyRetentionProbe:
    def __init__(self) -> None:
        self.state = SimpleNamespace(rpc_control_paths=frozenset())
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await receive()
        self.started.set()
        await self.release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})


class _StreamingProbe:
    def __init__(self) -> None:
        self.state = SimpleNamespace(rpc_control_paths=frozenset())
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"5")]})
        await send({"type": "http.response.body", "body": b"a", "more_body": True})
        self.started.set()
        await self.release.wait()
        await send({"type": "http.response.body", "body": b"tail", "more_body": False})


@contextlib.asynccontextmanager
async def _running_server(
    app: Callable[[Scope, Receive, Send], Awaitable[None]],
    *,
    http: str | type,
    limit_concurrency: int | None = None,
    lifespan: str = "off",
    timeout_keep_alive: float = 5.0,
) -> AsyncIterator[_RunningServer]:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            http=http,
            lifespan=lifespan,
            limit_concurrency=limit_concurrency,
            log_level="critical",
            timeout_keep_alive=timeout_keep_alive,
        )
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(100):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        assert server.started
        yield _RunningServer(server=server, task=task, port=port)
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except BaseException:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
            raise
        finally:
            listener.close()


async def _open_connection(
    port: int, request: bytes | None = None
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    if request is not None:
        writer.write(request)
        await writer.drain()
    return reader, writer


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(ConnectionError):
        await writer.wait_closed()


async def _read_response(reader: asyncio.StreamReader) -> bytes:
    headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=1.0)
    content_length = next(
        int(line.split(b":", 1)[1].strip())
        for line in headers.split(b"\r\n")
        if line.lower().startswith(b"content-length:")
    )
    return headers + await asyncio.wait_for(reader.readexactly(content_length), timeout=1.0)


@contextlib.contextmanager
def _record_request_tasks() -> Iterator[list[asyncio.Task[Any]]]:
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    request_tasks: list[asyncio.Task[Any]] = []

    def record(
        target_loop: asyncio.AbstractEventLoop,
        coroutine: Coroutine[Any, Any, Any],
        **kwargs: Any,
    ) -> asyncio.Task[Any]:
        if previous_factory is None:
            task = asyncio.Task(coroutine, loop=target_loop, **kwargs)
        else:
            task = previous_factory(target_loop, coroutine, **kwargs)
        if getattr(getattr(coroutine, "cr_code", None), "co_name", "") == "run_asgi":
            request_tasks.append(task)
        return task

    loop.set_task_factory(record)
    try:
        yield request_tasks
    finally:
        loop.set_task_factory(previous_factory)


def _protocol_snapshot(running: _RunningServer) -> Any:
    assert len(running.server.server_state.connections) == 1
    protocol = next(iter(running.server.server_state.connections))
    return protocol._admission.snapshot()


async def _wait_for_snapshot(running: _RunningServer, *, expected: tuple[int, int, int]) -> None:
    for _ in range(100):
        if len(running.server.server_state.connections) == 1:
            snapshot = _protocol_snapshot(running)
            if (snapshot.unknown, snapshot.data, snapshot.control) == expected:
                return
        await asyncio.sleep(0.01)
    snapshot = _protocol_snapshot(running)
    assert (snapshot.unknown, snapshot.data, snapshot.control) == expected


async def _wait_for_no_connections(running: _RunningServer) -> None:
    for _ in range(100):
        if not running.server.server_state.connections:
            return
        await asyncio.sleep(0.01)
    assert not running.server.server_state.connections


class TestUvicornAdmissionBoundary:
    def test_the_private_protocol_contract_is_pinned_to_the_reviewed_uvicorn_line(self) -> None:
        """The bounded protocol must be reviewed again before a Uvicorn private API upgrade."""
        assert Version("0.41") <= Version(uvicorn.__version__) < Version("0.42")
        requirements = [
            Requirement(requirement.partition("#")[0].strip())
            for requirement in Path("requirements.txt").read_text().splitlines()
            if requirement.partition("#")[0].strip()
        ]
        uvicorn_requirement = next(requirement for requirement in requirements if requirement.name == "uvicorn")
        assert uvicorn_requirement.specifier == SpecifierSet(">=0.41,<0.42")

    async def test_builtin_concurrency_rejection_still_creates_an_asgi_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uvicorn's ordinary 503 path is not a pre-task hard admission boundary."""
        from uvicorn.protocols.http import h11_impl

        release_rejection = asyncio.Event()
        rejected_calls = 0

        async def blocked_rejection(scope: Scope, receive: Receive, send: Send) -> None:
            nonlocal rejected_calls
            rejected_calls += 1
            await release_rejection.wait()

        monkeypatch.setattr(h11_impl, "service_unavailable", blocked_rejection)

        async with _running_server(_ImmediateProbe(), http="h11", limit_concurrency=1) as running:
            connections = [
                await _open_connection(
                    running.port,
                    b"GET /v1/health HTTP/1.1\r\nHost: test\r\n\r\n",
                )
                for _ in range(8)
            ]
            try:
                for _ in range(100):
                    if rejected_calls == len(connections):
                        break
                    await asyncio.sleep(0.01)
                assert rejected_calls == len(connections)
                assert len(running.server.server_state.tasks) == len(connections) > 1
            finally:
                release_rejection.set()
                for reader, writer in connections:
                    await _close_writer(writer)
                    del reader

    async def test_headerless_connections_are_aborted_before_parent_admission(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unknown request headers have a hard connection bound and never create an ASGI task."""
        from uvicorn.protocols.http import h11_impl

        parent_admissions: list[int] = []
        parent_connection_made = h11_impl.H11Protocol.connection_made

        def record_parent_admission(protocol: Any, transport: asyncio.Transport) -> None:
            parent_connection_made(protocol, transport)
            parent_admissions.append(len(protocol.server_state.connections))

        monkeypatch.setattr(h11_impl.H11Protocol, "connection_made", record_parent_admission)
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        monkeypatch.setattr(protocol_module, "MAX_UNCLASSIFIED_RPC_CONNECTIONS", 1)
        probe = _AdmissionProbe()

        async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
            first_reader, first_writer = await _open_connection(running.port, b"GET /v1/hea")
            await _wait_for_snapshot(running, expected=(1, 0, 0))
            second_reader, second_writer = await _open_connection(running.port)
            try:
                assert await asyncio.wait_for(second_reader.read(1), timeout=1.0) == b""
                assert len(running.server.server_state.connections) <= 1
                assert not running.server.server_state.tasks
                assert parent_admissions == [1]
            finally:
                await _close_writer(first_writer)
                await _close_writer(second_writer)
                del first_reader

            await _wait_for_no_connections(running)
            replacement_reader, replacement_writer = await _open_connection(running.port, b"GET /v1/hea")
            try:
                await _wait_for_snapshot(running, expected=(1, 0, 0))
            finally:
                await _close_writer(replacement_writer)
                del replacement_reader

    @pytest.mark.parametrize(
        "control_request",
        [
            b"GET /v1/health HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n",
            b"GET /v1/in-flight HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n",
            b"GET /v1/calls/c1?timeout=0 HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n",
            b"POST /v1/calls/c1/ack HTTP/1.1\r\nHost: test\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
            b"POST /v1/dynamic-control HTTP/1.1\r\nHost: test\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        ],
    )
    async def test_slow_data_body_does_not_consume_control_admission(
        self, monkeypatch: pytest.MonkeyPatch, control_request: bytes
    ) -> None:
        """Every fixed and dynamic control route bypasses a saturated data lane."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        monkeypatch.setattr(protocol_module, "MAX_UNCLASSIFIED_RPC_CONNECTIONS", 3)
        monkeypatch.setattr(protocol_module, "MAX_DATA_RPC_CONNECTIONS", 1)
        monkeypatch.setattr(protocol_module, "MAX_CONTROL_RPC_CONNECTIONS", 1)
        probe = _AdmissionProbe()
        slow_data = b"POST /v1/demo HTTP/1.1\r\nHost: test\r\nContent-Length: 100\r\n\r\nx"

        async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
            first_reader, first_writer = await _open_connection(running.port, slow_data)
            await asyncio.wait_for(probe.data_started.wait(), timeout=1.0)
            with _record_request_tasks() as rejected_tasks:
                second_reader, second_writer = await _open_connection(running.port, slow_data)
                assert await asyncio.wait_for(second_reader.read(1), timeout=1.0) == b""
            assert not rejected_tasks
            control_reader, control_writer = await _open_connection(running.port, control_request)
            try:
                response = await asyncio.wait_for(control_reader.read(), timeout=1.0)
                assert b" 200 " in response and b"ok" in response
                assert probe.data_calls == 1
                assert probe.control_calls == 1
                assert len(running.server.server_state.tasks) <= 1
            finally:
                await _close_writer(first_writer)
                await _close_writer(second_writer)
                await _close_writer(control_writer)
                del first_reader

    @pytest.mark.parametrize(
        ("max_data", "max_control", "rejected_request", "accepted_request"),
        [
            (
                0,
                1,
                b"POST /v1/demo HTTP/1.1\r\nHost: test\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                b"GET /v1/health HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n",
            ),
            (
                1,
                0,
                b"GET /v1/health HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n",
                b"POST /v1/demo HTTP/1.1\r\nHost: test\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            ),
        ],
    )
    async def test_lane_rejection_rolls_back_unknown_admission_before_cross_lane_reuse(
        self,
        monkeypatch: pytest.MonkeyPatch,
        max_data: int,
        max_control: int,
        rejected_request: bytes,
        accepted_request: bytes,
    ) -> None:
        """A failed lane transfer returns its unknown token before opposite-lane admission."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        monkeypatch.setattr(protocol_module, "MAX_UNCLASSIFIED_RPC_CONNECTIONS", 1)
        monkeypatch.setattr(protocol_module, "MAX_DATA_RPC_CONNECTIONS", max_data)
        monkeypatch.setattr(protocol_module, "MAX_CONTROL_RPC_CONNECTIONS", max_control)
        probe = _ImmediateProbe()

        async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
            with _record_request_tasks() as rejected_tasks:
                rejected_reader, rejected_writer = await _open_connection(running.port, rejected_request)
                assert await asyncio.wait_for(rejected_reader.read(1), timeout=1.0) == b""
            assert not rejected_tasks
            await _wait_for_no_connections(running)

            accepted_reader, accepted_writer = await _open_connection(running.port, accepted_request)
            try:
                assert b" 200 " in await _read_response(accepted_reader)
            finally:
                await _close_writer(rejected_writer)
                await _close_writer(accepted_writer)

    @pytest.mark.parametrize(
        "lookalike_request",
        [
            b"POST /v1/health HTTP/1.1\r\nHost: test\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            b"GET /v1/calls HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n",
            b"POST /v1/calls/c1/ack/extra HTTP/1.1\r\nHost: test\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        ],
    )
    async def test_control_route_lookalikes_remain_data_and_abort_before_task_creation(
        self, monkeypatch: pytest.MonkeyPatch, lookalike_request: bytes
    ) -> None:
        """Wrong methods and malformed call paths cannot consume reserved control admission."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        monkeypatch.setattr(protocol_module, "MAX_UNCLASSIFIED_RPC_CONNECTIONS", 2)
        monkeypatch.setattr(protocol_module, "MAX_DATA_RPC_CONNECTIONS", 1)
        monkeypatch.setattr(protocol_module, "MAX_CONTROL_RPC_CONNECTIONS", 1)
        probe = _AdmissionProbe()
        slow_data = b"POST /v1/demo HTTP/1.1\r\nHost: test\r\nContent-Length: 100\r\n\r\nx"

        async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
            first_reader, first_writer = await _open_connection(running.port, slow_data)
            await asyncio.wait_for(probe.data_started.wait(), timeout=1.0)
            with _record_request_tasks() as rejected_tasks:
                rejected_reader, rejected_writer = await _open_connection(running.port, lookalike_request)
                assert await asyncio.wait_for(rejected_reader.read(1), timeout=1.0) == b""
            try:
                assert not rejected_tasks
                assert probe.data_calls == 1
            finally:
                await _close_writer(first_writer)
                await _close_writer(rejected_writer)
                del first_reader

    async def test_a_full_control_lane_aborts_before_task_creation_and_reopens_after_release(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control calls have their own hard task bound and a released slot admits a replacement."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        monkeypatch.setattr(protocol_module, "MAX_UNCLASSIFIED_RPC_CONNECTIONS", 3)
        monkeypatch.setattr(protocol_module, "MAX_DATA_RPC_CONNECTIONS", 1)
        monkeypatch.setattr(protocol_module, "MAX_CONTROL_RPC_CONNECTIONS", 1)
        probe = _BlockingControlProbe()
        request = b"GET /v1/health HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n"

        async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
            first_reader, first_writer = await _open_connection(running.port, request)
            await asyncio.wait_for(probe.control_started.wait(), timeout=1.0)
            with _record_request_tasks() as rejected_tasks:
                second_reader, second_writer = await _open_connection(running.port, request)
                assert await asyncio.wait_for(second_reader.read(1), timeout=1.0) == b""
            assert not rejected_tasks
            assert probe.control_calls == 1

            probe.release_control.set()
            assert b" 200 " in await _read_response(first_reader)
            await _close_writer(first_writer)
            replacement_reader, replacement_writer = await _open_connection(running.port, request)
            try:
                assert b" 200 " in await _read_response(replacement_reader)
                assert probe.control_calls == 2
            finally:
                await _close_writer(second_writer)
                await _close_writer(replacement_writer)

    async def test_keepalive_reclassifies_every_request_and_idle_connections_remain_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A keep-alive connection changes lanes per request and owns an unknown slot while idle."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        monkeypatch.setattr(protocol_module, "MAX_UNCLASSIFIED_RPC_CONNECTIONS", 1)
        monkeypatch.setattr(protocol_module, "MAX_DATA_RPC_CONNECTIONS", 1)
        monkeypatch.setattr(protocol_module, "MAX_CONTROL_RPC_CONNECTIONS", 1)
        probe = _SequencedProbe()
        requests = [
            (b"POST /v1/demo HTTP/1.1\r\nHost: test\r\nContent-Length: 0\r\n\r\n", "/v1/demo", (0, 1, 0)),
            (b"GET /v1/calls/c1?timeout=0 HTTP/1.1\r\nHost: test\r\n\r\n", "/v1/calls/c1", (0, 0, 1)),
            (
                b"POST /v1/calls/c1/ack HTTP/1.1\r\nHost: test\r\nContent-Length: 0\r\n\r\n",
                "/v1/calls/c1/ack",
                (0, 0, 1),
            ),
            (b"GET /v1/health HTTP/1.1\r\nHost: test\r\n\r\n", "/v1/health", (0, 0, 1)),
            (b"POST /v1/demo HTTP/1.1\r\nHost: test\r\nContent-Length: 0\r\n\r\n", "/v1/demo", (0, 1, 0)),
        ]

        async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
            reader, writer = await _open_connection(running.port)
            try:
                for request, expected_path, active_snapshot in requests:
                    writer.write(request)
                    await writer.drain()
                    path, release = await asyncio.wait_for(probe.requests.get(), timeout=1.0)
                    assert path == expected_path
                    await _wait_for_snapshot(running, expected=active_snapshot)
                    release.set()
                    assert b" 200 " in await _read_response(reader)
                    await _wait_for_snapshot(running, expected=(1, 0, 0))

                extra_reader, extra_writer = await _open_connection(running.port)
                try:
                    assert await asyncio.wait_for(extra_reader.read(1), timeout=1.0) == b""
                finally:
                    await _close_writer(extra_writer)
            finally:
                await _close_writer(writer)

            await _wait_for_no_connections(running)
            replacement_reader, replacement_writer = await _open_connection(running.port)
            try:
                await _wait_for_snapshot(running, expected=(1, 0, 0))
            finally:
                await _close_writer(replacement_writer)
                del replacement_reader

    @pytest.mark.parametrize(("status_code", "max_bytes", "max_data_requests"), [(413, 1, 1), (503, 2, 0)])
    async def test_an_early_response_closes_a_connection_with_an_incomplete_request_body(
        self,
        status_code: int,
        max_bytes: int,
        max_data_requests: int,
    ) -> None:
        """An early ingress rejection cannot reopen admission before the current request body ends."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        probe = _ImmediateProbe()
        app = _RequestBodyLimitMiddleware(
            probe,
            max_bytes=max_bytes,
            boot_uuid="boot",
            max_data_in_flight_requests=max_data_requests,
        )
        request = b"POST /v1/demo HTTP/1.1\r\nHost: test\r\nContent-Length: 2\r\n\r\nx"

        async with _running_server(app, http=protocol_module._BoundedH11Protocol) as running:
            reader, writer = await _open_connection(running.port, request)
            try:
                assert f" {status_code} ".encode() in await _read_response(reader)
                assert await asyncio.wait_for(reader.read(1), timeout=1.0) == b""
                assert not probe.paths
            finally:
                await _close_writer(writer)

    async def test_fragmented_keepalive_headers_cancel_the_previous_idle_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Receiving the first next-request byte cancels the prior keep-alive timer."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        monkeypatch.setattr(protocol_module, "MAX_UNCLASSIFIED_RPC_CONNECTIONS", 1)
        probe = _ImmediateProbe()
        first_request = b"GET /v1/health HTTP/1.1\r\nHost: test\r\n\r\n"

        async with _running_server(
            probe,
            http=protocol_module._BoundedH11Protocol,
            timeout_keep_alive=0.05,
        ) as running:
            reader, writer = await _open_connection(running.port, first_request)
            try:
                assert b" 200 " in await _read_response(reader)
                await _wait_for_snapshot(running, expected=(1, 0, 0))
                writer.write(b"GET /v1/hea")
                await writer.drain()
                await asyncio.sleep(0.1)
                writer.write(b"lth HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n")
                await writer.drain()
                assert b" 200 " in await _read_response(reader)
            finally:
                await _close_writer(writer)

    async def test_classification_parser_releases_trailing_body_bytes_after_forwarding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The header classifier cannot retain a second copy of body bytes after parent delivery."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        original_next_event = h11.Connection.next_event
        request_remainders: list[int] = []

        def record_request_remainder(connection: h11.Connection) -> Any:
            event = original_next_event(connection)
            if isinstance(event, h11.Request) and event.target == b"/v1/demo":
                request_remainders.append(len(connection._receive_buffer))
            return event

        monkeypatch.setattr(h11.Connection, "next_event", record_request_remainder)
        probe = _BodyRetentionProbe()
        body = b"x" * 32768
        request = (
            b"POST /v1/demo HTTP/1.1\r\nHost: test\r\nX-Padding: "
            + b"p" * 4096
            + b"\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )

        async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
            reader, writer = await _open_connection(running.port, request)
            try:
                await asyncio.wait_for(probe.started.wait(), timeout=1.0)
                assert request_remainders and request_remainders[0] > 0
                protocol = next(iter(running.server.server_state.connections))
                assert not protocol._header_bytes
                assert len(protocol._header_parser._receive_buffer) == 0
                probe.release.set()
                assert b" 200 " in await _read_response(reader)
            finally:
                probe.release.set()
                await _close_writer(writer)

    async def test_shutdown_finishes_an_already_started_streaming_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Graceful shutdown preserves an admitted response that has already started streaming."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        original_shutdown = protocol_module._BoundedH11Protocol.shutdown
        shutdown_called = asyncio.Event()

        def record_shutdown(protocol: Any) -> None:
            original_shutdown(protocol)
            shutdown_called.set()

        monkeypatch.setattr(protocol_module._BoundedH11Protocol, "shutdown", record_shutdown)
        probe = _StreamingProbe()
        reader: asyncio.StreamReader
        writer: asyncio.StreamWriter

        async def release_after_shutdown() -> None:
            await shutdown_called.wait()
            probe.release.set()

        release_task = asyncio.create_task(release_after_shutdown())
        async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
            reader, writer = await _open_connection(
                running.port,
                b"GET /v1/demo HTTP/1.1\r\nHost: test\r\n\r\n",
            )
            await asyncio.wait_for(probe.started.wait(), timeout=1.0)
            headers = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=1.0)
            assert b" 200 " in headers
            assert await asyncio.wait_for(reader.readexactly(1), timeout=1.0) == b"a"

        try:
            await asyncio.wait_for(release_task, timeout=1.0)
            assert await asyncio.wait_for(reader.readexactly(4), timeout=1.0) == b"tail"
        finally:
            await _close_writer(writer)

    @pytest.mark.parametrize(
        ("request_bytes", "expected_snapshot"),
        [
            (
                b"POST /v1/demo HTTP/1.1\r\nHost: test\r\nContent-Length: 100\r\n\r\nx",
                (0, 1, 0),
            ),
            (b"GET /v1/health HTTP/1.1\r\nHost: test\r\n\r\n", (0, 0, 1)),
        ],
    )
    async def test_active_disconnect_releases_its_lane_and_admits_a_replacement(
        self,
        monkeypatch: pytest.MonkeyPatch,
        request_bytes: bytes,
        expected_snapshot: tuple[int, int, int],
    ) -> None:
        """Disconnecting active data and control calls releases the exact lane token."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        monkeypatch.setattr(protocol_module, "MAX_UNCLASSIFIED_RPC_CONNECTIONS", 1)
        monkeypatch.setattr(protocol_module, "MAX_DATA_RPC_CONNECTIONS", 1)
        monkeypatch.setattr(protocol_module, "MAX_CONTROL_RPC_CONNECTIONS", 1)
        probe = _DisconnectProbe()

        async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
            for _ in range(2):
                reader, writer = await _open_connection(running.port, request_bytes)
                assert await asyncio.wait_for(probe.started.get(), timeout=1.0) in {"/v1/demo", "/v1/health"}
                await _wait_for_snapshot(running, expected=expected_snapshot)
                await _close_writer(writer)
                assert await asyncio.wait_for(probe.stopped.get(), timeout=1.0) in {"/v1/demo", "/v1/health"}
                await _wait_for_no_connections(running)
                del reader

    @pytest.mark.parametrize(
        ("request_bytes", "expected_snapshot"),
        [
            (
                b"POST /v1/demo HTTP/1.1\r\nHost: test\r\nContent-Length: 100\r\n\r\nx",
                (0, 1, 0),
            ),
            (b"GET /v1/health HTTP/1.1\r\nHost: test\r\n\r\n", (0, 0, 1)),
        ],
    )
    async def test_cancelled_active_request_releases_its_lane_and_admits_a_replacement(
        self,
        monkeypatch: pytest.MonkeyPatch,
        request_bytes: bytes,
        expected_snapshot: tuple[int, int, int],
    ) -> None:
        """Cancelling active data and control ASGI tasks releases the exact lane token."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        monkeypatch.setattr(protocol_module, "MAX_UNCLASSIFIED_RPC_CONNECTIONS", 2)
        monkeypatch.setattr(protocol_module, "MAX_DATA_RPC_CONNECTIONS", 1)
        monkeypatch.setattr(protocol_module, "MAX_CONTROL_RPC_CONNECTIONS", 1)
        probe = _DisconnectProbe()

        async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
            with _record_request_tasks() as request_tasks:
                first_reader, first_writer = await _open_connection(running.port, request_bytes)
                await asyncio.wait_for(probe.started.get(), timeout=1.0)
                await _wait_for_snapshot(running, expected=expected_snapshot)
                assert len(request_tasks) == 1
                request_tasks[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request_tasks[0]
                assert await asyncio.wait_for(first_reader.read(1), timeout=1.0) == b""

            replacement_reader, replacement_writer = await _open_connection(running.port, request_bytes)
            try:
                await asyncio.wait_for(probe.started.get(), timeout=1.0)
                await _wait_for_snapshot(running, expected=expected_snapshot)
            finally:
                await _close_writer(first_writer)
                await _close_writer(replacement_writer)
                del replacement_reader

    async def test_create_rpc_app_publishes_real_dynamic_control_paths_to_the_protocol(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real RPC metadata reaches the protocol classifier before dynamic control admission."""
        app_module = importlib.import_module("miles.utils.workers.rpc.server.app")
        common_protocol = importlib.import_module("miles.utils.workers.rpc.common.protocol")
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        canonical_classifier = common_protocol.is_rpc_control_request
        middleware_calls: list[tuple[str, str]] = []
        protocol_calls: list[tuple[str, str]] = []

        def middleware_classifier(*, method: str, path: str, dynamic_paths: frozenset[str]) -> bool:
            middleware_calls.append((method, path))
            return canonical_classifier(method=method, path=path, dynamic_paths=dynamic_paths)

        def protocol_classifier(*, method: str, path: str, dynamic_paths: frozenset[str]) -> bool:
            protocol_calls.append((method, path))
            return canonical_classifier(method=method, path=path, dynamic_paths=dynamic_paths)

        monkeypatch.setattr(app_module, "is_rpc_control_request", middleware_classifier)
        monkeypatch.setattr(protocol_module, "is_rpc_control_request", protocol_classifier)
        monkeypatch.setattr(protocol_module, "MAX_UNCLASSIFIED_RPC_CONNECTIONS", 1)
        monkeypatch.setattr(protocol_module, "MAX_DATA_RPC_CONNECTIONS", 0)
        monkeypatch.setattr(protocol_module, "MAX_CONTROL_RPC_CONNECTIONS", 1)

        class Worker:
            @rpc(control_plane=True)
            def heartbeat(self) -> str:
                return "alive"

        app = create_rpc_app(Worker())
        assert app.state.rpc_control_paths == app.state.rpc_server.control_paths == frozenset({"/v1/heartbeat"})
        assert "rpc_control_paths" not in vars(app.state)
        request_body = b'{"call_id":"heartbeat","query":{}}'
        request = (
            b"POST /v1/heartbeat HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: "
            + str(len(request_body)).encode()
            + b"\r\nConnection: close\r\n\r\n"
            + request_body
        )
        async with _running_server(
            app,
            http=protocol_module._BoundedH11Protocol,
            lifespan="on",
        ) as running:
            reader, writer = await _open_connection(running.port, request)
            try:
                assert b" 200 " in await _read_response(reader)
            finally:
                await _close_writer(writer)
        assert protocol_calls == [("POST", "/v1/heartbeat")]
        assert middleware_calls == [("POST", "/v1/heartbeat")]

    @pytest.mark.parametrize(
        ("method", "path", "expected"),
        [
            ("GET", "/v1/health", True),
            ("GET", "/v1/in-flight", True),
            ("GET", "/v1/calls/c1", True),
            ("POST", "/v1/calls/c1/ack", True),
            ("POST", "/v1/dynamic-control", True),
            ("POST", "/v1/demo", False),
            ("GET", "/v1/calls", False),
            ("POST", "/v1/calls/c1/ack/extra", False),
        ],
    )
    def test_the_protocol_and_middleware_share_canonical_control_classification(
        self, method: str, path: str, expected: bool
    ) -> None:
        """Fixed, call-id, and dynamic control routes use one exact classifier."""
        protocol_module = importlib.import_module("miles.utils.workers.rpc.common.protocol")
        assert (
            protocol_module.is_rpc_control_request(
                method=method,
                path=path,
                dynamic_paths=frozenset({"/v1/dynamic-control"}),
            )
            is expected
        )

    async def test_http_pipelining_is_closed_after_the_first_admitted_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pipelined successor never bypasses per-request lane admission."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        monkeypatch.setattr(protocol_module, "MAX_UNCLASSIFIED_RPC_CONNECTIONS", 1)
        probe = _ImmediateProbe()
        pipelined = (
            b"GET /v1/health HTTP/1.1\r\nHost: test\r\n\r\n"
            b"GET /v1/health HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n"
        )

        async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
            reader, writer = await _open_connection(running.port, pipelined)
            try:
                response = await asyncio.wait_for(reader.read(), timeout=1.0)
                assert response.count(b"HTTP/1.1 200") == 1
                assert probe.paths == ["/v1/health"]
            finally:
                await _close_writer(writer)

    async def test_http_upgrade_is_aborted_without_an_asgi_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Upgrade requests fail closed instead of escaping the bounded HTTP lane state machine."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        probe = _ImmediateProbe()
        upgrade = (
            b"GET /v1/health HTTP/1.1\r\nHost: test\r\nConnection: Upgrade\r\n"
            b"Upgrade: websocket\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )

        async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
            with _record_request_tasks() as rejected_tasks:
                reader, writer = await _open_connection(running.port, upgrade)
                assert await asyncio.wait_for(reader.read(1), timeout=1.0) == b""
            try:
                assert not rejected_tasks
                assert not probe.paths
            finally:
                await _close_writer(writer)

    async def test_server_startup_failure_closes_the_listener_and_consumes_the_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A startup exception cannot leave the bound listener or serve task alive."""
        original_socket = socket.socket
        listeners: list[socket.socket] = []

        def record_socket(*args: Any, **kwargs: Any) -> socket.socket:
            listener = original_socket(*args, **kwargs)
            listeners.append(listener)
            return listener

        async def fail_startup(server: uvicorn.Server, *, sockets: list[socket.socket]) -> None:
            raise RuntimeError("startup failed")

        monkeypatch.setattr(socket, "socket", record_socket)
        monkeypatch.setattr(uvicorn.Server, "serve", fail_startup)

        with pytest.raises(RuntimeError, match="startup failed"):
            async with _running_server(_ImmediateProbe(), http="h11"):
                pass

        assert listeners and listeners[0].fileno() == -1

    async def test_server_teardown_finishes_a_live_blocked_request(self) -> None:
        """Context teardown closes live protocols and consumes their blocked ASGI tasks."""
        protocol_module = importlib.import_module("miles.utils.workers.serving.http_protocol")
        probe = _DisconnectProbe()
        request = b"POST /v1/demo HTTP/1.1\r\nHost: test\r\nContent-Length: 100\r\n\r\nx"

        with _record_request_tasks() as request_tasks:
            async with _running_server(probe, http=protocol_module._BoundedH11Protocol) as running:
                reader, writer = await _open_connection(running.port, request)
                await asyncio.wait_for(probe.started.get(), timeout=1.0)
                assert len(request_tasks) == 1 and not request_tasks[0].done()

        try:
            assert await asyncio.wait_for(reader.read(1), timeout=1.0) == b""
            assert request_tasks[0].done()
        finally:
            await _close_writer(writer)
