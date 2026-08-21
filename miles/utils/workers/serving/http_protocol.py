from __future__ import annotations

import asyncio
import contextvars
import hashlib
import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, cast
from urllib.parse import unquote

import h11
import uvicorn
from uvicorn.config import Config
from uvicorn.protocols.http import h11_impl
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.server import ServerState

from miles.utils.workers.rpc.common.protocol import is_rpc_control_request

MAX_UNCLASSIFIED_RPC_CONNECTIONS = 256
MAX_DATA_RPC_CONNECTIONS = 4096
MAX_CONTROL_RPC_CONNECTIONS = 4096
RPC_LISTEN_BACKLOG = 512

_EXPECTED_H11_INIT_PARAMETERS = ("self", "config", "server_state", "app_state", "_loop")
_EXPECTED_H11_HANDLE_EVENTS_DIGEST = "342e26628c8f76ce561f57918c33a4b037f608701ea1769b692d0cc655f301ee"
_EXPECTED_H11_RUN_ASGI_DIGEST = "0f33f2449a444bc210e26187fae751e687466157b0050706212bd6ea2b6a5f35"

_Lane = Literal["unknown", "data", "control"]


@dataclass(frozen=True, slots=True)
class _AdmissionSnapshot:
    unknown: int
    data: int
    control: int


class _Admission:
    def __init__(self) -> None:
        self._counts: dict[_Lane, int] = {"unknown": 0, "data": 0, "control": 0}

    def reserve(self, *, lane: _Lane) -> bool:
        if self._counts[lane] >= self._maximum(lane=lane):
            return False
        self._counts[lane] += 1
        return True

    def transfer(self, *, source: _Lane, target: _Lane) -> bool:
        assert self._counts[source] > 0
        if self._counts[target] >= self._maximum(lane=target):
            self._counts[source] -= 1
            return False
        self._counts[source] -= 1
        self._counts[target] += 1
        return True

    def release(self, *, lane: _Lane) -> None:
        assert self._counts[lane] > 0
        self._counts[lane] -= 1

    def snapshot(self) -> _AdmissionSnapshot:
        return _AdmissionSnapshot(
            unknown=self._counts["unknown"],
            data=self._counts["data"],
            control=self._counts["control"],
        )

    @staticmethod
    def _maximum(*, lane: _Lane) -> int:
        match lane:
            case "unknown":
                return MAX_UNCLASSIFIED_RPC_CONNECTIONS
            case "data":
                return MAX_DATA_RPC_CONNECTIONS
            case "control":
                return MAX_CONTROL_RPC_CONNECTIONS


class _CancellationPropagatingRequestResponseCycle(h11_impl.RequestResponseCycle):
    async def run_asgi(self, app: Any) -> None:
        try:
            result = await app(self.scope, self.receive, self.send)
        except asyncio.CancelledError:
            self.transport.close()
            raise
        except BaseException as exc:
            self.logger.error("Exception in ASGI application\n", exc_info=exc)
            if not self.response_started:
                await self.send_500_response()
            else:
                self.transport.close()
        else:
            if result is not None:
                self.logger.error("ASGI callable should return None, but returned '%s'.", result)
                self.transport.close()
            elif not self.response_started and not self.disconnected:
                self.logger.error("ASGI callable returned without starting response.")
                await self.send_500_response()
            elif not self.response_complete and not self.disconnected:
                self.logger.error("ASGI callable returned without completing response.")
                self.transport.close()
        finally:
            self.on_response = lambda: None


class _BoundedH11Protocol(H11Protocol):
    def __init__(
        self,
        config: Config,
        server_state: ServerState,
        app_state: dict[str, Any],
        _loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        super().__init__(config=config, server_state=server_state, app_state=app_state, _loop=_loop)
        state = vars(server_state)
        admission = state.get("_miles_rpc_admission")
        if admission is None:
            admission = _Admission()
            state["_miles_rpc_admission"] = admission
        self._admission = cast(_Admission, admission)
        self._lane: _Lane | None = None
        self._parent_admitted = False
        self._header_parser = h11.Connection(h11.SERVER)
        self._header_bytes = bytearray()
        self._pipelined = False
        self._rpc_control_paths = self._find_control_paths()

    def connection_made(self, transport: asyncio.Transport) -> None:
        if not self._admission.reserve(lane="unknown"):
            transport.abort()
            return
        self._lane = "unknown"
        self._parent_admitted = True
        super().connection_made(transport)

    def connection_lost(self, exc: Exception | None) -> None:
        self._release_lane()
        if self._parent_admitted:
            self._parent_admitted = False
            super().connection_lost(exc)

    def data_received(self, data: bytes) -> None:
        if self._lane != "unknown":
            super().data_received(data)
            return

        self._unset_keepalive_if_required()
        self._header_bytes.extend(data)
        try:
            self._header_parser.receive_data(data)
            event = self._header_parser.next_event()
        except h11.RemoteProtocolError:
            self._forward_header_bytes()
            return
        if event is h11.NEED_DATA:
            return
        if not isinstance(event, h11.Request) or self._is_upgrade(event=event):
            self._abort_rejected_request()
            return

        try:
            method = event.method.decode("ascii")
            raw_path = event.target.partition(b"?")[0]
            path = self.root_path + unquote(raw_path.decode("ascii"))
        except UnicodeDecodeError:
            self._abort_rejected_request()
            return
        target: _Lane = (
            "control"
            if is_rpc_control_request(
                method=method,
                path=path,
                dynamic_paths=self._rpc_control_paths,
            )
            else "data"
        )
        if not self._admission.transfer(source="unknown", target=target):
            self._lane = None
            self._abort_transport()
            return
        self._lane = target
        self._header_parser = h11.Connection(h11.SERVER)
        self._forward_header_bytes()

    def handle_events(self) -> None:
        while True:
            try:
                event = self.conn.next_event()
            except h11.RemoteProtocolError:
                self.logger.warning("Invalid HTTP request received.")
                self.send_400_response("Invalid HTTP request received.")
                return

            if event is h11.NEED_DATA:
                break
            if event is h11.PAUSED:
                self.flow.pause_reading()
                self._pipelined = True
                break
            if isinstance(event, h11.Request):
                self._start_request(event=event)
                if self._should_upgrade():
                    self.handle_websocket_upgrade(event)
                    return
            elif isinstance(event, h11.Data):
                if self.conn.our_state is h11.DONE:
                    continue
                self.cycle.body += event.data
                if len(self.cycle.body) > h11_impl.HIGH_WATER_LIMIT:
                    self.flow.pause_reading()
                self.cycle.message_event.set()
            elif isinstance(event, h11.EndOfMessage):
                if self.conn.our_state is h11.DONE:
                    self.transport.resume_reading()
                    self.conn.start_next_cycle()
                    continue
                self.cycle.more_body = False
                self.cycle.message_event.set()
                if self.conn.their_state == h11.MUST_CLOSE:
                    break

    def _start_request(self, *, event: h11.Request) -> None:
        self.headers = [(key.lower(), value) for key, value in event.headers]
        raw_path, _, query_string = event.target.partition(b"?")
        path = unquote(raw_path.decode("ascii"))
        self.scope = {
            "type": "http",
            "asgi": {"version": self.config.asgi_version, "spec_version": "2.3"},
            "http_version": event.http_version.decode("ascii"),
            "server": self.server,
            "client": self.client,
            "scheme": self.scheme,
            "method": event.method.decode("ascii"),
            "root_path": self.root_path,
            "path": self.root_path + path,
            "raw_path": self.root_path.encode("ascii") + raw_path,
            "query_string": query_string,
            "headers": self.headers,
            "state": self.app_state.copy(),
        }
        self._unset_keepalive_if_required()
        self.cycle = _CancellationPropagatingRequestResponseCycle(
            scope=self.scope,
            conn=self.conn,
            transport=self.transport,
            flow=self.flow,
            logger=self.logger,
            access_logger=self.access_logger,
            access_log=self.access_log,
            default_headers=self.server_state.default_headers,
            message_event=asyncio.Event(),
            on_response=self.on_response_complete,
        )
        task = contextvars.Context().run(self.loop.create_task, self.cycle.run_asgi(self.app))
        task.add_done_callback(self.tasks.discard)
        task.add_done_callback(partial(self._request_task_finished, cycle=self.cycle))
        self.tasks.add(task)

    def on_response_complete(self) -> None:
        self.flow.pause_reading()

    def shutdown(self) -> None:
        if self.cycle is not None and self.cycle.response_started:
            super().shutdown()
            return
        self.transport.close()

    def _finish_response(self) -> None:
        if self._pipelined:
            self._release_lane()
            self.transport.close()
            return
        if self._lane not in {"data", "control"}:
            self.transport.close()
            return
        if self.conn.their_state is not h11.DONE:
            self._release_lane()
            self.transport.close()
            return
        if not self._admission.transfer(source=self._lane, target="unknown"):
            self._lane = None
            self.transport.close()
            return
        self._lane = "unknown"
        self._header_parser = h11.Connection(h11.SERVER)
        self._header_bytes.clear()
        super().on_response_complete()

    def _forward_header_bytes(self) -> None:
        data = bytes(self._header_bytes)
        self._header_bytes.clear()
        super().data_received(data)

    def _request_task_finished(
        self,
        task: asyncio.Task[Any],
        *,
        cycle: _CancellationPropagatingRequestResponseCycle,
    ) -> None:
        if task.cancelled() or not cycle.response_complete:
            self.transport.close()
            return
        if cycle is self.cycle:
            self._finish_response()

    def _abort_rejected_request(self) -> None:
        self._release_lane()
        self._abort_transport()

    def _abort_transport(self) -> None:
        if self._parent_admitted:
            self.transport.abort()

    def _release_lane(self) -> None:
        if self._lane is None:
            return
        self._admission.release(lane=self._lane)
        self._lane = None

    def _find_control_paths(self) -> frozenset[str]:
        app: Any = self.app
        for _ in range(16):
            state = vars(app).get("state")
            if state is not None:
                try:
                    paths = state.rpc_control_paths
                except AttributeError:
                    pass
                else:
                    return cast(frozenset[str], paths)
            if (wrapped := vars(app).get("app")) is None:
                break
            app = wrapped
        return frozenset()

    @staticmethod
    def _is_upgrade(*, event: h11.Request) -> bool:
        headers = {key.lower(): value.lower() for key, value in event.headers}
        connection = {token.strip() for token in headers.get(b"connection", b"").split(b",")}
        return b"upgrade" in connection


def _validate_uvicorn_contract() -> None:
    if tuple(uvicorn.__version__.split(".")[:2]) != ("0", "41"):
        raise RuntimeError(f"Miles RPC requires Uvicorn >=0.41,<0.42, got {uvicorn.__version__}")
    parameters = tuple(inspect.signature(H11Protocol.__init__).parameters)
    if parameters != _EXPECTED_H11_INIT_PARAMETERS:
        raise RuntimeError(
            f"Uvicorn H11Protocol signature changed from {_EXPECTED_H11_INIT_PARAMETERS} to {parameters}"
        )
    sources = {
        "H11Protocol.handle_events": (H11Protocol.handle_events, _EXPECTED_H11_HANDLE_EVENTS_DIGEST),
        "RequestResponseCycle.run_asgi": (h11_impl.RequestResponseCycle.run_asgi, _EXPECTED_H11_RUN_ASGI_DIGEST),
    }
    for name, (function, expected) in sources.items():
        if (actual := hashlib.sha256(inspect.getsource(function).encode()).hexdigest()) != expected:
            raise RuntimeError(f"Uvicorn {name} changed from {expected} to {actual}")


_validate_uvicorn_contract()
