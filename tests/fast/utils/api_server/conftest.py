from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx
import pytest

from miles.utils.ft_utils.api_server.handles import _CellHandler
from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellSpec, CellStatus, TriState
from miles.utils.ft_utils.api_server.registry import _CellRegistry
from miles.utils.ft_utils.api_server.server import _create_api_app
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.ray_worker_manager import CellSummary


class MockCellState:
    def __init__(
        self,
        *,
        phase: str = "Running",
        conditions: list[dict[str, str | None]] | None = None,
        is_suspended: bool = False,
        suspend_error: Exception | None = None,
        resume_error: Exception | None = None,
    ) -> None:
        self.phase = phase
        self.conditions = conditions or [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]
        self.is_suspended = is_suspended
        self.suspend_error = suspend_error
        self.resume_error = resume_error
        self.suspend_calls: int = 0
        self.resume_calls: int = 0


class MockHandler(_CellHandler):
    def __init__(self, cell_type: str) -> None:
        self._cell_type = cell_type
        self.cells: dict[str, MockCellState] = {}
        self.injected: list[tuple[str, FailureMode, int]] = []
        self.supports_inject_fault = False

    @property
    def cell_type(self) -> str:
        return self._cell_type

    def add(self, cell_key: str = "0", **overrides) -> MockCellState:
        state = MockCellState(**overrides)
        self.cells[cell_key] = state
        return state

    async def list_cell_keys(self) -> list[str]:
        return list(self.cells)

    async def get_cell(self, cell_key: str) -> Cell:
        state = self.cells[cell_key]
        return Cell(
            metadata=self._compute_metadata(cell_key),
            spec=CellSpec(suspend=state.is_suspended),
            status=CellStatus(
                phase=state.phase,
                conditions=[CellCondition(**c) for c in state.conditions],
            ),
        )

    async def suspend(self, cell_key: str) -> None:
        state = self.cells[cell_key]
        if state.suspend_error:
            raise state.suspend_error
        state.suspend_calls += 1
        state.is_suspended = True
        state.phase = "Suspended"
        state.conditions = [
            {"type": "Allocated", "status": "False"},
            {"type": "Healthy", "status": "False"},
        ]

    async def resume(self, cell_key: str) -> None:
        state = self.cells[cell_key]
        if state.resume_error:
            raise state.resume_error
        state.resume_calls += 1
        state.is_suspended = False
        state.phase = "Running"
        state.conditions = [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]

    async def inject_fault(self, cell_key: str, *, mode: FailureMode, sub_index: int) -> None:
        if not self.supports_inject_fault:
            await super().inject_fault(cell_key, mode=mode, sub_index=sub_index)
        self.injected.append((cell_key, mode, sub_index))


class MockRemoteCall:
    def __init__(self, return_value: object, effect: Callable[..., None] | None = None) -> None:
        self._return_value = return_value
        self._effect = effect
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args: object, **kwargs: object) -> asyncio.Future[object]:
        self.calls.append((args, kwargs))
        if self._effect is not None:
            self._effect(*args, **kwargs)
        future: asyncio.Future[object] = asyncio.get_event_loop().create_future()
        future.set_result(self._return_value)
        return future


class MockInferenceController:
    """Plain object, not a Ray actor: the controller lives in the driver process."""

    def __init__(self, health_statuses: dict[str, TriState] | None = None) -> None:
        self._health_statuses = dict(health_statuses or {})
        self.health_status_calls: int = 0

    def get_cell_health_statuses(self) -> dict[str, TriState]:
        self.health_status_calls += 1
        return dict(self._health_statuses)


class MockWorkerManager:
    """Stands in for the RayWorkerManager actor handle, whose methods are called remotely."""

    def __init__(self, summaries: dict[str, CellSummary] | None = None) -> None:
        self._summaries = dict(summaries or {})
        self.stopped_cells: list[list[str]] = []
        self.started_cells: list[list[str]] = []

    @property
    def get_cell_summaries(self) -> MockRemoteCall:
        return MockRemoteCall(dict(self._summaries))

    @property
    def stop_cells(self) -> MockRemoteCall:
        return MockRemoteCall(None, effect=lambda ids: self._record(self.stopped_cells, ids, suspended=True))

    @property
    def start_cells(self) -> MockRemoteCall:
        return MockRemoteCall(None, effect=lambda ids: self._record(self.started_cells, ids, suspended=False))

    def _record(self, log: list[list[str]], cell_ids: list[str], *, suspended: bool) -> None:
        log.append(list(cell_ids))
        for cell_id in cell_ids:
            self._summaries[cell_id] = CellSummary(
                cell_id=cell_id, suspended=suspended, meta=self._summaries[cell_id].meta
            )


def make_cell_summaries(*cell_ids: str, suspended: bool = False, engine: bool = True) -> dict[str, CellSummary]:
    return {
        cell_id: CellSummary(
            cell_id=cell_id,
            suspended=suspended,
            meta={"model_id": "default"} if engine else {},
        )
        for cell_id in cell_ids
    }


class MockRayTrainCell:
    def __init__(
        self,
        *,
        phase: str = "Running",
        conditions: list[dict[str, str | None]] | None = None,
        is_stopped: bool = False,
    ) -> None:
        self._phase = phase
        self._conditions = conditions or [
            {"type": "Allocated", "status": "True"},
            {"type": "Healthy", "status": "True"},
        ]
        self._is_stopped = is_stopped

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def conditions(self) -> list[dict[str, str | None]]:
        return self._conditions

    @property
    def is_stopped(self) -> bool:
        return self._is_stopped

    def cell_status(self) -> CellStatus:
        from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus

        return CellStatus(
            phase=self._phase,
            conditions=[CellCondition(**c) for c in self._conditions],
        )


def make_mock_group(cells: list[MockRayTrainCell]) -> object:
    from miles.ray.train.group import RayTrainGroup

    group = object.__new__(RayTrainGroup)
    group._cells = cells
    group._indep_dp_quorum_id = 0
    group._alive_cell_ids = frozenset()
    return group


@pytest.fixture
def actor_handler() -> MockHandler:
    return MockHandler("actor")


@pytest.fixture
def rollout_handler() -> MockHandler:
    return MockHandler("rollout")


@pytest.fixture
def registry(actor_handler: MockHandler, rollout_handler: MockHandler) -> _CellRegistry:
    return _CellRegistry([actor_handler, rollout_handler])


@pytest.fixture
def async_client(registry: _CellRegistry) -> httpx.AsyncClient:
    app = _create_api_app(registry)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")
