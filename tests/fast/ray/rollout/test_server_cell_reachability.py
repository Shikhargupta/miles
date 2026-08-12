from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.ray.rollout.cell_state import (
    CellAddrInfo,
    StateInitializing,
    StatePendingWeights,
    StateServing,
    StateUninitialized,
)
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.ft_utils.api_server.models import TriState

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("dispose_tracked_server_cells")]

_ADDR_INFO = CellAddrInfo(server_url="http://10.0.0.1:30000", bootstrap_port=None, gate_url=None)
_SERVING_DEADLINE_SECONDS = 100.0


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _StubHealthChecker:
    def __init__(self, status: TriState) -> None:
        self.status = status

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class _StubProvider:
    async def get_addrs(self, worker_name: str):
        raise AssertionError("no cell of this module is ever addressed")


class _StubRouterApiClient:
    async def remove_worker(self, **kwargs) -> None:
        return None


def _make_meta(**overrides) -> ServerCellMetadata:
    return ServerCellMetadata(
        **{
            "model_id": "default",
            "worker_type": "regular",
            "cell_id": "inference-engine-0-0-0",
            "num_gpus_per_engine": 1,
            "gpu_offset": 0,
            "sglang_api_key": None,
            "worker_name": "inference-engine-0-0-0-0",
            "needs_offload": False,
            "update_weights": True,
            "workers_hash": "pseudo-hash-0",
            **overrides,
        }
    )


def _make_cell(*, state, status: TriState, clock: _Clock) -> ServerCell:
    cell = track_server_cell(
        ServerCell(
            args=make_args(),
            meta=_make_meta(),
            router_api_client=_StubRouterApiClient(),
            provider=_StubProvider(),
            clock=clock,
            serving_deadline_seconds=_SERVING_DEADLINE_SECONDS,
        )
    )
    cell._state = state
    cell._health_checker = _StubHealthChecker(status)
    return cell


class TestWhichCellsCountAsUnreachable:
    @pytest.mark.parametrize("status", list(TriState))
    async def test_a_serving_cell_is_unreachable_exactly_when_its_health_checks_fail(self, status):
        """An unknown health check is a probe that has not answered yet, which is not the same as a dead engine."""
        clock = _Clock()
        cell = _make_cell(state=StateServing(addr_info=_ADDR_INFO), status=status, clock=clock)

        assert cell.is_unreachable is (status is TriState.FALSE)

    @pytest.mark.parametrize("status", list(TriState))
    async def test_a_cell_waiting_for_weights_answers_the_same_way(self, status):
        """A cell that already serves requests to the router must be swept on the same rule as a serving one."""
        clock = _Clock()
        cell = _make_cell(state=StatePendingWeights(addr_info=_ADDR_INFO), status=status, clock=clock)

        assert cell.is_unreachable is (status is TriState.FALSE)

    @pytest.mark.parametrize("status", list(TriState))
    async def test_a_cell_whose_engine_never_starts_serving_is_unreachable_once_its_deadline_passes(self, status):
        """The gate answering only proves the pod is up; an engine that never loads must still leave the run."""
        clock = _Clock()
        cell = _make_cell(state=StateInitializing(addr_info=_ADDR_INFO), status=status, clock=clock)

        assert not cell.is_unreachable

        clock.now = _SERVING_DEADLINE_SECONDS
        assert cell.is_unreachable

    async def test_an_uninitialized_cell_is_never_swept(self):
        """A colocated cell sits uninitialized between weight updates by design, so a deadline would evict a run."""
        clock = _Clock()
        cell = _make_cell(state=StateUninitialized(), status=TriState.FALSE, clock=clock)

        clock.now = 10 * _SERVING_DEADLINE_SECONDS
        assert not cell.is_unreachable

    async def test_the_deadline_restarts_whenever_the_cell_changes_state(self):
        """Time spent waiting for the gate is not time the engine had to load, so each state gets its own budget."""
        clock = _Clock()
        cell = _make_cell(state=StateUninitialized(), status=TriState.UNKNOWN, clock=clock)

        clock.now = _SERVING_DEADLINE_SECONDS
        cell._change_state("mark_initializing", StateUninitialized, StateInitializing(addr_info=_ADDR_INFO))

        assert not cell.is_unreachable
        clock.now = 2 * _SERVING_DEADLINE_SECONDS
        assert cell.is_unreachable
