from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace

import pytest
from tests.fast.fixtures.controller_fixtures import make_inference_controller

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell
from miles.utils.context_lock import ContextLock
from miles.utils.workers.worker_provider.base import CellInfo, ObservationSupersededError
from miles.utils.workers.worker_spec import NamedHostAndPorts

_POOL_ID = "west-inference-engine-0-0"
_CELL_ID = f"{_POOL_ID}-0"


class _StubProvider:
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise AssertionError(f"cell startup is patched out in this module ({worker_name=})")

    async def invalidate_cell(self, cell_id: str) -> None:
        raise AssertionError(f"nothing invalidates cells in this module ({cell_id=})")


def _make_cell_info(*, workers_hash: str = "hash-1") -> CellInfo:
    return CellInfo(
        cell_id=_CELL_ID,
        pool_id=_POOL_ID,
        alive=True,
        worker_names=[f"{_CELL_ID}-0"],
        workers_hash=workers_hash,
        meta=dict(
            model_id="default",
            worker_type="regular",
            num_gpus_per_engine=1,
            gpu_offset=0,
            sglang_api_key=None,
            needs_offload=False,
            update_weights=True,
        ),
    )


def _make_server() -> RolloutServer:
    return RolloutServer(
        server_cells={},
        args=SimpleNamespace(colocate=False, ft_components=[]),
        context_lock=ContextLock("InferenceController"),
        engine_provider=_StubProvider(),
    )


def _make_controller(*, server: RolloutServer) -> InferenceController:
    controller = make_inference_controller(
        SimpleNamespace(debug_train_only=False, colocate=False),
        engine_provider=server.engine_provider,
        servers={"default": server},
        context_lock=server.context_lock,
    )
    return controller


class _Gate:
    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self.opened = asyncio.Event()
        self.cells_that_dialled: list[str] = []


def _install_gate(monkeypatch) -> _Gate:
    gate = _Gate()

    async def _wait_at_the_gate(self: ServerCell) -> None:
        gate.cells_that_dialled.append(self.meta.cell_id)
        gate.reached.set()
        await gate.opened.wait()

    monkeypatch.setattr(ServerCell, "init", _wait_at_the_gate)
    return gate


async def _dispose(server: RolloutServer) -> None:
    async with server.context_lock:
        await server.dispose()


class TestTheContextLockDuringACellBringUp:
    @pytest.mark.asyncio
    async def test_the_context_lock_is_free_while_a_cell_waits_at_its_launch_gate(self, monkeypatch):
        """A gate that never answers must not freeze weight updates, ft sweeps and every other cell of this run."""
        gate = _install_gate(monkeypatch)
        srv = _make_server()
        controller = _make_controller(server=srv)

        reconciling = asyncio.create_task(controller._reconcile(_CELL_ID, _make_cell_info()))
        await asyncio.wait_for(gate.reached.wait(), timeout=5.0)

        await asyncio.wait_for(controller.prepare_eval(), timeout=5.0)
        assert not controller.context_lock.locked

        gate.opened.set()
        await asyncio.wait_for(reconciling, timeout=5.0)
        assert list(srv.server_cells) == [_CELL_ID]
        await _dispose(srv)

    @pytest.mark.asyncio
    async def test_a_cell_reaches_this_run_only_after_its_gate_answered(self, monkeypatch):
        """Committing before the gate answers would put an engine that is not up yet in front of the router."""
        gate = _install_gate(monkeypatch)
        srv = _make_server()
        controller = _make_controller(server=srv)

        reconciling = asyncio.create_task(controller._reconcile(_CELL_ID, _make_cell_info()))
        await asyncio.wait_for(gate.reached.wait(), timeout=5.0)
        assert srv.server_cells == {}

        gate.opened.set()
        await asyncio.wait_for(reconciling, timeout=5.0)

        assert list(srv.server_cells) == [_CELL_ID]
        await _dispose(srv)


class TestConcurrentObservationsOfOneCell:
    @pytest.mark.asyncio
    async def test_two_observations_of_one_cell_install_it_once(self, monkeypatch):
        """Two providers announcing one cell at once would leave an engine running that nothing here tracks."""
        gate = _install_gate(monkeypatch)
        srv = _make_server()
        controller = _make_controller(server=srv)

        first = asyncio.create_task(controller._reconcile(_CELL_ID, _make_cell_info()))
        await asyncio.wait_for(gate.reached.wait(), timeout=5.0)
        second = asyncio.create_task(controller._reconcile(_CELL_ID, _make_cell_info()))
        await asyncio.sleep(0)
        gate.opened.set()

        with pytest.raises(ObservationSupersededError):
            await asyncio.wait_for(first, timeout=5.0)
        await asyncio.wait_for(second, timeout=5.0)

        assert list(srv.server_cells) == [_CELL_ID]
        assert gate.cells_that_dialled == [_CELL_ID, _CELL_ID]
        await _dispose(srv)

    @pytest.mark.asyncio
    async def test_a_removal_that_arrives_during_a_bring_up_wins(self, monkeypatch):
        """The bring-up started before the news that the cell is gone, so its commit must notice and stand down."""
        gate = _install_gate(monkeypatch)
        srv = _make_server()
        controller = _make_controller(server=srv)

        bringing_up = asyncio.create_task(controller._reconcile(_CELL_ID, _make_cell_info()))
        await asyncio.wait_for(gate.reached.wait(), timeout=5.0)
        removing = asyncio.create_task(controller._reconcile(_CELL_ID, None))
        await asyncio.sleep(0)
        gate.opened.set()

        with pytest.raises(ObservationSupersededError):
            await asyncio.wait_for(bringing_up, timeout=5.0)
        await asyncio.wait_for(removing, timeout=5.0)

        assert srv.server_cells == {}

    @pytest.mark.asyncio
    async def test_an_older_observation_that_arrives_last_never_supersedes_a_newer_one(self, monkeypatch):
        """A replayed observation can enter after a newer dispatch, and taking it would install a stale engine."""
        gate = _install_gate(monkeypatch)
        srv = _make_server()
        controller = _make_controller(server=srv)
        older = _make_cell_info()
        newer = dataclasses.replace(_make_cell_info(), worker_names=[f"{_CELL_ID}-0", f"{_CELL_ID}-1"])

        bringing_up_newer = asyncio.create_task(controller._reconcile(_CELL_ID, newer))
        await asyncio.wait_for(gate.reached.wait(), timeout=5.0)
        replaying_older = asyncio.create_task(controller._reconcile(_CELL_ID, older))
        await asyncio.sleep(0)
        gate.opened.set()

        await asyncio.wait_for(asyncio.gather(bringing_up_newer, replaying_older), timeout=5.0)

        assert srv.server_cells[_CELL_ID].observed_info == newer
        await _dispose(srv)


class TestABringUpThatOutlivesItsRun:
    @pytest.mark.asyncio
    async def test_a_cell_that_finished_starting_after_dispose_never_joins(self, monkeypatch):
        """Installing into a disposed server leaks a health checker into a run that is already over."""
        gate = _install_gate(monkeypatch)
        srv = _make_server()
        controller = _make_controller(server=srv)

        reconciling = asyncio.create_task(controller._reconcile(_CELL_ID, _make_cell_info()))
        await asyncio.wait_for(gate.reached.wait(), timeout=5.0)
        await asyncio.wait_for(controller.dispose(), timeout=5.0)
        gate.opened.set()

        await asyncio.wait_for(reconciling, timeout=5.0)

        assert srv.server_cells == {}

    @pytest.mark.asyncio
    async def test_a_watcher_that_refuses_to_stop_does_not_keep_the_servers_alive(self):
        """Every step of teardown has to run, or a cell's health checker probes an engine forever."""
        srv = _make_server()
        controller = _make_controller(server=srv)

        async def _refuse_to_stop() -> None:
            raise RuntimeError("injected stop failure")

        controller._watcher_disposers.append(_refuse_to_stop)

        await asyncio.wait_for(controller.dispose(), timeout=5.0)

        assert srv._disposed


class TestWhatCountsAsAChangedCell:
    @pytest.mark.asyncio
    async def test_a_cell_whose_addresses_moved_is_rebuilt_even_at_the_same_hash(self, monkeypatch):
        """The cell resolves its engine's address once, so a moved engine would be served through a dead url."""
        gate = _install_gate(monkeypatch)
        gate.opened.set()
        srv = _make_server()
        controller = _make_controller(server=srv)
        await controller._reconcile(_CELL_ID, _make_cell_info())
        first = srv.server_cells[_CELL_ID]

        moved = dataclasses.replace(_make_cell_info(), worker_names=[f"{_CELL_ID}-0", f"{_CELL_ID}-1"])
        await controller._reconcile(_CELL_ID, moved)

        assert srv.server_cells[_CELL_ID] is not first
        assert srv.server_cells[_CELL_ID].observed_info == moved
        await _dispose(srv)

    @pytest.mark.asyncio
    async def test_repeating_the_same_observation_keeps_the_cell_it_already_holds(self, monkeypatch):
        """Every snapshot repeats its cells, and rebuilding them each time would restart the fleet every period."""
        gate = _install_gate(monkeypatch)
        gate.opened.set()
        srv = _make_server()
        controller = _make_controller(server=srv)
        await controller._reconcile(_CELL_ID, _make_cell_info())
        first = srv.server_cells[_CELL_ID]

        await controller._reconcile(_CELL_ID, _make_cell_info())

        assert srv.server_cells[_CELL_ID] is first
        await _dispose(srv)
