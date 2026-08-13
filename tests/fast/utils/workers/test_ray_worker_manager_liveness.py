from __future__ import annotations

import asyncio

import pytest
import ray
from tests.fast.utils.workers.conftest import worker_manager_args
from tests.fast.utils.workers.fake_ray import READINESS_METHOD, FakeRayCluster

from miles.utils.workers import ray_worker_manager
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.types import WorkerCommBackend
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec


class DemoWorker:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


_WORKER_CLASS_PATH = f"{DemoWorker.__module__}.{DemoWorker.__qualname__}"


def _make_spec(name: str, *, num_cells: int = 1, num_workers_per_cell: int = 1) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=name,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master", allow_dynamic=True)],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=num_cells, num_workers_per_cell=num_workers_per_cell, num_gpus_per_worker=0
        ),
        worker_class=_WORKER_CLASS_PATH,
        ctor_kwargs=lambda _ctx: {},
    )


async def _launch(specs: list[ServeWorkerSpec], *, comm_backend: WorkerCommBackend) -> RayWorkerManager:
    manager = RayWorkerManager()
    await manager.init(worker_manager_args(), specs, {}, comm_backend=comm_backend)
    return manager


def _kill_worker_process(cluster: FakeRayCluster, *, handle_index: int) -> None:
    cluster.handles[handle_index].failing_methods[READINESS_METHOD] = ray.exceptions.RayActorError()


@pytest.fixture
def instant_scans(monkeypatch) -> None:
    monkeypatch.setattr(ray_worker_manager, "_LIVENESS_SCAN_INTERVAL_SECONDS", 0.0)


class TestScanLivenessOnce:
    async def test_a_cell_whose_workers_all_answer_stays_alive(self, fake_ray_cluster: FakeRayCluster):
        """The scan must not tear down a healthy cell, or every run would restart itself forever."""
        manager = await _launch([_make_spec("engine", num_cells=2)], comm_backend=WorkerCommBackend.RPC)

        await manager._scan_liveness_once()

        assert all(info.alive for info in manager.get_cell_infos(pool_ids=["engine"]).values())

    async def test_a_cell_that_lost_a_worker_stops_being_alive(self, fake_ray_cluster: FakeRayCluster):
        """A worker that exits on its own must reach the membership, or nobody ever replaces it."""
        manager = await _launch([_make_spec("engine", num_cells=2)], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=0)

        await manager._scan_liveness_once()

        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert not infos["engine-0"].alive
        assert infos["engine-1"].alive

    async def test_the_whole_cell_goes_when_one_of_its_workers_dies(self, fake_ray_cluster: FakeRayCluster):
        """A cell is the unit of recovery, so a surviving sibling of a dead rank must be reclaimed too."""
        manager = await _launch([_make_spec("engine", num_workers_per_cell=2)], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=1)

        await manager._scan_liveness_once()

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive
        assert [handle.killed for handle in fake_ray_cluster.handles] == [True, True]

    async def test_a_dropped_cell_can_be_started_again(self, fake_ray_cluster: FakeRayCluster):
        """Reporting the death is only useful if the cell is then restartable without a stop first."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=0)
        await manager._scan_liveness_once()

        await manager.start_cells(["engine-0"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive
        assert len(fake_ray_cluster.handles) == 2

    async def test_a_restarted_cell_reports_a_new_workers_hash(self, fake_ray_cluster: FakeRayCluster):
        """The consumer rebuilds its handles off the hash, so a self-death must move it as a stop does."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        before = manager.get_cell_infos(pool_ids=["engine"])["engine-0"].workers_hash
        _kill_worker_process(fake_ray_cluster, handle_index=0)

        await manager._scan_liveness_once()
        await manager.start_cells(["engine-0"])

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-0"].workers_hash != before

    async def test_a_ray_wire_cell_is_scanned_the_same_way(self, fake_ray_cluster: FakeRayCluster):
        """Liveness is a property of the actor process, not of the wire its methods travel on."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RAY)
        _kill_worker_process(fake_ray_cluster, handle_index=0)

        await manager._scan_liveness_once()

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive

    async def test_an_already_stopped_cell_is_not_probed(self, fake_ray_cluster: FakeRayCluster):
        """Probing a cell nobody launched would raise on its missing actors and abort the whole scan."""
        manager = await _launch([_make_spec("engine", num_cells=2)], comm_backend=WorkerCommBackend.RPC)
        await manager.stop_cells(["engine-0"])
        fake_ray_cluster.calls.clear()

        await manager._scan_liveness_once()

        probed = {call.handle.index for call in fake_ray_cluster.calls_of(READINESS_METHOD)}
        assert probed == {1}


class TestScanLivenessRacesWithMembershipChanges:
    async def test_a_cell_stopped_while_another_one_is_probed_is_skipped(self, fake_ray_cluster: FakeRayCluster):
        """A suspend landing mid-scan empties a cell's actors, and probing them would abort the whole scan."""
        manager = await _launch([_make_spec("engine", num_cells=2)], comm_backend=WorkerCommBackend.RPC)
        fake_ray_cluster.handles[0].hanging_methods[READINESS_METHOD] = 0.2

        scan = asyncio.create_task(manager._scan_liveness_once())
        await asyncio.sleep(0.05)
        await manager.stop_cells(["engine-1"])
        await scan

        infos = manager.get_cell_infos(pool_ids=["engine"])
        assert infos["engine-0"].alive
        assert not infos["engine-1"].alive

    async def test_a_cell_restarted_while_being_probed_survives(self, fake_ray_cluster: FakeRayCluster):
        """The dead workers the scan saw belong to the old generation, so the new one must not pay for them."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=0)
        fake_ray_cluster.handles[0].hanging_methods[READINESS_METHOD] = 0.2

        scan = asyncio.create_task(manager._scan_liveness_once())
        await asyncio.sleep(0.05)
        await manager.stop_cells(["engine-0"])
        await manager.start_cells(["engine-0"])
        await scan

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive


class TestScanLivenessOnlyTrustsAProvenDeath:
    async def test_a_worker_that_does_not_answer_in_time_is_kept(self, fake_ray_cluster: FakeRayCluster):
        """A busy worker must not be declared dead, or a slow train step would kill its own cell."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        fake_ray_cluster.handles[0].hanging_methods[READINESS_METHOD] = 3600.0

        await manager._scan_liveness_once()

        assert manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive

    async def test_an_application_error_from_the_probe_is_treated_as_death(self, fake_ray_cluster: FakeRayCluster):
        """A worker answering its readiness probe with a task error is as unusable as a missing one."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        fake_ray_cluster.handles[0].failing_methods[READINESS_METHOD] = ray.exceptions.RayTaskError.__new__(
            ray.exceptions.RayTaskError
        )

        await manager._scan_liveness_once()

        assert not manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive


class TestScanLivenessLoop:
    async def test_init_starts_the_scan(self, fake_ray_cluster: FakeRayCluster, instant_scans: None):
        """The scan is what makes the manager notice a death nobody reported, so it must run unasked."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        _kill_worker_process(fake_ray_cluster, handle_index=0)

        await _wait_until(lambda: not manager.get_cell_infos(pool_ids=["engine"])["engine-0"].alive)

        manager._liveness_scan_task.cancel()

    async def test_a_failing_scan_does_not_end_the_loop(self, fake_ray_cluster: FakeRayCluster, instant_scans: None):
        """One bad scan must not silently leave the run without any liveness reporting at all."""
        manager = await _launch([_make_spec("engine")], comm_backend=WorkerCommBackend.RPC)
        scans: list[int] = []

        async def scan_once() -> None:
            scans.append(len(scans))
            if len(scans) == 1:
                raise RuntimeError("scan failed")

        manager._scan_liveness_once = scan_once

        await _wait_until(lambda: len(scans) >= 3)

        assert not manager._liveness_scan_task.done()
        manager._liveness_scan_task.cancel()


async def _wait_until(predicate, *, timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "the condition never became true"
        await asyncio.sleep(0.01)
