from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
import ray
from tests.fast.ray.rollout.conftest import fake_engine

import miles.utils.workers.cell_launch as cell_launch
from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.ray_worker_manager import RayWorkerManager, WorkerManagerClient
from miles.utils.workers.worker_spec import (
    BaseCellSpec,
    CellAddressing,
    CommandWorkerSpec,
    PortInfo,
    RayActorOptions,
    SchedulingSpec,
    WorkerLaunchPlan,
)

_MANAGER_MODULE = "miles.utils.workers.ray_worker_manager"


def _engine() -> object:
    """A fake engine whose ``run`` is awaitable, as the real remote call is."""
    engine = fake_engine(host="10.0.0.1", port_seed=0)
    engine.run.remote.side_effect = lambda **kwargs: asyncio.sleep(0)
    return engine


def _payloads(addressing: CellAddressing) -> list[dict]:
    return [
        {"host": node_ip, **ports}
        for node_ip, ports in zip(addressing.node_ips, addressing.per_worker_ports, strict=True)
    ]


async def _noop_wait(addressing, is_worker_alive):
    return None


def _spec(*, cell_id: str = "cell-0", num_workers: int = 1) -> BaseCellSpec:
    worker = CommandWorkerSpec(
        name="fake-worker",
        port_infos=[PortInfo(name="server", static_port=30000, mode="per_worker", allow_dynamic=True)],
        env_var=lambda placement: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=num_workers, num_gpus_per_worker=1),
        ray_options=RayActorOptions(num_cpus=0.2, num_gpus=0.2),
        build_launch_plan=lambda placement, addressing: WorkerLaunchPlan(cmd=f"serve --rank {placement.global_rank}"),
        build_member_payloads=_payloads,
        wait_cell_ready=_noop_wait,
    )
    return BaseCellSpec(worker=worker, cell_id=cell_id, rank_offset=0, gpu_offset=0)


def _with_actors(actors: list):
    """Hand the manager these actors instead of really creating them."""
    return patch.object(
        cell_launch, "create_cell_worker_actor", side_effect=lambda *, placement, **kw: actors[placement.local_index]
    )


def _manager(**kwargs) -> RayWorkerManager:
    return RayWorkerManager(pg=(None, list(range(8)), list(range(8))), **kwargs)


async def _start(manager: RayWorkerManager, spec: BaseCellSpec) -> None:
    """Register the cell (which brings it up) or restart it by id, as its callers do."""
    if spec.cell_id not in manager.registered_cell_ids():
        await manager.register_cells([spec])
    else:
        await manager.start_cell(spec.cell_id)


class TestStartCell:
    async def test_register_brings_every_cell_up(self, patch_ray_get):
        """A registered spec is a running cell; no separate start call is needed."""
        manager = _manager()
        with _with_actors([_engine()]):
            await manager.register_cells([_spec()])
        assert manager.cell_ids() == ["cell-0"]

    async def test_tracks_one_worker_per_spec_worker(self, patch_ray_get):
        """The manager is the only place that knows which actors a cell owns."""
        actors = [_engine() for _ in range(2)]
        manager = _manager()
        with _with_actors(actors):
            await _start(manager, _spec(num_workers=2))
        assert [worker.actor for worker in manager.cell_workers("cell-0")] == actors

    async def test_records_the_member_payload_each_worker_was_started_with(self, patch_ray_get):
        """Consumers derive their urls from the payload, so it must be kept verbatim."""
        manager = _manager()
        with _with_actors([_engine()]):
            await _start(manager, _spec())
        (worker,) = manager.cell_workers("cell-0")
        assert worker.payload["host"] == "10.0.0.1"
        assert "server" in worker.payload

    async def test_run_launches_every_worker(self, patch_ray_get):
        """A cell is one distributed engine, so no node-rank may be left unlaunched."""
        actors = [_engine() for _ in range(2)]
        manager = _manager()
        with _with_actors(actors):
            await _start(manager, _spec(num_workers=2))
        for actor in actors:
            actor.run.remote.assert_called_once()

    async def test_a_second_start_of_a_live_cell_is_rejected(self, patch_ray_get):
        """Starting over live workers would leak them, so the manager refuses."""
        manager = _manager()
        with _with_actors([_engine()]):
            await _start(manager, _spec())
            with pytest.raises(AssertionError, match="already has live workers"):
                await _start(manager, _spec())

    async def test_cells_share_one_allocator_so_their_ports_never_collide(self, patch_ray_get):
        """Two cells on one node must not be handed the same port."""
        manager = _manager()
        with _with_actors([_engine()]):
            await _start(manager, _spec(cell_id="cell-0"))
            await _start(manager, _spec(cell_id="cell-1"))
        ports = [manager.cell_workers(cell_id)[0].payload["server"] for cell_id in ("cell-0", "cell-1")]
        assert ports[0] != ports[1]

    async def test_an_explicit_allocator_keeps_its_cursor(self, patch_ray_get):
        """Recovery reuses the startup allocator so it does not rescan from the base port."""
        allocator = PortAllocator()
        allocator.alloc(engine=fake_engine(host="10.0.0.1", port_seed=20000), node_ip="10.0.0.1")
        manager = _manager(_port_allocator=allocator)
        with _with_actors([_engine()]):
            await _start(manager, _spec())
        assert manager.cell_workers("cell-0")[0].payload["server"] >= 20000


class TestFailedBringUp:
    async def test_a_failed_launch_kills_the_actors_it_created(self, patch_ray_get):
        """A half-started cell must not leave actors nothing can reclaim."""
        actors = [_engine()]

        async def _blow_up(**kwargs):
            raise RuntimeError("run blew up")

        actors[0].run.remote.side_effect = _blow_up
        manager = _manager()

        with (
            _with_actors(actors),
            patch(f"{_MANAGER_MODULE}.ray") as ray_mock,
            pytest.raises(RuntimeError, match="run blew up"),
        ):
            await _start(manager, _spec())

        assert ray_mock.kill.call_count == 1
        assert manager.cell_workers("cell-0") == []

    async def test_the_cell_can_be_started_again_after_a_failed_launch(self, patch_ray_get):
        """The failed attempt must release the cell id, or recovery is impossible."""

        async def _blow_up(**kwargs):
            raise RuntimeError("run blew up")

        failing = [_engine()]
        failing[0].run.remote.side_effect = _blow_up
        manager = _manager()
        with _with_actors(failing), patch(f"{_MANAGER_MODULE}.ray"), pytest.raises(RuntimeError):
            await _start(manager, _spec())

        with _with_actors([_engine()]):
            await _start(manager, _spec())
        assert len(manager.cell_workers("cell-0")) == 1

    async def test_a_cell_still_coming_up_is_not_offered_to_consumers(self, patch_ray_get):
        """An engine mid-launch would be registered with the router as if it were serving."""
        started = asyncio.Event()
        release = asyncio.Event()

        async def _slow_init(**kwargs):
            started.set()
            await release.wait()

        actors = [_engine()]
        actors[0].run.remote.side_effect = _slow_init
        manager = _manager()

        with _with_actors(actors):
            task = asyncio.create_task(_start(manager, _spec()))
            await asyncio.wait_for(started.wait(), timeout=5)
            assert manager.cell_workers("cell-0") == []

            release.set()
            await asyncio.wait_for(task, timeout=5)
        assert len(manager.cell_workers("cell-0")) == 1

    async def test_a_second_start_while_one_is_in_flight_is_rejected(self, patch_ray_get):
        """Two concurrent starts would each create actors and one would lose its record."""
        release = asyncio.Event()

        async def _slow_init(**kwargs):
            await release.wait()

        actors = [_engine()]
        actors[0].run.remote.side_effect = _slow_init
        manager = _manager()

        with _with_actors(actors):
            task = asyncio.create_task(_start(manager, _spec()))
            await asyncio.sleep(0.05)
            with pytest.raises(AssertionError, match="already has live workers"):
                await _start(manager, _spec())

            release.set()
            await asyncio.wait_for(task, timeout=5)


class TestStopCell:
    async def test_kills_and_forgets_every_worker_of_the_cell(self, patch_ray_get):
        """Teardown is whole-cell: a survivor would belong to a dead process group."""
        manager = _manager()
        with _with_actors([_engine() for _ in range(2)]):
            await _start(manager, _spec(cell_id="cell-0", num_workers=2))
            await _start(manager, _spec(cell_id="cell-1"))

        with patch(f"{_MANAGER_MODULE}.ray") as ray_mock:
            await manager.stop_cell("cell-0")

        assert ray_mock.kill.call_count == 2
        assert manager.cell_workers("cell-0") == []
        assert len(manager.cell_workers("cell-1")) == 1

    async def test_shutdown_precedes_the_kill(self, patch_ray_get):
        """A killed engine cannot free its GPU memory, so it is asked to shut down first."""
        actors = [_engine()]
        manager = _manager()
        with _with_actors(actors):
            await _start(manager, _spec())

        events: list[str] = []

        async def _shutdown():
            events.append("shutdown")

        actors[0].shutdown.remote.side_effect = _shutdown
        with patch(f"{_MANAGER_MODULE}.ray") as ray_mock:
            ray_mock.kill.side_effect = lambda handle: events.append("kill")
            await manager.stop_cell("cell-0")

        assert events == ["shutdown", "kill"]

    async def test_a_failing_shutdown_still_kills_the_worker(self, patch_ray_get):
        """Teardown is how a wedged engine is reclaimed, so it must not abort on errors."""
        actors = [_engine()]
        manager = _manager()
        with _with_actors(actors):
            await _start(manager, _spec())

        async def _blow_up():
            raise RuntimeError("shutdown blew up")

        actors[0].shutdown.remote.side_effect = _blow_up
        with patch(f"{_MANAGER_MODULE}.ray") as ray_mock:
            await manager.stop_cell("cell-0")
            assert ray_mock.kill.call_count == 1
        assert manager.cell_workers("cell-0") == []

    async def test_a_cancelled_shutdown_still_kills_the_worker(self, patch_ray_get):
        """Cancelling the wait must not strand actors that nothing records any more."""
        actors = [_engine()]
        manager = _manager()
        with _with_actors(actors):
            await _start(manager, _spec())

        async def _hang():
            await asyncio.Event().wait()

        actors[0].shutdown.remote.side_effect = _hang
        with patch(f"{_MANAGER_MODULE}.ray") as ray_mock:
            task = asyncio.create_task(manager.stop_cell("cell-0"))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert ray_mock.kill.call_count == 1
        assert manager.cell_workers("cell-0") == []

    async def test_stopping_an_unknown_cell_is_a_noop(self):
        """A caller may stop a cell that was never brought up here."""
        manager = _manager()
        with patch(f"{_MANAGER_MODULE}.ray") as ray_mock:
            await manager.stop_cell("cell-0")
        ray_mock.kill.assert_not_called()

    async def test_a_stopped_cell_can_be_started_again(self, patch_ray_get):
        """Restart is how a dead cell comes back, so the manager must forget the old workers."""
        manager = _manager()
        with _with_actors([_engine()]):
            await _start(manager, _spec())
            with patch(f"{_MANAGER_MODULE}.ray"):
                await manager.stop_cell("cell-0")
            await _start(manager, _spec())
        assert len(manager.cell_workers("cell-0")) == 1


class TestLaunchIsDrivenByTheSpec:
    async def test_the_spec_decides_the_worker_the_placement_and_the_ray_request(self, patch_ray_get):
        """Nothing about the worker reaches the manager except through its spec."""
        captured: list[dict] = []
        actor = _engine()

        def _create(*, worker, placement, pg_handle, bundle_index):
            captured.append(dict(worker=worker, placement=placement, bundle_index=bundle_index))
            return actor

        manager = RayWorkerManager(pg=(None, [4, 5, 6, 7], [10, 11, 12, 13]))
        spec = BaseCellSpec(worker=_spec().worker, cell_id="cell-0", rank_offset=2, gpu_offset=1)

        with patch.object(cell_launch, "create_cell_worker_actor", side_effect=_create):
            await _start(manager, spec)

        assert captured[0]["worker"] is spec.worker
        assert captured[0]["placement"].global_rank == 2
        assert captured[0]["placement"].base_gpu_id == 11
        assert captured[0]["bundle_index"] == 5


class TestWorkerManagerClient:
    async def test_forwards_lifecycle_and_reads_to_the_named_actor(self, monkeypatch):
        """The client keeps the manager call surface while the work happens in the actor."""
        calls: list[tuple] = []

        class _Method:
            def __init__(self, name: str) -> None:
                self._name = name

            def remote(self, *args, **kwargs):
                calls.append((self._name, args, kwargs))
                future = asyncio.get_event_loop().create_future()
                future.set_result(f"{self._name}-result")
                return future

        class _Handle:
            def __getattr__(self, name: str) -> _Method:
                return _Method(name)

        monkeypatch.setattr(ray, "get", lambda ref: ref.result())
        client = WorkerManagerClient(actor_handle=_Handle())

        await client.register_cells(["spec"])
        await client.start_cell("cell-0")
        await client.stop_cell("cell-0")
        assert client.cell_ids() == "cell_ids-result"
        assert client.cell_workers("cell-0") == "cell_workers-result"
        assert [name for name, _, _ in calls] == [
            "register_cells",
            "start_cell",
            "stop_cell",
            "cell_ids",
            "cell_workers",
        ]
