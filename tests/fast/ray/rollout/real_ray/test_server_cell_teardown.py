from __future__ import annotations

import asyncio
import threading
import time

import ray
from tests.fast.ray.rollout.conftest import make_args, make_cell_spec
from tests.fast.ray.rollout.real_ray.conftest import detach_cell, start_cells

import miles.utils.workers.ray_worker_manager as manager_module
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_provider.base import CellInfo, CellMember
from miles.utils.workers.worker_spec import WorkerPlacement


@ray.remote(num_cpus=0)
class _HangingEngine:
    def shutdown(self):
        time.sleep(3600)


def _build_server() -> RolloutServer:
    args = make_args(num_gpus_per_node=8)
    cell = ServerCell(args=args, spec=make_cell_spec(args=args))
    return RolloutServer(server_cells={"cell-0": cell}, args=args)


def _adopt_actor(worker_manager: RayWorkerManager, cell, actor_handle) -> None:
    """Register a hand-made actor with the manager and attach the cell to it."""
    from tests.fast.ray.rollout.conftest import adopt_cell_workers

    adopt_cell_workers(
        worker_manager,
        cell_id=cell.cell_id,
        payloads=[{"host": "127.0.0.1", "port": 30000}],
        actors=[actor_handle],
    )
    cell.attach(
        CellInfo(
            cell_id=cell.cell_id,
            members=[
                CellMember(
                    handle=actor_handle,
                    payload={"host": "127.0.0.1", "port": 30000},
                    placement=WorkerPlacement(local_index=0, global_rank=0, base_gpu_id=0),
                )
            ],
        )
    )


def _is_dead(actor_handle, *, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ray.get(actor_handle.__ray_ready__.remote(), timeout=1.0)
        except ray.exceptions.RayActorError:
            return True
        except ray.exceptions.GetTimeoutError:
            pass
        time.sleep(0.1)
    return False


class TestTeardownIsTerminal:
    async def test_a_failing_shutdown_still_kills_the_actor(self, patched_sglang_engine, placement_group_factory):
        """A graceful shutdown that raises must not leave the actor and its server process behind."""
        worker_manager = RayWorkerManager(pg=placement_group_factory(1))
        srv = _build_server()
        cell = srv.server_cells["cell-0"]
        await start_cells([cell], worker_manager)
        actor_handle = cell.primary_actor_handle
        ray.get(actor_handle.set_fault.remote("shutdown", RuntimeError("shutdown blew up")))

        await detach_cell(cell, worker_manager)

        assert _is_dead(actor_handle)
        assert not cell.is_allocated

    def test_a_hanging_shutdown_does_not_block_teardown(self, monkeypatch, ray_local_mode):
        """A wedged engine must not stall teardown forever, since teardown is how a wedged engine is reclaimed."""
        monkeypatch.setattr(manager_module, "SHUTDOWN_TIMEOUT", 0.5)
        worker_manager = RayWorkerManager(pg=(None, [], []))
        srv = _build_server()
        cell = srv.server_cells["cell-0"]
        actor_handle = _HangingEngine.remote()
        _adopt_actor(worker_manager, cell, actor_handle)

        finished = threading.Event()

        def _teardown():
            asyncio.run(detach_cell(cell, worker_manager))
            finished.set()

        thread = threading.Thread(target=_teardown, daemon=True)
        thread.start()
        thread.join(timeout=30)

        assert finished.is_set(), "the teardown waited on a shutdown that never returns"
        assert _is_dead(actor_handle)
        assert not cell.is_allocated
