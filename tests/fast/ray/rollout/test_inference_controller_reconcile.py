from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args, make_cell_spec

from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell
from miles.utils.workers.worker_provider.base import CellInfo, CellMember
from miles.utils.workers.worker_spec import WorkerPlacement


def _cell_info(*, cell_id: str = "cell-0", num_members: int = 1, port: int = 30000) -> CellInfo:
    return CellInfo(
        cell_id=cell_id,
        members=[
            CellMember(
                handle=fake_actor_handle(),
                payload={"host": f"10.0.0.{index + 1}", "port": port + index},
                placement=WorkerPlacement(local_index=index, global_rank=index, base_gpu_id=index),
            )
            for index in range(num_members)
        ],
    )


@pytest.fixture(autouse=True)
def stub_rollout_engine_lock():
    """The controller's ray Lock actor is irrelevant to reconcile and needs a cluster."""
    with patch("miles.ray.rollout.inference_controller.Lock", MagicMock()):
        yield


def _make_controller(*, update_weights: bool = True, num_nodes: int = 1, needs_offload: bool = False):
    args = make_args(num_gpus_per_node=8)
    cell = ServerCell(
        args=args,
        spec=make_cell_spec(args=args, num_nodes=num_nodes, needs_offload=needs_offload),
    )
    router = MagicMock()
    router.add_worker = AsyncMock()
    router.remove_worker = AsyncMock()
    srv = RolloutServer(
        server_cells={cell.cell_id: cell},
        args=args,
        model_name="default",
        update_weights=update_weights,
    )
    controller = InferenceController(args, pg=None)
    controller.servers = {"default": srv}
    return controller, srv, cell, router


def _with_router(srv, router):
    return patch.object(RolloutServer, "router_api_client", property(lambda self: router))


class TestReconcileAdd:
    async def test_an_observed_cell_is_attached_and_registered(self):
        """The controller learns about workers only through the provider's report."""
        controller, srv, cell, router = _make_controller()
        with _with_router(srv, router):
            await controller._reconcile("cell-0", _cell_info())

        assert cell.is_alive
        assert cell.addr_info.server_url == "http://10.0.0.1:30000"
        router.add_worker.assert_awaited_once()

    async def test_an_updatable_cell_flags_new_engines_for_the_updater(self):
        """The weight updater must re-snapshot its engines after a cell appears."""
        controller, srv, cell, router = _make_controller(update_weights=True)
        with _with_router(srv, router):
            await controller._reconcile("cell-0", _cell_info())

        assert srv.has_new_engines is True

    async def test_a_frozen_cell_does_not_flag_new_engines(self):
        """A frozen model receives no weight updates, so nothing has to re-snapshot."""
        controller, srv, cell, router = _make_controller(update_weights=False)
        with _with_router(srv, router):
            await controller._reconcile("cell-0", _cell_info())

        assert srv.has_new_engines is False

    async def test_a_half_reported_multi_node_cell_is_left_alone(self):
        """Attaching before every node-rank exists would address a partial engine."""
        controller, srv, cell, router = _make_controller(num_nodes=2)
        with _with_router(srv, router):
            await controller._reconcile("cell-0", _cell_info(num_members=1))

        assert not cell.is_allocated
        router.add_worker.assert_not_awaited()

    async def test_a_cell_that_comes_back_mid_run_gives_its_memory_back(self):
        """A restarted engine holds the GPU memory the trainer took over."""
        controller, srv, cell, router = _make_controller(needs_offload=True)
        controller.rollout_id = 0
        client = MagicMock()
        client.release_memory_occupation = AsyncMock()
        client.resume_memory_occupation = AsyncMock()
        with _with_router(srv, router), patch.object(ServerCell, "api_client", property(lambda self: client)):
            await controller._reconcile("cell-0", _cell_info())

        client.release_memory_occupation.assert_awaited_once()
        client.resume_memory_occupation.assert_awaited_once()

    async def test_the_startup_pass_keeps_the_memory_the_engine_already_holds(self):
        """At startup the trainer has not taken the device yet, so nothing is released."""
        controller, srv, cell, router = _make_controller(needs_offload=True)
        client = MagicMock()
        client.release_memory_occupation = AsyncMock()
        with _with_router(srv, router), patch.object(ServerCell, "api_client", property(lambda self: client)):
            await controller._reconcile("cell-0", _cell_info())

        client.release_memory_occupation.assert_not_awaited()


class TestFailedAttach:
    async def test_a_failed_registration_leaves_the_cell_detached(self):
        """A cell that stayed attached after a failed add would never be retried."""
        controller, srv, cell, router = _make_controller()
        router.add_worker.side_effect = RuntimeError("router rejected the worker")

        with _with_router(srv, router), pytest.raises(RuntimeError, match="router rejected"):
            await controller._reconcile("cell-0", _cell_info())

        assert not cell.is_allocated

    async def test_the_same_workers_are_attached_again_on_the_next_poll(self):
        """The provider keeps reporting the same workers, so the retry must go through."""
        controller, srv, cell, router = _make_controller()
        router.add_worker.side_effect = RuntimeError("router rejected the worker")
        info = _cell_info()

        with _with_router(srv, router):
            with pytest.raises(RuntimeError):
                await controller._reconcile("cell-0", info)
            router.add_worker.side_effect = None
            await controller._reconcile("cell-0", info)

        assert cell.is_alive
        assert router.add_worker.await_count == 2

    async def test_a_failed_memory_release_leaves_the_cell_detached(self):
        """An engine that cannot give its memory back must not be marked alive."""
        controller, srv, cell, router = _make_controller(needs_offload=True)
        client = MagicMock()
        client.release_memory_occupation = AsyncMock(side_effect=RuntimeError("engine wedged"))
        with (
            _with_router(srv, router),
            patch.object(ServerCell, "api_client", property(lambda self: client)),
            pytest.raises(RuntimeError, match="engine wedged"),
        ):
            await controller._reconcile("cell-0", _cell_info())

        assert not cell.is_allocated
        router.add_worker.assert_not_awaited()


class TestReconcileRemove:
    async def test_a_vanished_cell_is_unregistered_and_detached(self):
        """A cell whose workers are gone must stop receiving router traffic."""
        controller, srv, cell, router = _make_controller()
        with _with_router(srv, router):
            await controller._reconcile("cell-0", _cell_info())
            await controller._reconcile("cell-0", None)

        assert not cell.is_allocated
        router.remove_worker.assert_awaited_once()

    async def test_a_vanished_cell_flags_new_engines_for_the_updater(self):
        """The updater must drop the dead engine from its snapshot."""
        controller, srv, cell, router = _make_controller()
        with _with_router(srv, router):
            await controller._reconcile("cell-0", _cell_info())
            srv.clear_has_new_engines()
            await controller._reconcile("cell-0", None)

        assert srv.has_new_engines is True

    async def test_a_router_that_rejects_the_unregister_still_detaches_the_cell(self):
        """The workers are already gone, so bookkeeping must not hang on the router."""
        controller, srv, cell, router = _make_controller()
        with _with_router(srv, router):
            await controller._reconcile("cell-0", _cell_info())
            router.remove_worker.side_effect = RuntimeError("router rejected the removal")
            await controller._reconcile("cell-0", None)

        assert not cell.is_allocated

    async def test_an_already_detached_cell_needs_no_router_call(self):
        """A cell that was never attached has no url the router could know."""
        controller, srv, cell, router = _make_controller()
        with _with_router(srv, router):
            await controller._reconcile("cell-0", None)

        router.remove_worker.assert_not_awaited()


class TestReconcileReplace:
    async def test_new_members_replace_the_attached_ones(self):
        """A restart hands the cell different workers, so the old url must go."""
        controller, srv, cell, router = _make_controller()
        with _with_router(srv, router):
            await controller._reconcile("cell-0", _cell_info(port=30000))
            await controller._reconcile("cell-0", _cell_info(port=40000))

        assert cell.addr_info.server_url == "http://10.0.0.1:40000"
        router.remove_worker.assert_awaited_once()
        assert router.add_worker.await_count == 2

    async def test_unchanged_members_are_left_attached(self):
        """Re-attaching a healthy cell would re-init a running engine."""
        controller, srv, cell, router = _make_controller()
        with _with_router(srv, router):
            info = _cell_info()
            await controller._reconcile("cell-0", info)
            await controller._reconcile("cell-0", info)

        router.add_worker.assert_awaited_once()
        router.remove_worker.assert_not_awaited()


class TestReconcileGate:
    async def test_a_paused_controller_defers_reconciles(self):
        """The engine list must stay still while a weight update snapshots it."""
        controller, srv, cell, router = _make_controller()
        await controller.health_monitoring_pause()

        with _with_router(srv, router):
            task = asyncio.create_task(controller._reconcile("cell-0", _cell_info()))
            await asyncio.sleep(0.05)
            assert not cell.is_allocated

            await controller.health_monitoring_resume()
            await asyncio.wait_for(task, timeout=5)

        assert cell.is_alive

    async def test_pause_waits_for_an_in_flight_reconcile(self):
        """Pausing mid-attach would hand the updater a half-attached cell."""
        controller, srv, cell, router = _make_controller()
        release = asyncio.Event()

        async def _slow_add_worker(**kwargs):
            await release.wait()

        router.add_worker.side_effect = _slow_add_worker

        with _with_router(srv, router):
            task = asyncio.create_task(controller._reconcile("cell-0", _cell_info()))
            await asyncio.sleep(0.05)
            pause = asyncio.create_task(controller.health_monitoring_pause())
            await asyncio.sleep(0.05)
            assert not pause.done()

            release.set()
            await asyncio.wait_for(task, timeout=5)
            await asyncio.wait_for(pause, timeout=5)

    async def test_an_unknown_cell_id_is_rejected(self):
        """A cell id that names no server means the provider and the specs disagree."""
        controller, srv, cell, router = _make_controller()
        with pytest.raises(AssertionError, match="exactly one cell"):
            await controller._reconcile("nope-0", _cell_info(cell_id="nope-0"))
