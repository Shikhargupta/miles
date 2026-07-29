from __future__ import annotations

import asyncio
from unittest.mock import patch

from tests.fast.ray.rollout.conftest import FakeWorkerCellControl, FakeWorkerHandle, FakeWorkerProvider, make_args

from miles.ray.rollout.inference_controller import InferenceController, _ReconcileGate
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell
from miles.utils.workers.worker_provider.base import CellInfo


class _RecordingRouterApiClient:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def add_worker(self, **kwargs):
        self.events.append(("add_worker", kwargs))

    async def remove_worker(self, **kwargs):
        self.events.append(("remove_worker", kwargs))


def _make_cell(*, update_weights: bool = True) -> tuple[ServerCell, FakeWorkerHandle]:
    handle = FakeWorkerHandle(addr_and_ports={"server_addr": "10.0.0.1", "server_port": 30000})
    cell = ServerCell(
        args=make_args(num_gpus_per_node=8),
        worker_type="regular",
        cell_id="sglang-default-group0-0",
        spec_name="sglang-default-group0",
        cell_index=0,
        update_weights=update_weights,
        provider=FakeWorkerProvider({"sglang-default-group0-0-0": handle}),
        worker_cell_control=FakeWorkerCellControl(),
    )
    return cell, handle


def _make_controller(servers: dict[str, RolloutServer]) -> InferenceController:
    controller = InferenceController.__new__(InferenceController)
    controller.args = make_args(num_gpus_per_node=8)
    controller.servers = servers
    controller.rollout_id = -1
    controller._reconcile_gate = _ReconcileGate()
    controller._cell_ops_lock = asyncio.Lock()
    controller._watcher_disposers = []
    return controller


def _make_setup(*, update_weights: bool = True):
    cell, handle = _make_cell(update_weights=update_weights)
    router_client = _RecordingRouterApiClient()
    srv = RolloutServer(
        server_cells={cell.cell_id: cell},
        args=cell.args,
        router_ip="10.0.0.9",
        router_port=9000,
        update_weights=update_weights,
    )
    srv._recording_router_client = router_client
    controller = _make_controller({"default": srv})
    return controller, srv, cell, handle, router_client


def _observed(cell: ServerCell, members_hash: str = "hash-a") -> CellInfo:
    return CellInfo(
        cell_id=cell.cell_id, spec_name=cell.spec_name, members_hash=members_hash, member_urls=["http://10.0.0.1:30000"]
    )


def _with_recording_client():
    return patch.object(RolloutServer, "_router_api_client", property(lambda self: self._recording_router_client))


class TestReconcileAdd:
    async def test_updatable_cell_attaches_without_router_registration(self):
        """A recovered updatable cell must wait for a weight sync before serving."""
        controller, srv, cell, handle, router_client = _make_setup()

        with _with_recording_client():
            await controller._reconcile(cell.cell_id, _observed(cell))

        assert cell.is_allocated and not cell.is_alive
        assert handle.calls == ["get_addr_and_ports", "init"]
        assert srv.has_new_engines is True
        assert router_client.events == []

    async def test_non_updatable_cell_registers_immediately(self):
        """A frozen model's engines carry disk weights, so they serve right away."""
        controller, srv, cell, _handle, router_client = _make_setup(update_weights=False)

        with _with_recording_client():
            await controller._reconcile(cell.cell_id, _observed(cell))

        assert cell.is_alive
        assert [name for name, _ in router_client.events] == ["add_worker"]

    async def test_clear_has_new_engines_promotes_synced_cells(self):
        """After update_weights finishes, unsynced cells join the router and go alive."""
        controller, srv, cell, _handle, router_client = _make_setup()
        with _with_recording_client():
            await controller._reconcile(cell.cell_id, _observed(cell))

            await controller.clear_updatable_has_new_engines()

        assert cell.is_alive
        assert [name for name, _ in router_client.events] == ["add_worker"]
        assert srv.has_new_engines is False

    async def test_partially_started_cell_is_skipped(self):
        """A cell whose members have not all published urls is not attached yet."""
        controller, _srv, cell, handle, _router_client = _make_setup()
        observed = CellInfo(cell_id=cell.cell_id, spec_name=cell.spec_name, members_hash="hash-a", member_urls=[])

        with _with_recording_client():
            await controller._reconcile(cell.cell_id, observed)

        assert not cell.is_allocated
        assert handle.calls == []

    async def test_failed_attach_rolls_back_to_stopped(self):
        """An attach whose init dies must not leave the cell half-allocated."""
        controller, _srv, cell, handle, _router_client = _make_setup()

        async def _boom() -> dict:
            raise RuntimeError("engine gone")

        handle.get_addr_and_ports = _boom

        with _with_recording_client():
            try:
                await controller._reconcile(cell.cell_id, _observed(cell))
            except RuntimeError:
                pass

        assert not cell.is_allocated


class TestReconcileRemove:
    async def test_vanished_cell_is_unregistered_and_stopped(self):
        """A cell the manager no longer reports must leave the router and the books."""
        controller, srv, cell, _handle, router_client = _make_setup()
        with _with_recording_client():
            await controller._reconcile(cell.cell_id, _observed(cell))
            await controller.clear_updatable_has_new_engines()

            await controller._reconcile(cell.cell_id, None)

        assert not cell.is_allocated
        assert [name for name, _ in router_client.events] == ["add_worker", "remove_worker"]

    async def test_members_change_replaces_the_cell(self):
        """A generation bump means new workers: the old attachment must be replaced."""
        controller, srv, cell, handle, _router_client = _make_setup()
        with _with_recording_client():
            await controller._reconcile(cell.cell_id, _observed(cell, members_hash="hash-a"))
            calls_after_first = list(handle.calls)

            await controller._reconcile(cell.cell_id, _observed(cell, members_hash="hash-b"))

        assert cell.is_allocated
        assert handle.calls == calls_after_first + ["get_addr_and_ports", "init"]

    async def test_unchanged_observation_is_a_noop(self):
        """Steady state: the same members hash must not churn the cell."""
        controller, _srv, cell, handle, _router_client = _make_setup()
        with _with_recording_client():
            await controller._reconcile(cell.cell_id, _observed(cell, members_hash="hash-a"))
            calls_after_first = list(handle.calls)

            await controller._reconcile(cell.cell_id, _observed(cell, members_hash="hash-a"))

        assert handle.calls == calls_after_first


class TestSuspendResumeMembership:
    async def test_api_stop_cell_flags_membership_change_for_the_updater(self):
        """Suspending an updatable cell must force the next update to rebuild its groups."""
        controller, srv, cell, _handle, _router_client = _make_setup()
        with _with_recording_client():
            await controller._reconcile(cell.cell_id, _observed(cell))
            await controller.clear_updatable_has_new_engines()
            assert srv.has_new_engines is False

            await controller.stop_cell(cell.cell_id)

        assert not cell.is_allocated
        assert srv.has_new_engines is True

    async def test_frozen_promotion_failure_is_retried_on_the_next_poll(self):
        """A transient router failure must not leave a frozen cell out of the router forever."""
        controller, srv, cell, _handle, router_client = _make_setup(update_weights=False)
        attempts = {"n": 0}

        async def _flaky_add_worker(**kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("router hiccup")
            router_client.events.append(("add_worker", kwargs))

        router_client.add_worker = _flaky_add_worker

        with _with_recording_client():
            try:
                await controller._reconcile(cell.cell_id, _observed(cell))
            except RuntimeError:
                pass
            assert cell.is_allocated and not cell.is_alive

            await controller._reconcile(cell.cell_id, _observed(cell))

        assert cell.is_alive
        assert [name for name, _ in router_client.events] == ["add_worker"]


class TestReconcileGate:
    async def test_paused_gate_defers_reconcile_until_resume(self):
        """No cell may join or leave while a weight update snapshot is in flight."""
        controller, _srv, cell, _handle, _router_client = _make_setup()
        await controller.health_monitoring_pause()

        with _with_recording_client():
            reconcile_task = asyncio.create_task(controller._reconcile(cell.cell_id, _observed(cell)))
            await asyncio.sleep(0.05)
            assert not cell.is_allocated

            await controller.health_monitoring_resume()
            await asyncio.wait_for(reconcile_task, timeout=5)

        assert cell.is_allocated

    async def test_pause_waits_for_inflight_reconcile_to_drain(self):
        """Pause must not return while an attach is still mutating the cell set."""
        controller, _srv, cell, handle, _router_client = _make_setup()
        release = asyncio.Event()

        async def _slow_addr_and_ports() -> dict:
            await release.wait()
            return {"server_addr": "10.0.0.1", "server_port": 30000}

        handle.get_addr_and_ports = _slow_addr_and_ports

        with _with_recording_client():
            reconcile_task = asyncio.create_task(controller._reconcile(cell.cell_id, _observed(cell)))
            await asyncio.sleep(0.05)

            pause_task = asyncio.create_task(controller.health_monitoring_pause())
            await asyncio.sleep(0.05)
            assert not pause_task.done()

            release.set()
            await asyncio.wait_for(pause_task, timeout=5)
            await asyncio.wait_for(reconcile_task, timeout=5)

        assert cell.is_allocated


class TestComputeCellStatus:
    async def test_alive_cell_reports_running_and_healthy(self):
        """An alive cell shows Running with Allocated/Healthy true for the api server."""
        controller, _srv, cell, _handle, _router_client = _make_setup(update_weights=False)
        with _with_recording_client():
            await controller._reconcile(cell.cell_id, _observed(cell))

        status = controller.compute_cell_status(cell.cell_id)

        assert status.phase == "Running"
        assert [c.status for c in status.conditions] == ["True", "True"]
        assert controller.get_cell_is_suspended(cell.cell_id) is False

    async def test_weight_unsynced_cell_reports_unknown_health(self):
        """An attached-but-unsynced cell must not look unhealthy, or the ft loop would heal it."""
        controller, _srv, cell, _handle, _router_client = _make_setup()
        with _with_recording_client():
            await controller._reconcile(cell.cell_id, _observed(cell))

        status = controller.compute_cell_status(cell.cell_id)

        healthy = [c for c in status.conditions if c.type == "Healthy"]
        assert [c.status for c in healthy] == ["Unknown"]

    async def test_stopped_cell_reports_suspended(self):
        """A vanished cell is Suspended and marked suspend in its spec."""
        controller, _srv, cell, _handle, _router_client = _make_setup()

        status = controller.compute_cell_status(cell.cell_id)

        assert status.phase == "Suspended"
        assert controller.get_cell_is_suspended(cell.cell_id) is True
