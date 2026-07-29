"""Real ``ray.kill`` plus a real manager restart drive rollout recovery here;
a MagicMock handle can't simulate actors actually dying and coming back."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.rollout_server import RolloutServer


class _NoopRouterApiClient:
    async def add_worker(self, **kwargs):
        return None

    async def remove_worker(self, **kwargs):
        return None


def _raw_actor(cell):
    return cell.primary_worker_handle.actor


def _with_noop_router():
    return patch.object(RolloutServer, "_router_api_client", property(lambda self: _NoopRouterApiClient()))


async def _start_server(harness, args) -> RolloutServer:
    cells = harness.build_cells(args)
    srv = RolloutServer(server_cells=cells, args=args, router_ip="10.0.0.9", router_port=9000)
    with _with_noop_router():
        await srv.start_all_cells()
    return srv


@pytest.mark.asyncio
class TestKillRestartAttach:
    async def test_restart_and_attach_relaunch_a_killed_cell_through_the_manager(self, manager_harness_factory):
        """Kill cell 0's engine for real, restart it at the manager and re-attach,
        verify a fresh actor serves it and the surviving cell keeps its call history."""
        args = make_args(num_gpus_per_node=8, rollout_num_gpus=2)
        harness = await manager_harness_factory(args)
        srv = await _start_server(harness, args)
        cells = list(srv.server_cells.values())
        survivor_calls_before = len(ray.get(_raw_actor(cells[1]).get_calls.remote()))

        ray.kill(_raw_actor(cells[0]))
        cells[0]._mark_stopped()

        await harness.worker_cell_control.restart_cell(cell_id=cells[0].cell_id)
        await cells[0].attach_unsynced()

        assert cells[0].is_allocated
        recovered_calls = [name for name, _, _ in ray.get(_raw_actor(cells[0]).get_calls.remote())]
        assert recovered_calls.count("init") == 1
        assert len(ray.get(_raw_actor(cells[1]).get_calls.remote())) == survivor_calls_before

    async def test_attached_cell_publishes_the_new_url_only_after_promotion(self, manager_harness_factory):
        """A re-attached updatable engine reaches the router with its fresh url only once promoted."""
        events: list[dict] = []

        class _Recorder:
            async def add_worker(self, **kwargs):
                events.append(kwargs)

            async def remove_worker(self, **kwargs):
                events.append(kwargs)

        args = make_args(num_gpus_per_node=8, rollout_num_gpus=1)
        harness = await manager_harness_factory(args)
        srv = await _start_server(harness, args)
        (cell,) = srv.server_cells.values()
        url_before = cell.addr_info.server_url

        ray.kill(_raw_actor(cell))
        cell._mark_stopped()
        await harness.worker_cell_control.restart_cell(cell_id=cell.cell_id)

        with patch.object(RolloutServer, "_router_api_client", property(lambda self: _Recorder())):
            await cell.attach_unsynced()
            assert events == []
            assert cell.is_allocated and not cell.is_alive

            await srv.promote_weight_synced_cells()

        assert [event["worker_url"] for event in events] == [cell.addr_info.server_url]
        assert cell.addr_info.server_url != url_before
        assert cell.is_alive

    async def test_attach_with_offload_calls_release_then_resume(self, manager_harness_factory):
        """``needs_offload=True`` + ``update_weights=True`` means attach_unsynced must
        release_memory_occupation, then resume with the WEIGHTS tag."""
        from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

        args = make_args(num_gpus_per_node=8, rollout_num_gpus=1, colocate=True, offload_rollout=True)
        harness = await manager_harness_factory(args)
        srv = await _start_server(harness, args)
        (cell,) = srv.server_cells.values()
        assert cell.needs_offload

        ray.kill(_raw_actor(cell))
        cell._mark_stopped()

        await harness.worker_cell_control.restart_cell(cell_id=cell.cell_id)
        await cell.attach_unsynced()

        recovered_actor = _raw_actor(cell)
        paths = ray.get(recovered_actor.get_http_paths.remote())
        assert "/release_memory_occupation" in paths
        assert "/resume_memory_occupation" in paths
        assert paths.index("/release_memory_occupation") < paths.index("/resume_memory_occupation")
        assert ray.get(recovered_actor.get_http_payloads_of.remote("/release_memory_occupation")) == [{"tags": None}]
        assert ray.get(recovered_actor.get_http_payloads_of.remote("/resume_memory_occupation")) == [
            {"tags": [GPU_MEMORY_TYPE_WEIGHTS]}
        ]


@pytest.mark.asyncio
class TestSimulateCrashKeepsActorReachable:
    """``MockSGLangEngine.simulate_crash`` self-calls ``shutdown()`` (mirror
    of real SGLangEngine). The actor stays alive at the Ray level, so
    follow-up ``.remote()`` calls must still return."""

    async def test_simulate_crash_then_follow_up_call_still_returns(self, manager_harness_factory):
        args = make_args(num_gpus_per_node=8, rollout_num_gpus=1)
        harness = await manager_harness_factory(args)
        srv = await _start_server(harness, args)
        (cell,) = srv.server_cells.values()
        actor = _raw_actor(cell)

        ray.get(actor.simulate_crash.remote())

        ray.get(actor.get_calls.remote(), timeout=10.0)


@pytest.mark.asyncio
class TestAttachMultiNodeEngine:
    async def test_attach_releases_and_resumes_only_on_node0(self, manager_harness_factory):
        """Attaching a 2-node engine must not send release/resume to node 1."""
        args = make_args(
            num_gpus_per_node=8,
            rollout_num_gpus=16,
            rollout_num_gpus_per_engine=16,
            colocate=True,
            offload_rollout=True,
        )
        harness = await manager_harness_factory(args)
        (cell,) = harness.build_cells(args).values()
        assert cell.num_nodes == 2
        assert cell.needs_offload

        await cell.attach_unsynced()

        node0_actor, node1_actor = [handle.actor for handle in cell.worker_handles]
        node0_paths = ray.get(node0_actor.get_http_paths.remote())
        node1_paths = ray.get(node1_actor.get_http_paths.remote())

        assert "/release_memory_occupation" in node0_paths
        assert "/resume_memory_occupation" in node0_paths
        assert node1_paths == []
