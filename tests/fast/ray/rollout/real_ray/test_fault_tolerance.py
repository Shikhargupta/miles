"""Real ``ray.kill`` is required so follow-up ``.remote()`` calls surface
``RayActorError``; a MagicMock handle can't simulate that."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.ray.rollout.real_ray.conftest import (
    build_cells,
    detach_cell,
    kill_cells,
    make_worker_manager,
    start_cells,
)

from miles.ray.rollout.rollout_server import RolloutServer

# ----------------------------- single-engine kill + recover -----------------------------


@pytest.mark.asyncio
class TestKillAndRecover:
    async def test_recover_creates_new_actor_after_real_kill(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Kill cell 0's engine for real, recover, verify a fresh actor replaces
        it and the surviving cell is untouched."""
        pg = placement_group_factory(2)
        setups = build_cells(num_cells=2)
        worker_manager = make_worker_manager(pg)
        cells = await start_cells(setups, worker_manager, mark_alive=True)

        original_handles = [cell.primary_actor_handle for cell in cells]
        ray.kill(original_handles[0])
        await detach_cell(cells[0], worker_manager)

        try:
            (cells[0],) = await start_cells([setups[0]], worker_manager, mark_alive=True)
            # New actor for cell 0
            assert cells[0].primary_actor_handle is not original_handles[0]
            calls = ray.get(cells[0].primary_actor_handle.get_calls.remote())
            assert "run" in [c[0] for c in calls]

            # Cell 1 untouched, still the same actor
            assert cells[1].primary_actor_handle is original_handles[1]
        finally:
            kill_cells(cells)

    async def test_restarting_dead_cells_leaves_the_live_one_untouched(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """We kill 0 and 2, leave 1 alive, expect only 0 and 2 to be re-created."""
        pg = placement_group_factory(3)
        setups = build_cells(num_cells=3)
        worker_manager = make_worker_manager(pg)
        cells = await start_cells(setups, worker_manager, mark_alive=True)

        old = [cell.primary_actor_handle for cell in cells]
        for i in (0, 2):
            ray.kill(old[i])
            await detach_cell(cells[i], worker_manager)

        try:
            cells[0], cells[2] = await start_cells([setups[0], setups[2]], worker_manager, mark_alive=True)
            for i in (0, 2):
                assert cells[i].primary_actor_handle is not old[i]
            assert cells[1].primary_actor_handle is old[1]
        finally:
            kill_cells(cells)

    async def test_recover_publishes_the_new_url_to_the_router(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """A recovered engine gets a fresh port, so the router must be told the new url."""
        events: list[dict] = []

        class _Recorder:
            async def add_worker(self, **kwargs):
                events.append(kwargs)

            async def remove_worker(self, **kwargs):
                events.append(kwargs)

        pg = placement_group_factory(1)
        setups = build_cells(num_cells=1)
        worker_manager = make_worker_manager(pg)
        cells = await start_cells(setups, worker_manager, mark_alive=True)
        srv = RolloutServer(
            cell_specs={f"cell-{i}": setup.spec for i, setup in enumerate(setups)},
            server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)},
            args=make_args(num_gpus_per_node=8),
            router_ip="10.0.0.9",
            router_port=9000,
        )
        ray.kill(cells[0].primary_actor_handle)
        await detach_cell(cells[0], worker_manager)

        try:
            with patch.object(RolloutServer, "router_api_client", property(lambda self: _Recorder())):
                (cells[0],) = await start_cells([setups[0]], worker_manager, mark_alive=True)
                await cells[0].register(srv.router_api_client)

            assert [event["worker_url"] for event in events] == [cells[0].addr_info.server_url]
            assert cells[0].is_alive
        finally:
            kill_cells(cells)

    async def test_recover_with_offload_calls_release_then_resume(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """``needs_offload=True`` + ``update_weights=True`` means a mid-run attach
        must release_memory_occupation, then resume with WEIGHTS tag.
        Verify by reading the recovered engine's mock HTTP server log."""
        pg = placement_group_factory(2)
        setups = build_cells(num_cells=2, needs_offload=True, update_weights=True)
        worker_manager = make_worker_manager(pg)
        cells = await start_cells(setups, worker_manager, mark_alive=True)
        old = [cell.primary_actor_handle for cell in cells]

        ray.kill(old[0])
        await detach_cell(cells[0], worker_manager)

        try:
            (cells[0],) = await start_cells([setups[0]], worker_manager, mark_alive=True)
            await cells[0].release_offloaded_memory()
            recovered_actor = cells[0].primary_actor_handle
            calls = ray.get(recovered_actor.get_calls.remote())
            assert "run" in [c[0] for c in calls]

            paths = ray.get(recovered_actor.get_http_paths.remote())
            assert "/release_memory_occupation" in paths
            assert "/resume_memory_occupation" in paths

            # Ordering claim: release must precede resume — otherwise GPU
            # memory would be re-occupied before being released, defeating
            # the offload. Use the first occurrence of each.
            release_idx = paths.index("/release_memory_occupation")
            resume_idx = paths.index("/resume_memory_occupation")
            assert release_idx < resume_idx, f"release must precede resume; saw order {paths}"
            # The client drains the working queue before releasing.
            assert paths.index("/flush_cache") < release_idx

            from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

            # Recovery releases everything, not just the weights: an engine that kept its kv cache
            # would leave the trainer short of GPU memory when it takes the device back.
            assert ray.get(recovered_actor.get_http_payloads_of.remote("/release_memory_occupation")) == [
                {"tags": None}
            ]
            assert ray.get(recovered_actor.get_http_payloads_of.remote("/resume_memory_occupation")) == [
                {"tags": [GPU_MEMORY_TYPE_WEIGHTS]}
            ]
        finally:
            kill_cells(cells)


# ----------------------------- concurrent recover -----------------------------


@pytest.mark.asyncio
class TestConcurrentRecover:
    async def test_two_cell_batches_recover_in_parallel_completes_without_deadlock(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Two cell batches recovering simultaneously through real
        ``asyncio.gather`` must both complete — no deadlock, no exception
        leaking out of the gather chain.

        The batches share one worker manager, as they do in production: each
        batch's ports are only bound once its engine inits, so concurrent
        recovers with independent allocators could probe the same free port
        twice. The real-ray claim being verified is end-to-end gather
        completion across two batches."""
        pg = placement_group_factory(4)
        setups_a = build_cells(num_cells=2)
        setups_b = build_cells(num_cells=2, rank_offset=2, gpu_offset=2, cell_id_offset=2)
        worker_manager = make_worker_manager(pg)
        a = await start_cells(setups_a, worker_manager, mark_alive=True)
        b = await start_cells(setups_b, worker_manager, mark_alive=True)

        # Kill one engine in each batch
        for cells in (a, b):
            old = cells[0].primary_actor_handle
            ray.kill(old)
            await detach_cell(cells[0], worker_manager)

        try:
            # Real concurrent recover via asyncio.gather
            recovered_a, recovered_b = await asyncio.gather(
                start_cells([setups_a[0]], worker_manager, mark_alive=True),
                start_cells([setups_b[0]], worker_manager, mark_alive=True),
            )
            (a[0],) = recovered_a
            (b[0],) = recovered_b
            assert a[0].is_alive
            assert b[0].is_alive
        finally:
            kill_cells(a)
            kill_cells(b)


# ----------------------------- crash injection at cell level -----------------------------


@pytest.mark.asyncio
class TestKillSubprocessKeepsMockActorReachable:
    """``MockSGLangEngine.kill_subprocess`` mirrors the moment the engine
    subprocess died but the actor has not exited yet: the server is gone
    while follow-up ``.remote()`` calls still return."""

    async def test_kill_subprocess_then_health_check_still_returns(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        pg = placement_group_factory(1)
        setups = build_cells(num_cells=1)
        cells = await start_cells(setups, make_worker_manager(pg), mark_alive=True)
        actor = cells[0].primary_actor_handle

        try:
            ray.get(actor.kill_subprocess.remote())
            # Actor handle still reachable at Ray level — follow-up returns.
            ray.get(actor.get_calls.remote(), timeout=10.0)
        finally:
            kill_cells(cells)


@pytest.mark.asyncio
class TestRecoverMultiNodeEngine:
    async def test_release_and_resume_only_reach_node0(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Recovering a 2-node engine must not send release/resume to node 1."""
        pg = placement_group_factory(16)
        (setup,) = build_cells(num_cells=1, num_gpus_per_engine=16, needs_offload=True)

        try:
            (cell,) = await start_cells([setup], make_worker_manager(pg), mark_alive=True)
            assert len(cell.actor_handles) == 2
            await cell.release_offloaded_memory()

            node0_actor, node1_actor = cell.actor_handles
            node0_paths = ray.get(node0_actor.get_http_paths.remote())
            node1_paths = ray.get(node1_actor.get_http_paths.remote())

            assert "/release_memory_occupation" in node0_paths
            assert "/resume_memory_occupation" in node0_paths
            assert node1_paths == []
        finally:
            kill_cells([cell])
