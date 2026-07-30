from __future__ import annotations

import pytest
import ray
from tests.fast.ray.rollout.real_ray.conftest import (
    attach_cells,
    build_cells,
    detach_cell,
    make_worker_manager,
    start_cells,
)

from miles.backends.sglang_utils.sglang_engine import build_server_url
from miles.ray.rollout.rollout_server import RolloutServer, start_rollout_servers


def _all_actor_handles(cells) -> list:
    return [handle for cell in cells for handle in cell.actor_handles]


class TestStartEnginesShortCircuits:
    """Branches that bail before hitting the PG / actor creation path."""

    async def test_debug_train_only_brings_up_no_workers(self, placement_group_factory):
        """In debug_train_only the wiring schedules no actors at all."""
        pg = placement_group_factory(2)
        setups = build_cells(num_cells=2, debug_train_only=True)
        srv = RolloutServer(
            cell_specs={f"cell-{i}": setup.spec for i, setup in enumerate(setups)},
            server_cells={},
            args=setups[0].args,
        )

        worker_manager = make_worker_manager(pg)
        await start_rollout_servers(setups[0].args, pg, worker_manager)

        assert worker_manager.cell_ids() == []
        assert srv.has_new_engines is False


class TestStartEnginesRealActors:
    """Drives the actor-creation loop end-to-end. Verifies the actors are
    real Ray actors (via ``get_calls()`` round-trip) and that ``run`` was
    invoked with a command carrying the addr/ports from the allocator."""

    async def test_creates_real_actors_and_run_launches(self, patched_sglang_engine, placement_group_factory):
        pg = placement_group_factory(2)
        setups = build_cells(num_cells=2)

        cells = await start_cells(setups, make_worker_manager(pg))

        for cell in cells:
            calls = ray.get(cell.primary_actor_handle.get_calls.remote())
            method_names = [name for name, _, _ in calls]
            assert "run" in method_names
            server_args = ray.get(cell.primary_actor_handle.get_server_args.remote())
            assert server_args["host"] == "127.0.0.1"
            assert cell.addr_info.server_url == build_server_url(host=server_args["host"], port=server_args["port"])

        # Cleanup: kill the actors we created.
        for handle in _all_actor_handles(cells):
            ray.kill(handle)

    async def test_starting_a_subset_of_cells_leaves_the_rest_unallocated(
        self, patched_sglang_engine, placement_group_factory
    ):
        pg = placement_group_factory(4)
        setups = build_cells(num_cells=4)

        worker_manager = make_worker_manager(pg)
        started = await start_cells([setups[1], setups[3]], worker_manager)

        assert worker_manager.cell_ids() == ["cell-1", "cell-3"]

        for cell in started:
            ray.kill(cell.primary_actor_handle)

    async def test_reattaching_an_unchanged_cell_keeps_its_actors(
        self, patched_sglang_engine, placement_group_factory
    ):
        """A reconcile pass over a healthy cell must not swap the actors underneath it."""
        pg = placement_group_factory(2)
        setups = build_cells(num_cells=2)

        worker_manager = make_worker_manager(pg)
        cells = await start_cells(setups, worker_manager)
        first_handles = _all_actor_handles(cells)

        cells = await attach_cells(setups, worker_manager)
        for first, handle in zip(first_handles, _all_actor_handles(cells), strict=True):
            assert handle is first  # still the same actor

        for handle in first_handles:
            ray.kill(handle)

    async def test_restarting_a_multi_node_cell_replaces_every_node_rank(
        self, patched_sglang_engine, placement_group_factory
    ):
        """A cell is one distributed engine: a restart must bring back every
        node-rank, since a survivor would belong to a process group that is gone."""
        pg = placement_group_factory(16)
        (setup,) = build_cells(num_cells=1, num_gpus_per_engine=16)
        worker_manager = make_worker_manager(pg)
        (cell,) = await start_cells([setup], worker_manager)
        assert len(cell.actor_handles) == 2
        original_handles = list(cell.actor_handles)

        await detach_cell(cell, worker_manager)
        assert worker_manager.cell_workers(cell.cell_id) == []

        (cell,) = await start_cells([setup], worker_manager)
        try:
            assert len(cell.actor_handles) == 2
            for original, handle in zip(original_handles, cell.actor_handles, strict=True):
                assert handle is not original
        finally:
            for handle in cell.actor_handles:
                ray.kill(handle)


# FIXME(@fzyzcjy): TestStopCellsRealKill is a timing-sensitive Ray actor
# termination race that flakes in CI (stage-a-cpu). Real fix tracked in
# https://github.com/radixark/miles/pull/1282 — re-enable once that lands.
@pytest.mark.skip(reason="FIXME(@fzyzcjy): flaky Ray actor termination race; real fix in #1282")
class TestStopCellsRealKill:
    """``ray.kill`` is the real thing here — we verify the actor is actually
    dead by issuing a follow-up ``.remote()`` and expecting RayActorError."""

    async def test_stop_marks_cells_stopped_and_actors_truly_die(self, patched_sglang_engine, placement_group_factory):
        pg = placement_group_factory(2)
        setups = build_cells(num_cells=2)
        worker_manager = make_worker_manager(pg)
        cells = await start_cells(setups, worker_manager)

        actors = _all_actor_handles(cells)
        for cell in cells:
            await detach_cell(cell, worker_manager)

        assert worker_manager.cell_ids() == []

        # Real-Ray claim: a follow-up call on a killed actor must surface as
        # RayActorError, not silently return.
        for actor in actors:
            with pytest.raises((ray.exceptions.RayActorError, ray.exceptions.RayTaskError)):
                ray.get(actor.get_calls.remote(), timeout=10.0)

    async def test_stop_handles_shutdown_failure_gracefully(self, patched_sglang_engine, placement_group_factory):
        """If ``shutdown`` raises on the actor, the teardown must still
        mark the cell stopped (and ray.kill is still called).

        We use ``set_fault`` to make shutdown raise on its next invocation."""
        pg = placement_group_factory(2)
        setups = build_cells(num_cells=2)
        worker_manager = make_worker_manager(pg)
        cells = await start_cells(setups, worker_manager)

        # Plant a one-shot shutdown failure on cell 1.
        ray.get(cells[1].primary_actor_handle.set_fault.remote("shutdown", RuntimeError("boom")))

        for cell in cells:
            await detach_cell(cell, worker_manager)
        assert worker_manager.cell_ids() == [], "all cells must be stopped despite shutdown raise"


class TestStartEnginesRealAllocator:
    """Drive the manager's port allocation (no stub) so that the
    actor → driver port round-trip via
    ``_get_free_port_block.remote`` actually runs."""

    async def test_real_allocator_assigns_distinct_ports_via_remote_calls(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        pg = placement_group_factory(2)
        setups = build_cells(num_cells=2)

        cells = await start_cells(setups, make_worker_manager(pg))

        # the launch command == the addr_and_ports map produced by the real allocator
        kwargs0, kwargs1 = ray.get(
            [
                cells[0].primary_actor_handle.get_server_args.remote(),
                cells[1].primary_actor_handle.get_server_args.remote(),
            ]
        )

        # Real-allocator claim 1: each engine got a fully-formed addr/port set
        for k in kwargs0, kwargs1:
            for key in ("host", "port", "nccl_port", "dist_init_addr"):
                assert key in k, f"missing {key} in the launch command from the real allocator"
            assert k["host"] == "127.0.0.1"

        # Real-allocator claim 2: ports are distinct between engines (the
        # node cursor must advance across engines on the same node).
        ports_engine0 = {kwargs0["port"], kwargs0["nccl_port"]}
        ports_engine1 = {kwargs1["port"], kwargs1["nccl_port"]}
        assert ports_engine0.isdisjoint(
            ports_engine1
        ), f"port collision across engines: {ports_engine0} vs {ports_engine1}"

        # Real-allocator claim 3: the allocator actually called
        # _get_free_port_block on each cell's actor; this
        # assertion catches a regression where the allocator silently fell
        # back to a stub or swallowed the .remote() calls.
        calls = ray.get(cells[0].primary_actor_handle.get_calls.remote())
        method_names = [name for name, _, _ in calls]
        assert "_get_free_port_block" in method_names, f"allocator never called the port-finder; saw {method_names}"

        for handle in _all_actor_handles(cells):
            ray.kill(handle)

    async def test_real_allocator_advances_cursor_across_sequential_cells(
        self,
        patched_sglang_engine,
        placement_group_factory,
    ):
        """Two sequentially-started batches of cells on independent PGs both
        invoke the real allocator. The bring-up mutates the worker
        manager's allocator in place; reusing it for B must shift B's ports
        past A's — that's the cursor's job."""
        pg = placement_group_factory(4)
        setups_a = build_cells(num_cells=2)
        setups_b = build_cells(num_cells=2, rank_offset=2, gpu_offset=2, cell_id_offset=2)

        worker_manager = make_worker_manager(pg)
        a = await start_cells(setups_a, worker_manager)

        b = await start_cells(setups_b, worker_manager)

        kwargs_a = ray.get([handle.get_server_args.remote() for handle in _all_actor_handles(a)])
        kwargs_b = ray.get([handle.get_server_args.remote() for handle in _all_actor_handles(b)])
        ports_a = {p for kw in kwargs_a for p in (kw["port"], kw["nccl_port"])}
        ports_b = {p for kw in kwargs_b for p in (kw["port"], kw["nccl_port"])}

        assert ports_a.isdisjoint(ports_b), f"sequential cells overlapped on ports: a={ports_a} b={ports_b}"

        for handle in _all_actor_handles(a) + _all_actor_handles(b):
            ray.kill(handle)
