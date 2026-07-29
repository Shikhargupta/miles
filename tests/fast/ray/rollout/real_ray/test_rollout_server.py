from __future__ import annotations

import asyncio

import pytest
import ray
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.rollout_server import RolloutServer


def _raw_actor(cell):
    return cell.primary_worker_handle.actor


async def _start_cells(cells) -> None:
    await asyncio.gather(*[cell.start_engines() for cell in cells])
    for cell in cells:
        cell._mark_alive()


def _make_server(cells) -> RolloutServer:
    return RolloutServer(server_cells={cell.cell_id: cell for cell in cells}, args=make_args(num_gpus_per_node=8))


# ----------------------------- check_weights -----------------------------


@pytest.mark.asyncio
class TestCheckWeightsAggregation:
    async def test_aggregates_across_cells_via_real_asyncio_gather(self, manager_harness_factory):
        """Drives RolloutServer.check_weights through real ``asyncio.gather``
        over real HTTP requests. Verifies every cell's primary engine was
        actually invoked (read from each mock server's request log)."""
        args = make_args(num_gpus_per_node=8, rollout_num_gpus=5)
        harness = await manager_harness_factory(args)
        cells = list(harness.build_cells(args).values())
        await _start_cells(cells)

        srv = _make_server(cells)
        results = await srv.check_weights(action="report")

        assert len(results) == 5
        for cell in cells:
            payloads = ray.get(_raw_actor(cell).get_http_payloads_of.remote("/weights_checker"))
            assert payloads == [{"action": "report", "allow_quant_error": False, "selector": "all"}]


# ----------------------------- offload / onload -----------------------------


@pytest.mark.asyncio
class TestOffloadOnloadAggregation:
    async def test_offload_and_onload_reach_every_offloading_cell(self, manager_harness_factory):
        """Both fan out across cells and return one flat result per cell."""
        args = make_args(num_gpus_per_node=8, rollout_num_gpus=3, colocate=True, offload_rollout=True)
        harness = await manager_harness_factory(args)
        cells = list(harness.build_cells(args).values())
        assert all(cell.needs_offload for cell in cells)
        await _start_cells(cells)

        srv = _make_server(cells)
        offload_results = await srv.offload(tags=["weights"])
        onload_results = await srv.onload(["weights"])

        assert len(offload_results) == 3
        assert len(onload_results) == 3
        for cell in cells:
            paths = ray.get(_raw_actor(cell).get_http_paths.remote())
            assert [path for path in paths if path.endswith("_memory_occupation")] == [
                "/release_memory_occupation",
                "/resume_memory_occupation",
            ]
            assert ray.get(_raw_actor(cell).get_http_payloads_of.remote("/release_memory_occupation")) == [
                {"tags": ["weights"]}
            ]
            assert ray.get(_raw_actor(cell).get_http_payloads_of.remote("/resume_memory_occupation")) == [
                {"tags": ["weights"]}
            ]

    async def test_a_cell_that_does_not_need_offload_is_skipped(self, manager_harness_factory):
        """Only the cells colocated with megatron give their memory back."""
        args = make_args(num_gpus_per_node=8, rollout_num_gpus=2)
        harness = await manager_harness_factory(args)
        cells = list(harness.build_cells(args).values())
        assert not any(cell.needs_offload for cell in cells)
        await _start_cells(cells)

        srv = _make_server(cells)
        assert await srv.offload(tags=None) == []
        for cell in cells:
            assert "/release_memory_occupation" not in ray.get(_raw_actor(cell).get_http_paths.remote())

    async def test_a_dead_engine_is_not_addressed(self, manager_harness_factory):
        """Offload must not block forever on an engine the server already gave up on."""
        args = make_args(num_gpus_per_node=8, rollout_num_gpus=2, colocate=True, offload_rollout=True)
        harness = await manager_harness_factory(args)
        cells = list(harness.build_cells(args).values())
        await _start_cells(cells)
        cells[1]._mark_stopped()

        srv = _make_server(cells)
        assert len(await srv.offload(tags=None)) == 1
