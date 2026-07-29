from __future__ import annotations

import asyncio
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


def _make_test_args(**overrides):
    return make_args(num_gpus_per_node=8, **overrides)


def _raw_actor(cell):
    return cell.primary_worker_handle.actor


class TestStartEnginesRealActors:
    """Drives the provider-attach loop end-to-end against manager-launched
    mock engine actors: handles resolve to real named actors and ``init``
    runs with the manager-pushed addr/ports."""

    async def test_attaches_real_actors_and_init_runs(self, manager_harness_factory):
        args = _make_test_args(rollout_num_gpus=2)
        harness = await manager_harness_factory(args)
        cells = harness.build_cells(args)

        await asyncio.gather(*[cell.start_engines() for cell in cells.values()])

        for cell in cells.values():
            assert cell.is_allocated
            calls = ray.get(_raw_actor(cell).get_calls.remote())
            method_names = [name for name, _, _ in calls]
            assert "configure_addrs_and_ports" in method_names
            assert "init" in method_names
            addr_ports = ray.get(_raw_actor(cell).get_addr_and_ports.remote())
            assert cell.addr_info.server_url == f"http://{addr_ports['server_addr']}:{addr_ports['server_port']}"

    async def test_starting_a_subset_of_cells_leaves_the_rest_unattached(self, manager_harness_factory):
        args = _make_test_args(rollout_num_gpus=4)
        harness = await manager_harness_factory(args)
        cells = list(harness.build_cells(args).values())

        await asyncio.gather(*[cells[i].start_engines() for i in (1, 3)])

        assert not cells[0].is_allocated
        assert cells[1].is_allocated
        assert not cells[2].is_allocated
        assert cells[3].is_allocated

    async def test_stop_cells_kills_the_manager_workers_for_real(self, manager_harness_factory):
        """After stop_cells the named worker actor is truly gone from ray."""
        args = _make_test_args(rollout_num_gpus=2)
        harness = await manager_harness_factory(args)
        cells = harness.build_cells(args)
        srv = RolloutServer(server_cells=cells, args=args, router_ip="10.0.0.9", router_port=9000)
        await asyncio.gather(*[cell.start_engines() for cell in cells.values()])
        for cell in cells.values():
            cell._mark_alive()
        stopped_id = next(iter(cells))
        actor = _raw_actor(cells[stopped_id])

        with patch.object(RolloutServer, "_router_api_client", property(lambda self: _NoopRouterApiClient())):
            await srv.stop_cells([stopped_id])

        assert not cells[stopped_id].is_allocated
        with pytest.raises((ray.exceptions.RayActorError, ray.exceptions.RayTaskError)):
            ray.get(actor.get_calls.remote(), timeout=10.0)

    async def test_stop_survives_an_engine_whose_shutdown_raises(self, manager_harness_factory):
        """A graceful shutdown that raises must not leave the cell marked allocated."""
        args = _make_test_args(rollout_num_gpus=2)
        harness = await manager_harness_factory(args)
        cells = harness.build_cells(args)
        srv = RolloutServer(server_cells=cells, args=args, router_ip="10.0.0.9", router_port=9000)
        await asyncio.gather(*[cell.start_engines() for cell in cells.values()])
        for cell in cells.values():
            cell._mark_alive()
        faulty_id = next(iter(cells))
        ray.get(_raw_actor(cells[faulty_id]).set_fault.remote("shutdown", RuntimeError("shutdown blew up")))

        with patch.object(RolloutServer, "_router_api_client", property(lambda self: _NoopRouterApiClient())):
            await srv.stop_cells([faulty_id])

        assert not cells[faulty_id].is_allocated
