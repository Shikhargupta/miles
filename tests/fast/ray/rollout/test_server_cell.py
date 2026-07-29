from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.fast.ray.rollout.conftest import (
    FakeWorkerHandle,
    FakeWorkerProvider,
    fake_worker_handle,
    make_args,
)

from miles.ray.rollout.cell_state import AddrInfo
from miles.ray.rollout.rollout_server import RolloutServer, list_cell_ids
from miles.ray.rollout.server_cell import ServerCell
from miles.ray.specs.inference import _compute_nodes_per_engine


def _allocated_cell(num_nodes: int = 1, *, alive: bool = True, addressed: bool = True) -> ServerCell:
    cell = ServerCell(
        num_nodes=num_nodes, args=make_args(num_gpus_per_node=8), worker_type="regular", cell_id="cell-0"
    )
    cell._mark_allocated_uninitialized([fake_worker_handle() for _ in range(num_nodes)])
    if not addressed:
        return cell
    cell._mark_addressing([AddrInfo(server_url=f"http://10.0.0.{i + 1}:3000{i}") for i in range(num_nodes)])
    if alive:
        cell._mark_alive()
    return cell


class TestServerCellState:
    def test_a_fresh_cell_is_stopped(self):
        """A cell owns one state machine for all of its node-ranks."""
        cell = ServerCell(num_nodes=2, args=make_args(num_gpus_per_node=8), worker_type="regular", cell_id="cell-0")
        assert not cell.is_allocated
        assert not cell.is_alive

    def test_allocating_covers_every_node_rank(self):
        """The cell's workers are the node-ranks of one engine, so they are held together."""
        cell = _allocated_cell(num_nodes=2, alive=False)
        assert cell.is_allocated and not cell.is_alive
        assert len(cell.worker_handles) == 2
        assert cell.primary_worker_handle is cell.worker_handles[0]

    def test_the_primary_addr_is_the_router_visible_one(self):
        """Only node 0 serves the endpoint the router routes to."""
        cell = _allocated_cell(num_nodes=2)
        assert cell.is_alive
        assert cell.addr_info is cell.addr_infos[0]
        assert cell.api_client.server_url == "http://10.0.0.1:30000"

    def test_stopping_releases_the_whole_cell(self):
        """Teardown is whole-cell: no node-rank may outlive the engine."""
        cell = _allocated_cell(num_nodes=2)
        cell._mark_stopped()
        assert not cell.is_allocated
        assert not cell.is_alive

    def test_the_api_client_is_unavailable_before_the_url_is_known(self):
        """An allocated but unaddressed cell has no endpoint to talk to yet."""
        cell = _allocated_cell(num_nodes=2, addressed=False)
        with pytest.raises(AssertionError):
            _ = cell.api_client

    def test_going_alive_requires_an_addr(self):
        """A cell must not be reported alive before it knows its own url."""
        cell = _allocated_cell(num_nodes=2, addressed=False)
        with pytest.raises(AssertionError):
            cell._mark_alive()

    def test_restarting_replaces_the_addr(self):
        """A restarted cell must serve on its new endpoint, not the dead one."""
        cell = _allocated_cell(num_nodes=1)
        assert cell.api_client.server_url == "http://10.0.0.1:30000"

        cell._mark_stopped()
        cell._mark_allocated_uninitialized([fake_worker_handle()])
        cell._mark_addressing([AddrInfo(server_url="http://10.0.0.9:39999")])
        cell._mark_alive()

        assert cell.api_client.server_url == "http://10.0.0.9:39999"


def _make_handle(*, port: int, bootstrap_port: int | None = None) -> FakeWorkerHandle:
    addr_and_ports: dict = {"server_addr": "10.0.0.1", "server_port": port}
    if bootstrap_port is not None:
        addr_and_ports["disaggregation_bootstrap_port"] = bootstrap_port
    return FakeWorkerHandle(addr_and_ports=addr_and_ports)


def _provider_cell(
    *, num_nodes: int = 1, worker_type: str = "regular", bootstrap_port: int | None = None
) -> tuple[ServerCell, list[FakeWorkerHandle]]:
    handles = [_make_handle(port=30000 + i, bootstrap_port=bootstrap_port) for i in range(num_nodes)]
    provider = FakeWorkerProvider(
        {f"sglang-default-group0-0-{i}": handle for i, handle in enumerate(handles)},
    )
    cell = ServerCell(
        args=make_args(num_gpus_per_node=8),
        worker_type=worker_type,
        cell_id="sglang-default-group0-0",
        num_nodes=num_nodes,
        spec_name="sglang-default-group0",
        cell_index=0,
        provider=provider,
    )
    return cell, handles


class TestStartEngines:
    async def test_attaches_provider_handles_and_inits_every_worker(self):
        """start_engines looks up each node-rank's handle by name and drives its init."""
        cell, handles = _provider_cell(num_nodes=2)

        await cell.start_engines()

        assert cell.is_allocated
        assert cell.worker_handles == handles
        for handle in handles:
            assert handle.calls == ["get_addr_and_ports", "init"]

    async def test_derives_the_server_url_from_the_manager_ports(self):
        """The cell's addr comes from the addr/ports the manager pushed into the engine."""
        cell, _handles = _provider_cell(num_nodes=2)

        await cell.start_engines()

        assert cell.addr_info.server_url == "http://10.0.0.1:30000"
        assert cell.addr_infos[1].server_url == "http://10.0.0.1:30001"

    async def test_carries_the_bootstrap_port_of_a_prefill_worker(self):
        """PD disaggregation needs the decode side to dial the prefill bootstrap port."""
        cell, _handles = _provider_cell(worker_type="prefill", bootstrap_port=8998)

        await cell.start_engines()

        assert cell.addr_info.bootstrap_port == 8998

    async def test_rejects_an_already_allocated_cell(self):
        """A second start_engines call must not replace a running cell's workers."""
        cell, _handles = _provider_cell()
        await cell.start_engines()
        with pytest.raises(AssertionError, match="stopped cells"):
            await cell.start_engines()

    async def test_failed_init_rolls_the_attach_back_to_stopped(self):
        """A failed attach must not linger allocated, or a later promotion would register a broken engine."""
        cell, handles = _provider_cell()

        async def _boom() -> None:
            raise RuntimeError("engine died during init")

        handles[0].init_effect = _boom

        with pytest.raises(RuntimeError, match="engine died during init"):
            await cell.attach_unsynced()

        assert not cell.is_allocated

    async def test_failed_weight_restore_rolls_the_attach_back_to_stopped(self):
        """An engine whose weight-memory restore failed is not ready and must not stay attached."""
        cell, _handles = _provider_cell()
        cell.needs_offload = True
        client = MagicMock()
        client.release_memory_occupation = AsyncMock(side_effect=RuntimeError("oom"))

        with patch.object(ServerCell, "api_client", property(lambda self: client)):
            with pytest.raises(RuntimeError, match="oom"):
                await cell.attach_unsynced()

        assert not cell.is_allocated


class TestServerCellApiCalls:
    async def test_offload_releases_memory_on_the_primary_engine_only(self):
        """Non-primary node-ranks are workers without their own HTTP endpoint."""
        cell = _allocated_cell(num_nodes=2)
        client = MagicMock()
        client.release_memory_occupation = AsyncMock(return_value="released")
        with patch.object(ServerCell, "api_client", property(lambda self: client)):
            assert await cell.offload(tags=["weights"]) == "released"
        client.release_memory_occupation.assert_awaited_once_with(tags=["weights"])

    async def test_onload_resumes_memory_on_the_primary_engine_only(self):
        """Non-primary node-ranks are workers without their own HTTP endpoint."""
        cell = _allocated_cell(num_nodes=2)
        client = MagicMock()
        client.resume_memory_occupation = AsyncMock(return_value="resumed")
        with patch.object(ServerCell, "api_client", property(lambda self: client)):
            assert await cell.onload(tags=None) == "resumed"
        client.resume_memory_occupation.assert_awaited_once_with(tags=None)

    async def test_check_weights_forwards_all_arguments_to_the_primary_engine(self):
        """The whole keyword set must reach the engine api unchanged."""
        cell = _allocated_cell()
        client = MagicMock()
        client.check_weights = AsyncMock(return_value={"ok": True})
        with patch.object(ServerCell, "api_client", property(lambda self: client)):
            result = await cell.check_weights(
                action="report", allow_quant_error=True, selector="first", skip_list=["a"]
            )
        assert result == {"ok": True}
        client.check_weights.assert_awaited_once_with(
            action="report", allow_quant_error=True, selector="first", skip_list=["a"]
        )


def _addressed_cell(
    *, worker_type: str = "regular", bootstrap_port: int | None = None, **args_overrides
) -> ServerCell:
    cell = ServerCell(
        args=make_args(num_gpus_per_node=8, **args_overrides),
        worker_type=worker_type,
        num_nodes=2,
        cell_id="cell-0",
    )
    cell._mark_allocated_uninitialized([fake_worker_handle() for _ in range(2)])
    cell._mark_addressing(
        [
            AddrInfo(server_url=f"http://10.0.0.{index + 1}:3000{index}", bootstrap_port=bootstrap_port)
            for index in range(2)
        ]
    )
    cell._mark_alive()
    return cell


class TestServerCellRouterMembership:
    async def test_register_publishes_the_primary_engine_url_and_worker_type(self):
        """The router routes to the cell through its node-0 engine only."""
        client = MagicMock()
        client.add_worker = AsyncMock()
        await _addressed_cell().register(client)
        client.add_worker.assert_awaited_once_with(
            worker_url="http://10.0.0.1:30000",
            worker_type="regular",
            use_legacy_api=False,
            bootstrap_port=None,
        )

    async def test_register_passes_the_bootstrap_port_of_a_prefill_worker(self):
        """PD disaggregation needs the decode side to dial this port."""
        client = MagicMock()
        client.add_worker = AsyncMock()
        await _addressed_cell(worker_type="prefill", bootstrap_port=8998).register(client)
        assert client.add_worker.await_args.kwargs["worker_type"] == "prefill"
        assert client.add_worker.await_args.kwargs["bootstrap_port"] == 8998

    async def test_unregister_removes_the_same_url_register_published(self):
        """A mismatch would leave the router routing to a dead worker."""
        client = MagicMock()
        client.remove_worker = AsyncMock()
        await _addressed_cell().unregister(client)
        client.remove_worker.assert_awaited_once_with(worker_url="http://10.0.0.1:30000", use_legacy_api=False)

    async def test_use_miles_router_pins_the_legacy_api_on_both_calls(self):
        """--use-miles-router selects the query-string API for register and unregister alike."""
        client = MagicMock()
        client.add_worker = AsyncMock()
        client.remove_worker = AsyncMock()
        cell = _addressed_cell(use_miles_router=True)
        await cell.register(client)
        await cell.unregister(client)
        assert client.add_worker.await_args.kwargs["use_legacy_api"] is True
        assert client.remove_worker.await_args.kwargs["use_legacy_api"] is True


def _build_servers(
    *, num_servers: int = 1, engines_per_server: int = 2, num_gpus_per_engine: int = 1
) -> dict[str, RolloutServer]:
    args = make_args(num_gpus_per_node=8)
    nodes_per_engine = _compute_nodes_per_engine(num_gpus_per_engine=num_gpus_per_engine, num_gpus_per_node=8)
    servers: dict[str, RolloutServer] = {}
    for s_idx in range(num_servers):
        model_name = f"model_{s_idx}"
        cells = [_allocated_cell(num_nodes=nodes_per_engine) for _ in range(engines_per_server // nodes_per_engine)]
        for cell in cells:
            cell.num_gpus_per_engine = num_gpus_per_engine
        servers[model_name] = RolloutServer(
            server_cells={f"{model_name}-{i}": cell for i, cell in enumerate(cells)},
            args=args,
            model_name=model_name,
            update_weights=True,
        )
    return servers


class TestListCellIds:
    def test_single_server_lists_every_cell(self):
        """Happy path: one server with N cells → N ids under model_0."""
        servers = _build_servers(num_servers=1, engines_per_server=3)
        assert list_cell_ids(servers) == ["model_0-0", "model_0-1", "model_0-2"]

    def test_multi_server_ordered_by_model_id_alphabetically(self):
        """When multiple servers exist, ids are emitted in model id order."""
        servers = _build_servers(num_servers=2, engines_per_server=1)
        assert list_cell_ids(servers) == ["model_0-0", "model_1-0"]

    def test_multinode_engine_slots_form_one_cell(self):
        """num_gpus_per_engine=16 and num_gpus_per_node=8 → nodes_per_engine=2;
        the 2 engine slots form one cell."""
        servers = _build_servers(num_servers=1, engines_per_server=2, num_gpus_per_engine=16)
        assert list_cell_ids(servers) == ["model_0-0"]

    def test_server_without_cells_emits_zero_ids(self):
        """A server with no cells (e.g. only placeholder groups) emits no cell ids."""
        srv = MagicMock()
        srv.server_cells = {}
        assert list_cell_ids({"only": srv}) == []

    def test_empty_server_dict_returns_empty_list(self):
        assert list_cell_ids({}) == []
