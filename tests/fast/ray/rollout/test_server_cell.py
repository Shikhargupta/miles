from __future__ import annotations

import dataclasses

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args, make_cell_spec

from miles.ray.rollout.rollout_server import RolloutServer, list_cell_ids
from miles.ray.rollout.server_cell import ServerCell
from miles.ray.specs.inference import compute_nodes_per_engine, format_cell_id
from miles.utils.workers.worker_provider.base import CellInfo, CellMember
from miles.utils.workers.worker_spec import WorkerPlacement


def _cell_info(
    num_nodes: int = 1, *, bootstrap_port: int | None = None, base_gpu_ids: list[int] | None = None
) -> CellInfo:
    base_gpu_ids = base_gpu_ids if base_gpu_ids is not None else list(range(num_nodes))
    payloads = [{"host": f"10.0.0.{i + 1}", "port": 30000 + i} for i in range(num_nodes)]
    if bootstrap_port is not None:
        for payload in payloads:
            payload["disaggregation_bootstrap_port"] = bootstrap_port
    return CellInfo(
        cell_id="cell-0",
        members=[
            CellMember(
                handle=fake_actor_handle(),
                payload=payloads[i],
                placement=WorkerPlacement(local_index=i, global_rank=i, base_gpu_id=base_gpu_ids[i]),
            )
            for i in range(num_nodes)
        ],
    )


def _attached_cell(
    num_nodes: int = 1,
    *,
    alive: bool = True,
    num_gpus_per_engine: int = 1,
    worker_type: str = "regular",
    bootstrap_port: int | None = None,
    base_gpu_ids: list[int] | None = None,
    **args_overrides,
) -> ServerCell:
    args = make_args(num_gpus_per_node=8, **args_overrides)
    cell = ServerCell.attach(
        args=args,
        spec=make_cell_spec(
            args=args, num_nodes=num_nodes, num_gpus_per_engine=num_gpus_per_engine, worker_type=worker_type
        ),
        update_weights=True,
        cell_info=_cell_info(num_nodes, bootstrap_port=bootstrap_port, base_gpu_ids=base_gpu_ids),
    )
    if alive:
        cell.mark_alive()
    return cell


class TestEngineGpuIds:
    def test_gpu_ranges_follow_the_attached_placements(self):
        """The driver-side gpu layout must match where the launch actually put each actor."""
        cell = _attached_cell(num_gpus_per_engine=2, base_gpu_ids=[4])
        assert cell.engine_gpu_ids == [[4, 5]]

    def test_each_node_rank_of_a_multi_node_engine_covers_its_node(self):
        """A 2-node engine reports one whole-node gpu range per node-rank."""
        cell = _attached_cell(num_nodes=2, num_gpus_per_engine=16, base_gpu_ids=[0, 0])
        assert cell.engine_gpu_ids == [list(range(8)), list(range(8))]


class TestServerCellState:
    def test_a_cell_is_born_attached_but_not_yet_alive(self):
        """Attachment is construction; readiness is still a separate step."""
        cell = _attached_cell(num_nodes=2, alive=False)
        assert not cell.is_alive
        assert len(cell.actor_handles) == 2
        assert cell.primary_actor_handle is cell.actor_handles[0]

    def test_the_primary_addr_is_the_router_visible_one(self):
        """Only node 0 serves the endpoint the router routes to."""
        cell = _attached_cell(num_nodes=2)
        assert cell.is_alive
        assert cell.addr_info is cell.addr_infos[0]
        assert cell.api_client.server_url == "http://10.0.0.1:30000"

    def test_attach_rejects_a_cell_info_for_another_cell(self):
        """Adopting another cell's workers would route requests to the wrong engine."""
        args = make_args(num_gpus_per_node=8)
        cell_info = _cell_info()
        with pytest.raises(AssertionError, match="does not name"):
            ServerCell.attach(
                args=args,
                spec=make_cell_spec(args=args, cell_id="cell-9"),
                update_weights=True,
                cell_info=cell_info,
            )

    def test_a_replacement_cell_serves_on_its_own_addr(self):
        """A restarted cell is a new cell, serving its new endpoint, not the dead one."""
        first = _attached_cell()
        assert first.api_client.server_url == "http://10.0.0.1:30000"

        args = make_args(num_gpus_per_node=8)
        info = _cell_info()
        info.members[0] = dataclasses.replace(info.members[0], payload={"host": "10.0.0.9", "port": 39999})
        replacement = ServerCell.attach(args=args, spec=make_cell_spec(args=args), update_weights=True, cell_info=info)
        replacement.mark_alive()

        assert replacement.api_client.server_url == "http://10.0.0.9:39999"


class TestServerCellApiCalls:
    async def test_offload_releases_memory_on_the_primary_engine_only(self):
        """Non-primary node-ranks are workers without their own HTTP endpoint."""
        cell = _attached_cell(num_nodes=2)
        client = MagicMock()
        client.release_memory_occupation = AsyncMock(return_value="released")
        with patch.object(ServerCell, "api_client", property(lambda self: client)):
            assert await cell.offload(tags=["weights"]) == "released"
        client.release_memory_occupation.assert_awaited_once_with(tags=["weights"])

    async def test_onload_resumes_memory_on_the_primary_engine_only(self):
        """Non-primary node-ranks are workers without their own HTTP endpoint."""
        cell = _attached_cell(num_nodes=2)
        client = MagicMock()
        client.resume_memory_occupation = AsyncMock(return_value="resumed")
        with patch.object(ServerCell, "api_client", property(lambda self: client)):
            assert await cell.onload(tags=None) == "resumed"
        client.resume_memory_occupation.assert_awaited_once_with(tags=None)

    async def test_check_weights_forwards_all_arguments_to_the_primary_engine(self):
        """The whole keyword set must reach the engine api unchanged."""
        cell = _attached_cell()
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
    return _attached_cell(num_nodes=2, worker_type=worker_type, bootstrap_port=bootstrap_port, **args_overrides)


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
    nodes_per_engine = compute_nodes_per_engine(num_gpus_per_engine=num_gpus_per_engine, num_gpus_per_node=8)
    servers: dict[str, RolloutServer] = {}
    for s_idx in range(num_servers):
        model_name = f"model_{s_idx}"
        cells = [
            _attached_cell(num_nodes=nodes_per_engine, num_gpus_per_engine=num_gpus_per_engine)
            for _ in range(engines_per_server // nodes_per_engine)
        ]
        servers[model_name] = RolloutServer(
            cell_specs={format_cell_id(server_id=model_name, index=i): cell.spec for i, cell in enumerate(cells)},
            server_cells={format_cell_id(server_id=model_name, index=i): cell for i, cell in enumerate(cells)},
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
