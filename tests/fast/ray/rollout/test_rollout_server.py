from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import (
    FakeWorkerCellControl,
    FakeWorkerProvider,
    fake_worker_handle,
    make_args,
    make_dataclass_cells,
    make_sglang_config_yaml,
)

from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.ray.rollout import rollout_server
from miles.ray.rollout.cell_state import AddrInfo
from miles.ray.rollout.rollout_server import RolloutServer, start_rollout_servers
from miles.ray.rollout.server_cell import ServerCell
from miles.ray.specs.inference import (
    _compute_megatron_num_gpus,
    _compute_rollout_pg_offset,
    compute_inference_deployments,
)


class TestRolloutServerPureFunctions:
    def test_resolve_sglang_config_yaml_gpu_mismatch_asserts(self, tmp_path):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 1\n"
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=8)
        with pytest.raises(AssertionError, match="total GPUs"):
            resolve_sglang_config(args)

    def test_compute_rollout_offset_colocate_returns_zero(self):
        args = make_args(
            colocate=True,
            debug_train_only=False,
            debug_rollout_only=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=False,
        )
        assert _compute_rollout_pg_offset(args) == 0

    def test_compute_rollout_offset_critic_train_only(self):
        args = make_args(
            colocate=False,
            debug_train_only=False,
            debug_rollout_only=False,
            critic_train_only=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_rollout_pg_offset(args) == 4

    def test_compute_rollout_offset_actor_plus_critic(self):
        args = make_args(
            colocate=False,
            debug_train_only=False,
            debug_rollout_only=False,
            critic_train_only=False,
            use_critic=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_rollout_pg_offset(args) == 12

    def test_compute_megatron_num_gpus_for_actor_only(self):
        args = make_args(
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            use_critic=False,
            debug_rollout_only=False,
            critic_train_only=False,
        )
        assert _compute_megatron_num_gpus(args) == 16

    def test_compute_megatron_num_gpus_with_critic(self):
        args = make_args(
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
            debug_rollout_only=False,
            critic_train_only=False,
        )
        assert _compute_megatron_num_gpus(args) == 12

    def test_compute_megatron_num_gpus_zero_when_debug_rollout_only(self):
        args = make_args(debug_rollout_only=True)
        assert _compute_megatron_num_gpus(args) == 0


class TestRolloutServerCrossCellProperties:
    def test_api_clients_expose_one_client_per_cell(self):
        """Each cell is addressed through its primary (node-0) endpoint."""
        cells = make_dataclass_cells(num_cells=2, gpu_offset=0) + make_dataclass_cells(num_cells=2, gpu_offset=2)
        for index, cell in enumerate(cells):
            cell._mark_allocated_uninitialized([fake_worker_handle()])
            cell._mark_addressing([AddrInfo(server_url=f"http://10.0.0.{index + 1}:30000")])
        srv = RolloutServer(server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)})
        assert [client.server_url for client in srv.api_clients] == [
            f"http://10.0.0.{index + 1}:30000" for index in range(4)
        ]

    def test_engine_gpu_counts_parallel_to_engines(self):
        cells = make_dataclass_cells(num_cells=2, num_gpus_per_engine=1) + make_dataclass_cells(
            num_cells=2, num_gpus_per_engine=2
        )
        srv = RolloutServer(server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)})
        assert srv.engine_gpu_counts == [1, 1, 2, 2]

    def test_engine_gpu_offsets_consistent_across_cells(self):
        cells = make_dataclass_cells(num_cells=2, num_gpus_per_engine=1, gpu_offset=0) + make_dataclass_cells(
            num_cells=2, num_gpus_per_engine=2, gpu_offset=4
        )
        srv = RolloutServer(server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)})
        assert srv.engine_gpu_offsets == [0, 1, 4, 6]


class TestFailedAttachPromotion:
    async def test_failed_attach_leaves_the_cell_stopped_so_promotion_skips_it(self):
        """A cell whose attach died must stay stopped; promotion must not register it."""

        async def _boom() -> None:
            raise RuntimeError("engine died during init")

        handle = fake_worker_handle(
            addr_and_ports={"server_addr": "10.0.0.1", "server_port": 30000}, init_effect=_boom
        )
        cell = ServerCell(
            args=make_args(num_gpus_per_node=8),
            worker_type="regular",
            cell_id="sglang-default-group0-0",
            spec_name="sglang-default-group0",
            cell_index=0,
            update_weights=False,
            provider=FakeWorkerProvider({"sglang-default-group0-0-0": handle}),
            worker_cell_control=FakeWorkerCellControl(),
        )
        srv = RolloutServer(server_cells={cell.cell_id: cell}, args=cell.args, update_weights=False)

        with pytest.raises(RuntimeError, match="engine died during init"):
            await cell.attach_unsynced()

        assert not cell.is_allocated
        await srv.promote_weight_synced_cells()
        assert not cell.is_alive


class TestStartRolloutServersCellBuilding:
    @pytest.fixture
    def stub_engine_startup(self, monkeypatch):
        async def _no_cells(self, *args, **kwargs):
            return None

        monkeypatch.setattr(rollout_server, "start_router", lambda *args, **kwargs: ("127.0.0.1", 30000))
        monkeypatch.setattr(RolloutServer, "start_all_cells", _no_cells)

    async def _cells_for(self, tmp_path, *, num_gpus: int, num_gpus_per_engine: int):
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": num_gpus, "num_gpus_per_engine": num_gpus_per_engine}
                ]
            )
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=num_gpus, num_gpus_per_node=8)
        deployments = compute_inference_deployments(args)
        servers = await start_rollout_servers(
            args, deployments=deployments, provider=FakeWorkerProvider(), worker_cell_control=FakeWorkerCellControl()
        )
        return servers["default"].server_cells

    async def test_a_single_node_engine_becomes_its_own_cell(self, stub_engine_startup, tmp_path):
        """With one gpu per engine on 8-gpu nodes, every engine is a one-engine cell."""
        cells = await self._cells_for(tmp_path, num_gpus=8, num_gpus_per_engine=1)
        assert [cell.num_nodes for cell in cells.values()] == [1] * 8

    async def test_a_multi_node_engine_chunks_its_node_ranks_into_one_cell(self, stub_engine_startup, tmp_path):
        """With 16 gpus per engine on 8-gpu nodes, each cell holds both node-ranks."""
        cells = await self._cells_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert [cell.num_nodes for cell in cells.values()] == [2, 2]

    async def test_a_trailing_partial_multi_node_engine_is_rejected(self, stub_engine_startup, tmp_path):
        """24 gpus do not divide into whole 2-node engines, so startup must fail fast."""
        with pytest.raises(AssertionError, match="whole number of"):
            await self._cells_for(tmp_path, num_gpus=24, num_gpus_per_engine=16)

    async def test_cells_carry_contiguous_gpu_offsets_and_manager_ids(self, stub_engine_startup, tmp_path):
        """Each multi-node cell starts where the previous one ended and keeps the manager's cell id."""
        cells = await self._cells_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert list(cells) == ["sglang-default-group0-0", "sglang-default-group0-1"]
        assert [cell.gpu_offset for cell in cells.values()] == [0, 16]
        assert [cell.cell_index for cell in cells.values()] == [0, 1]

    async def test_a_group_starting_at_a_misaligned_rank_is_rejected(self, stub_engine_startup, tmp_path):
        """One single-node engine ahead of a 2-node group leaves an odd engine_offset and must fail fast."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "prefill", "num_gpus": 1, "num_gpus_per_engine": 1},
                    {"worker_type": "decode", "num_gpus": 32, "num_gpus_per_engine": 16},
                ]
            )
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=33, num_gpus_per_node=8)
        with pytest.raises(AssertionError, match="not aligned to"):
            compute_inference_deployments(args)

    async def test_external_rollout_mode_is_rejected(self, stub_engine_startup):
        """The external allocator was removed; startup must fail loudly until the replacement lands."""
        args = make_args(rollout_external=True)
        with pytest.raises(NotImplementedError):
            await start_rollout_servers(
                args, deployments=[], provider=None, worker_cell_control=FakeWorkerCellControl()
            )
