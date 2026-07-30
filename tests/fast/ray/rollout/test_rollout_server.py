from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_dataclass_cells, make_sglang_config_yaml

from miles.ray.rollout import rollout_server
from miles.ray.rollout.rollout_server import RolloutServer, start_rollout_servers
from miles.utils.workers.ray_worker_manager import RayWorkerManager


class TestRolloutServerCrossCellProperties:
    def test_api_clients_expose_one_client_per_cell(self):
        """Each cell is addressed through its primary (node-0) endpoint."""
        cells = make_dataclass_cells(num_cells=2, gpu_offset=0) + make_dataclass_cells(
            num_cells=2, gpu_offset=2, host_offset=2
        )
        srv = RolloutServer(
            cell_specs={f"cell-{i}": cell.spec for i, cell in enumerate(cells)},
            server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)},
            args=make_args(num_gpus_per_node=8),
        )
        assert [client.server_url for client in srv.api_clients] == [
            f"http://10.0.0.{index + 1}:30000" for index in range(4)
        ]

    def test_engine_gpu_counts_parallel_to_engines(self):
        cells = make_dataclass_cells(num_cells=2, num_gpus_per_engine=1) + make_dataclass_cells(
            num_cells=2, num_gpus_per_engine=2
        )
        srv = RolloutServer(
            cell_specs={f"cell-{i}": cell.spec for i, cell in enumerate(cells)},
            server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)},
            args=make_args(num_gpus_per_node=8),
        )
        assert srv.engine_gpu_counts == [1, 1, 2, 2]

    def test_engine_gpu_offsets_consistent_across_cells(self):
        cells = make_dataclass_cells(num_cells=2, num_gpus_per_engine=1, gpu_offset=0) + make_dataclass_cells(
            num_cells=2, num_gpus_per_engine=2, gpu_offset=4
        )
        srv = RolloutServer(
            cell_specs={f"cell-{i}": cell.spec for i, cell in enumerate(cells)},
            server_cells={f"cell-{i}": cell for i, cell in enumerate(cells)},
            args=make_args(num_gpus_per_node=8),
        )
        assert srv.engine_gpu_offsets == [0, 1, 4, 6]


class TestStartRolloutServersCellChunking:
    @pytest.fixture
    def stub_engine_startup(self, monkeypatch):
        async def _no_workers(self, request):
            return None

        monkeypatch.setattr(rollout_server, "start_router", lambda *args, **kwargs: ("127.0.0.1", 30000))
        monkeypatch.setattr(RayWorkerManager, "start_cell", _no_workers)

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
        servers = await start_rollout_servers(args, worker_manager=RayWorkerManager(pg=None))
        return list(servers["default"].cell_specs.values())

    async def test_a_single_node_engine_becomes_its_own_cell(self, stub_engine_startup, tmp_path):
        """With one gpu per engine on 8-gpu nodes, every engine is a one-engine cell."""
        specs = await self._cells_for(tmp_path, num_gpus=8, num_gpus_per_engine=1)
        assert [spec.worker.scheduling.num_workers_per_cell for spec in specs] == [1] * 8

    async def test_a_multi_node_engine_chunks_its_node_ranks_into_one_cell(self, stub_engine_startup, tmp_path):
        """With 16 gpus per engine on 8-gpu nodes, each cell holds both node-ranks."""
        specs = await self._cells_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert [spec.worker.scheduling.num_workers_per_cell for spec in specs] == [2, 2]

    async def test_a_trailing_partial_multi_node_engine_is_rejected(self, stub_engine_startup, tmp_path):
        """24 gpus do not divide into whole 2-node engines, so startup must fail fast."""
        with pytest.raises(AssertionError, match="whole number of"):
            await self._cells_for(tmp_path, num_gpus=24, num_gpus_per_engine=16)

    async def test_cells_carry_contiguous_rank_and_gpu_offsets(self, stub_engine_startup, tmp_path):
        """Each multi-node cell starts where the previous one ended, so node-0 detection stays valid."""
        specs = await self._cells_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert [spec.rank_offset for spec in specs] == [0, 2]
        assert [spec.gpu_offset for spec in specs] == [0, 16]

    async def test_every_multi_node_cell_starts_on_an_aligned_rank(self, stub_engine_startup, tmp_path):
        """sglang derives node_rank from the global rank, so a cell must not start mid-engine."""
        specs = await self._cells_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        for spec in specs:
            assert spec.rank_offset % spec.worker.scheduling.num_workers_per_cell == 0

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
            await start_rollout_servers(args, worker_manager=RayWorkerManager(pg=None))
