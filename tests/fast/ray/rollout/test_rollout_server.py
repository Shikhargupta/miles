from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args, make_dataclass_cells

from miles.ray.rollout.cell_state import AddrInfo
from miles.ray.rollout.rollout_server import (
    RolloutServer,
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
    _resolve_sglang_config,
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
            _resolve_sglang_config(args)

    def test_compute_rollout_offset_colocate_returns_zero(self):
        args = make_args(
            colocate=True,
            debug_train_only=False,
            debug_rollout_only=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=False,
        )
        assert _compute_rollout_offset(args) == 0

    def test_compute_rollout_offset_critic_train_only(self):
        args = make_args(
            colocate=False,
            debug_train_only=False,
            debug_rollout_only=False,
            critic_train_only=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_rollout_offset(args) == 4

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
        assert _compute_rollout_offset(args) == 12

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
            cell._mark_allocated_uninitialized([fake_actor_handle()])
            cell._mark_addressing([AddrInfo(server_url=f"http://10.0.0.{index + 1}:30000")])
        srv = RolloutServer(server_cells=cells)
        assert [client.server_url for client in srv.api_clients] == [
            f"http://10.0.0.{index + 1}:30000" for index in range(4)
        ]

    def test_engine_gpu_counts_parallel_to_engines(self):
        cells = make_dataclass_cells(num_cells=2, num_gpus_per_engine=1) + make_dataclass_cells(
            num_cells=2, num_gpus_per_engine=2
        )
        srv = RolloutServer(server_cells=cells)
        assert srv.engine_gpu_counts == [1, 1, 2, 2]

    def test_engine_gpu_offsets_consistent_across_cells(self):
        cells = make_dataclass_cells(num_cells=2, num_gpus_per_engine=1, gpu_offset=0) + make_dataclass_cells(
            num_cells=2, num_gpus_per_engine=2, gpu_offset=4
        )
        srv = RolloutServer(server_cells=cells)
        assert srv.engine_gpu_offsets == [0, 1, 4, 6]


class TestRolloutServerNodesPerEngineHeterogeneity:
    def test_homogeneous_cells_return_single_value(self):
        cells = make_dataclass_cells(num_gpus_per_engine=1) + make_dataclass_cells(num_gpus_per_engine=1)
        srv = RolloutServer(server_cells=cells)
        assert srv.nodes_per_engine == 1

    def test_heterogeneous_cells_raise_value_error(self):
        # 1 gpu/engine vs 16 gpu/engine on 8-gpu nodes → 1 vs 2 nodes/engine
        cells = make_dataclass_cells(num_gpus_per_engine=1) + make_dataclass_cells(num_cells=1, num_gpus_per_engine=16)
        srv = RolloutServer(server_cells=cells)
        with pytest.raises(ValueError, match="Heterogeneous nodes_per_engine"):
            _ = srv.nodes_per_engine
