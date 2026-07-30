from __future__ import annotations

import pytest
from pydantic import ValidationError
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.ray.specs.inference import (
    compute_engine_env_vars,
    compute_inference_model_specs,
    compute_megatron_num_gpus,
    compute_nodes_per_engine,
    compute_rollout_offset,
)
from miles.utils.test_utils.mock_sglang_engine import parse_cmd_flags
from miles.utils.workers.worker_spec import CellAddressing, CommandWorkerSpec, WorkerPlacement


def _write_yaml(tmp_path, content: str):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(content)
    return str(cfg_path)


class TestComputeInferenceModelSpecs:
    def test_default_args_build_one_single_gpu_cell_per_gpu(self):
        """Without a yaml config, every rollout gpu becomes its own one-worker cell."""
        (model,) = compute_inference_model_specs(make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=1))
        assert model.name == "default"
        assert model.update_weights is True
        assert model.has_pd_disaggregation is False
        assert [cell.cell_id for cell in model.cells] == [f"default-{i}" for i in range(8)]
        assert [cell.rank_offset for cell in model.cells] == list(range(8))
        assert [cell.gpu_offset for cell in model.cells] == list(range(8))

    def test_worker_fields_mirror_args_for_the_default_model(self):
        """The worker spec carries the launch inputs the cell used to compute inline."""
        (model,) = compute_inference_model_specs(make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=1))
        worker = model.cells[0].worker
        assert worker.worker_type == "regular"
        assert worker.num_gpus_per_engine == 1
        assert worker.needs_offload is False
        assert worker.model_path == "/fake/model"
        assert worker.scheduling.num_cells == 8
        assert worker.scheduling.num_workers_per_cell == 1
        assert worker.scheduling.num_gpus_per_worker == 1

    def test_multi_node_engines_chunk_node_ranks_into_cells(self):
        """A 16-gpu engine on 8-gpu nodes spans 2 node-ranks that form one cell."""
        (model,) = compute_inference_model_specs(
            make_args(rollout_num_gpus=32, rollout_num_gpus_per_engine=16, num_gpus_per_node=8)
        )
        assert [cell.rank_offset for cell in model.cells] == [0, 2]
        assert [cell.gpu_offset for cell in model.cells] == [0, 16]
        worker = model.cells[0].worker
        assert worker.scheduling.num_cells == 2
        assert worker.scheduling.num_workers_per_cell == 2
        assert worker.scheduling.num_gpus_per_worker == 8
        assert worker.num_gpus_per_engine == 16

    def test_an_engine_size_that_is_not_a_whole_number_of_nodes_is_rejected(self):
        """12 gpus per engine on 8-gpu nodes would silently become 8, so it must fail fast."""
        with pytest.raises(AssertionError, match="neither within one node"):
            compute_inference_model_specs(
                make_args(rollout_num_gpus=24, rollout_num_gpus_per_engine=12, num_gpus_per_node=8)
            )

    def test_a_group_too_small_for_one_engine_is_rejected(self, tmp_path):
        """A group that cannot host a single engine would vanish from the layout, so it must fail fast."""
        cfg = _write_yaml(
            tmp_path,
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 2, "num_gpus_per_engine": 4}]
            ),
        )
        with pytest.raises(AssertionError, match="not enough for a single engine"):
            compute_inference_model_specs(make_args(sglang_config=cfg, rollout_num_gpus=2))

    def test_trailing_partial_multi_node_engine_is_rejected(self):
        """24 gpus do not divide into whole 2-node engines, so spec computation must fail fast."""
        with pytest.raises(AssertionError, match="whole number of"):
            compute_inference_model_specs(
                make_args(rollout_num_gpus=24, rollout_num_gpus_per_engine=16, num_gpus_per_node=8)
            )

    def test_a_group_starting_at_a_misaligned_rank_is_rejected(self, tmp_path):
        """One single-node engine ahead of a 2-node group leaves an odd engine offset and must fail fast."""
        cfg = _write_yaml(
            tmp_path,
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "prefill", "num_gpus": 1, "num_gpus_per_engine": 1},
                    {"worker_type": "decode", "num_gpus": 32, "num_gpus_per_engine": 16},
                ]
            ),
        )
        args = make_args(sglang_config=cfg, rollout_num_gpus=33, num_gpus_per_node=8)
        with pytest.raises(AssertionError, match="not aligned to"):
            compute_inference_model_specs(args)

    def test_placeholder_groups_reserve_offsets_without_cells(self, tmp_path):
        """A placeholder group consumes rank and gpu space but launches nothing."""
        cfg = _write_yaml(
            tmp_path,
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": 2, "num_gpus_per_engine": 1},
                    {"worker_type": "placeholder", "num_gpus": 2, "num_gpus_per_engine": 1},
                    {"worker_type": "regular", "num_gpus": 2, "num_gpus_per_engine": 1},
                ]
            ),
        )
        (model,) = compute_inference_model_specs(make_args(sglang_config=cfg, rollout_num_gpus=6))
        assert [cell.cell_id for cell in model.cells] == ["default-0", "default-1", "default-2", "default-3"]
        assert [cell.rank_offset for cell in model.cells] == [0, 1, 4, 5]
        assert [cell.gpu_offset for cell in model.cells] == [0, 1, 4, 5]

    def test_prefill_workers_carry_a_disaggregation_bootstrap_port(self, tmp_path):
        """The decode side dials the prefill bootstrap port, so only prefill reserves it."""
        cfg = _write_yaml(
            tmp_path,
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "prefill", "num_gpus": 4, "num_gpus_per_engine": 1},
                    {"worker_type": "decode", "num_gpus": 4, "num_gpus_per_engine": 1},
                ]
            ),
        )
        (model,) = compute_inference_model_specs(make_args(sglang_config=cfg, rollout_num_gpus=8))
        assert model.has_pd_disaggregation is True
        prefill_ports = {p.name for p in model.cells[0].worker.port_infos}
        decode_ports = {p.name for p in model.cells[4].worker.port_infos}
        assert "disaggregation_bootstrap" in prefill_ports
        assert "disaggregation_bootstrap" not in decode_ports

    def test_dist_init_port_reserves_a_consecutive_master_block(self):
        """The dist-init endpoint lives on the master and spans a 30+dp block of ports."""
        (model,) = compute_inference_model_specs(
            make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=1, sglang_dp_size=4)
        )
        (dist_init,) = [p for p in model.cells[0].worker.port_infos if p.name == "dist_init"]
        assert dist_init.mode == "master"
        assert dist_init.num_consecutive == 34

    def test_needs_offload_only_inside_the_megatron_slot_range(self):
        """Only rollout gpus that share slots with megatron must offload; the rest opt out of the memory saver."""
        colocated = make_args(rollout_num_gpus=8, offload_rollout=True, colocate=True)
        (model,) = compute_inference_model_specs(colocated)
        assert model.cells[0].worker.needs_offload is True
        assert "enable_memory_saver" not in model.cells[0].worker.sglang_overrides

        disjoint = make_args(rollout_num_gpus=8, offload_rollout=True, colocate=False)
        (model,) = compute_inference_model_specs(disjoint)
        assert model.cells[0].worker.needs_offload is False
        assert model.cells[0].worker.sglang_overrides["enable_memory_saver"] is False

    def test_multi_model_cells_continue_global_offsets_but_restart_ids(self, tmp_path):
        """Ranks and gpu slots are global across models while cell ids restart per model."""
        cfg = _write_yaml(
            tmp_path,
            "sglang:\n"
            "  - name: actor\n"
            "    update_weights: true\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 1\n"
            "  - name: ref\n"
            "    update_weights: false\n"
            "    model_path: /ref/model\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 1\n",
        )
        actor, ref = compute_inference_model_specs(make_args(sglang_config=cfg, rollout_num_gpus=12))
        assert [cell.cell_id for cell in ref.cells] == [f"ref-{i}" for i in range(4)]
        assert [cell.rank_offset for cell in ref.cells] == [8, 9, 10, 11]
        assert [cell.gpu_offset for cell in ref.cells] == [8, 9, 10, 11]
        assert actor.update_weights is True
        assert ref.update_weights is False
        assert ref.cells[0].worker.model_path == "/ref/model"


def _placement(global_rank: int = 0) -> WorkerPlacement:
    return WorkerPlacement(local_index=0, global_rank=global_rank, base_gpu_id=0)


class TestEngineLaunchIsDrivenByTheSpec:
    """The worker manager launches from the spec alone, so the spec must carry all of it."""

    def _worker(self, **kwargs):
        (model,) = compute_inference_model_specs(
            make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=1, **kwargs)
        )
        return model.cells[0].worker

    def test_the_spec_is_a_command_worker_and_names_what_it_asks_ray_for(self):
        """A manager that knew the worker kind or the gpu fraction itself could not serve other workers."""
        worker = self._worker()
        assert isinstance(worker, CommandWorkerSpec)
        assert worker.ray_options.num_cpus == 0.2
        assert worker.ray_options.num_gpus == 0.2

    def test_the_launch_plan_renders_the_engine_command_from_placement_and_addressing(self):
        """Rank and base gpu come from the placement; the addressing feeds the command flags."""
        worker = self._worker()
        addressing = CellAddressing(
            node_ips=["10.0.0.1"],
            master_ports={"dist_init": 31500},
            per_worker_ports=[{"server": 30000, "nccl": 30500, "engine_info_bootstrap": 31000}],
        )
        plan = worker.build_launch_plan(WorkerPlacement(local_index=0, global_rank=0, base_gpu_id=0), addressing)
        flags = parse_cmd_flags(plan.cmd)
        assert flags["host"] == "10.0.0.1" and flags["port"] == 30000
        assert flags["nccl_port"] == 30500
        assert flags["dist_init_addr"] == "10.0.0.1:31500"

    async def test_wait_cell_ready_fails_fast_when_the_worker_died(self):
        """A dead worker must abort the readiness wait instead of polling forever."""
        worker = self._worker()
        addressing = CellAddressing(
            node_ips=["127.0.0.1"],
            master_ports={"dist_init": 31500},
            per_worker_ports=[{"server": 1, "nccl": 30500, "engine_info_bootstrap": 31000}],
        )
        with pytest.raises(Exception, match="terminated"):
            await worker.wait_cell_ready(addressing, lambda: False)

    def test_member_payloads_map_the_allocated_ports_onto_the_engine_addressing(self):
        """Only the spec knows which addressing key each declared port name feeds."""
        worker = self._worker()
        addressing = CellAddressing(
            node_ips=["10.0.0.1", "10.0.0.2"],
            master_ports={"dist_init": 31500},
            per_worker_ports=[
                {"server": 30000, "nccl": 30500, "engine_info_bootstrap": 31000},
                {"server": 30001, "nccl": 30501, "engine_info_bootstrap": 31001},
            ],
        )

        payloads = worker.build_member_payloads(addressing)

        assert payloads == [
            dict(
                host="10.0.0.1",
                port=30000,
                nccl_port=30500,
                engine_info_bootstrap_port=31000,
                dist_init_addr="10.0.0.1:31500",
            ),
            dict(
                host="10.0.0.2",
                port=30001,
                nccl_port=30501,
                engine_info_bootstrap_port=31001,
                dist_init_addr="10.0.0.1:31500",
            ),
        ]

    def test_a_prefill_payload_carries_its_disaggregation_bootstrap_port(self, tmp_path):
        """The decode side dials that port, so it must reach the engine's addressing."""
        cfg = _write_yaml(
            tmp_path,
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "prefill", "num_gpus": 1, "num_gpus_per_engine": 1},
                    {"worker_type": "decode", "num_gpus": 1, "num_gpus_per_engine": 1},
                ]
            ),
        )
        (model,) = compute_inference_model_specs(make_args(sglang_config=cfg, rollout_num_gpus=2))
        addressing = CellAddressing(
            node_ips=["10.0.0.1"],
            master_ports={"dist_init": 31500},
            per_worker_ports=[
                {
                    "server": 30000,
                    "nccl": 30500,
                    "engine_info_bootstrap": 31000,
                    "disaggregation_bootstrap": 32000,
                }
            ],
        )

        (payload,) = model.cells[0].worker.build_member_payloads(addressing)

        assert payload["disaggregation_bootstrap_port"] == 32000

    def test_an_unexpected_master_port_set_is_rejected(self):
        """dist_init is the only master endpoint an engine cell has, so anything else is a bug."""
        worker = self._worker()
        addressing = CellAddressing(node_ips=["10.0.0.1"], master_ports={}, per_worker_ports=[{}])
        with pytest.raises(AssertionError, match="master_ports"):
            worker.build_member_payloads(addressing)


class TestRejectedWorkerSpecs:
    def test_external_rollout_has_no_launchable_spec(self):
        """The external allocator was removed; spec computation must fail loudly until it returns."""
        with pytest.raises(NotImplementedError, match="external rollout"):
            compute_inference_model_specs(make_args(rollout_external=True))

    @pytest.mark.parametrize("overrides", [{"port": 40000}, {"host": "10.9.9.9"}, {"host": "10.9.9.9", "port": 40000}])
    def test_an_override_of_host_or_port_is_rejected(self, overrides):
        """The engine's url is derived from the allocator, so an override would address the wrong endpoint."""
        from tests.fast.ray.rollout.conftest import make_cell_spec

        with pytest.raises(ValidationError, match="must not override host/port"):
            make_cell_spec(sglang_overrides=overrides)


class TestComputeEngineEnvVars:
    def test_custom_all_reduce_v2_disabled_only_for_colocated_multi_gpu_engines(self):
        """Colocated multi-gpu engines hit the v2 all-reduce bug, so only they turn it off."""
        colocated = compute_engine_env_vars(
            make_args(colocate=True, rollout_num_gpus_per_engine=2), _placement(), worker_type="regular"
        )
        assert colocated["SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"] == "0"
        plain = compute_engine_env_vars(
            make_args(colocate=False, rollout_num_gpus_per_engine=2), _placement(), worker_type="regular"
        )
        assert plain["SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"] == "1"

    def test_visible_device_env_vars_are_passed_through(self):
        """Engines must see all gpus of their node, so the noset flags are always on."""
        from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST

        env_vars = compute_engine_env_vars(make_args(), _placement(), worker_type="regular")
        for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST:
            assert env_vars[name] == "1"

    def test_the_deep_gemm_cache_dir_is_per_worker_type_and_rank(self):
        """Co-located engines must not share a JIT cache dir, so it carries their identity."""
        env_vars = compute_engine_env_vars(make_args(), _placement(global_rank=3), worker_type="prefill")
        assert env_vars["SGLANG_DG_CACHE_DIR"] == "/tmp/sglang_deep_gemm/prefill_rank_3"


class TestComputeNodesPerEngine:
    def test_engines_that_fit_on_one_node_use_one_node(self):
        assert compute_nodes_per_engine(num_gpus_per_engine=4, num_gpus_per_node=8) == 1

    def test_engines_larger_than_a_node_span_whole_nodes(self):
        assert compute_nodes_per_engine(num_gpus_per_engine=16, num_gpus_per_node=8) == 2


class TestComputeRolloutOffset:
    def test_colocate_returns_zero(self):
        """Colocated rollout shares the megatron gpus, so it starts at slot 0."""
        args = make_args(colocate=True, actor_num_nodes=1, actor_num_gpus_per_node=8)
        assert compute_rollout_offset(args) == 0

    def test_critic_train_only_offsets_past_the_critic(self):
        args = make_args(critic_train_only=True, critic_num_nodes=1, critic_num_gpus_per_node=4)
        assert compute_rollout_offset(args) == 4

    def test_actor_plus_critic_offsets_past_both(self):
        args = make_args(
            use_critic=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert compute_rollout_offset(args) == 12


class TestComputeMegatronNumGpus:
    def test_actor_only(self):
        args = make_args(actor_num_nodes=2, actor_num_gpus_per_node=8)
        assert compute_megatron_num_gpus(args) == 16

    def test_actor_plus_critic(self):
        args = make_args(
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert compute_megatron_num_gpus(args) == 12

    def test_zero_when_debug_rollout_only(self):
        args = make_args(debug_rollout_only=True)
        assert compute_megatron_num_gpus(args) == 0
