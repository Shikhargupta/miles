from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_sglang_config_yaml
from tests.fast.ray.specs.conftest import make_args

from miles.ray.specs.inference import (
    _compute_engine_env_vars,
    _compute_megatron_num_gpus,
    _compute_nodes_per_engine,
    _compute_rollout_pg_offset,
    compute_inference_deployments,
    compute_inference_specs,
)


class TestComputeInferenceSpecs:
    def test_default_config_yields_one_group_of_single_gpu_cells(self):
        """8 rollout gpus at 1 gpu/engine become one spec with 8 one-worker cells."""
        (spec,) = compute_inference_specs(make_args())
        assert spec.name == "sglang-default-group0"
        assert spec.scheduling.num_cells == 8
        assert spec.scheduling.num_workers_per_cell == 1
        assert spec.scheduling.num_gpus_per_worker == 1

    def test_multi_node_engine_groups_node_ranks_into_one_cell(self):
        """A 16-gpu engine on 8-gpu nodes yields cells of two 8-gpu workers."""
        args = make_args(rollout_num_gpus=32, rollout_num_gpus_per_engine=16, num_gpus_per_node=8)
        (spec,) = compute_inference_specs(args)
        assert spec.scheduling.num_cells == 2
        assert spec.scheduling.num_workers_per_cell == 2
        assert spec.scheduling.num_gpus_per_worker == 8

    def test_empty_when_debug_train_only(self):
        """debug_train_only launches no inference engines."""
        assert compute_inference_specs(make_args(debug_train_only=True)) == []

    def test_empty_when_rollout_external(self):
        """External rollout mode must not produce any engine specs."""
        assert compute_inference_specs(make_args(rollout_external=True)) == []

    def test_prefill_decode_split_yields_one_spec_per_group(self):
        """PD disaggregation produces separate prefill and decode specs."""
        args = make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=1, prefill_num_servers=2)
        prefill, decode = compute_inference_specs(args)
        assert prefill.name == "sglang-default-group0"
        assert decode.name == "sglang-default-group1"
        assert prefill.scheduling.num_cells == 2
        assert decode.scheduling.num_cells == 6

    def test_prefill_group_gets_disaggregation_bootstrap_port(self):
        """Only prefill workers need the disaggregation bootstrap port."""
        args = make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=1, prefill_num_servers=2)
        prefill, decode = compute_inference_specs(args)
        assert "disaggregation_bootstrap" in [port_info.name for port_info in prefill.port_infos]
        assert "disaggregation_bootstrap" not in [port_info.name for port_info in decode.port_infos]

    def test_multi_model_yaml_expands_into_per_model_specs(self, tmp_path):
        """Each configured model contributes its own uniquely named specs."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            make_sglang_config_yaml(name="actor") + make_sglang_config_yaml(name="ref").replace("sglang:\n", "")
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=16)
        specs = compute_inference_specs(args)
        assert [spec.name for spec in specs] == ["sglang-actor-group0", "sglang-ref-group0"]

    def test_placeholder_group_produces_no_spec(self, tmp_path):
        """Placeholder groups reserve gpus but must not spawn workers."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: placeholder\n"
            "        num_gpus: 4\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 1\n"
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=8)
        (spec,) = compute_inference_specs(args)
        assert spec.name == "sglang-default-group1"
        assert spec.scheduling.num_cells == 4

    def test_dist_init_port_reserves_a_consecutive_block(self):
        """The dist-init master port reserves 30 + dp_size consecutive ports."""
        (spec,) = compute_inference_specs(make_args(sglang_dp_size=2))
        (dist_init,) = [port_info for port_info in spec.port_infos if port_info.name == "dist_init"]
        assert dist_init.mode == "master"
        assert dist_init.num_consecutive == 32

    def test_ctor_kwargs_carries_shared_engine_arguments(self):
        """ctor_kwargs stays lazy and yields args-level shared engine settings."""
        args = make_args()
        (spec,) = compute_inference_specs(args)
        kwargs = spec.ctor_kwargs(0, 0)
        assert kwargs["args"] is args
        assert kwargs["worker_type"] == "regular"
        assert kwargs["num_gpus_per_engine"] == 1
        assert kwargs["sglang_overrides"]["model_path"] == args.hf_checkpoint

    def test_ctor_kwargs_computes_global_engine_ranks(self):
        """Ranks advance per worker within a cell and keep counting across groups."""
        args = make_args(rollout_num_gpus=32, rollout_num_gpus_per_engine=16, num_gpus_per_node=8)
        (spec,) = compute_inference_specs(args)
        assert spec.ctor_kwargs(0, 1)["rank"] == 1
        assert spec.ctor_kwargs(1, 0)["rank"] == 2

    def test_ctor_kwargs_ranks_continue_after_earlier_groups(self):
        """A later group's ranks start after all engines of earlier groups."""
        args = make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=1, prefill_num_servers=2)
        prefill, decode = compute_inference_specs(args)
        assert prefill.ctor_kwargs(0, 0)["rank"] == 0
        assert decode.ctor_kwargs(0, 0)["rank"] == 2

    def test_server_port_declares_http_url_scheme(self):
        """The engine's server port is the cell's url so providers can hand out urls."""
        (spec,) = compute_inference_specs(make_args())
        (server,) = [port_info for port_info in spec.port_infos if port_info.name == "server"]
        assert server.url_scheme == "http"

    def test_deployments_expose_model_metadata(self):
        """Deployments carry the model-level fields the rollout server needs."""
        args = make_args()
        (deployment,) = compute_inference_deployments(args)
        assert deployment.model_name == "default"
        assert deployment.update_weights is True
        assert deployment.model_path == args.hf_checkpoint
        assert deployment.group_gpu_offset == 0

    def test_host_port_overrides_are_rejected(self, tmp_path):
        """A config that pins host/port would bypass managed ports and must fail."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 1\n"
            "        overrides:\n"
            "          port: 12345\n"
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=8)
        with pytest.raises(AssertionError, match="host/port"):
            compute_inference_specs(args)

    def test_colocated_offload_group_keeps_memory_saver(self):
        """A group sharing gpus with megatron keeps memory saver enabled."""
        args = make_args(colocate=True, offload_rollout=True)
        (spec,) = compute_inference_specs(args)
        assert "enable_memory_saver" not in spec.ctor_kwargs(0, 0)["sglang_overrides"]

    def test_disjoint_offload_group_disables_memory_saver(self):
        """A group beyond the megatron gpus gets enable_memory_saver=False."""
        args = make_args(colocate=False, offload_rollout=True, actor_num_nodes=1, actor_num_gpus_per_node=8)
        (spec,) = compute_inference_specs(args)
        assert spec.ctor_kwargs(0, 0)["sglang_overrides"]["enable_memory_saver"] is False


class TestComputeEngineEnvVars:
    def test_defaults_include_sglang_flags_and_noset_vars(self):
        """The baseline env pins the sglang flags and the ray NOSET flags."""
        env = _compute_engine_env_vars(make_args())
        assert env["NVSHMEM_DISABLE_NCCL"] == "1"
        assert env["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "false"
        assert env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"

    def test_rank_dependent_dg_cache_dir_is_left_to_the_launcher(self):
        """SGLANG_DG_CACHE_DIR is per-rank, so the shared env must not set it."""
        assert "SGLANG_DG_CACHE_DIR" not in _compute_engine_env_vars(make_args())

    def test_custom_all_reduce_disabled_for_colocated_multi_gpu_engines(self):
        """Colocate with multi-gpu engines turns the custom all-reduce off."""
        env = _compute_engine_env_vars(make_args(colocate=True, rollout_num_gpus_per_engine=2))
        assert env["SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"] == "0"
        env = _compute_engine_env_vars(make_args(colocate=False, rollout_num_gpus_per_engine=2))
        assert env["SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"] == "1"

    def test_environment_overrides_are_respected(self, monkeypatch):
        """A value exported by the launching process wins over the default."""
        monkeypatch.setenv("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "true")
        assert _compute_engine_env_vars(make_args())["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "true"


class TestComputeNodesPerEngine:
    def test_engine_within_one_node(self):
        """An engine no larger than a node stays on one node."""
        assert _compute_nodes_per_engine(num_gpus_per_engine=1, num_gpus_per_node=8) == 1
        assert _compute_nodes_per_engine(num_gpus_per_engine=8, num_gpus_per_node=8) == 1

    def test_engine_spanning_nodes(self):
        """A 16-gpu engine on 8-gpu nodes spans two nodes."""
        assert _compute_nodes_per_engine(num_gpus_per_engine=16, num_gpus_per_node=8) == 2


class TestLayoutHelpers:
    def test_rollout_offset_follows_actor_and_critic(self):
        """Rollout gpus start after the actor and critic slots."""
        args = make_args(
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_rollout_pg_offset(args) == 12
        assert _compute_megatron_num_gpus(args) == 12

    def test_colocate_starts_rollout_at_zero(self):
        """Colocated rollout shares the training gpus from slot zero."""
        assert _compute_rollout_pg_offset(make_args(colocate=True)) == 0
