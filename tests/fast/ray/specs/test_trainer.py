from __future__ import annotations

import pytest
from tests.fast.ray.specs.conftest import make_args

from miles.ray.specs.trainer import (
    _TRAINER_MASTER_PORT,
    _compute_trainer_env_vars,
    compute_trainer_specs,
    spec_trainer_ranks,
)


class TestComputeTrainerSpecs:
    def test_actor_only_by_default(self):
        """Without a critic there is exactly one trainer spec, for the actor."""
        specs = compute_trainer_specs(make_args())
        assert [spec.name for spec in specs] == ["train-actor"]

    def test_critic_added_when_use_critic(self):
        """use_critic adds a second spec sized from the critic gpu args."""
        specs = compute_trainer_specs(make_args(use_critic=True, critic_num_nodes=1, critic_num_gpus_per_node=4))
        assert [spec.name for spec in specs] == ["train-actor", "train-critic"]
        assert specs[1].scheduling.num_workers_per_cell == 4

    def test_empty_when_debug_rollout_only(self):
        """debug_rollout_only launches no trainer workers at all."""
        assert compute_trainer_specs(make_args(debug_rollout_only=True)) == []


class TestSpecTrainerRanks:
    def test_single_cell_covers_all_gpus_without_indep_dp(self):
        """Without indep_dp the whole role is one cell of one-gpu workers."""
        spec = spec_trainer_ranks(make_args(actor_num_nodes=2, actor_num_gpus_per_node=8), role="actor")
        assert spec.scheduling.num_cells == 1
        assert spec.scheduling.num_workers_per_cell == 16
        assert spec.scheduling.num_gpus_per_worker == 1

    def test_indep_dp_splits_into_model_parallel_cells(self):
        """indep_dp makes one cell per dp replica, sized tp*pp*cp."""
        args = make_args(
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            indep_dp=True,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
        spec = spec_trainer_ranks(args, role="actor")
        assert spec.scheduling.num_cells == 4
        assert spec.scheduling.num_workers_per_cell == 2

    def test_master_port_is_master_mode_and_dynamic(self):
        """The torch master port is allocated on the master and told to the others."""
        spec = spec_trainer_ranks(make_args(), role="actor")
        master, rpc = spec.port_infos
        assert master.name == "master"
        assert master.static_port == _TRAINER_MASTER_PORT
        assert master.mode == "master"
        assert master.allow_dynamic is True
        assert rpc.name == "rpc"

    def test_worker_class_follows_train_backend(self):
        """megatron and fsdp backends map to their actor classes."""
        assert "MegatronTrainRayActor" in spec_trainer_ranks(make_args(), role="actor").worker_class
        assert "FSDPTrainRayActor" in spec_trainer_ranks(make_args(train_backend="fsdp"), role="actor").worker_class

    def test_ctor_kwargs_carries_per_worker_rank_and_cell(self):
        """ctor_kwargs yields the per-worker rank/cell besides the shared arguments."""
        args = make_args()
        spec = spec_trainer_ranks(args, role="actor")
        kwargs = spec.ctor_kwargs(1, 3)
        assert kwargs["args"] is args
        assert kwargs["role"] == "actor"
        assert kwargs["cell_index"] == 1
        assert kwargs["rank"] == 3
        assert kwargs["world_size"] == spec.scheduling.num_workers_per_cell
        assert kwargs["master_addr"] is None and kwargs["master_port"] is None
        assert kwargs["indep_dp_store_addr"] is None

    def test_fault_tolerance_selects_the_ft_wrapped_actor_class(self):
        """use_fault_tolerance routes to the lazily wrapped concurrency-group class."""
        spec = spec_trainer_ranks(make_args(use_fault_tolerance=True), role="actor")
        assert spec.worker_class == "miles.ray.train.ft_actor_classes.MegatronTrainRayActorFt"

    def test_unknown_role_is_rejected(self):
        """Roles other than actor/critic are a programming error."""
        with pytest.raises(ValueError, match="Unknown trainer role"):
            spec_trainer_ranks(make_args(), role="reward")


class TestComputeTrainerEnvVars:
    def test_defaults_include_nccl_and_noset_vars(self):
        """The baseline env pins NCCL_CUMEM_ENABLE and the ray NOSET flags."""
        env = _compute_trainer_env_vars(make_args())
        assert env["NCCL_CUMEM_ENABLE"] == "0"
        assert env["NVSHMEM_DISABLE_NCCL"] == "1"
        assert env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"

    def test_environment_overrides_are_respected(self, monkeypatch):
        """A value exported by the launching process wins over the default."""
        monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")
        assert _compute_trainer_env_vars(make_args())["NCCL_CUMEM_ENABLE"] == "1"

    def test_train_env_vars_are_merged(self):
        """User-provided --train-env-vars land in the result."""
        env = _compute_trainer_env_vars(make_args(train_env_vars={"MY_FLAG": "yes"}))
        assert env["MY_FLAG"] == "yes"

    def test_dumper_source_patcher_config_is_forwarded(self):
        """The train-side dumper source patcher config becomes an env var."""
        env = _compute_trainer_env_vars(make_args(dumper_source_patcher_config_train="cfg.json"))
        assert env["DUMPER_SOURCE_PATCHER_CONFIG"] == "cfg.json"

    def test_offload_train_preloads_torch_memory_saver(self):
        """offload_train on megatron injects the LD_PRELOAD memory saver hooks."""
        pytest.importorskip("torch_memory_saver")
        env = _compute_trainer_env_vars(make_args(offload_train=True))
        assert "torch_memory_saver" in env["LD_PRELOAD"]
        assert env["TMS_INIT_ENABLE"] == "1"
        assert env["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"

    def test_no_preload_without_offload_train(self):
        """Without offload_train the memory saver hooks are absent."""
        assert "LD_PRELOAD" not in _compute_trainer_env_vars(make_args())
