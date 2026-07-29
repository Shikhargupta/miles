from __future__ import annotations

from tests.fast.ray.specs.conftest import make_args

from miles.ray.specs.rollout import spec_rollout_executor


class TestSpecRolloutExecutor:
    def test_single_cpu_worker_with_only_the_rpc_port(self):
        """The rollout executor is one gpu-less worker exposing just the rpc port."""
        spec = spec_rollout_executor(make_args())
        assert spec.name == "rollout-executor"
        assert [port_info.name for port_info in spec.port_infos] == ["rpc"]
        assert spec.scheduling.num_cells == 1
        assert spec.scheduling.num_workers_per_cell == 1
        assert spec.scheduling.num_gpus_per_worker == 0

    def test_ctor_kwargs_carries_args(self):
        """ctor_kwargs stays lazy and passes the args through."""
        args = make_args()
        spec = spec_rollout_executor(args)
        assert spec.ctor_kwargs() == dict(args=args)
        assert spec.worker_class == "miles.ray.rollout.rollout_executor.RolloutExecutor"
