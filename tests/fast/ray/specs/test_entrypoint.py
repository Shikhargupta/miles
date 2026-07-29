from __future__ import annotations

from tests.fast.ray.specs.conftest import make_args

from miles.ray.specs.entrypoint import compute_specs


class TestComputeSpecs:
    def test_full_system_lists_all_worker_sets(self):
        """The default layout produces trainer, engine, router, and executor specs."""
        names = [spec.name for spec in compute_specs(make_args())]
        assert names == ["train-actor", "sglang-default-group0", "router-default", "rollout-executor"]

    def test_session_servers_appear_when_enabled(self):
        """--use-session-server adds the session server workers."""
        names = [spec.name for spec in compute_specs(make_args(use_session_server=True))]
        assert "session-server" in names

    def test_debug_train_only_drops_engines_and_routers(self):
        """debug_train_only keeps the trainer but no sglang engines or routers."""
        names = [spec.name for spec in compute_specs(make_args(debug_train_only=True))]
        assert names == ["train-actor", "rollout-executor"]

    def test_debug_rollout_only_drops_trainers(self):
        """debug_rollout_only keeps the engines but no trainer."""
        names = [spec.name for spec in compute_specs(make_args(debug_rollout_only=True))]
        assert names == ["sglang-default-group0", "router-default", "rollout-executor"]

    def test_spec_names_are_unique(self):
        """A critic plus multi-group inference still yields unique names."""
        args = make_args(
            use_critic=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
            prefill_num_servers=2,
            use_session_server=True,
        )
        names = [spec.name for spec in compute_specs(args)]
        assert len(names) == len(set(names))
