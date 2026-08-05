from __future__ import annotations

from types import SimpleNamespace

from miles.ray.specs.rollout import (
    ROLLOUT_EXECUTOR_RPC_CLASS,
    ROLLOUT_EXECUTOR_SPEC_NAME,
    ROLLOUT_EXECUTOR_WORKER_CLASS,
    rollout_executor_cell_id,
    rollout_executor_worker_name,
    spec_rollout_executor,
)
from miles.utils.external_utils.command_utils.helm_backend.values import RunLayout, build_values, section_of


def _args(cluster_backend: str) -> SimpleNamespace:
    return SimpleNamespace(cluster_backend=cluster_backend, pin_rollout_manager_to_head=False)


def _layout() -> RunLayout:
    return RunLayout(
        run_id="260101-000000-000",
        release="miles-run-260101",
        orchestrator_command=["python", "/repo/train.py"],
        worker_argv=["--rollout-num-gpus", "8"],
        num_gpus_per_node=8,
    )


class TestRolloutExecutorSpec:
    def test_a_kubernetes_run_asks_for_exactly_one_gpuless_worker(self):
        """One executor per run, and it must claim no gpu or the scheduler would reserve a whole node."""
        spec = spec_rollout_executor(_args("kubernetes"))

        assert spec.name == ROLLOUT_EXECUTOR_SPEC_NAME
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (1, 1)
        assert spec.scheduling.num_gpu_slots_per_worker == 0

    def test_a_ray_run_lists_the_spec_with_no_cells(self):
        """Under ray the driver still creates the actor itself, so the platform must launch nothing."""
        spec = spec_rollout_executor(_args("ray"))

        assert spec.scheduling.num_cells == 0

    def test_the_caller_facing_class_is_the_ray_free_api(self):
        """The driver introspects this class to build rpc stubs, so it must not drag ray into a namespace."""
        spec = spec_rollout_executor(_args("kubernetes"))

        assert spec.worker_class == ROLLOUT_EXECUTOR_WORKER_CLASS
        assert spec.caller_facing_class == ROLLOUT_EXECUTOR_RPC_CLASS

    def test_it_renders_into_static_workers_with_its_rpc_port(self):
        """The release has to contain the executor pod, or the address book would point at nothing."""
        spec = spec_rollout_executor(_args("kubernetes"))

        values = build_values([spec], _layout())

        (entry,) = values["run"]["staticWorkers"]
        assert section_of(spec) == "staticWorkers"
        assert entry["name"] == ROLLOUT_EXECUTOR_SPEC_NAME
        assert entry["ports"] == [{"name": "rpc", "port": 8000}]
        assert ROLLOUT_EXECUTOR_WORKER_CLASS in entry["command"]
        assert "resources" not in entry

    def test_the_worker_and_cell_names_are_stable(self):
        """The driver looks the executor up by name, so these names are part of the release's contract."""
        assert rollout_executor_worker_name() == "rollout-executor-0-0"
        assert rollout_executor_cell_id() == "rollout-executor-0"
