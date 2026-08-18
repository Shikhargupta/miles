import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_workloads import assert_only_the_orchestration_side_restarted
from tests.fast.e2e.deploy.hot_restart.cluster_facts import ORCHESTRATOR, ROLLOUT_EXECUTOR, TRAINER
from tests.fast.e2e.deploy.hot_restart.restart_facts import (
    ENGINE_POOL_POD,
    evidence_of,
    quiet_run_snapshot,
    restart_snapshot,
    restarted_snapshot,
    two_restarts,
)


class TestAssertOnlyTheOrchestrationSideRestarted:
    def test_a_run_whose_script_was_replaced_twice_passes(self):
        """This is what the whole feature promises: two new scripts, the same trainers underneath."""
        assert_only_the_orchestration_side_restarted(evidence_of(snapshots=two_restarts(), records=[]), num_restarts=2)

    def test_a_pod_belonging_to_no_listed_workload_fails(self):
        """A pod nothing owns is a pod this verdict silently says nothing about."""
        snapshots = [
            *two_restarts(),
            restart_snapshot(
                uid_of_pod={f"{ORCHESTRATOR}-0": "uid-o-3", ENGINE_POOL_POD: "uid-e"},
                stamp_of_workload={ORCHESTRATOR: "t2"},
            ),
        ]

        with pytest.raises(AssertionError, match="belong to no workload"):
            assert_only_the_orchestration_side_restarted(evidence_of(snapshots=snapshots, records=[]), num_restarts=2)

    def test_a_trainer_pod_that_was_replaced_fails(self):
        """A trainer that restarted lost the weights the take-over claims it reloaded a checkpoint into."""
        snapshots = [
            *two_restarts(),
            restart_snapshot(
                uid_of_pod={
                    f"{ORCHESTRATOR}-0": "uid-o-3",
                    f"{ROLLOUT_EXECUTOR}-0": "uid-r-3",
                    f"{TRAINER}-0": "uid-t-2",
                },
                stamp_of_workload={ORCHESTRATOR: "t2", ROLLOUT_EXECUTOR: "t2", TRAINER: None},
            ),
        ]

        with pytest.raises(AssertionError, match="lost a pod"):
            assert_only_the_orchestration_side_restarted(evidence_of(snapshots=snapshots, records=[]), num_restarts=2)

    def test_a_trainer_whose_pod_template_was_rewritten_fails(self):
        """An ordinary relaunch has to render a zero diff for every object it does not replace."""
        snapshots = [
            *two_restarts(),
            restart_snapshot(
                uid_of_pod={
                    f"{ORCHESTRATOR}-0": "uid-o-3",
                    f"{ROLLOUT_EXECUTOR}-0": "uid-r-3",
                    f"{TRAINER}-0": "uid-t",
                },
                stamp_of_workload={ORCHESTRATOR: "t2", ROLLOUT_EXECUTOR: "t2", TRAINER: "t2"},
            ),
        ]

        with pytest.raises(AssertionError):
            assert_only_the_orchestration_side_restarted(evidence_of(snapshots=snapshots, records=[]), num_restarts=2)

    def test_a_second_restart_that_never_landed_fails(self):
        """Testing one take-over would not show that a script can take over what a script already took over."""
        snapshots = [quiet_run_snapshot(), restarted_snapshot(stamp="t1", uid_suffix="2")]

        with pytest.raises(AssertionError, match="hot restart"):
            assert_only_the_orchestration_side_restarted(evidence_of(snapshots=snapshots, records=[]), num_restarts=2)
