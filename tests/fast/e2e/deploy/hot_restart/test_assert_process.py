import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_process import (
    _compute_trainer_boot_uuids,
    assert_the_run_was_watched_closely_enough,
    assert_the_trainer_never_rebooted,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import POD_KIND
from tests.fast.e2e.deploy.hot_restart.cluster_facts import cluster_snapshot
from tests.fast.e2e.deploy.hot_restart.restart_facts import (
    evidence_of,
    quiet_run_snapshot,
    restart_snapshot,
    restarted_snapshot,
    two_restarts,
)


class TestAssertTheTrainerNeverRebooted:
    def test_one_boot_uuid_across_every_script_passes(self):
        """The trainer process outliving both scripts is what makes this a hot restart at all."""
        assert_the_trainer_never_rebooted(evidence_of(snapshots=two_restarts(), records=[]))

    def test_a_second_boot_uuid_fails(self):
        """A restarted rpc server serves a new process, whatever its pod's uid says."""
        snapshots = [*two_restarts(), restart_snapshot(uid_of_pod={}, stamp_of_workload={}, boot_uuid="boot-b")]

        with pytest.raises(AssertionError, match="boot uuid"):
            assert_the_trainer_never_rebooted(evidence_of(snapshots=snapshots, records=[]))

    def test_a_trainer_never_reached_again_after_a_take_over_fails(self):
        """A uuid only ever read before the last take-over says nothing about what survived it."""
        snapshots = [
            quiet_run_snapshot(),
            restarted_snapshot(stamp="t1", uid_suffix="2"),
            restarted_snapshot(stamp="t2", uid_suffix="3", boot_uuid=None),
        ]

        with pytest.raises(AssertionError, match="never reached after"):
            assert_the_trainer_never_rebooted(evidence_of(snapshots=snapshots, records=[]))


class TestAssertTheRunWasWatchedCloselyEnough:
    def test_a_run_observed_throughout_passes(self):
        """Every verdict about pods is worth exactly what the observations behind it cost."""
        assert_the_run_was_watched_closely_enough(evidence_of(snapshots=two_restarts(), records=[]))

    def test_a_run_nobody_looked_at_twice_fails(self):
        """Every workload verdict is a comparison between observations, and one of them says nothing."""
        with pytest.raises(AssertionError, match="never looked at twice"):
            assert_the_run_was_watched_closely_enough(evidence_of(snapshots=[quiet_run_snapshot()], records=[]))

    def test_a_run_whose_observations_never_saw_a_whole_release_fails(self):
        """A cluster nobody could read whole looks exactly like a cluster where nothing was replaced."""
        partial = [one.model_copy(update={"reads_missing": (POD_KIND,)}) for one in two_restarts()]

        with pytest.raises(AssertionError, match="never looked at twice"):
            assert_the_run_was_watched_closely_enough(evidence_of(snapshots=partial, records=[]))

    def test_a_run_nothing_ever_observed_fails(self):
        """Evidence written before anything was collected would pass every count there is."""
        with pytest.raises(AssertionError, match="never looked at twice"):
            assert_the_run_was_watched_closely_enough(evidence_of(snapshots=[], records=[]))


class TestComputeTrainerBootUuids:
    def test_a_trainer_that_outlived_every_script_answers_with_one_boot_uuid(self):
        """A second uuid means the process a hot restart promised to keep alive was replaced."""
        snapshots = [
            cluster_snapshot(pods=[], workloads=[], trainer_boot_uuid="boot-a"),
            cluster_snapshot(pods=[], workloads=[], trainer_boot_uuid=None),
            cluster_snapshot(pods=[], workloads=[], trainer_boot_uuid="boot-a"),
        ]

        assert _compute_trainer_boot_uuids(snapshots) == {"boot-a"}
