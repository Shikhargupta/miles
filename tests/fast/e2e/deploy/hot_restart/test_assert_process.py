import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_process import (
    assert_the_run_was_watched_closely_enough,
    assert_the_trainer_never_rebooted,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import POD_KIND, ObservationCounts
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

        with pytest.raises(AssertionError, match="never reached again"):
            assert_the_trainer_never_rebooted(evidence_of(snapshots=snapshots, records=[]))


class TestAssertTheRunWasWatchedCloselyEnough:
    def test_a_run_observed_successfully_throughout_passes(self):
        """Every verdict about pods is worth exactly what the observations behind it cost."""
        assert_the_run_was_watched_closely_enough(evidence_of(snapshots=two_restarts(), records=[]))

    def test_a_run_whose_observations_mostly_failed_fails(self):
        """A cluster nobody could read looks exactly like a cluster where nothing was replaced."""
        observations = ObservationCounts(attempts_of_read={POD_KIND: 10}, failures_of_read={POD_KIND: 9})

        with pytest.raises(AssertionError, match="answered less than"):
            assert_the_run_was_watched_closely_enough(
                evidence_of(snapshots=two_restarts(), records=[], observations=observations)
            )

    def test_a_run_that_counted_no_observation_at_all_fails(self):
        """Evidence written before anything was counted would pass every ratio there is."""
        with pytest.raises(AssertionError, match="how often this run was observed"):
            assert_the_run_was_watched_closely_enough(
                evidence_of(snapshots=two_restarts(), records=[], observations=ObservationCounts())
            )
