from pathlib import Path
from typing import Any

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import driver as driver_module
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_probe import SnapshotAttempt
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import HotRestartDriver
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import (
    BOOT_UUID_READ,
    POD_KIND,
    HotRestartRecord,
    ObservationCounts,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.progress import RunProgress
from tests.fast.e2e.deploy.hot_restart.cluster_facts import (
    NAMESPACE,
    RELEASE,
    TRAINER,
    cluster_snapshot,
    pod_fact,
    workload_fact,
)


def _driver(tmp_path: Path, **overrides: Any) -> HotRestartDriver:
    kwargs: dict[str, Any] = dict(
        relaunch=lambda: None,
        checkpoint_dir=tmp_path / "checkpoints",
        events_dir=tmp_path / "events",
        release=RELEASE,
        namespace=NAMESPACE,
        trainer_id="actor",
        num_restarts=2,
    )
    kwargs.update(overrides)
    return HotRestartDriver(**kwargs)


class TestHotRestartDriverStart:
    def test_a_dump_directory_holding_a_previous_run_is_refused(self, tmp_path):
        """Resuming from what a previous run left would compare a run that never started from nothing."""
        driver = _driver(tmp_path)
        (tmp_path / "checkpoints").mkdir()

        with pytest.raises(AssertionError, match="before this run was even installed"):
            driver.start()


class TestHotRestartDriverProgressGuard:
    def test_a_run_whose_event_log_shrank_before_any_restart_fails(self, tmp_path):
        """Nothing but a take-over rolls the log back, so this is a run reading someone else's dumps."""
        driver = _driver(tmp_path)
        driver._assert_the_run_never_lost_a_step_outside_a_take_over(
            RunProgress(last_saved_iteration=1, last_finished_rollout_id=3)
        )

        with pytest.raises(AssertionError, match="lost work"):
            driver._assert_the_run_never_lost_a_step_outside_a_take_over(
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)
            )

    def test_the_rollback_a_take_over_performs_is_not_read_as_lost_work(self, tmp_path):
        """The log going back to the checkpoint is the very thing a hot restart is supposed to do."""
        driver = _driver(tmp_path)
        driver.records.append(
            HotRestartRecord(index=0, saved_iteration_at_trigger=1, finished_rollout_id_at_trigger=3)
        )
        driver._assert_the_run_never_lost_a_step_outside_a_take_over(
            RunProgress(last_saved_iteration=1, last_finished_rollout_id=4)
        )

        driver._assert_the_run_never_lost_a_step_outside_a_take_over(
            RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)
        )


class TestHotRestartDriverSnapshots:
    def _install_reader(self, monkeypatch, attempts: list[SnapshotAttempt]) -> None:
        remaining = list(attempts)
        monkeypatch.setattr(driver_module, "read_cluster_snapshot", lambda **_kwargs: remaining.pop(0))

    def test_every_read_the_driver_tried_and_every_one_that_failed_is_counted(self, tmp_path, monkeypatch):
        """A verdict read off two lucky observations of a run nobody could reach proves nothing."""
        driver = _driver(tmp_path)
        self._install_reader(
            monkeypatch,
            [
                SnapshotAttempt(
                    snapshot=None, reads_attempted=(POD_KIND, BOOT_UUID_READ), reads_failed=(POD_KIND, BOOT_UUID_READ)
                ),
                SnapshotAttempt(
                    snapshot=cluster_snapshot(
                        pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=[workload_fact(TRAINER)]
                    ),
                    reads_attempted=(POD_KIND, BOOT_UUID_READ),
                    reads_failed=(),
                ),
            ],
        )

        driver._collect_snapshot()
        driver._collect_snapshot()

        assert driver.evidence.observations == ObservationCounts(
            attempts_of_read={POD_KIND: 2, BOOT_UUID_READ: 2}, failures_of_read={POD_KIND: 1, BOOT_UUID_READ: 1}
        )
        assert len(driver.snapshots) == 1

    def test_a_release_being_uninstalled_is_not_recorded_as_a_run_losing_its_pods(self, tmp_path, monkeypatch):
        """The last observation happens while the run is torn down, and every pod is gone by then."""
        driver = _driver(tmp_path)
        self._install_reader(
            monkeypatch,
            [
                SnapshotAttempt(
                    snapshot=cluster_snapshot(pods=[], workloads=[]), reads_attempted=(POD_KIND,), reads_failed=()
                )
            ],
        )

        driver._collect_snapshot()

        assert driver.snapshots == []
