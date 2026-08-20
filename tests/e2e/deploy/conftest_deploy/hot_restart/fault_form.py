import logging
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import read_restart_stamps_of_release
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_hot_restart_config, compute_release_of_config
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import (
    HotRestartRecord,
    read_attempts_of_rollout_id,
    read_run_progress,
)
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import BaseFaultForm

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig

logger = logging.getLogger(__name__)

HOT_RESTART_FORM_NAME: str = "hot_restart"
TAKE_OVER_TIMEOUT_SECONDS: float = 1800.0
TAKE_OVER_POLL_INTERVAL_SECONDS: float = 10.0
RELAUNCH_JOIN_TIMEOUT_SECONDS: float = 1800.0


class HotRestartFaultForm(BaseFaultForm):
    def __init__(
        self,
        *,
        launch: Callable[[ExecuteTrainConfig], None],
        config: ExecuteTrainConfig,
        checkpoint_dir: Path,
        events_dir: Path,
        poll_interval_seconds: float = TAKE_OVER_POLL_INTERVAL_SECONDS,
        timeout_seconds: float = TAKE_OVER_TIMEOUT_SECONDS,
    ) -> None:
        self._launch = launch
        self._config = config
        self._release = compute_release_of_config(config)
        self._namespace = config.namespace
        self._checkpoint_dir = checkpoint_dir
        self._events_dir = events_dir
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds
        self._threads: list[threading.Thread] = []
        self._failures: list[tuple[int, BaseException]] = []
        self._records: list[HotRestartRecord] = []

    @property
    def name(self) -> str:
        return HOT_RESTART_FORM_NAME

    @property
    def records(self) -> tuple[HotRestartRecord, ...]:
        return tuple(self._records)

    def join_relaunches(self, *, timeout_seconds: float = RELAUNCH_JOIN_TIMEOUT_SECONDS) -> None:
        for thread in self._threads:
            thread.join(timeout=timeout_seconds)

    def assert_every_take_over_installed_cleanly(self) -> None:
        # The last relaunch is what drives the run to its end, so this is where the run's own
        # verdict surfaces: its launcher raises, and that exception lands in _failures.
        assert not self._failures, "a hot restart of this run did not install cleanly:\n" + "\n".join(
            f"  - take-over {at}: {failure!r}" for at, failure in self._failures
        )
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        assert not alive, (
            f"{alive} are still installing a hot restart, so this run may still be replaced under the dumps that "
            f"are about to be read"
        )

    @property
    def harms_the_cell(self) -> bool:
        return False

    def inject(self, cell: dict, rng: random.Random) -> None:
        progress = read_run_progress(checkpoint_dir=self._checkpoint_dir, events_dir=self._events_dir)
        before = read_attempts_of_rollout_id(self._events_dir)

        logger.info(f"Hot restarting {self._release} of a run that stands at {progress}")
        index = len(self._threads)
        self._relaunch_on_a_thread()
        self._wait_until_the_take_over_reached_the_run(index=index, before=before)
        self._records.append(
            HotRestartRecord(
                index=index,
                saved_iteration_at_trigger=progress.last_saved_iteration,
                frozen_rollout_id=max(before, default=-1),
            )
        )

    def _relaunch_on_a_thread(self) -> None:
        index = len(self._threads)
        thread = threading.Thread(target=self._relaunch, args=(index,), daemon=True, name=f"take-over-{index}")
        self._threads.append(thread)
        thread.start()

    def _relaunch(self, index: int) -> None:
        try:
            self._launch(compute_hot_restart_config(self._config, installed_release=self._release))
        except BaseException as e:
            logger.warning(f"The hot restart of {self._release} failed to install", exc_info=True)
            self._failures.append((index, e))

    def _wait_until_the_take_over_reached_the_run(self, *, index: int, before: dict[int, int]) -> None:
        deadline = time.monotonic() + self._timeout_seconds
        relaunch = self._threads[index]

        while time.monotonic() < deadline:
            time.sleep(self._poll_interval_seconds)
            if carries_the_stamps_of(self._read_restart_stamps_or_none(), take_overs=index + 1):
                logger.info(
                    f"The take-over of {self._release} landed; its workloads carry {index + 1} restart stamp(s)"
                )
                return
            if describes_a_run_that_redid_a_step(before=before, after=self._read_attempts_or_none()):
                logger.info(f"The take-over of {self._release} landed; the run redid a step it had trained")
                return

            assert not self._failures_of(index), (
                f"the hot restart of {self._release} was refused rather than installed, so the run keeps training "
                f"under the script this injection meant to replace: {self._failures_of(index)}"
            )
            assert relaunch.is_alive(), (
                f"the relaunch of {self._release} returned without its workloads ever carrying {index + 1} restart "
                f"stamp(s), so nothing took its orchestration script over"
            )

        raise AssertionError(
            f"the relaunch of {self._release} was installed {self._timeout_seconds}s ago and its workloads still do "
            f"not carry {index + 1} restart stamp(s), nor has the run redone any of the steps it had trained "
            f"({before}), so a take-over cannot be told from a relaunch that hung"
        )

    def _failures_of(self, index: int) -> list[BaseException]:
        return [failure for at, failure in self._failures if at == index]

    def _read_attempts_or_none(self) -> dict[int, int] | None:
        try:
            return read_attempts_of_rollout_id(self._events_dir)
        except Exception:
            logger.warning("Failed to read how far the run being hot restarted has come", exc_info=True)
            return None

    def _read_restart_stamps_or_none(self) -> set[str] | None:
        try:
            return read_restart_stamps_of_release(release=self._release, namespace=self._namespace)
        except Exception:
            logger.warning(f"Failed to read the restart stamps of {self._release}", exc_info=True)
            return None


def carries_the_stamps_of(stamps: set[str] | None, *, take_overs: int) -> bool:
    """A take-over writes one restart stamp on every workload it replaces, and only take-overs write them."""
    if stamps is None:
        return False
    return len(stamps) >= take_overs


def describes_a_run_that_redid_a_step(*, before: dict[int, int], after: dict[int, int] | None) -> bool:
    """A take-over resuming from a checkpoint rolls the log back; one starting over re-trains step 0."""
    if after is None or not before:
        return False
    if max(after, default=-1) < max(before):
        return True
    return any(after.get(rollout_id, 0) > attempts for rollout_id, attempts in before.items())
