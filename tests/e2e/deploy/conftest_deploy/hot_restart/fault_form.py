import logging
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_hot_restart_config, compute_release_of_config
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import RunProgress, read_run_progress
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import BaseFaultForm

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig

logger = logging.getLogger(__name__)

HOT_RESTART_FORM_NAME: str = "hot_restart"
TAKE_OVER_TIMEOUT_SECONDS: float = 1800.0
TAKE_OVER_POLL_INTERVAL_SECONDS: float = 10.0


class HotRestartIsNotDueYet(Exception):
    pass


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
        self._checkpoint_dir = checkpoint_dir
        self._events_dir = events_dir
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds
        self._threads: list[threading.Thread] = []
        self._failures: list[BaseException] = []

    @property
    def name(self) -> str:
        return HOT_RESTART_FORM_NAME

    @property
    def harms_the_cell(self) -> bool:
        return False

    def inject(self, cell: dict, rng: random.Random) -> None:
        progress = read_run_progress(checkpoint_dir=self._checkpoint_dir, events_dir=self._events_dir)
        if not can_be_taken_over(progress):
            raise HotRestartIsNotDueYet(
                f"the run has saved {progress.last_saved_iteration} and finished step "
                f"{progress.last_finished_rollout_id}; a take-over only costs steps no checkpoint covers, so this "
                f"draw waits for a save with a finished step past it"
            )

        logger.info(f"Hot restarting {self._release} of a run that stands at {progress}")
        self._relaunch_on_a_thread()
        self._wait_until_the_take_over_reached_the_run(progress)

    def _relaunch_on_a_thread(self) -> None:
        self._failures.clear()
        thread = threading.Thread(target=self._relaunch, daemon=True, name=f"take-over-{len(self._threads)}")
        self._threads.append(thread)
        thread.start()

    def _relaunch(self) -> None:
        try:
            self._launch(compute_hot_restart_config(self._config, installed_release=self._release))
        except BaseException as e:
            logger.warning(f"The hot restart of {self._release} failed to install", exc_info=True)
            self._failures.append(e)

    def _wait_until_the_take_over_reached_the_run(self, before: RunProgress) -> None:
        deadline = time.monotonic() + self._timeout_seconds
        relaunch = self._threads[-1]

        while time.monotonic() < deadline:
            time.sleep(self._poll_interval_seconds)
            if _describes_a_run_rolled_back_to_its_checkpoint(before=before, after=self._read_progress_or_none()):
                logger.info(f"The take-over of {self._release} landed, rolling the run back from {before}")
                return

            assert not self._failures, (
                f"the hot restart of {self._release} was refused rather than installed, so the run keeps training "
                f"under the script this injection meant to replace: {self._failures}"
            )
            assert relaunch.is_alive(), (
                f"the relaunch of {self._release} returned without the run ever rolling back from {before}, so "
                f"nothing took its orchestration script over"
            )

        raise AssertionError(
            f"the relaunch of {self._release} was installed {self._timeout_seconds}s ago and the run still stands "
            f"at {before}, so a take-over cannot be told from a relaunch that hung"
        )

    def _read_progress_or_none(self) -> RunProgress | None:
        try:
            return read_run_progress(checkpoint_dir=self._checkpoint_dir, events_dir=self._events_dir)
        except Exception:
            logger.warning("Failed to read how far the run being hot restarted has come", exc_info=True)
            return None


def can_be_taken_over(progress: RunProgress) -> bool:
    if (saved := progress.last_saved_iteration) is None:
        return False
    return progress.last_finished_rollout_id is not None and progress.last_finished_rollout_id > saved


def _describes_a_run_rolled_back_to_its_checkpoint(*, before: RunProgress, after: RunProgress | None) -> bool:
    if after is None or after.last_finished_rollout_id is None or before.last_finished_rollout_id is None:
        return False
    return after.last_finished_rollout_id < before.last_finished_rollout_id
