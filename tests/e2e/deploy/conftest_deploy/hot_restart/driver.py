import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_probe import compute_trainer_rpc_url, read_cluster_snapshot
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import (
    ClusterSnapshot,
    HotRestartEvidence,
    HotRestartRecord,
    ObservationCounts,
    compute_hot_restart_workloads,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.gate import HotRestartGate, compute_next_gate
from tests.e2e.deploy.conftest_deploy.hot_restart.progress import RunProgress, read_run_progress

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS: float = 5.0
GATE_TIMEOUT_SECONDS: float = 3600.0
RELAUNCH_JOIN_TIMEOUT_SECONDS: float = 1800.0
CONSECUTIVE_FAILURE_LIMIT: int = 5


@dataclass
class HotRestartDriver:
    relaunch: Callable[[], None]
    checkpoint_dir: Path
    events_dir: Path
    release: str
    namespace: str
    trainer_id: str
    num_restarts: int
    build_gate: Callable[[Sequence[HotRestartRecord]], HotRestartGate] = compute_next_gate
    poll_interval_seconds: float = POLL_INTERVAL_SECONDS
    gate_timeout_seconds: float = GATE_TIMEOUT_SECONDS
    consecutive_failure_limit: int = CONSECUTIVE_FAILURE_LIMIT
    records: list[HotRestartRecord] = field(default_factory=list)
    snapshots: list[ClusterSnapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._stop_event = threading.Event()
        self._failures: list[BaseException] = []
        self._relaunch_threads: list[threading.Thread] = []
        self._thread = threading.Thread(target=self._run, daemon=True, name="hot-restart-driver")
        self._gate = self.build_gate(self.records)
        self._deadline = 0.0
        self._max_finished_rollout_id: int | None = None
        self._attempts_of_read: Counter[str] = Counter()
        self._failures_of_read: Counter[str] = Counter()

    @property
    def hot_restart_workloads(self) -> frozenset[str]:
        return compute_hot_restart_workloads(self.release)

    @property
    def evidence(self) -> HotRestartEvidence:
        return HotRestartEvidence(
            records=tuple(self.records),
            snapshots=tuple(self.snapshots),
            release=self.release,
            observations=ObservationCounts(
                attempts_of_read=dict(self._attempts_of_read), failures_of_read=dict(self._failures_of_read)
            ),
        )

    def start(self) -> None:
        for path in (self.checkpoint_dir, self.events_dir):
            assert not path.exists(), (
                f"{path} exists before this run was even installed, and a run that resumes from what a previous run "
                f"left would be compared against a baseline that started from nothing"
            )
        self._deadline = time.monotonic() + self.gate_timeout_seconds
        self._thread.start()

    def stop_collecting(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=RELAUNCH_JOIN_TIMEOUT_SECONDS)
        for thread in self._relaunch_threads:
            thread.join(timeout=RELAUNCH_JOIN_TIMEOUT_SECONDS)
        self._collect_snapshot()

    def assert_nothing_is_still_running(self) -> None:
        assert not self._thread.is_alive(), (
            f"the hot restart driver was still working {RELAUNCH_JOIN_TIMEOUT_SECONDS}s after being asked to stop, "
            f"so reading what it collected now would race it"
        )
        for thread in self._relaunch_threads:
            assert not thread.is_alive(), (
                f"{thread.name} is still installing a hot restart {RELAUNCH_JOIN_TIMEOUT_SECONDS}s after the run "
                f"ended, so this run may still be replaced under the dumps that are about to be compared"
            )

    def assert_every_restart_happened(self) -> None:
        assert not self._failures, "the hot restart driver failed:\n" + "\n".join(
            f"  - {one!r}" for one in self._failures
        )
        assert len(self.records) == self.num_restarts, (
            f"the run ended after {len(self.records)} of {self.num_restarts} hot restart(s), so it never proved "
            f"what a take-over of the trainers of a run that is still training costs"
        )
        assert len(self.snapshots) >= 2, (
            f"the cluster was observed {len(self.snapshots)} time(s), which is too few to tell a pod that survived "
            f"every restart from one that was never looked at twice"
        )

    def _run(self) -> None:
        consecutive_failures = 0
        while not self._stop_event.is_set():
            try:
                self._observe_once()
                consecutive_failures = 0
            except BaseException as e:
                consecutive_failures += 1
                logger.warning(
                    f"The hot restart driver failed to drive the run ({consecutive_failures} time(s) in a row)",
                    exc_info=True,
                )
                if consecutive_failures >= self.consecutive_failure_limit:
                    self._failures.append(e)
                    return
            self._stop_event.wait(timeout=self.poll_interval_seconds)

    def _observe_once(self) -> None:
        progress = read_run_progress(checkpoint_dir=self.checkpoint_dir, events_dir=self.events_dir)
        self._assert_the_run_never_lost_a_step_outside_a_take_over(progress)
        if progress.last_finished_rollout_id is not None:
            self._collect_snapshot()

        if len(self.records) == self.num_restarts:
            return

        if not self._gate.observe(progress):
            assert time.monotonic() < self._deadline, (
                f"hot restart {len(self.records)} waited {self.gate_timeout_seconds}s for {self._gate.awaited}, "
                f"and the run only reached {progress} with its gate at {self._gate.stage.name}"
            )
            return

        record = self._gate.compute_record(index=len(self.records), progress=progress)
        self.records.append(record)
        logger.info(f"Hot restart {record.index} is due: {record}")
        self._trigger(record.index)
        if len(self.records) < self.num_restarts:
            self._gate = self.build_gate(self.records)
            self._deadline = time.monotonic() + self.gate_timeout_seconds

    def _assert_the_run_never_lost_a_step_outside_a_take_over(self, progress: RunProgress) -> None:
        finished = progress.last_finished_rollout_id
        if finished is None:
            return
        if self.records and finished <= self.records[-1].finished_rollout_id_at_trigger:
            return

        assert self._max_finished_rollout_id is None or finished >= self._max_finished_rollout_id, (
            f"the run had finished step {self._max_finished_rollout_id} and now reports {finished}, and outside the "
            f"rollback a take-over performs an event log only ever grows, so this run lost work nobody asked it to"
        )
        self._max_finished_rollout_id = finished

    def _trigger(self, index: int) -> None:
        thread = threading.Thread(target=self._relaunch, daemon=True, name=f"hot-restart-relaunch-{index}")
        self._relaunch_threads.append(thread)
        thread.start()

    def _relaunch(self) -> None:
        try:
            self.relaunch()
        except BaseException as e:
            logger.warning("A hot restart relaunch failed", exc_info=True)
            self._failures.append(e)

    def _collect_snapshot(self) -> None:
        attempt = read_cluster_snapshot(
            release=self.release,
            namespace=self.namespace,
            trainer_rpc_url=compute_trainer_rpc_url(
                release=self.release, namespace=self.namespace, trainer_id=self.trainer_id
            ),
        )
        self._attempts_of_read.update(attempt.reads_attempted)
        self._failures_of_read.update(attempt.reads_failed)

        if (snapshot := attempt.snapshot) is None:
            return
        if snapshot.describes_a_release_that_is_gone:
            logger.warning(
                f"Observed {self.release} with {len(snapshot.pods)} pod(s) and {len(snapshot.workloads)} "
                f"workload(s), which is a release being uninstalled rather than a run whose pods were replaced"
            )
            return
        self.snapshots.append(snapshot)


@contextmanager
def driving_hot_restarts(driver: HotRestartDriver, *, dump_dir: str) -> Iterator[HotRestartDriver]:
    driver.start()
    try:
        yield driver
    finally:
        driver.stop_collecting()
        driver.evidence.write(dump_dir=dump_dir)
        driver.assert_nothing_is_still_running()
