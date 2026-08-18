from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.progress import RunProgress


class GateStage(Enum):
    AWAITING_SAVE = auto()
    AWAITING_STEP = auto()
    OPEN = auto()


class HotRestartGate(Protocol):
    stage: GateStage

    @property
    def awaited(self) -> str: ...

    def observe(self, progress: RunProgress) -> bool: ...

    def compute_record(self, *, index: int, progress: RunProgress) -> HotRestartRecord: ...


@dataclass
class RestartGate:
    minimum_saved_iteration: int | None = None
    stage: GateStage = GateStage.AWAITING_SAVE
    saved_iteration: int | None = None
    rollout_id_at_save: int | None = None

    @property
    def awaited(self) -> str:
        return "a checkpoint and a step after it"

    def observe(self, progress: RunProgress) -> bool:
        if self.stage is GateStage.AWAITING_SAVE:
            self._observe_save(progress)
        if self.stage is GateStage.AWAITING_STEP:
            self._observe_step(progress)
        return self.stage is GateStage.OPEN

    def compute_record(self, *, index: int, progress: RunProgress) -> HotRestartRecord:
        assert self.stage is GateStage.OPEN, f"restart {index} was recorded while its gate was still {self.stage.name}"
        assert self.saved_iteration is not None and progress.last_finished_rollout_id is not None
        assert progress.last_finished_rollout_id > self.saved_iteration, (
            f"restart {index} would redo nothing: the run saved iteration {self.saved_iteration} and has finished "
            f"step {progress.last_finished_rollout_id}, so a take-over resumes exactly where this script stands"
        )
        return HotRestartRecord(
            index=index,
            saved_iteration_at_trigger=self.saved_iteration,
            finished_rollout_id_at_trigger=progress.last_finished_rollout_id,
        )

    def _observe_save(self, progress: RunProgress) -> None:
        saved = progress.last_saved_iteration
        if saved is None or (self.minimum_saved_iteration is not None and saved < self.minimum_saved_iteration):
            return
        if progress.last_finished_rollout_id is None:
            return
        self.saved_iteration = saved
        self.rollout_id_at_save = progress.last_finished_rollout_id
        self.stage = GateStage.AWAITING_STEP

    def _observe_step(self, progress: RunProgress) -> None:
        finished = progress.last_finished_rollout_id
        if finished is None or (self.rollout_id_at_save is not None and finished <= self.rollout_id_at_save):
            return
        self.stage = GateStage.OPEN


@dataclass
class NoCheckpointGate:
    stage: GateStage = GateStage.AWAITING_STEP

    @property
    def awaited(self) -> str:
        return "a finished step of a run that has saved nothing"

    def observe(self, progress: RunProgress) -> bool:
        if self.stage is GateStage.OPEN:
            return True

        self._assert_the_run_has_saved_nothing(progress)
        if progress.last_finished_rollout_id is None:
            return False

        self.stage = GateStage.OPEN
        return True

    def compute_record(self, *, index: int, progress: RunProgress) -> HotRestartRecord:
        assert self.stage is GateStage.OPEN, f"restart {index} was recorded while its gate was still {self.stage.name}"
        self._assert_the_run_has_saved_nothing(progress)
        assert progress.last_finished_rollout_id is not None
        return HotRestartRecord(
            index=index,
            saved_iteration_at_trigger=None,
            finished_rollout_id_at_trigger=progress.last_finished_rollout_id,
        )

    def _assert_the_run_has_saved_nothing(self, progress: RunProgress) -> None:
        assert progress.last_saved_iteration is None, (
            f"the run saved iteration {progress.last_saved_iteration} before anything took it over, and this gate "
            f"exists to restart a run holding no checkpoint at all; a take-over from here resumes from that save, "
            f"which is the path the deterministic scenario already covers, so this run stops instead of quietly "
            f"proving that one a second time"
        )


def compute_next_gate(records: Sequence[HotRestartRecord]) -> RestartGate:
    if not records:
        return RestartGate()
    return RestartGate(minimum_saved_iteration=records[-1].finished_rollout_id_at_trigger)


def compute_next_no_checkpoint_gate(records: Sequence[HotRestartRecord]) -> NoCheckpointGate:
    assert not records, (
        f"a run that has saved nothing is restarted once, and {len(records)} restart(s) already happened, so the run "
        f"this gate would watch has been training against a checkpoint of its own since the first take-over"
    )
    return NoCheckpointGate()
