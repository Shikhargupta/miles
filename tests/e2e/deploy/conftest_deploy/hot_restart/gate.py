from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.progress import RunProgress


class GateStage(Enum):
    AWAITING_SAVE = auto()
    AWAITING_STEP = auto()
    OPEN = auto()


@dataclass
class RestartGate:
    minimum_saved_iteration: int | None = None
    stage: GateStage = GateStage.AWAITING_SAVE
    saved_iteration: int | None = None
    rollout_id_at_save: int | None = None

    def observe(self, progress: RunProgress) -> bool:
        if self.stage is GateStage.AWAITING_SAVE:
            self._observe_save(progress)
        if self.stage is GateStage.AWAITING_STEP:
            self._observe_step(progress)
        return self.stage is GateStage.OPEN

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


def compute_record_of_open_gate(gate: RestartGate, *, index: int, progress: RunProgress) -> HotRestartRecord:
    assert gate.stage is GateStage.OPEN, f"restart {index} was recorded while its gate was still {gate.stage.name}"
    assert gate.saved_iteration is not None and progress.last_finished_rollout_id is not None
    assert progress.last_finished_rollout_id > gate.saved_iteration, (
        f"restart {index} would redo nothing: the run saved iteration {gate.saved_iteration} and has finished "
        f"step {progress.last_finished_rollout_id}, so a take-over resumes exactly where this script stands"
    )
    return HotRestartRecord(
        index=index,
        saved_iteration_at_trigger=gate.saved_iteration,
        finished_rollout_id_at_trigger=progress.last_finished_rollout_id,
    )


def compute_next_gate(records: Sequence[HotRestartRecord]) -> RestartGate:
    if not records:
        return RestartGate()
    return RestartGate(minimum_saved_iteration=records[-1].finished_rollout_id_at_trigger)
