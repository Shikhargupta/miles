from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.comparison import EVENTS_DIRNAME
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.step_events import (
    read_checkpoint_snapshot_dirs,
    read_discarded_event_dirs,
    read_step_events,
)


@dataclass(frozen=True)
class RedoneSteps:
    resume_rollout_ids: tuple[int, ...]
    finished_rollout_ids_before_take_over: tuple[int, ...]
    attempts_of_rollout_id: dict[int, int]


def assert_only_the_steps_after_a_checkpoint_were_redone(
    *, dump_dir: str, checkpoint_dir: str, records: Sequence[HotRestartRecord], num_rollouts: int
) -> RedoneSteps:
    assert records, (
        f"{dump_dir} holds a run nothing restarted, and a run that redid no step proves nothing about which steps "
        f"a take-over redoes"
    )

    discarded_dirs = read_discarded_event_dirs(dump_dir)
    assert len(discarded_dirs) == len(records), (
        f"every take-over rolls the event log back to the checkpoint it resumes from, leaving the log it replaced "
        f"behind, and {len(records)} restart(s) left {[one.name for one in discarded_dirs]}"
    )
    discarded_logs = [_read_finished_steps(one) for one in discarded_dirs]
    for discarded_dir, log in zip(discarded_dirs, discarded_logs, strict=True):
        assert log, (
            f"{discarded_dir.name} describes no finished step at all, so the take-over that left it behind rolled "
            f"back the log of a run that had not trained anything yet"
        )

    discarded_logs.sort(key=max)
    surviving_log = _read_finished_steps(Path(dump_dir) / EVENTS_DIRNAME)
    logs = [*discarded_logs, surviving_log]

    finished_rollout_ids = [max(one) for one in discarded_logs]
    assert len(set(finished_rollout_ids)) == len(finished_rollout_ids), (
        f"two take-overs replaced logs that had reached the same step {finished_rollout_ids}, so the restarts did "
        f"not wait for a checkpoint past the window the previous one wasted"
    )

    resume_rollout_ids: list[int] = []
    for index, (log, later_log) in enumerate(zip(discarded_logs, logs[1:], strict=True)):
        finished = max(log)
        assert sorted(log) == list(range(finished + 1)), (
            f"the log replaced by restart {index} describes the steps {sorted(log)}, and a run that trains step by "
            f"step from zero leaves every step up to the one it had reached in it"
        )

        survived = sorted(rollout_id for rollout_id, event in log.items() if later_log.get(rollout_id) == event)
        resume = max(survived, default=-1)
        assert survived == list(range(resume + 1)), (
            f"restart {index} carried the steps {survived} over into the log that followed it, and a take-over "
            f"resumes from a checkpoint, so what it keeps is a prefix of the run and never a hole in it"
        )
        assert resume < finished, (
            f"restart {index} kept every step up to {resume} and the log it replaced had reached {finished}, so "
            f"this take-over redid nothing and says nothing about what a take-over costs"
        )
        resume_rollout_ids.append(resume)

    assert sorted(surviving_log) == list(range(num_rollouts)), (
        f"the surviving event log has to describe each of the {num_rollouts} steps exactly once: a take-over that "
        f"trained from scratch instead of from its checkpoint would describe the early steps twice, and one that "
        f"resumed past its checkpoint would skip some; it describes {sorted(surviving_log)}"
    )

    _assert_every_resume_point_is_a_checkpoint(checkpoint_dir=checkpoint_dir, resume_rollout_ids=resume_rollout_ids)

    for record, finished in zip(records, finished_rollout_ids, strict=True):
        assert finished >= record.finished_rollout_id_at_trigger, (
            f"restart {record.index} was triggered against a run standing at step "
            f"{record.finished_rollout_id_at_trigger}, and the log it replaced had only reached {finished}, so the "
            f"run it took over is not the run the driver was watching"
        )

    attempts = {
        rollout_id: len({log[rollout_id] for log in logs if rollout_id in log}) for rollout_id in range(num_rollouts)
    }
    expected = {
        rollout_id: 1
        + sum(
            1
            for resume, finished in zip(resume_rollout_ids, finished_rollout_ids, strict=True)
            if resume < rollout_id <= finished
        )
        for rollout_id in range(num_rollouts)
    }
    assert attempts == expected, (
        f"the steps a take-over redoes are the ones between the checkpoint it resumed from and the step the script "
        f"it replaced had reached, which is {expected}, and counting how many times each step was actually written "
        f"across every log this run left says {attempts}"
    )
    assert sorted(set(attempts.values())) == [1, 2], (
        f"a hot restart wastes the steps its checkpoint does not cover and nothing else, so every step is trained "
        f"once or twice; these were trained {sorted(set(attempts.values()))} time(s): {attempts}"
    )

    return RedoneSteps(
        resume_rollout_ids=tuple(resume_rollout_ids),
        finished_rollout_ids_before_take_over=tuple(finished_rollout_ids),
        attempts_of_rollout_id=attempts,
    )


def _assert_every_resume_point_is_a_checkpoint(*, checkpoint_dir: str, resume_rollout_ids: Sequence[int]) -> None:
    snapshot_dirs = read_checkpoint_snapshot_dirs(checkpoint_dir)
    assert snapshot_dirs, (
        f"{checkpoint_dir} holds no event log snapshot beside any checkpoint, so nothing there could have been "
        f"restored and whatever the run resumed from was not a checkpoint of this run"
    )

    steps_of_snapshot = {one.parent.name: sorted(_read_finished_steps(one)) for one in snapshot_dirs}
    for index, resume in enumerate(resume_rollout_ids):
        matching = sorted(name for name, steps in steps_of_snapshot.items() if steps == list(range(resume + 1)))
        assert matching, (
            f"restart {index} resumed a run that had finished the steps {list(range(resume + 1))}, and no "
            f"checkpoint of this run holds that log: the snapshots beside the checkpoints hold "
            f"{steps_of_snapshot}, so the take-over resumed from something no save wrote"
        )


def _read_finished_steps(events_dir: Path) -> dict[int, str]:
    events_of_rollout_id = read_step_events(events_dir)

    repeated = {rollout_id: len(logged) for rollout_id, logged in events_of_rollout_id.items() if len(logged) > 1}
    assert not repeated, (
        f"{events_dir} describes the steps {repeated} more than once each, and a take-over rolls the log back to "
        f"its checkpoint before redoing anything, so one log describes each step it covers exactly once"
    )
    return {rollout_id: logged[0] for rollout_id, logged in events_of_rollout_id.items()}
