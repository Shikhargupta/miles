from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tests.e2e.deploy.conftest_deploy.comparison import EVENTS_DIRNAME
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.progress import read_last_saved_iteration
from tests.e2e.deploy.conftest_deploy.hot_restart.step_events import (
    read_checkpoint_snapshot_dirs,
    read_discarded_event_dirs,
    read_step_events,
)


@dataclass(frozen=True)
class RedoneFromScratch:
    finished_rollout_id_before_take_over: int
    attempts_of_rollout_id: dict[int, int]


def assert_a_run_that_had_saved_nothing_was_redone_from_scratch(
    *, dump_dir: str, checkpoint_dir: str, records: Sequence[HotRestartRecord], num_rollouts: int
) -> RedoneFromScratch:
    record = _read_the_one_record_of_a_run_that_had_saved_nothing(records)

    discarded_dirs = read_discarded_event_dirs(dump_dir)
    assert not discarded_dirs, (
        f"a take-over rolls the event log back by restoring the copy that sits beside the checkpoint it resumes "
        f"from, and a run that had saved nothing resumes from --ref-load, which holds no snapshot of this run, so "
        f"there is nothing to restore and nothing is moved aside; these were left behind: "
        f"{[one.name for one in discarded_dirs]}, so this run resumed from a checkpoint after all"
    )

    attempts = {
        rollout_id: len(logged) for rollout_id, logged in read_step_events(Path(dump_dir) / EVENTS_DIRNAME).items()
    }
    assert sorted(attempts) == list(range(num_rollouts)), (
        f"the run was asked for {num_rollouts} steps and its one event log describes {sorted(attempts)}; a take-over "
        f"of a run holding no checkpoint restarts it at step 0 and it trains to the end from there, so every step "
        f"has to be in there"
    )

    redone = sorted(rollout_id for rollout_id, count in attempts.items() if count > 1)
    finished = max(redone, default=-1)
    assert redone == list(range(finished + 1)), (
        f"the steps trained more than once are {redone}, and a run restarted at step 0 redoes exactly the steps it "
        f"had finished, which is a prefix of the run and never a hole in it"
    )
    assert finished >= 0, (
        f"no step of {dump_dir} was trained twice, so the take-over either landed before the run had finished "
        f"anything or resumed where the script it replaced stood, and either way it redid nothing"
    )
    assert set(attempts.values()) <= {1, 2}, (
        f"one take-over throws away the steps the run it replaced had finished and nothing else, so no step is "
        f"trained a third time; these were trained {attempts}"
    )
    assert finished >= record.finished_rollout_id_at_trigger, (
        f"restart {record.index} was triggered against a run standing at step "
        f"{record.finished_rollout_id_at_trigger}, and only the steps up to {finished} were trained twice, so the "
        f"run it took over is not the run the driver was watching"
    )

    _assert_the_run_saved_after_it_was_restarted(checkpoint_dir=checkpoint_dir, dump_dir=dump_dir)

    return RedoneFromScratch(finished_rollout_id_before_take_over=finished, attempts_of_rollout_id=attempts)


def _read_the_one_record_of_a_run_that_had_saved_nothing(records: Sequence[HotRestartRecord]) -> HotRestartRecord:
    assert len(records) == 1, (
        f"a run has no checkpoint at all only until the first save, so it is taken over once; {len(records)} "
        f"restart(s) were recorded and every one after the first resumed from something the run had written"
    )
    [record] = records
    assert record.saved_iteration_at_trigger is None, (
        f"restart {record.index} was triggered against a run that had saved iteration "
        f"{record.saved_iteration_at_trigger}, so the take-over this dump describes had a checkpoint to resume from"
    )
    return record


def _assert_the_run_saved_after_it_was_restarted(*, checkpoint_dir: str, dump_dir: str) -> None:
    assert (saved := read_last_saved_iteration(Path(checkpoint_dir))) is not None, (
        f"{checkpoint_dir} holds no checkpoint even after the run ended, so this run never saved at all and says "
        f"nothing about a run that saves once a take-over has restarted it"
    )
    snapshot_dirs = read_checkpoint_snapshot_dirs(checkpoint_dir)
    assert snapshot_dirs, (
        f"{checkpoint_dir} saved iteration {saved} without the copy of {dump_dir}/{EVENTS_DIRNAME} that every save "
        f"writes beside it, so the save this run performed after being restarted was not a whole one"
    )
