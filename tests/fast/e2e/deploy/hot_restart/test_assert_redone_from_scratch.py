from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.e2e.deploy.conftest_deploy.comparison import EVENTS_DIRNAME
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone_from_scratch import (
    RedoneFromScratch,
    assert_a_run_that_had_saved_nothing_was_redone_from_scratch,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_common import compute_checkpoint_dir

from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import MetricEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

TRACKER_FILENAME: str = "latest_checkpointed_iteration.txt"


def _write_finished_step(events_dir: Path, *, rollout_id: int) -> None:
    logger = EventLogger(log_dir=events_dir, file_name="main.jsonl", source=SimpleProcessIdentity(component="main"))
    logger.log(MetricEvent, {"rollout_id": rollout_id, "metrics": {"train/grad_norm": 1.0}}, print_log=False)


@dataclass(frozen=True)
class _Run:
    dump_dir: Path

    @property
    def events_dir(self) -> Path:
        return self.dump_dir / EVENTS_DIRNAME

    @property
    def checkpoint_dir(self) -> Path:
        return Path(compute_checkpoint_dir(str(self.dump_dir)))

    @property
    def megatron_args(self) -> Namespace:
        return Namespace(
            save=str(self.checkpoint_dir),
            load=str(self.checkpoint_dir),
            save_debug_event_data=str(self.events_dir),
        )

    def train(self, *rollout_ids: int) -> None:
        for rollout_id in rollout_ids:
            _write_finished_step(self.events_dir, rollout_id=rollout_id)

    def save(self, iteration: int) -> None:
        event_logger_checkpoint.snapshot(self.megatron_args, iteration)
        (self.checkpoint_dir / TRACKER_FILENAME).write_text(str(iteration))

    def take_over(self) -> None:
        event_logger_checkpoint.restore(self.megatron_args)

    def assert_redone_from_scratch(
        self, *, records: list[HotRestartRecord] | None = None, num_rollouts: int = 6
    ) -> RedoneFromScratch:
        return assert_a_run_that_had_saved_nothing_was_redone_from_scratch(
            dump_dir=str(self.dump_dir),
            checkpoint_dir=str(self.checkpoint_dir),
            records=_records() if records is None else records,
            num_rollouts=num_rollouts,
        )


def _records() -> list[HotRestartRecord]:
    return [HotRestartRecord(index=0, saved_iteration_at_trigger=None, finished_rollout_id_at_trigger=1)]


def _run_restarted_before_it_saved(tmp_path: Path) -> _Run:
    run = _Run(dump_dir=tmp_path)
    run.train(0, 1, 2)
    run.take_over()
    run.train(0, 1, 2)
    run.save(3)
    run.train(3, 4, 5)
    return run


class TestAssertARunThatHadSavedNothingWasRedoneFromScratch:
    def test_a_take_over_that_threw_away_every_step_the_run_had_finished_passes(self, tmp_path):
        """Steps 0, 1 and 2 were trained twice and the rest once, which is a run restarted at step 0."""
        run = _run_restarted_before_it_saved(tmp_path)

        redone = run.assert_redone_from_scratch()

        assert redone.finished_rollout_id_before_take_over == 2
        assert redone.attempts_of_rollout_id == {0: 2, 1: 2, 2: 2, 3: 1, 4: 1, 5: 1}

    def test_a_take_over_that_found_a_checkpoint_after_all_fails(self, tmp_path):
        """A save between the trigger and the take-over turns this into the scenario the other test covers."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1)
        run.save(2)
        run.train(2)
        run.take_over()
        run.train(2, 3, 4, 5)

        with pytest.raises(AssertionError, match="resumed from a checkpoint after all"):
            run.assert_redone_from_scratch()

    def test_a_record_that_had_a_checkpoint_at_trigger_time_fails(self, tmp_path):
        """The record is the only thing saying which of the two take-over paths this dump describes."""
        run = _run_restarted_before_it_saved(tmp_path)

        with pytest.raises(AssertionError, match="had saved iteration"):
            run.assert_redone_from_scratch(
                records=[HotRestartRecord(index=0, saved_iteration_at_trigger=1, finished_rollout_id_at_trigger=1)]
            )

    def test_more_than_one_restart_fails(self, tmp_path):
        """Every take-over after the first resumes from what the one before it made the run save."""
        run = _run_restarted_before_it_saved(tmp_path)

        with pytest.raises(AssertionError, match="taken over once"):
            run.assert_redone_from_scratch(records=_records() * 2)

    def test_a_run_that_redid_nothing_fails(self, tmp_path):
        """A take-over landing before the run finished a step wastes nothing and proves nothing."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1, 2, 3, 4, 5)
        run.save(5)

        with pytest.raises(AssertionError, match="redid nothing"):
            run.assert_redone_from_scratch()

    def test_a_run_that_redid_a_window_instead_of_its_whole_history_fails(self, tmp_path):
        """Redoing steps 1 and 2 but not 0 is a resume from a checkpoint, however it came about."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1, 2)
        run.train(1, 2)
        run.save(3)
        run.train(3, 4, 5)

        with pytest.raises(AssertionError, match="never a hole in it"):
            run.assert_redone_from_scratch()

    def test_a_step_trained_a_third_time_fails(self, tmp_path):
        """One take-over throws away one run's worth of work, and a third attempt is a second take-over."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1, 2)
        run.train(0, 1, 2)
        run.train(0, 1, 2, 3, 4, 5)
        run.save(5)

        with pytest.raises(AssertionError, match="a third time"):
            run.assert_redone_from_scratch()

    def test_a_run_that_never_finished_every_step_fails(self, tmp_path):
        """A comparison over fewer steps than the run was asked for would quietly prove less."""
        run = _run_restarted_before_it_saved(tmp_path)

        with pytest.raises(AssertionError, match="every step"):
            run.assert_redone_from_scratch(num_rollouts=7)

    def test_a_take_over_of_a_run_further_along_than_the_driver_saw_passes(self, tmp_path):
        """A relaunch lands seconds after it is triggered, and the run keeps training in between."""
        run = _run_restarted_before_it_saved(tmp_path)

        run.assert_redone_from_scratch(
            records=[HotRestartRecord(index=0, saved_iteration_at_trigger=None, finished_rollout_id_at_trigger=0)]
        )

    def test_a_take_over_that_replaced_a_run_behind_the_one_that_was_watched_fails(self, tmp_path):
        """The work a take-over threw away cannot be less than what the run the driver watched had done."""
        run = _run_restarted_before_it_saved(tmp_path)

        with pytest.raises(AssertionError, match="not the run the driver was watching"):
            run.assert_redone_from_scratch(
                records=[HotRestartRecord(index=0, saved_iteration_at_trigger=None, finished_rollout_id_at_trigger=3)]
            )

    def test_a_run_that_never_saved_after_it_was_restarted_fails(self, tmp_path):
        """The point of restarting before the first save is to watch the save that follows it."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1, 2)
        run.train(0, 1, 2, 3, 4, 5)

        with pytest.raises(AssertionError, match="never saved at all"):
            run.assert_redone_from_scratch()
