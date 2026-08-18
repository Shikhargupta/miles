from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.e2e.deploy.conftest_deploy.comparison import EVENTS_DIRNAME
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone import (
    RedoneSteps,
    assert_only_the_steps_after_a_checkpoint_were_redone,
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

    def assert_redone_steps(self, *, records: list[HotRestartRecord], num_rollouts: int = 6) -> RedoneSteps:
        return assert_only_the_steps_after_a_checkpoint_were_redone(
            dump_dir=str(self.dump_dir),
            checkpoint_dir=str(self.checkpoint_dir),
            records=records,
            num_rollouts=num_rollouts,
        )


def _records() -> list[HotRestartRecord]:
    return [
        HotRestartRecord(index=0, saved_iteration_at_trigger=1, finished_rollout_id_at_trigger=2),
        HotRestartRecord(index=1, saved_iteration_at_trigger=3, finished_rollout_id_at_trigger=3),
    ]


def _run_restarted_twice(tmp_path: Path) -> _Run:
    run = _Run(dump_dir=tmp_path)
    run.train(0)
    run.save(1)
    run.train(1, 2)
    run.take_over()
    run.train(1, 2)
    run.save(3)
    run.train(3)
    run.take_over()
    run.train(3, 4, 5)
    return run


class TestAssertOnlyTheStepsAfterACheckpointWereRedone:
    def test_two_take_overs_that_each_redid_their_own_window_pass(self, tmp_path):
        """Steps 1, 2 and 3 were trained twice, every other step once, and no step three times."""
        run = _run_restarted_twice(tmp_path)

        redone = run.assert_redone_steps(records=_records())

        assert redone.resume_rollout_ids == (0, 2)
        assert redone.finished_rollout_ids_before_take_over == (2, 3)
        assert redone.attempts_of_rollout_id == {0: 1, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1}

    def test_a_run_that_left_no_discarded_event_log_fails(self, tmp_path):
        """A take-over that never rolled the log back never resumed from a checkpoint either."""
        run = _Run(dump_dir=tmp_path)
        run.train(0, 1, 2, 3, 4, 5)

        with pytest.raises(AssertionError, match="rolls the event log back"):
            run.assert_redone_steps(records=_records())

    def test_more_discarded_logs_than_the_driver_recorded_restarts_fails(self, tmp_path):
        """A take-over nobody triggered replaced this run, and no window explains what it redid."""
        run = _run_restarted_twice(tmp_path)

        with pytest.raises(AssertionError, match="rolls the event log back"):
            run.assert_redone_steps(records=_records()[:1])

    def test_a_take_over_that_trained_from_scratch_fails(self, tmp_path):
        """Resuming at step 0 instead of at the checkpoint keeps nothing of what the run had trained."""
        run = _Run(dump_dir=tmp_path)
        run.train(0)
        run.save(1)
        run.train(1, 2)
        run.events_dir.rename(tmp_path / ".trash_20260818_000000_abcdef01")
        run.train(0, 1, 2, 3, 4, 5)

        with pytest.raises(AssertionError, match="no save wrote"):
            run.assert_redone_steps(records=_records()[:1])

    def test_a_run_that_never_finished_every_step_fails(self, tmp_path):
        """A comparison over fewer steps than the run was asked for would quietly prove less."""
        run = _run_restarted_twice(tmp_path)

        with pytest.raises(AssertionError, match="exactly once"):
            run.assert_redone_steps(records=_records(), num_rollouts=7)

    def test_a_step_trained_a_third_time_fails(self, tmp_path):
        """Two take-overs sharing a window waste one step twice over, which no checkpoint explains."""
        run = _Run(dump_dir=tmp_path)
        run.train(0)
        run.save(1)
        run.train(1, 2)
        run.take_over()
        run.train(1, 2, 3)
        run.take_over()
        run.train(1, 2, 3, 4, 5)

        with pytest.raises(AssertionError, match="once or twice"):
            run.assert_redone_steps(
                records=[
                    HotRestartRecord(index=0, saved_iteration_at_trigger=1, finished_rollout_id_at_trigger=2),
                    HotRestartRecord(index=1, saved_iteration_at_trigger=1, finished_rollout_id_at_trigger=3),
                ]
            )

    def test_a_take_over_of_a_run_further_along_than_the_driver_saw_passes(self, tmp_path):
        """A relaunch lands seconds after it is triggered, and the run keeps training in between."""
        run = _run_restarted_twice(tmp_path)

        run.assert_redone_steps(
            records=[
                HotRestartRecord(index=0, saved_iteration_at_trigger=1, finished_rollout_id_at_trigger=1),
                HotRestartRecord(index=1, saved_iteration_at_trigger=3, finished_rollout_id_at_trigger=3),
            ]
        )

    def test_a_take_over_that_replaced_a_run_behind_the_one_that_was_watched_fails(self, tmp_path):
        """The log a take-over replaced cannot have less in it than the run the driver was watching."""
        run = _run_restarted_twice(tmp_path)

        with pytest.raises(AssertionError, match="not the run the driver was watching"):
            run.assert_redone_steps(
                records=[
                    HotRestartRecord(index=0, saved_iteration_at_trigger=1, finished_rollout_id_at_trigger=2),
                    HotRestartRecord(index=1, saved_iteration_at_trigger=3, finished_rollout_id_at_trigger=4),
                ]
            )

    def test_one_log_describing_a_step_twice_fails(self, tmp_path):
        """A script that trained a step twice without any rollback is a run nothing rolled back at all."""
        run = _run_restarted_twice(tmp_path)
        run.train(5)

        with pytest.raises(AssertionError, match="more than once"):
            run.assert_redone_steps(records=_records())
