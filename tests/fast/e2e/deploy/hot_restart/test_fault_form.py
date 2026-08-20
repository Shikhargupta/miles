import threading
from pathlib import Path
from random import Random
from typing import Any

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import fault_form as fault_form_module
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import HOT_RESTART_ARG
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import RunProgress
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import (
    HOT_RESTART_FORM_NAME,
    HotRestartFaultForm,
    describes_a_run_that_redid_a_step,
)

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig

CELL: dict = {"metadata": {"name": "actor-0"}}


def _form(launch, **overrides: Any) -> HotRestartFaultForm:
    kwargs: dict[str, Any] = dict(
        launch=launch,
        config=ExecuteTrainConfig(run_id="demo", namespace="rl"),
        checkpoint_dir=Path("/dumps/checkpoints"),
        events_dir=Path("/dumps/events"),
        poll_interval_seconds=0.0,
        timeout_seconds=5.0,
    )
    kwargs.update(overrides)
    return HotRestartFaultForm(**kwargs)


def _install_run(monkeypatch, *, attempts: list[dict[int, int]], saved: int | None = 1) -> None:
    """Feed the form the per-rollout attempt counts it reads, one per look."""
    remaining = list(attempts)
    monkeypatch.setattr(
        fault_form_module,
        "read_attempts_of_rollout_id",
        lambda _events_dir: remaining.pop(0) if remaining else attempts[-1],
    )
    monkeypatch.setattr(
        fault_form_module,
        "read_run_progress",
        lambda **_kwargs: RunProgress(
            last_saved_iteration=saved, last_finished_rollout_id=max(attempts[0], default=None)
        ),
    )


class TestWhatCountsAsATakeOverThatLanded:
    def test_a_log_rolled_back_to_a_checkpoint_counts(self):
        """A take-over resuming from a checkpoint restores the log that sat beside it."""
        assert describes_a_run_that_redid_a_step(before={0: 1, 1: 1, 2: 1}, after={0: 1, 1: 1})

    def test_a_run_starting_over_from_scratch_counts(self):
        """With no checkpoint the log is not rolled back at all; step 0 is simply trained again."""
        assert describes_a_run_that_redid_a_step(before={0: 1, 1: 1}, after={0: 2, 1: 1})

    def test_a_run_that_merely_trained_on_does_not_count(self):
        """The old script keeps training while the relaunch installs, and that is not a take-over."""
        assert not describes_a_run_that_redid_a_step(before={0: 1, 1: 1}, after={0: 1, 1: 1, 2: 1})

    def test_a_read_that_failed_does_not_count(self):
        """A dump directory that did not answer is not evidence of anything."""
        assert not describes_a_run_that_redid_a_step(before={0: 1}, after=None)

    def test_a_run_that_had_trained_nothing_yet_does_not_count(self):
        """There is no step to redo, so nothing observable would say the take-over landed."""
        assert not describes_a_run_that_redid_a_step(before={}, after={0: 1})


class TestInject:
    def test_every_draw_relaunches_the_release_that_is_already_up(self, monkeypatch):
        """A relaunch under another release would leave the trainers this run is watching behind."""
        launched: list[ExecuteTrainConfig] = []
        _install_run(monkeypatch, attempts=[{0: 1, 1: 1}, {0: 1}])

        _form(launched.append).inject(CELL, Random(0))

        assert [one.hot_restart for one in launched] == [HOT_RESTART_ARG]
        assert [one.run_id for one in launched] == ["demo"]

    def test_a_draw_before_the_first_save_fires_like_any_other(self, monkeypatch):
        """Taking a run over before it saved costs everything it trained, which is a path worth covering."""
        launched: list[ExecuteTrainConfig] = []
        _install_run(monkeypatch, attempts=[{0: 1, 1: 1}, {0: 2, 1: 1}], saved=None)

        _form(launched.append).inject(CELL, Random(0))

        assert len(launched) == 1

    def test_the_injector_is_not_blocked_by_a_relaunch_that_drives_the_run_to_its_end(self, monkeypatch):
        """The relaunch installs a script that trains to the end, so a call that waits for it never returns."""
        redone = threading.Event()
        _install_run(monkeypatch, attempts=[{0: 1, 1: 1}, {0: 1}])

        _form(lambda _config: redone.wait(timeout=30.0)).inject(CELL, Random(0))

        redone.set()

    def test_a_relaunch_that_never_reached_the_run_is_reported_rather_than_counted(self, monkeypatch):
        """An injection counted as landed would let this soak pass on a run nothing ever replaced."""
        _install_run(monkeypatch, attempts=[{0: 1, 1: 1}])

        with pytest.raises(AssertionError, match="without the run ever redoing a step"):
            _form(lambda _config: None).inject(CELL, Random(0))

    def test_a_relaunch_the_cluster_refused_is_reported_rather_than_counted(self, monkeypatch):
        """A refused upgrade leaves the run training under the very script this injection meant to replace."""
        _install_run(monkeypatch, attempts=[{0: 1, 1: 1}])

        with pytest.raises(AssertionError, match="refused rather than installed"):
            _form(_raise_refused).inject(CELL, Random(0))

    def test_a_later_draw_is_judged_on_its_own_relaunch_and_not_on_an_earlier_one(self, monkeypatch):
        """A refused upgrade may well be transient, and a soak that gives up on the first one proves less."""
        _install_run(monkeypatch, attempts=[{0: 1, 1: 1}])
        form = _form(_raise_refused)
        with pytest.raises(AssertionError, match="refused rather than installed"):
            form.inject(CELL, Random(0))

        _install_run(monkeypatch, attempts=[{0: 1, 1: 1}, {0: 1}])
        form._launch = lambda _config: None

        form.inject(CELL, Random(0))


def _raise_refused(_config: ExecuteTrainConfig) -> None:
    raise SystemExit("the relaunch would change more than the size of this run")


class TestForm:
    def test_the_form_is_named_after_what_it_does(self):
        """The name is what the injection log carries, and a form nobody can name cannot be read back."""
        assert _form(lambda _config: None).name == HOT_RESTART_FORM_NAME

    def test_the_form_declares_that_it_leaves_the_cell_it_was_drawn_for_running(self):
        """A cell counted as crashed is dropped from the live set forever, and no later draw would fire."""
        assert not _form(lambda _config: None).harms_the_cell


class TestTheClosingContract:
    def test_a_soak_whose_take_overs_all_installed_cleanly_passes(self, monkeypatch):
        """Nothing raised and nothing is still running, so what was collected can be read."""
        _install_run(monkeypatch, attempts=[{0: 1, 1: 1}, {0: 1}])
        form = _form(lambda _config: None)
        form.inject(CELL, Random(0))

        form.join_relaunches(timeout_seconds=30.0)
        form.assert_every_take_over_installed_cleanly()

    def test_the_run_verdict_raised_by_the_last_relaunch_is_not_lost(self, monkeypatch):
        """The last relaunch's launcher is what observes the run's own metric checker."""
        _install_run(monkeypatch, attempts=[{0: 1, 1: 1}, {0: 1}])
        form = _form(_raise_the_run_verdict)
        form.inject(CELL, Random(0))

        form.join_relaunches(timeout_seconds=30.0)

        with pytest.raises(AssertionError, match="did not install cleanly"):
            form.assert_every_take_over_installed_cleanly()

    def test_a_relaunch_still_running_at_the_end_is_reported(self, monkeypatch):
        """A run still being replaced under the dumps about to be read is not a finished soak."""
        never_returns = threading.Event()
        _install_run(monkeypatch, attempts=[{0: 1, 1: 1}, {0: 1}])
        form = _form(lambda _config: never_returns.wait(timeout=30.0))
        form.inject(CELL, Random(0))

        try:
            form.join_relaunches(timeout_seconds=0.05)
            with pytest.raises(AssertionError, match="still installing a hot restart"):
                form.assert_every_take_over_installed_cleanly()
        finally:
            never_returns.set()

    def test_a_failure_from_an_earlier_take_over_is_still_reported_at_the_end(self, monkeypatch):
        """Per-draw judgement must not throw away what an earlier draw already proved broken."""
        _install_run(monkeypatch, attempts=[{0: 1, 1: 1}])
        form = _form(_raise_refused)
        with pytest.raises(AssertionError, match="refused rather than installed"):
            form.inject(CELL, Random(0))

        _install_run(monkeypatch, attempts=[{0: 1, 1: 1}, {0: 1}])
        form._launch = lambda _config: None
        form.inject(CELL, Random(0))
        form.join_relaunches(timeout_seconds=30.0)

        with pytest.raises(AssertionError, match="take-over 0"):
            form.assert_every_take_over_installed_cleanly()


def _raise_the_run_verdict(_config: ExecuteTrainConfig) -> None:
    raise SystemExit("eval/gsm8k 0.31 is below the required 0.55")
