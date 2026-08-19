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
    HotRestartIsNotDueYet,
    can_be_taken_over,
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


def _install_reported_progress(monkeypatch, reported: list[RunProgress]) -> None:
    remaining = list(reported)
    monkeypatch.setattr(
        fault_form_module, "read_run_progress", lambda **_kwargs: remaining.pop(0) if remaining else reported[-1]
    )


class TestEligibility:
    def test_a_run_holding_a_checkpoint_with_a_step_past_it_may_be_taken_over(self):
        """That step is the work the take-over throws away, which is what this soak measures."""
        assert can_be_taken_over(RunProgress(last_saved_iteration=1, last_finished_rollout_id=2))

    def test_a_run_that_has_saved_nothing_may_not_be_taken_over(self):
        """Such a run restarts from scratch, which the deterministic scenario covers on a run it can compare."""
        assert not can_be_taken_over(RunProgress(last_saved_iteration=None, last_finished_rollout_id=3))

    def test_a_run_that_has_finished_nothing_may_not_be_taken_over(self):
        """A take-over of a run that has trained nothing costs a relaunch and proves nothing."""
        assert not can_be_taken_over(RunProgress(last_saved_iteration=1, last_finished_rollout_id=None))

    def test_a_run_standing_exactly_on_its_last_save_may_not_be_taken_over(self):
        """The take-over would resume where the script it replaced stood, so no step is ever redone."""
        assert not can_be_taken_over(RunProgress(last_saved_iteration=2, last_finished_rollout_id=2))


class TestInject:
    def test_an_eligible_draw_relaunches_the_release_that_is_already_up(self, monkeypatch):
        """A relaunch under another release would leave the trainers this run is watching behind."""
        launched: list[ExecuteTrainConfig] = []
        _install_reported_progress(
            monkeypatch,
            [
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=2),
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=1),
            ],
        )

        _form(launched.append).inject(CELL, Random(0))

        assert [one.hot_restart for one in launched] == [HOT_RESTART_ARG]
        assert [one.run_id for one in launched] == ["demo"]

    def test_the_injector_is_not_blocked_by_a_relaunch_that_drives_the_run_to_its_end(self, monkeypatch):
        """The relaunch installs a script that trains to the end, so a call that waits for it never returns."""
        rolled_back = threading.Event()
        _install_reported_progress(
            monkeypatch,
            [
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=2),
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=1),
            ],
        )

        _form(lambda _config: rolled_back.wait(timeout=30.0)).inject(CELL, Random(0))

        rolled_back.set()

    def test_an_ineligible_draw_waits_instead_of_firing(self, monkeypatch):
        """The plan draws a moment, not a victim: a moment the run cannot use is retried, never spent."""
        launched: list[ExecuteTrainConfig] = []
        _install_reported_progress(monkeypatch, [RunProgress(last_saved_iteration=None, last_finished_rollout_id=1)])

        with pytest.raises(HotRestartIsNotDueYet, match="waits for a save"):
            _form(launched.append).inject(CELL, Random(0))

        assert launched == []

    def test_a_relaunch_that_never_reached_the_run_is_reported_rather_than_counted(self, monkeypatch):
        """An injection counted as landed would let this soak pass on a run nothing ever replaced."""
        _install_reported_progress(monkeypatch, [RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)])

        with pytest.raises(AssertionError, match="ever rolling back"):
            _form(lambda _config: None).inject(CELL, Random(0))

    def test_a_relaunch_the_cluster_refused_is_reported_rather_than_counted(self, monkeypatch):
        """A refused upgrade leaves the run training under the very script this injection meant to replace."""
        _install_reported_progress(monkeypatch, [RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)])

        with pytest.raises(AssertionError, match="refused rather than installed"):
            _form(_raise_refused).inject(CELL, Random(0))

    def test_a_later_draw_is_judged_on_its_own_relaunch_and_not_on_an_earlier_one(self, monkeypatch):
        """A refused upgrade may well be transient, and a soak that gives up on the first one proves less."""
        _install_reported_progress(monkeypatch, [RunProgress(last_saved_iteration=1, last_finished_rollout_id=2)])
        form = _form(_raise_refused)
        with pytest.raises(AssertionError, match="refused rather than installed"):
            form.inject(CELL, Random(0))

        _install_reported_progress(
            monkeypatch,
            [
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=2),
                RunProgress(last_saved_iteration=1, last_finished_rollout_id=1),
            ],
        )
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
