import shlex

import pytest
from tests.e2e.deploy.conftest_deploy import scenario_hot_restart_no_checkpoint as scenario
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_common import compute_checkpoint_dir, read_installed_args

from miles.utils.external_utils.command_utils.common import ArgvManipulator
from miles.utils.misc import should_run_periodic_action


def _first_saved_rollout_id() -> int:
    return next(
        rollout_id
        for rollout_id in range(scenario.NUM_ROLLOUTS)
        if should_run_periodic_action(
            rollout_id, scenario.SAVE_INTERVAL, num_rollout_per_epoch=None, num_rollout=scenario.NUM_ROLLOUTS
        )
    )


class TestTiming:
    def test_the_run_trains_more_than_one_step_before_it_first_saves(self):
        """The restart has to fire after a finished step and before the first save, which needs room."""
        assert _first_saved_rollout_id() >= 1

    def test_the_run_still_saves_after_the_restart_it_is_given(self):
        """A run whose only save is its last step would never exercise saving after a take-over."""
        assert _first_saved_rollout_id() < scenario.NUM_ROLLOUTS - 1

    def test_the_run_is_taken_over_exactly_once(self):
        """The second take-over resumes from a checkpoint, which the deterministic scenario already covers."""
        assert scenario.NUM_RESTARTS == 1

    def test_the_gradient_floor_sits_above_the_window_the_take_over_can_redo(self):
        """A floor the redone steps alone could fill would pass a run that trained nothing past them."""
        assert _first_saved_rollout_id() < scenario.MIN_TRAINED_ROLLOUTS <= scenario.NUM_ROLLOUTS


class TestBuildArgs:
    def test_the_relaunch_of_a_run_repeats_the_arguments_it_was_installed_with(self, monkeypatch):
        """Each call draws a new run id, and a hot restart whose wandb group followed it would change the pods."""
        monkeypatch.setenv("WANDB_API_KEY", "key")

        first = scenario._build_args(scenario._MODE, "/dumps/no-checkpoint/target")
        second = scenario._build_args(scenario._MODE, "/dumps/no-checkpoint/target", True)

        assert "--wandb-group" in first
        assert first == second
        assert read_installed_args("/dumps/no-checkpoint/target") == first

    def test_a_relaunch_repeats_the_string_the_run_was_installed_with(self):
        """Rebuilding the arguments would drop whatever the pipeline was asked for, such as the dumper."""
        args = scenario._build_args(scenario._MODE, "/dumps/no-checkpoint/plain", False)

        assert read_installed_args("/dumps/no-checkpoint/plain") == args
        assert "--dumper-dir" not in args

    def test_relaunching_a_run_this_process_never_installed_fails(self):
        """A relaunch of arguments nobody installed would render a pod template of its own."""
        with pytest.raises(AssertionError, match="nothing installed a run"):
            read_installed_args("/dumps/no-checkpoint/never-installed")

    def test_the_run_is_installed_with_the_save_interval_the_timing_is_reasoned_from(self):
        """The gate reads the checkpoint directory, so a run saving at another pace opens it at another step."""
        argv = shlex.split(scenario._build_args(scenario._MODE, "/dumps/no-checkpoint/interval"))

        assert ArgvManipulator.values_of(argv, "--save-interval") == [str(scenario.SAVE_INTERVAL)]

    def test_each_side_of_the_comparison_checkpoints_into_its_own_directory(self):
        """A shared checkpoint directory would hand the target a checkpoint the baseline wrote."""
        mode = scenario._MODE

        assert compute_checkpoint_dir("/dumps/nc/target") in scenario._build_args(mode, "/dumps/nc/target")
        assert compute_checkpoint_dir("/dumps/nc/baseline") not in scenario._build_args(mode, "/dumps/nc/target")

    def test_the_weight_decay_of_the_common_arguments_is_replaced_and_not_repeated(self):
        """A repeated flag leaves it to the parser which value wins, and this run needs the one it asked for."""
        argv = shlex.split(scenario._build_args(scenario._MODE, "/dumps/no-checkpoint/decay"))

        assert ArgvManipulator.values_of(argv, "--weight-decay") == ["0"]

    def test_a_run_that_would_only_ever_save_its_last_step_is_refused(self, monkeypatch):
        """The scenario watches the save that follows the take-over, and such a run performs none."""
        monkeypatch.setattr(scenario, "SAVE_INTERVAL", scenario.NUM_ROLLOUTS)

        with pytest.raises(AssertionError, match="only ever saves the last step"):
            scenario._build_script_args(scenario._MODE, "/dumps/no-checkpoint/rare", False)

    def test_a_run_that_saves_the_first_step_it_finishes_is_refused(self, monkeypatch):
        """The take-over has to land after a finished step and before the first save, and such a run has no window."""
        monkeypatch.setattr(scenario, "SAVE_INTERVAL", 1)

        with pytest.raises(AssertionError, match="leaving no such window"):
            scenario._build_script_args(scenario._MODE, "/dumps/no-checkpoint/eager", False)
