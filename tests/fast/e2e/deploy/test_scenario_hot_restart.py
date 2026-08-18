import shlex

import pytest
from tests.e2e.deploy.conftest_deploy import scenario_hot_restart_deterministic as scenario
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_common import compute_checkpoint_dir, read_installed_args

from miles.utils.external_utils.command_utils.common import ArgvManipulator


class TestBuildArgs:
    def test_the_relaunch_of_a_run_repeats_the_arguments_it_was_installed_with(self, monkeypatch):
        """Each call draws a new run id, and a hot restart whose wandb group followed it would change the pods."""
        monkeypatch.setenv("WANDB_API_KEY", "key")

        first = scenario._build_args(scenario._MODE, "/dumps/target")
        second = scenario._build_args(scenario._MODE, "/dumps/target", True)

        assert "--wandb-group" in first
        assert first == second

    def test_a_relaunch_repeats_the_string_the_run_was_installed_with(self):
        """Rebuilding the arguments would drop whatever the pipeline was asked for, such as the dumper."""
        args = scenario._build_args(scenario._MODE, "/dumps/target/plain", False)

        assert read_installed_args("/dumps/target/plain") == args
        assert "--dumper-dir" not in args

    def test_relaunching_a_run_this_process_never_installed_fails(self):
        """A relaunch of arguments nobody installed would render a pod template of its own."""
        with pytest.raises(AssertionError, match="nothing installed a run"):
            read_installed_args("/dumps/target/never-installed")

    def test_the_weight_decay_of_the_common_arguments_is_replaced_and_not_repeated(self):
        """A repeated flag leaves it to the parser which value wins, and this run needs the one it asked for."""
        argv = shlex.split(scenario._build_args(scenario._MODE, "/dumps/target"))

        assert ArgvManipulator.values_of(argv, "--weight-decay") == ["0"]

    def test_each_side_of_the_comparison_checkpoints_into_its_own_directory(self):
        """A shared checkpoint directory would let the target resume from what the baseline wrote."""
        mode = scenario._MODE

        assert compute_checkpoint_dir("/dumps/target") in scenario._build_args(mode, "/dumps/target")
        assert compute_checkpoint_dir("/dumps/baseline") not in scenario._build_args(mode, "/dumps/target")

    def test_the_run_saves_often_enough_for_two_restarts_to_each_find_a_new_checkpoint(self):
        """Both restarts need a checkpoint of their own, and neither may reach the end of the run first."""
        assert scenario.SAVE_INTERVAL * (scenario.NUM_RESTARTS * 2) < scenario.NUM_ROLLOUTS
