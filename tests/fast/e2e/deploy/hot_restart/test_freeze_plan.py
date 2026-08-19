import json
import shlex

from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_freeze_plan
from tests.e2e.deploy.conftest_deploy.hot_restart.freeze_plan import (
    compute_freeze_plan_path,
    with_the_freeze_plan_of,
    write_freeze_plan,
)

from miles.utils.external_utils.command_utils.common import ArgvManipulator
from miles.utils.test_utils.ft_test_actions import (
    CI_FT_TEST_ACTIONS_PATH_FLAG,
    SLEEP_FOREVER_AT_END_ACTION,
    FTTestAction,
)


class TestTheFreezePlanFile:
    def test_the_plan_a_relaunch_writes_replaces_the_one_the_run_was_installed_with(self, tmp_path):
        """The run rereads this one path every step, so a second plan beside it would never be seen."""
        path = compute_freeze_plan_path(str(tmp_path))
        write_freeze_plan(path, frozen_rollout_id=2)
        write_freeze_plan(path, frozen_rollout_id=4)

        assert [FTTestAction(**one) for one in json.loads(path.read_text())] == [
            FTTestAction(at_rollout=4, action=SLEEP_FOREVER_AT_END_ACTION)
        ]

    def test_the_plan_of_two_dump_directories_never_lands_in_one_file(self, tmp_path):
        """The two sides of the comparison run at once, and one shared plan would freeze the baseline too."""
        assert compute_freeze_plan_path(f"{tmp_path}/target") != compute_freeze_plan_path(f"{tmp_path}/baseline")

    def test_a_partial_write_is_never_what_the_run_reads(self, tmp_path):
        """The run rereads the plan every step, so it may only ever see a whole one."""
        path = compute_freeze_plan_path(str(tmp_path))
        write_freeze_plan(path, frozen_rollout_id=2)

        assert json.loads(path.read_text()) == compute_freeze_plan(2)
        assert list(path.parent.glob("*.partial")) == []


# TODO ad hoc hack: revert after the args refactor
class TestTheArgumentsThatNameThePlan:
    def test_the_run_is_told_the_path_and_never_the_plan_itself(self, tmp_path):
        """Every worker pod's command carries these arguments, and a hot restart may not change one of them."""
        path = compute_freeze_plan_path(str(tmp_path))
        args = with_the_freeze_plan_of("--save /ckpt --num-rollout 6 ", plan_path=path)

        assert ArgvManipulator.values_of(shlex.split(args), CI_FT_TEST_ACTIONS_PATH_FLAG) == [str(path)]
        assert "--save /ckpt" in args

    def test_the_arguments_of_a_relaunch_are_the_ones_the_run_is_already_up_with(self, tmp_path):
        """A relaunch whose argv differs from the installed one is refused as more than a hot restart."""
        path = compute_freeze_plan_path(str(tmp_path))
        installed = with_the_freeze_plan_of("--save /ckpt ", plan_path=path)
        write_freeze_plan(path, frozen_rollout_id=2)
        write_freeze_plan(path, frozen_rollout_id=None)

        assert installed == with_the_freeze_plan_of("--save /ckpt ", plan_path=path)

    def test_the_path_survives_being_split_back_into_arguments(self, tmp_path):
        """The path reaches the pods as one argument, so an unquoted one would arrive as several."""
        args = with_the_freeze_plan_of("--save /ckpt ", plan_path=compute_freeze_plan_path(f"{tmp_path}/a dir"))

        assert len(ArgvManipulator.values_of(shlex.split(args), CI_FT_TEST_ACTIONS_PATH_FLAG)) == 1
