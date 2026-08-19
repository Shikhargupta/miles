import shlex

from tests.e2e.deploy.conftest_deploy.hot_restart import scenario_hot_restart_realistic_gsm8k as scenario
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import HotRestartFaultForm
from tests.e2e.ft.conftest_ft import scenario_realistic_gsm8k
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import ACTOR_CELL_TYPE

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import ArgvManipulator


def _run(dump_dir: str) -> scenario_realistic_gsm8k.Gsm8kRun:
    return scenario_realistic_gsm8k.Gsm8kRun(
        base_url="http://orchestrator:18080",
        config=ExecuteTrainConfig(run_id="demo", namespace="rl"),
        dump_dir=dump_dir,
        train_args="",
        launch=lambda config: None,
    )


class TestTheRecipeIsTheOneFtConverges:
    def test_the_run_is_the_realistic_gsm8k_run_and_not_a_copy_of_it(self):
        """A second recipe would drift from the one whose reward bounds this test inherits."""
        assert scenario.run_realistic_gsm8k is scenario_realistic_gsm8k.run_realistic_gsm8k

    def test_the_bounds_the_run_is_graded_against_are_the_ones_ft_declares(self):
        """The reward improvement is asserted by the run itself, off this threshold."""
        assert scenario.DEFAULT_METRIC_THRESHOLD is scenario_realistic_gsm8k.DEFAULT_METRIC_THRESHOLD
        assert scenario.DEFAULT_NUM_ROLLOUT is scenario_realistic_gsm8k.DEFAULT_NUM_ROLLOUT

    def test_this_scenario_spells_no_training_arguments_of_its_own_beyond_its_checkpoints(self):
        """Anything else spelled here is a recipe divergence nothing would compare against ft's."""
        declared = [one for one in shlex.split(scenario.build_checkpoint_args("/dumps")) if one.startswith("--")]

        assert sorted(declared) == ["--load", "--save", "--save-interval"]


class TestTheInjectionPlan:
    def test_the_only_fault_the_plan_may_draw_is_a_hot_restart(self):
        """A pod kill mixed in would make the trainer boot uuid this test pins change for a second reason."""
        forms = scenario.create_hot_restart_forms(_run("/dumps"))

        assert list(forms) == [ACTOR_CELL_TYPE]
        assert [type(one) for one in forms[ACTOR_CELL_TYPE]] == [HotRestartFaultForm]

    def test_the_plan_relaunches_the_release_the_run_was_installed_under(self):
        """A relaunch of another release would leave the trainers of this run behind."""
        run = _run("/dumps")
        [form] = scenario.create_hot_restart_forms(run)[ACTOR_CELL_TYPE]

        assert form._launch is run.launch
        assert form._config is run.config

    def test_the_form_reads_the_progress_of_the_run_it_restarts(self):
        """Eligibility is read off this run's checkpoints and events, not off a neighbouring dump directory."""
        run = _run("/dumps/gsm8k")
        [form] = scenario.create_hot_restart_forms(run)[ACTOR_CELL_TYPE]

        assert form._checkpoint_dir == scenario.compute_checkpoint_dir(run.dump_dir)
        assert form._events_dir == run.events_dir


class TestCheckpointArgs:
    def test_the_run_saves_and_resumes_from_one_directory_of_its_own(self):
        """A take-over restores the latest checkpoint, which the run has to be both writing and reading."""
        argv = shlex.split(scenario.build_checkpoint_args("/dumps/gsm8k"))
        checkpoint_dir = str(scenario.compute_checkpoint_dir("/dumps/gsm8k"))

        assert ArgvManipulator.values_of(argv, "--save") == [checkpoint_dir]
        assert ArgvManipulator.values_of(argv, "--load") == [checkpoint_dir]

    def test_the_run_saves_often_enough_for_a_take_over_to_find_a_checkpoint(self):
        """A run saving once would leave the whole soak ineligible, and no restart would ever fire."""
        argv = shlex.split(scenario.build_checkpoint_args("/dumps/gsm8k"))

        assert ArgvManipulator.values_of(argv, "--save-interval") == [str(scenario.SAVE_INTERVAL)]
        assert scenario.SAVE_INTERVAL < scenario.DEFAULT_NUM_ROLLOUT

    def test_a_run_nothing_restarted_is_a_failure_and_not_a_pass(self):
        """Every assertion past this one is vacuous on a run whose script was never replaced."""
        assert scenario.MIN_HOT_RESTARTS >= 1
