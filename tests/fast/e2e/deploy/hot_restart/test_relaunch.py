import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.relaunch import HOT_RESTART_ARG, relaunch_with_hot_restart
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.workers.types import HotRestartComponent


class TestHotRestartArg:
    def test_the_flag_names_both_components_a_take_over_replaces(self):
        """A hot restart replaces the orchestration script and the rollout executor together or not at all."""
        assert sorted(HOT_RESTART_ARG.split(",")) == sorted(one.value for one in HotRestartComponent)


class TestRelaunchWithHotRestart:
    def test_a_relaunch_that_would_install_another_release_is_refused(self):
        """A relaunch building its own config gets a run id of its own and leaves the run behind."""
        mode = FTTestMode(
            model_name="demo", model_hf_repo="demo/demo", megatron_model_type="demo", num_cells=1, parallel_args=""
        )

        with pytest.raises(AssertionError, match="already up"):
            relaunch_with_hot_restart(
                train_args="",
                mode=mode,
                config=ExecuteTrainConfig(run_id="demo"),
                installed_release="miles-run-someone-else-all",
            )
