import dataclasses

from tests.e2e.ft.conftest_ft.execution import run_training
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import HOT_RESTART_SEPARATOR, HotRestartComponent

HOT_RESTART_ARG: str = HOT_RESTART_SEPARATOR.join(one.value for one in HotRestartComponent)


def compute_release_of_config(config: ExecuteTrainConfig) -> str:
    return ReleaseName(
        run_id=config.run_id,
        deploy_component=config.deploy_component,
        deploy_instance_id=config.deploy_instance_id,
    ).serialize()


def relaunch_with_hot_restart(
    *, train_args: str, mode: FTTestMode, config: ExecuteTrainConfig, installed_release: str
) -> None:
    assert (relaunched := compute_release_of_config(config)) == installed_release, (
        f"a hot restart upgrades the release that is already up, and this relaunch would install {relaunched} while "
        f"the run being watched is {installed_release}; a relaunch that builds its own config gets a run id of its "
        f"own and would leave the running trainers behind"
    )
    run_training(
        train_args=train_args,
        mode=mode,
        config=dataclasses.replace(config, hot_restart=HOT_RESTART_ARG),
    )
