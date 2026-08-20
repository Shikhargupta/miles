from typing import Annotated

import typer
from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID
from tests.e2e.deploy.conftest_deploy.common.utils import assert_the_cluster_can_deploy_runs
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_workloads import (
    assert_the_take_overs_replaced_only_the_script,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import ClusterObserver, observing_the_cluster
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_checkpoint_dir, compute_release_of_config
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import HotRestartFaultForm
from tests.e2e.ft.conftest_ft.app import resolve_dump_dir
from tests.e2e.ft.conftest_ft.cli_options import MetricThresholdOption, NumRolloutOption, SeedOption
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import ACTOR_CELL_TYPE, CellFaultForms
from tests.e2e.ft.conftest_ft.scenario_realistic_gsm8k import (
    DEFAULT_METRIC_THRESHOLD,
    DEFAULT_NUM_ROLLOUT,
    DEFAULT_SEED,
    Gsm8kRun,
    run_realistic_gsm8k,
)

from miles.utils.external_utils import command_utils

app: typer.Typer = typer.Typer()

TEST_NAME: str = "hot_restart_realistic_gsm8k"
SAVE_INTERVAL: int = 10
MIN_HOT_RESTARTS: int = 1
DEFAULT_HOT_RESTART_INTERVAL_SECONDS: float = 1800.0

HotRestartIntervalSecondsOption = Annotated[
    float, typer.Option(help="Mean seconds between take-overs of the orchestration script")
]


@app.command(name="run")
def run_ci(
    seed: SeedOption = DEFAULT_SEED,
    num_rollout: NumRolloutOption = DEFAULT_NUM_ROLLOUT,
    metric_threshold: MetricThresholdOption = DEFAULT_METRIC_THRESHOLD,
    hot_restart_interval_seconds: HotRestartIntervalSecondsOption = DEFAULT_HOT_RESTART_INTERVAL_SECONDS,
) -> None:
    config = command_utils.default_config()
    assert_the_cluster_can_deploy_runs(config)

    observer = ClusterObserver(
        release=compute_release_of_config(config), namespace=config.namespace, trainer_id=DEFAULT_TRAINER_ID
    )
    with observing_the_cluster(observer):
        outcome = run_realistic_gsm8k(
            config=config,
            test_name=TEST_NAME,
            seed=seed,
            num_rollout=num_rollout,
            metric_threshold=metric_threshold,
            fully_async=False,
            mean_interval_seconds_of_cell_type={ACTOR_CELL_TYPE: hot_restart_interval_seconds},
            create_forms=create_hot_restart_forms,
            extra_train_args=build_checkpoint_args(resolve_dump_dir(TEST_NAME)),
        )

    evidence = HotRestartEvidence(
        records=(),
        snapshots=tuple(observer.snapshots),
        release=observer.release,
        observation_attempts=observer.attempts,
        observation_failures=observer.failures,
    )
    assert_the_take_overs_replaced_only_the_script(
        evidence, num_restarts=outcome.injector.num_successful_injections, minimum_restarts=MIN_HOT_RESTARTS
    )

    print(f"Hot restart realistic gsm8k test PASSED (seed={seed}, rollouts={num_rollout})")


def create_hot_restart_forms(run: Gsm8kRun) -> CellFaultForms:
    form = HotRestartFaultForm(
        launch=run.launch,
        config=run.config,
        checkpoint_dir=compute_checkpoint_dir(run.dump_dir),
        events_dir=run.events_dir,
    )
    return {ACTOR_CELL_TYPE: [form]}


def build_checkpoint_args(dump_dir: str) -> str:
    checkpoint_dir = compute_checkpoint_dir(dump_dir)
    return f"--save {checkpoint_dir} --load {checkpoint_dir} --save-interval {SAVE_INTERVAL} "


if __name__ == "__main__":
    app()
