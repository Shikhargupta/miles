from collections.abc import Callable

import typer

from tests.e2e.deploy.conftest_deploy.cluster_gate import is_cluster_ready_for_helm_runs
from tests.e2e.deploy.conftest_deploy.split_deployment import (
    BuildDeploymentsFn,
    BuildSideArgsFn,
    create_split_run_side,
)
from tests.e2e.ft.conftest_ft.app import BuildArgsFn, TargetSideContextFn, create_comparison_app_and_run_ci
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils import command_utils


def create_split_comparison_app_and_run_ci(
    *,
    test_name: str,
    mode: FTTestMode,
    build_baseline_args: BuildSideArgsFn,
    build_target_args: BuildArgsFn,
    build_deployments: BuildDeploymentsFn,
    compare_fn: Callable[[str, FTTestMode], None],
) -> tuple[typer.Typer, Callable[[], None]]:
    app, run_ci = create_comparison_app_and_run_ci(
        test_name=test_name,
        build_baseline_args=_baseline_args_are_built_once_the_side_has_its_config,
        build_target_args=build_target_args,
        compare_fn=compare_fn,
        run_side=create_split_run_side(build_baseline_args=build_baseline_args, build_deployments=build_deployments),
        resolve_mode_fn=lambda _name: mode,
    )
    return app, _gate_on_the_cluster(run_ci)


def create_deploy_comparison_app_and_run_ci(
    *,
    test_name: str,
    mode: FTTestMode,
    build_baseline_args: BuildArgsFn,
    build_target_args: BuildArgsFn,
    compare_fn: Callable[[str, FTTestMode], None],
    target_side_context: TargetSideContextFn,
) -> tuple[typer.Typer, Callable[[], None]]:
    app, run_ci = create_comparison_app_and_run_ci(
        test_name=test_name,
        build_baseline_args=build_baseline_args,
        build_target_args=build_target_args,
        compare_fn=compare_fn,
        target_side_context=target_side_context,
        resolve_mode_fn=lambda _name: mode,
    )
    return app, _gate_on_the_cluster(run_ci)


def _gate_on_the_cluster(run_ci: Callable[[str | None], None]) -> Callable[[], None]:
    def run_ci_where_a_run_can_be_deployed() -> None:
        if not is_cluster_ready_for_helm_runs(command_utils.default_config()):
            return
        run_ci(None)

    return run_ci_where_a_run_can_be_deployed


def create_split_single_run_app_and_run_ci(
    *,
    config_class: type[command_utils.ExecuteTrainConfig],
    run_fn: Callable[[command_utils.ExecuteTrainConfig], None],
    verify_fn: Callable[[command_utils.ExecuteTrainConfig], None],
) -> tuple[typer.Typer, Callable[[], None]]:
    def run_ci_where_a_run_can_be_deployed() -> None:
        config = command_utils.default_config(config_class)
        if not is_cluster_ready_for_helm_runs(config):
            return
        run_fn(config)

    def verify_what_a_previous_run_left() -> None:
        verify_fn(command_utils.default_config(config_class))

    app: typer.Typer = typer.Typer()
    app.command(name="run")(run_ci_where_a_run_can_be_deployed)
    app.command(name="verify")(verify_what_a_previous_run_left)

    return app, run_ci_where_a_run_can_be_deployed


def _baseline_args_are_built_once_the_side_has_its_config(mode: FTTestMode, dump_dir: str, enable_dumper: bool) -> str:
    return ""
