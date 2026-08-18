from collections.abc import Iterator
from contextlib import contextmanager

from examples.infra_features.hot_restart.run_qwen3_0_6b_hot_restart import ScriptArgs
from tests.e2e.deploy.conftest_deploy.app import create_deploy_comparison_app_and_run_ci
from tests.e2e.deploy.conftest_deploy.comparison import compare_deterministic_sides
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone_from_scratch import (
    assert_a_run_that_had_saved_nothing_was_redone_from_scratch,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence
from tests.e2e.deploy.conftest_deploy.hot_restart.gate import compute_next_no_checkpoint_gate
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_common import (
    assert_the_take_overs_replaced_only_the_script,
    build_hot_restart_args,
    build_hot_restart_script_args,
    compute_checkpoint_dir,
    driving_the_take_overs_of,
)
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE
from tests.e2e.ft.conftest_ft.modes import DENSE_MODEL_HF_REPO, DENSE_MODEL_NAME, DENSE_MODEL_TYPE, FTTestMode

from miles.utils.external_utils import command_utils

TEST_NAME: str = "hot_restart_no_checkpoint"
NUM_ROLLOUTS: int = 6
NUM_RESTARTS: int = 1
SAVE_INTERVAL: int = 4
MIN_TRAINED_ROLLOUTS: int = 4

_MODE: FTTestMode = FTTestMode(
    model_name=DENSE_MODEL_NAME,
    model_hf_repo=DENSE_MODEL_HF_REPO,
    megatron_model_type=DENSE_MODEL_TYPE,
    num_cells=2,
    train_gpus_per_node=4,
    rollout_num_engines=2,
    rollout_gpus_per_engine=1,
    parallel_args="--context-parallel-size 2",
)


def _build_script_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool) -> ScriptArgs:
    assert SAVE_INTERVAL < NUM_ROLLOUTS, (
        f"{TEST_NAME} restarts a run before it saves anything and then watches it save, and a run of "
        f"{NUM_ROLLOUTS} step(s) saving every {SAVE_INTERVAL} only ever saves the last step it trains"
    )
    assert SAVE_INTERVAL > 1, (
        f"{TEST_NAME} takes a run over after it has finished a step and before it has saved anything, and a run "
        f"saving every {SAVE_INTERVAL} step(s) saves the first step it finishes, leaving no such window"
    )

    return build_hot_restart_script_args(
        test_name=TEST_NAME,
        mode=mode,
        dump_dir=dump_dir,
        enable_dumper=enable_dumper,
        num_rollouts=NUM_ROLLOUTS,
        save_interval=SAVE_INTERVAL,
    )


def _build_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return build_hot_restart_args(
        test_name=TEST_NAME,
        mode=mode,
        dump_dir=dump_dir,
        script_args=_build_script_args(mode, dump_dir, enable_dumper),
    )


@contextmanager
def _restart_the_target_before_it_saves(
    mode: FTTestMode, dump_dir: str, config: command_utils.ExecuteTrainConfig
) -> Iterator[None]:
    with driving_the_take_overs_of(
        mode=mode,
        dump_dir=dump_dir,
        config=config,
        num_restarts=NUM_RESTARTS,
        build_gate=compute_next_no_checkpoint_gate,
    ):
        yield


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    baseline_dir: str = f"{dump_dir}/{BASELINE_SIDE}"
    target_dir: str = f"{dump_dir}/{TARGET_SIDE}"

    evidence = HotRestartEvidence.load(dump_dir=target_dir)
    assert_the_take_overs_replaced_only_the_script(evidence, num_restarts=NUM_RESTARTS)
    assert_a_run_that_had_saved_nothing_was_redone_from_scratch(
        dump_dir=target_dir,
        checkpoint_dir=compute_checkpoint_dir(target_dir),
        records=evidence.records,
        num_rollouts=NUM_ROLLOUTS,
    )

    compare_deterministic_sides(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        expected_engine_count=mode.rollout_num_engines,
        min_trained_rollouts=MIN_TRAINED_ROLLOUTS,
    )

    print("Hot restart without a checkpoint comparison test PASSED")


app, run_ci = create_deploy_comparison_app_and_run_ci(
    test_name=TEST_NAME,
    mode=_MODE,
    build_baseline_args=_build_args,
    build_target_args=_build_args,
    compare_fn=_compare,
    target_side_context=_restart_the_target_before_it_saves,
)

if __name__ == "__main__":
    app()
