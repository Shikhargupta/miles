import hashlib
import shlex
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from examples.infra_features.hot_restart.run_qwen3_0_6b_hot_restart import ScriptArgs, build_train_args
from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID
from tests.e2e.deploy.conftest_deploy.comparison import EVENTS_DIRNAME
from tests.e2e.deploy.conftest_deploy.example_args import (
    assert_the_example_builds_the_parallelism_of,
    build_deterministic_test_args,
    build_script_args,
    with_replaced_value,
    without_weight_decay,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_process import (
    assert_the_run_was_watched_closely_enough,
    assert_the_trainer_never_rebooted,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_workloads import assert_only_the_orchestration_side_restarted
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import HotRestartDriver, driving_hot_restarts
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence, HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.gate import HotRestartGate, compute_next_gate
from tests.e2e.deploy.conftest_deploy.hot_restart.relaunch import compute_release_of_config, relaunch_with_hot_restart
from tests.e2e.ft.conftest_ft.execution import DATA_DIR, MODEL_DIR
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.common import ArgvManipulator, get_mooncake_object_store_args

CHECKPOINT_DIRNAME: str = "checkpoints"
SAVE_FLAG: str = "--save"
LOAD_FLAG: str = "--load"
WANDB_GROUP_FLAG: str = "--wandb-group"

_TRAIN_ARGS_OF_DUMP_DIR: dict[str, str] = {}


def compute_checkpoint_dir(dump_dir: str) -> str:
    return f"{dump_dir}/{CHECKPOINT_DIRNAME}"


def build_hot_restart_script_args(
    *, test_name: str, mode: FTTestMode, dump_dir: str, enable_dumper: bool, num_rollouts: int, save_interval: int
) -> ScriptArgs:
    assert mode.has_real_rollout, (
        f"{test_name} replaces the rollout executor of a live run, and mode {mode.model_name} has no engines for "
        f"it to drive once it comes back"
    )
    assert not mode.colocate, (
        f"{test_name} keeps the trainers and the engines of a run running while their script is replaced, and a "
        f"colocated mode shares gpus between them"
    )

    return build_script_args(
        command_utils.default_config(),
        script_args_class=ScriptArgs,
        model_name=mode.model_name,
        megatron_model_type=mode.megatron_model_type,
        num_rollout=num_rollouts,
        save_interval=save_interval,
        actor_num_gpus=mode.train_gpus_per_node,
        num_engines=mode.rollout_num_engines,
        gpus_per_engine=mode.rollout_gpus_per_engine,
        model_dir=MODEL_DIR,
        data_dir=DATA_DIR,
        extra_args=build_deterministic_test_args(mode=mode, dump_dir=dump_dir, enable_dumper=enable_dumper),
    )


def build_hot_restart_args(*, test_name: str, mode: FTTestMode, dump_dir: str, script_args: ScriptArgs) -> str:
    checkpoint_dir = compute_checkpoint_dir(dump_dir)
    args = without_weight_decay(build_train_args(script_args))
    for flag in (SAVE_FLAG, LOAD_FLAG):
        args = with_replaced_value(args, flag=flag, value=checkpoint_dir)
    if ArgvManipulator.declares(shlex.split(args), WANDB_GROUP_FLAG):
        args = with_replaced_value(
            args, flag=WANDB_GROUP_FLAG, value=_compute_wandb_group(test_name=test_name, dump_dir=dump_dir)
        )
    args += get_mooncake_object_store_args()

    assert_the_example_builds_the_parallelism_of(mode, train_args=args)
    _TRAIN_ARGS_OF_DUMP_DIR[dump_dir] = args
    return args


def read_installed_args(dump_dir: str) -> str:
    assert (args := _TRAIN_ARGS_OF_DUMP_DIR.get(dump_dir)) is not None, (
        f"nothing installed a run under {dump_dir} in this process, and a relaunch that builds its arguments again "
        f"instead of repeating the ones the run is up with would render a pod template of its own"
    )
    return args


@contextmanager
def driving_the_take_overs_of(
    *,
    mode: FTTestMode,
    dump_dir: str,
    config: command_utils.ExecuteTrainConfig,
    num_restarts: int,
    build_gate: Callable[[Sequence[HotRestartRecord]], HotRestartGate] = compute_next_gate,
) -> Iterator[None]:
    release = compute_release_of_config(config)
    driver = HotRestartDriver(
        relaunch=lambda: relaunch_with_hot_restart(
            train_args=read_installed_args(dump_dir), mode=mode, config=config, installed_release=release
        ),
        checkpoint_dir=Path(compute_checkpoint_dir(dump_dir)),
        events_dir=Path(dump_dir) / EVENTS_DIRNAME,
        release=release,
        namespace=config.namespace,
        trainer_id=DEFAULT_TRAINER_ID,
        num_restarts=num_restarts,
        build_gate=build_gate,
    )

    with driving_hot_restarts(driver, dump_dir=dump_dir):
        yield

    driver.assert_every_restart_happened()
    assert_the_take_overs_replaced_only_the_script(driver.evidence, num_restarts=num_restarts)


def assert_the_take_overs_replaced_only_the_script(evidence: HotRestartEvidence, *, num_restarts: int) -> None:
    assert_the_run_was_watched_closely_enough(evidence)
    assert_only_the_orchestration_side_restarted(evidence, num_restarts=num_restarts)
    assert_the_trainer_never_rebooted(evidence)


def _compute_wandb_group(*, test_name: str, dump_dir: str) -> str:
    return f"{test_name}_{hashlib.sha256(dump_dir.encode()).hexdigest()[:12]}"
