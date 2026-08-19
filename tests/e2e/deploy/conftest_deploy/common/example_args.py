import dataclasses
import shlex
from typing import TypeVar

from tests.e2e.ft.conftest_ft.execution import DETERMINISTIC_ROLLOUT_ARGS, get_debug_dump_args, get_train_env_vars_arg
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.common import ArgvManipulator

WEIGHT_DECAY_FLAG: str = "--weight-decay"

ScriptArgsT = TypeVar("ScriptArgsT", bound=ExecuteTrainConfig)


def build_script_args(
    config: ExecuteTrainConfig, *, script_args_class: type[ScriptArgsT], **overrides: object
) -> ScriptArgsT:
    return script_args_class(**dataclasses.asdict(config), **overrides)


def build_deterministic_test_args(*, mode: FTTestMode, dump_dir: str, enable_dumper: bool) -> str:
    return (
        DETERMINISTIC_ROLLOUT_ARGS
        + get_debug_dump_args(dump_dir=dump_dir, enable_dumper=enable_dumper)
        + "--debug-deterministic-collective "
        + "--sglang-disable-radix-cache "
        + get_train_env_vars_arg(mode, deterministic=True)
    )


def assert_the_example_builds_the_parallelism_of(mode: FTTestMode, *, train_args: str) -> None:
    argv = shlex.split(train_args)
    declared = shlex.split(mode.parallel_args)
    token_after_flag = dict(zip(declared, declared[1:], strict=False))

    for flag in [one for one in declared if one.startswith("--")]:
        value = token_after_flag.get(flag, "")
        declared_values = [] if value.startswith("--") or not value else [value]
        built_values = ArgvManipulator.values_of(argv, flag)

        assert ArgvManipulator.declares(argv, flag) and built_values == declared_values, (
            f"this scenario's mode declares the parallelism {mode.parallel_args!r}, while the example builds "
            f"{flag} as {built_values}: the mode describes a topology nobody launched"
        )


def without_weight_decay(train_args: str) -> str:
    return with_replaced_value(train_args, flag=WEIGHT_DECAY_FLAG, value="0")


def with_replaced_value(train_args: str, *, flag: str, value: str) -> str:
    return shlex.join(ArgvManipulator.replacing_value(shlex.split(train_args), flag, value)) + " "
