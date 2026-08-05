import argparse
import dataclasses
import json
from collections.abc import Callable
from typing import TypeVar

from miles.utils.pydantic_utils import FrozenStrictBaseModel

CONFIG_JSON_FLAG = "--config-json"

_ConfigT = TypeVar("_ConfigT", bound=FrozenStrictBaseModel)
_ArgsT = TypeVar("_ArgsT")


def config_to_argv(config: FrozenStrictBaseModel) -> list[str]:
    argv = [CONFIG_JSON_FLAG, config.model_dump_json()]

    parsed = parse_config_argv(type(config), argv)
    assert parsed == config, f"config argv roundtrip mismatch: {parsed!r} != {config!r}"
    return argv


def parse_config_argv(config_cls: type[_ConfigT], argv: list[str] | None) -> _ConfigT:
    parser = argparse.ArgumentParser()
    parser.add_argument(CONFIG_JSON_FLAG, required=True)
    args = parser.parse_args(argv)
    return config_cls.model_validate_json(args.config_json)


def render_cli_argv(
    args_obj: _ArgsT,
    *,
    make_parser: Callable[[], argparse.ArgumentParser],
    from_parsed: Callable[[argparse.Namespace], _ArgsT],
    required_argv: list[str] | None = None,
) -> list[str]:
    def parse(argv: list[str]) -> _ArgsT:
        return from_parsed(make_parser().parse_args(argv))

    base_argv = list(required_argv or [])
    cli_defaults = parse(base_argv)
    simple_argv = base_argv + _render_cli_argv(args_obj, cli_defaults=cli_defaults, skip_structured=True)
    argv = base_argv + _render_cli_argv(args_obj, cli_defaults=cli_defaults, structured_defaults=parse(simple_argv))

    parsed = parse(argv)
    assert parsed == args_obj, f"cli argv roundtrip mismatch: {parsed!r} != {args_obj!r}"
    return argv


def _render_cli_argv(
    args_obj: _ArgsT,
    *,
    cli_defaults: _ArgsT,
    structured_defaults: _ArgsT | None = None,
    skip_structured: bool = False,
) -> list[str]:
    argv: list[str] = []
    for field in dataclasses.fields(args_obj):
        value = getattr(args_obj, field.name)
        if value is None:
            continue
        if value == getattr(cli_defaults, field.name):
            continue

        flag = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            assert value, f"{flag} cannot be rendered: the CLI only has a flag for the non-default value"
            argv.append(flag)
        elif isinstance(value, list):
            assert all(
                item is not None for item in value
            ), f"{flag} cannot be rendered: a None element of {field.name} cannot round-trip through the CLI"
            argv.append(flag)
            argv.extend(str(item) for item in value)
        elif isinstance(value, dict):
            assert all(
                item is not None for item in value.values()
            ), f"{flag} cannot be rendered: a None value in {field.name} cannot round-trip through the CLI"
            argv.append(flag)
            argv.extend(f"{key}={item}" for key, item in value.items())
        elif dataclasses.is_dataclass(value):
            if skip_structured:
                continue
            if structured_defaults is not None and value == getattr(structured_defaults, field.name):
                continue
            argv.extend([flag, json.dumps(dataclasses.asdict(value))])
        else:
            argv.extend([flag, str(value)])
    return argv
