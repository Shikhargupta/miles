import argparse
import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
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


def dataclass_to_values(args_obj: object) -> dict[str, object]:
    return {field.name: getattr(args_obj, field.name) for field in dataclasses.fields(args_obj)}


def render_cli_argv(
    wanted_values: Mapping[str, object],
    *,
    wanted_obj: _ArgsT,
    make_parser: Callable[[], argparse.ArgumentParser],
    from_parsed: Callable[[argparse.Namespace], _ArgsT],
    baseline_fields: Sequence[str] = (),
    field_to_dest: Mapping[str, str] | None = None,
    uncompared_fields: frozenset[str] = frozenset(),
) -> list[str]:
    actions_by_dest = _actions_by_dest(make_parser())

    def render(field_name: str, value: object) -> list[str]:
        action = _resolve_action(actions_by_dest, field_name=field_name, field_to_dest=field_to_dest or {})
        return _render_action_argv(action, value)

    def parse(argv: list[str]) -> _ArgsT:
        return from_parsed(make_parser().parse_args(argv))

    baseline_argv = [token for name in baseline_fields for token in render(name, wanted_values[name])]
    cli_defaults = parse(baseline_argv)

    argv = list(baseline_argv)
    for name, value in wanted_values.items():
        if name in baseline_fields or value == getattr(cli_defaults, name):
            continue
        argv.extend(render(name, value))

    mismatch = _describe_mismatch(parse(argv), wanted_obj, uncompared_fields=uncompared_fields)
    assert not mismatch, f"cli argv roundtrip mismatch on {mismatch}"
    return argv


def _describe_mismatch(parsed: _ArgsT, wanted: _ArgsT, *, uncompared_fields: frozenset[str]) -> str:
    return ", ".join(
        f"{field.name}: parsed {getattr(parsed, field.name)!r} != wanted {getattr(wanted, field.name)!r}"
        for field in dataclasses.fields(wanted)
        if field.name not in uncompared_fields and getattr(parsed, field.name) != getattr(wanted, field.name)
    )


def _actions_by_dest(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    actions_by_dest: dict[str, argparse.Action] = {}
    for action in parser._actions:
        actions_by_dest.setdefault(action.dest, action)
    return actions_by_dest


def _resolve_action(
    actions_by_dest: dict[str, argparse.Action],
    *,
    field_name: str,
    field_to_dest: Mapping[str, str],
) -> argparse.Action:
    dest = field_to_dest.get(field_name, field_name)
    action = actions_by_dest.get(dest)
    if action is not None and action.option_strings:
        return action

    raise AssertionError(
        f"{field_name!r} cannot be rendered: the parser registers no option for dest {dest!r}. "
        f"Add an entry to field_to_dest, or pass the value through the native passthrough path."
    )


def _render_action_argv(action: argparse.Action, value: object) -> list[str]:
    if isinstance(action, argparse.BooleanOptionalAction):
        return [_boolean_option_string(action, value=bool(value))]

    if action.nargs == 0:
        flag = _long_option_string(action)
        assert (
            value == action.const
        ), f"{flag} cannot be rendered: the CLI only has a flag for {action.const!r}, not {value!r}"
        return [flag]

    flag = _long_option_string(action)

    if isinstance(action, argparse._AppendAction):
        argv: list[str] = []
        for item in value:
            argv.append(flag)
            argv.extend(_scalar_tokens(item))
        return argv

    if action.nargs in ("*", "+") or isinstance(action.nargs, int):
        if isinstance(value, dict):
            return [flag, *(f"{key}={item}" for key, item in value.items())]
        return [flag, *(str(item) for item in value)]

    if isinstance(value, dict | list | tuple):
        return [flag, json.dumps(value)]

    return [flag, str(value)]


def _scalar_tokens(item: object) -> list[str]:
    if isinstance(item, list | tuple):
        return [str(element) for element in item]
    return [str(item)]


def _long_option_string(action: argparse.Action) -> str:
    long_options = [option for option in action.option_strings if option.startswith("--")]
    return long_options[0] if long_options else action.option_strings[0]


def _boolean_option_string(action: argparse.Action, *, value: bool) -> str:
    negative = [option for option in action.option_strings if option.startswith("--no-")]
    positive = [option for option in action.option_strings if not option.startswith("--no-")]
    if value:
        assert positive, f"{action.dest!r} cannot be rendered: no positive option string"
        return positive[0]
    assert negative, f"{action.dest!r} cannot be rendered: no negative option string"
    return negative[0]
