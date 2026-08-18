import shlex

from miles.utils.external_utils.command_utils.common import ArgvManipulator


def value_of(train_args: str, flag: str) -> str:
    values = ArgvManipulator.values_of(shlex.split(train_args), flag)
    assert len(values) == 1, f"{flag} is declared {len(values)} time(s) in these arguments"
    return values[0]


def values_after(train_args: str, flag: str) -> list[str]:
    tokens = shlex.split(train_args)
    kept: list[str] = []
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        kept.append(token)
    return kept


def shared_argv(train_args: str, *, differing_flags: tuple[str, ...]) -> list[str]:
    kept: list[str] = []
    skipping = False
    for token in shlex.split(train_args):
        if token in differing_flags:
            skipping = True
        elif token.startswith("--"):
            skipping = False
            kept.append(token)
        elif not skipping:
            kept.append(token)
    return kept
