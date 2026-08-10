import argparse

from sglang.srt.server_args import ServerArgs

from miles.utils.workers.argv_utils import render_cli_argv

_BASELINE_FIELDS = ("model_path", "host", "port", "disaggregation_mode")

_UNCOMPARED_FIELDS = frozenset({"random_seed"})


def server_args_to_argv(server_args_dict: dict) -> list[str]:
    return render_cli_argv(
        server_args_dict,
        wanted_obj=ServerArgs(**server_args_dict),
        make_parser=_make_cli_parser,
        from_parsed=ServerArgs.from_cli_args,
        baseline_fields=_BASELINE_FIELDS,
        uncompared_fields=_UNCOMPARED_FIELDS,
    )


def parse_server_args_argv(argv: list[str]) -> ServerArgs:
    return ServerArgs.from_cli_args(_make_cli_parser().parse_args(argv))


def _make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser
