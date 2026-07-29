import os
import sys

HEAVY_MODULES = ("torch", "uvicorn", "fastapi", "miles.backends", "miles.utils.misc", "miles.utils.workers.rpc.server")


class SmokeWorker:
    def __init__(self, argv: list[str]):
        self._argv = argv

    def demo_sync(self, a: int, b: int) -> int:
        return a + b

    def report_argv(self) -> list[str]:
        return self._argv

    def report_env(self, name: str) -> str | None:
        return os.environ.get(name)


def make_worker(argv: list[str]) -> SmokeWorker:
    return SmokeWorker(argv)


def compute_env_vars(argv: list[str]) -> dict[str, str]:
    premature = [name for name in HEAVY_MODULES if name in sys.modules]
    return {
        "MILES_SERVE_SMOKE_ENV": ",".join(argv),
        "MILES_SERVE_SMOKE_PREMATURE_IMPORTS": ",".join(premature),
    }
