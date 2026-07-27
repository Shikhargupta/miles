import ast
import importlib.util
import inspect
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from tests.fast.launch_scripts.sh_harness import REPO_ROOT, sanitize

import miles.utils.external_utils.command_utils as command_utils
import miles.utils.misc as misc

FROZEN_RUN_ID = "260101-000000-000"

_FROZEN_ENV = {
    "MASTER_ADDR": "127.0.0.1",
    "MILES_SCRIPT_ENABLE_RAY_SUBMIT": "1",
    "PYTHONPATH": "/frozen/pythonpath",
    "WANDB_API_KEY": "frozen-wandb-api-key",
}

_CLEARED_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "GITHUB_COMMIT_NAME",
    "GLOO_SOCKET_IFNAME",
    "MILES_SCRIPT_EXTERNAL_RAY",
    "NCCL_DEBUG",
    "NCCL_DEBUG_FILE",
    "NCCL_NVLS_ENABLE",
    "NCCL_SOCKET_IFNAME",
    "NO_PROXY",
    "RAY_ADDRESS",
    "SLURM_JOB_NUM_NODES",
)


@dataclass(frozen=True)
class PyLaunchScript:
    path: Path
    entrypoints: tuple[str, ...]

    @property
    def rel(self) -> str:
        return self.path.relative_to(REPO_ROOT).as_posix()


def iter_py_launch_scripts() -> list[PyLaunchScript]:
    paths = sorted((REPO_ROOT / "scripts").rglob("run_*.py"))
    return [PyLaunchScript(path=path, entrypoints=tuple(_entrypoint_names(path))) for path in paths]


def freeze_environment(monkeypatch) -> None:
    for key, value in _FROZEN_ENV.items():
        monkeypatch.setenv(key, value)
    for key in _CLEARED_ENV:
        monkeypatch.delenv(key, raising=False)


def install_command_recorder(monkeypatch) -> list[str]:
    commands: list[str] = []
    pseudo_files: list[str] = []

    def fake_exec_command(cmd: str, capture_output: bool = False) -> str | None:
        commands.append(cmd)
        return "0" if capture_output else None

    def fake_exec_command_all_ray_node(
        cmd: str, capture_output: bool = False, num_nodes: int | None = None
    ) -> list[str | None]:
        commands.append(f"[all_ray_node num_nodes={num_nodes}] {cmd}")
        return ["0"]

    def fake_save_to_temp_file(text: str, ext: str) -> str:
        pseudo_files.append(text)
        return f"/frozen/pseudo_file_{len(pseudo_files)}.{ext}"

    for module in (command_utils, misc):
        monkeypatch.setattr(module, "exec_command", fake_exec_command)
        monkeypatch.setattr(module, "exec_command_all_ray_node", fake_exec_command_all_ray_node)
    monkeypatch.setattr(command_utils, "create_run_id", lambda: FROZEN_RUN_ID)
    monkeypatch.setattr(command_utils, "save_to_temp_file", fake_save_to_temp_file)

    return commands


def import_launch_script(path: Path) -> ModuleType:
    name = "miles_launch_script_" + path.relative_to(REPO_ROOT).with_suffix("").as_posix().replace("/", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return module


def call_entrypoint(module: ModuleType, name: str, overrides: dict[str, object]) -> None:
    entrypoint = getattr(module, name)
    first = next(iter(inspect.signature(entrypoint).parameters.values()), None)
    if first is not None and first.name == "args":
        entrypoint(module.ScriptArgs(**overrides))
    else:
        entrypoint(**overrides)


def format_commands(commands: list[str], sandbox: Path) -> str:
    lines = []
    for index, command in enumerate(commands):
        lines.append(f"### {index}")
        lines.append(re.sub(r" (?=--)", "\n  ", sanitize(command, sandbox=sandbox)))
        lines.append("")
    return "\n".join(lines)


def _entrypoint_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_") and node.name != "main"
    ]
