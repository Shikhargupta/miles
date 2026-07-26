import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT_PLACEHOLDER = "<REPO_ROOT>"
SANDBOX_PLACEHOLDER = "<SANDBOX>"

_ARG_SEPARATOR = "\x1f"
_RECORD_SEPARATOR = "\x1e"

_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

_FROZEN_ENV = {
    "HOME": "/root",
    "LANG": "C",
    "LC_ALL": "C",
    "TERM": "dumb",
    "MASTER_ADDR": "127.0.0.1",
    "NODE_RANK": "0",
    "WANDB_KEY": "frozen-wandb-key",
    "WANDB_API_KEY": "frozen-wandb-api-key",
}

_SHIMMED_COMMANDS = (
    "apt",
    "apt-get",
    "curl",
    "date",
    "docker",
    "git",
    "hf",
    "huggingface-cli",
    "ip",
    "mkdir",
    "nc",
    "nvidia-smi",
    "pip",
    "pip3",
    "pkill",
    "python",
    "python3",
    "ray",
    "rm",
    "rsync",
    "sleep",
    "torchrun",
    "wget",
)

_SHIM_STDOUT = {
    # large enough that "wait until this many GPUs joined the ray cluster" loops exit immediately
    "python": "1000000",
    "python3": "1000000",
    "date": "20260101_000000",
}

_SHIM_TEMPLATE = """#!/bin/bash
record='{name}'
for arg in "$@"; do
    record="$record$MILES_SH_HARNESS_ARG_SEP$arg"
done
printf '%s%s' "$record" "$MILES_SH_HARNESS_RECORD_SEP" >>"$MILES_SH_HARNESS_CAPTURE"
{stdout_statement}exit 0
"""


@dataclass(frozen=True)
class LaunchScriptRun:
    invocations: list[list[str]]
    stdout: str
    stderr: str
    returncode: int

    def invocations_of(self, command: str) -> list[list[str]]:
        return [argv for argv in self.invocations if argv[0] == command]

    def ray_job_submit_argv(self) -> list[str]:
        matches = [argv for argv in self.invocations_of("ray") if argv[1:3] == ["job", "submit"]]
        assert len(matches) == 1, f"expected exactly one `ray job submit`, got {len(matches)}"
        return matches[0]


def run_launch_script(
    script: Path,
    sandbox: Path,
    extra_env: dict[str, str] | None = None,
    timeout: float = 120.0,
) -> LaunchScriptRun:
    sandbox.mkdir(parents=True, exist_ok=True)
    fake_bin = sandbox / "fake_bin"
    capture = sandbox / "capture"
    workdir = sandbox / "workdir"
    _write_shims(fake_bin)
    capture.write_bytes(b"")
    workdir.mkdir(exist_ok=True)

    env = {
        **_FROZEN_ENV,
        "PATH": f"{fake_bin}:{_SYSTEM_PATH}",
        "MILES_SH_HARNESS_CAPTURE": str(capture),
        "MILES_SH_HARNESS_ARG_SEP": _ARG_SEPARATOR,
        "MILES_SH_HARNESS_RECORD_SEP": _RECORD_SEPARATOR,
        **(extra_env or {}),
    }
    process = subprocess.run(
        ["bash", str(script)],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    invocations = _parse_capture(capture.read_text(), sandbox=sandbox)

    return LaunchScriptRun(
        invocations=invocations,
        stdout=_sanitize(process.stdout, sandbox=sandbox),
        stderr=_sanitize(process.stderr, sandbox=sandbox),
        returncode=process.returncode,
    )


def format_invocations(invocations: list[list[str]]) -> str:
    lines = []
    for index, argv in enumerate(invocations):
        lines.append(f"### {index}")
        lines.extend(json.dumps(arg) for arg in argv)
        lines.append("")
    return "\n".join(lines)


def _write_shims(fake_bin: Path) -> None:
    fake_bin.mkdir(exist_ok=True)
    for name in _SHIMMED_COMMANDS:
        stdout = _SHIM_STDOUT.get(name)
        stdout_statement = "" if stdout is None else f"printf '%s\\n' {stdout!a}\n"
        shim = fake_bin / name
        shim.write_text(_SHIM_TEMPLATE.format(name=name, stdout_statement=stdout_statement))
        shim.chmod(0o755)


def _parse_capture(raw: str, sandbox: Path) -> list[list[str]]:
    records = [record for record in raw.split(_RECORD_SEPARATOR) if record != ""]
    return [[_sanitize(arg, sandbox=sandbox) for arg in record.split(_ARG_SEPARATOR)] for record in records]


def _sanitize(text: str, sandbox: Path) -> str:
    return text.replace(str(sandbox), SANDBOX_PLACEHOLDER).replace(str(REPO_ROOT), REPO_ROOT_PLACEHOLDER)
