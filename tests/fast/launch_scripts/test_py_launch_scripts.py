import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.fast.launch_scripts.py_harness import (
    call_entrypoint,
    format_commands,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
    iter_py_launch_scripts,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "launch_scripts" / "py"

_SCRIPTS_SKIPPED_PENDING_NPU_SUPPORT = {"scripts/run_qwen3_4b_npu.py"}


def _glm_checkpoint(sandbox: Path, model_name: str, num_layers: int) -> dict[str, object]:
    model_dir = sandbox / "models"
    (model_dir / model_name).mkdir(parents=True)
    (model_dir / model_name / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "architectures": ["GlmMoeDsaForCausalLM"],
                "num_hidden_layers": num_layers,
            }
        )
    )
    return {"model_dir": str(model_dir)}


_SCRIPTS_WHOSE_DEFAULTS_ARE_UNSUPPORTED: dict[str, Callable[[Path], dict[str, object]]] = {
    "scripts/run_deepseek_v4.py": lambda sandbox: {"model_name": "DeepSeek-V4-Flash-FP8-4layer"},
    "scripts/run_glm45_355b_a32b.py": lambda sandbox: {"hardware": "GB200"},
    "scripts/run_glm5_744b_a40b.py": lambda sandbox: _glm_checkpoint(sandbox, "GLM-5", 78),
    "scripts/run_glm5_2_744b_a40b.py": lambda sandbox: _glm_checkpoint(sandbox, "GLM-5.2", 78),
}

_ENTRYPOINTS_DISABLED_BY_THEIR_OWN_DEFAULTS = {("scripts/run_deepseek_v4.py", "prepare_mxfp8")}

_SCRIPTS = [script for script in iter_py_launch_scripts() if script.rel not in _SCRIPTS_SKIPPED_PENDING_NPU_SUPPORT]
_CASES = [(script.rel, entrypoint) for script in _SCRIPTS for entrypoint in script.entrypoints]


@pytest.fixture(params=_CASES, ids=[f"{rel}::{entrypoint}" for rel, entrypoint in _CASES])
def recorded(request, monkeypatch, tmp_path):
    rel, entrypoint = request.param
    freeze_environment(monkeypatch)
    commands = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / rel)
    call_entrypoint(module, entrypoint, _SCRIPTS_WHOSE_DEFAULTS_ARE_UNSUPPORTED.get(rel, lambda sandbox: {})(tmp_path))
    return rel, entrypoint, commands, tmp_path


class TestEveryLauncherEntrypoint:
    def test_commands_match_snapshot(self, recorded):
        """Every launcher entrypoint must build exactly the recorded shell commands."""
        rel, entrypoint, commands, sandbox = recorded
        snapshot = _SNAPSHOT_DIR / rel / f"{entrypoint}.txt"

        assert_matches_snapshot(snapshot, format_commands(commands, sandbox=sandbox), f"{rel}::{entrypoint}")

    def test_entrypoint_issues_commands(self, recorded):
        """An entrypoint that silently does nothing is a broken launcher, not a passing test."""
        rel, entrypoint, commands, _ = recorded
        if (rel, entrypoint) in _ENTRYPOINTS_DISABLED_BY_THEIR_OWN_DEFAULTS:
            assert not commands
        else:
            assert commands


class TestDiscovery:
    def test_all_py_launch_scripts_are_discovered(self):
        """Guards against the discovery glob silently going empty."""
        assert len(_SCRIPTS) > 15

    def test_execute_train_config_defaults_are_not_taken_from_a_slurm_allocation(self, monkeypatch):
        """SLURM_JOB_NUM_NODES is read at import time, so a stale allocation would skew every snapshot."""
        import miles.utils.external_utils.command_utils as command_utils

        assert command_utils.ExecuteTrainConfig().num_nodes == 1
