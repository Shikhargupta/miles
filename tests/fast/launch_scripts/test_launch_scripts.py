import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, format_invocations, iter_launch_scripts, run_launch_script

_SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
_UPDATE_ENV_VAR = "MILES_UPDATE_LAUNCH_SCRIPT_SNAPSHOTS"


@dataclass(frozen=True)
class LaunchScriptCase:
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


_CHECKPOINT_DIR = "/frozen/checkpoints"
_HEAD_NODE_IP = "10.0.0.1"

_SCRIPTS_REFUSING_TO_RUN_WITHOUT_EXPLICIT_INPUTS: dict[str, LaunchScriptCase] = {
    "examples/lora/run-qwen2.5-3B-megatron-lora-disaggregated-multi-node.sh": LaunchScriptCase(args=("p2p", "0")),
    "examples/on_policy_distillation/qwen3_5_35b_selfdistill/phase1_rlvr_teacher.sh": LaunchScriptCase(
        env={"OUTPUT_DIR": "{workdir}"}
    ),
    "examples/on_policy_distillation/qwen3_5_35b_selfdistill/phase2_gb200.sh": LaunchScriptCase(
        env={"OUTPUT_DIR": "{workdir}"}
    ),
    "examples/on_policy_distillation/qwen3_5_35b_selfdistill/phase2_opd_selfdistill.sh": LaunchScriptCase(
        env={"OUTPUT_DIR": "{workdir}"}
    ),
    "examples/p2p_weight_transfer/run-glm4.5-air-8node-profile.sh": LaunchScriptCase(
        args=("p2p", "0", _HEAD_NODE_IP), env={"MILES_LOG_DIR": "{workdir}"}
    ),
    "examples/p2p_weight_transfer/run-glm4.7-flash-2node-profile.sh": LaunchScriptCase(
        args=("p2p", "0", _HEAD_NODE_IP)
    ),
    "examples/p2p_weight_transfer/run-glm5-disagg-profile.sh": LaunchScriptCase(
        args=("GLM-5", "p2p", "0", _HEAD_NODE_IP), env={"MILES_LOG_DIR": "{workdir}"}
    ),
    "examples/p2p_weight_transfer/run-kimi-k2-64node-profile.sh": LaunchScriptCase(args=("p2p", "0", _HEAD_NODE_IP)),
    "examples/p2p_weight_transfer/run-qwen3-235B-A22B-16node-profile.sh": LaunchScriptCase(
        args=("p2p", "0", _HEAD_NODE_IP)
    ),
    "examples/p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh": LaunchScriptCase(
        args=("p2p", "0", _HEAD_NODE_IP)
    ),
    "scripts/run-nemotron-3-super-120b-a12b.sh": LaunchScriptCase(args=("head", _HEAD_NODE_IP)),
    "scripts/run-qwen3-235B-A22B-sft.sh": LaunchScriptCase(env={"BASE_FOLDER": _CHECKPOINT_DIR}),
    "scripts/run-qwen3-235B-A22B.sh": LaunchScriptCase(env={"BASE_FOLDER": _CHECKPOINT_DIR}),
    "scripts/run-qwen3-next-80B-A3B-8gpus.sh": LaunchScriptCase(env={"BASE_FOLDER": _CHECKPOINT_DIR}),
    "scripts/run-qwen3-next-80B-A3B-fsdp.sh": LaunchScriptCase(env={"BASE_FOLDER": _CHECKPOINT_DIR}),
    "scripts/run-qwen3-next-80B-A3B.sh": LaunchScriptCase(env={"BASE_FOLDER": _CHECKPOINT_DIR}),
    "scripts/run-qwen3.6-27B.sh": LaunchScriptCase(env={"OUTPUT_DIR": _CHECKPOINT_DIR}),
}

_SCRIPTS_RACING_WITH_A_BACKGROUNDED_SERVER = {"examples/on_policy_distillation/run-qwen3-8B-opd.sh"}

_SCRIPTS = [script.relative_to(REPO_ROOT).as_posix() for script in iter_launch_scripts()]


@pytest.fixture(params=_SCRIPTS)
def recorded(request, tmp_path):
    rel = request.param
    case = _SCRIPTS_REFUSING_TO_RUN_WITHOUT_EXPLICIT_INPUTS.get(rel, LaunchScriptCase())
    workdir = tmp_path / "workdir"
    run = run_launch_script(
        REPO_ROOT / rel,
        sandbox=tmp_path,
        args=case.args,
        extra_env={key: value.format(workdir=workdir) for key, value in case.env.items()},
    )
    return rel, run


def test_launch_script_invocations_match_snapshot(recorded):
    """Every launch script must issue exactly the recorded sequence of external commands."""
    rel, run = recorded
    snapshot = _SNAPSHOT_DIR / f"{rel}.txt"
    invocations = sorted(run.invocations) if rel in _SCRIPTS_RACING_WITH_A_BACKGROUNDED_SERVER else run.invocations
    actual = f"# returncode: {run.returncode}\n\n{format_invocations(invocations)}"

    if os.environ.get(_UPDATE_ENV_VAR):
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(actual)
        return

    assert snapshot.exists(), f"missing snapshot for {rel}; regenerate with {_UPDATE_ENV_VAR}=1"
    assert actual == snapshot.read_text()


def test_launch_script_submits_exactly_one_ray_job(recorded):
    """A launch script that no longer reaches `ray job submit` is broken, whatever else it does."""
    _, run = recorded
    assert run.returncode == 0
    assert len(run.ray_job_submit_argv()) > 10


def test_all_launch_scripts_are_discovered():
    """Guards against the discovery glob silently going empty."""
    assert len(_SCRIPTS) > 60
