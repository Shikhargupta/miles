import pytest

from tests.fast.launch_scripts.sh_harness import REPO_ROOT, run_launch_script

_SCRIPT = REPO_ROOT / "scripts" / "run-qwen3-4B.sh"


@pytest.fixture
def run(tmp_path):
    return run_launch_script(_SCRIPT, sandbox=tmp_path)


def test_script_runs_to_completion_without_touching_the_real_system(run):
    """The shimmed PATH lets a real launch script run end to end and exit cleanly."""
    assert run.returncode == 0


def test_destructive_commands_are_intercepted_instead_of_executed(run):
    """pkill / ray stop are recorded by shims, so they never reach the test runner."""
    assert ["pkill", "-9", "sglang"] in run.invocations
    assert ["ray", "stop", "--force"] in run.invocations


def test_ray_start_is_recorded_with_the_frozen_master_addr(run):
    """Node address comes from the frozen environment, not from the developer machine."""
    (ray_start,) = [argv for argv in run.invocations_of("ray") if argv[1] == "start"]
    assert "--node-ip-address" in ray_start
    assert ray_start[ray_start.index("--node-ip-address") + 1] == "127.0.0.1"


def test_ray_job_submit_argv_contains_the_expanded_model_args(run):
    """`source scripts/models/*.sh` expansion must be visible in the captured argv."""
    argv = run.ray_job_submit_argv()
    assert argv[:3] == ["ray", "job", "submit"]
    assert "--num-layers" in argv
    assert argv[argv.index("--num-layers") + 1] == "36"
    assert argv[argv.index("--hf-checkpoint") + 1] == "/root/Qwen3-4B"


def test_nvlink_detection_is_frozen_to_absent(run):
    """The nvidia-smi shim reports no NVLink, so NCCL_NVLS_ENABLE is deterministic."""
    argv = run.ray_job_submit_argv()
    (runtime_env,) = [arg for arg in argv if arg.startswith("--runtime-env-json=")]
    assert '"NCCL_NVLS_ENABLE": "0"' in runtime_env


def test_absolute_repo_paths_are_replaced_by_a_placeholder(tmp_path):
    """Snapshots must not embed the checkout location of whoever ran the test."""
    run = run_launch_script(_SCRIPT, sandbox=tmp_path)
    assert str(REPO_ROOT) not in "\n".join(arg for argv in run.invocations for arg in argv)


def test_reruns_produce_identical_recordings(tmp_path):
    """Snapshot testing only works if the harness is deterministic across runs."""
    first = run_launch_script(_SCRIPT, sandbox=tmp_path / "a")
    second = run_launch_script(_SCRIPT, sandbox=tmp_path / "b")
    assert first.invocations == second.invocations
