import pytest

from tests.fast.launch_scripts.sh_harness import REPO_ROOT, REPO_ROOT_PLACEHOLDER, run_launch_script

_SCRIPT = REPO_ROOT / "scripts" / "run-qwen3-4B.sh"

_SYNTHETIC_SCRIPT = """#!/bin/bash
set -ex
EXPECTED_GPUS=32
while true; do
    AVAILABLE_GPUS=$(python3 -c "import ray; print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
    if [ "$AVAILABLE_GPUS" -ge "$EXPECTED_GPUS" ]; then
        break
    fi
    sleep 5
done
hf download some/model --local-dir /root/models/some-model
torchrun --nproc-per-node 8 CHECKOUT/tools/convert_hf_to_torch_dist.py
ray job submit --address="http://127.0.0.1:8265" -- python3 CHECKOUT/train.py
"""


class TestRunLaunchScriptOnARealScript:
    @pytest.fixture
    def run(self, tmp_path):
        return run_launch_script(_SCRIPT, sandbox=tmp_path)

    def test_script_runs_to_completion_without_touching_the_real_system(self, run):
        """The shimmed PATH lets a real launch script run end to end and exit cleanly."""
        assert run.returncode == 0

    def test_destructive_commands_are_intercepted_instead_of_executed(self, run):
        """pkill / ray stop are recorded by shims, so they never reach the test runner."""
        assert ["pkill", "-9", "sglang"] in run.invocations
        assert ["ray", "stop", "--force"] in run.invocations

    def test_ray_start_is_recorded_with_the_frozen_master_addr(self, run):
        """Node address comes from the frozen environment, not from the developer machine."""
        (ray_start,) = [argv for argv in run.invocations_of("ray") if argv[1] == "start"]
        assert "--node-ip-address" in ray_start
        assert ray_start[ray_start.index("--node-ip-address") + 1] == "127.0.0.1"

    def test_ray_job_submit_argv_contains_the_expanded_model_args(self, run):
        """`source scripts/models/*.sh` expansion must be visible in the captured argv."""
        argv = run.ray_job_submit_argv()
        assert argv[:3] == ["ray", "job", "submit"]
        assert "--num-layers" in argv
        assert argv[argv.index("--num-layers") + 1] == "36"
        assert argv[argv.index("--hf-checkpoint") + 1] == "/root/Qwen3-4B"

    def test_nvlink_detection_is_frozen_to_absent(self, run):
        """The nvidia-smi shim reports no NVLink, so NCCL_NVLS_ENABLE is deterministic."""
        argv = run.ray_job_submit_argv()
        (runtime_env,) = [arg for arg in argv if arg.startswith("--runtime-env-json=")]
        assert '"NCCL_NVLS_ENABLE": "0"' in runtime_env

    def test_reruns_produce_identical_recordings(self, tmp_path):
        """Snapshot testing only works if the harness is deterministic across runs."""
        first = run_launch_script(_SCRIPT, sandbox=tmp_path / "a")
        second = run_launch_script(_SCRIPT, sandbox=tmp_path / "b")

        assert first.invocations == second.invocations


class TestRunLaunchScriptOnTheShimEdgeCases:
    @pytest.fixture
    def run(self, tmp_path):
        script = tmp_path / "synthetic.sh"
        script.write_text(_SYNTHETIC_SCRIPT.replace("CHECKOUT", str(REPO_ROOT)))
        return run_launch_script(script, sandbox=tmp_path / "sandbox", timeout=30)

    def test_a_gpu_wait_loop_leaves_on_its_first_poll(self, run):
        """The python shim must emit a real number; emitting its repr spins the loop until timeout."""
        assert run.returncode == 0
        assert run.invocations_of("sleep") == []

    def test_downloads_and_torchrun_are_intercepted(self, run):
        """Unshimmed, these would pull real weights and start a real training job."""
        assert run.invocations_of("hf")[0][1] == "download"
        assert run.invocations_of("torchrun")[0][1] == "--nproc-per-node"

    def test_a_repo_path_inside_argv_becomes_a_placeholder(self, run):
        """Recordings must not embed the checkout location of whoever ran the test."""
        torchrun_argv = run.invocations_of("torchrun")[0]

        assert torchrun_argv[-1] == f"{REPO_ROOT_PLACEHOLDER}/tools/convert_hf_to_torch_dist.py"
        assert str(REPO_ROOT) not in " ".join(torchrun_argv)
