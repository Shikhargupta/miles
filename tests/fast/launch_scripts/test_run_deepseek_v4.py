import pytest

from tests.fast.launch_scripts.py_harness import (
    REPO_ROOT,
    call_entrypoint,
    freeze_environment,
    import_launch_script,
    install_command_recorder,
)


@pytest.mark.parametrize(
    ("overrides", "expected_size"),
    [
        (
            {"model_name": "DeepSeek-V4-Flash-FP8-4layer", "num_nodes": 1, "num_gpus_per_node": 4},
            4,
        ),
        ({"model_name": "DeepSeek-V4-Flash-FP8", "num_nodes": 8, "num_gpus_per_node": 4}, 8),
    ],
)
def test_four_gpu_node_rollout_topology_matches_available_resources(monkeypatch, tmp_path, overrides, expected_size):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")

    call_entrypoint(module, "train", overrides, sandbox=tmp_path)

    train_command = recording.commands[-1]
    assert f"--rollout-num-gpus-per-engine {expected_size}" in train_command
    assert f"--sglang-tp-size {expected_size}" in train_command
    assert f"--sglang-ep-size {expected_size}" in train_command


def test_rollout_topology_rejects_engine_larger_than_pool():
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")
    args = module.ScriptArgs(model_name="DeepSeek-V4-Pro-FP8", num_nodes=1, num_gpus_per_node=8)

    with pytest.raises(
        ValueError,
        match="SGLang rollout_num_gpus_per_engine=32 exceeds rollout_num_gpus=8",
    ):
        module._get_sglang_parallel_config(args)


def test_rollout_topology_rejects_partial_engine():
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")
    args = module.ScriptArgs(model_name="DeepSeek-V4-Flash-FP8", num_nodes=3, num_gpus_per_node=4)

    with pytest.raises(
        ValueError,
        match="rollout_num_gpus=12 must be divisible by SGLang rollout_num_gpus_per_engine=8",
    ):
        module._get_sglang_parallel_config(args)
