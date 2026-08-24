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
def test_four_gpu_node_rollout_topology_distinguishes_4layer_from_gb300(
    monkeypatch, tmp_path, overrides, expected_size
):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    module = import_launch_script(REPO_ROOT / "scripts/run_deepseek_v4.py")

    call_entrypoint(module, "train", overrides, sandbox=tmp_path)

    train_command = recording.commands[-1]
    assert f"--rollout-num-gpus-per-engine {expected_size}" in train_command
    assert f"--sglang-tp-size {expected_size}" in train_command
    assert f"--sglang-ep-size {expected_size}" in train_command
