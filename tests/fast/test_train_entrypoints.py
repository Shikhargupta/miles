import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# How many rollouts each entrypoint has finished training when it publishes, as
# source text. One rule covers all three: count the rollouts you have trained.
# Both loop drivers push after train(N), so N+1; multi-LoRA pushes before
# train(N), so N. The pre-loop push has trained whatever the checkpoint had,
# which is exactly start_rollout_id.
_EXPECTED_NUM_TRAINED_ROLLOUTS = {
    "train.py": {"args.start_rollout_id", "rollout_id + 1"},
    "train_async.py": {"args.start_rollout_id", "rollout_id + 1"},
    "train_multi_lora_async.py": {"rollout_id"},
}


@pytest.mark.parametrize("entrypoint", sorted(_EXPECTED_NUM_TRAINED_ROLLOUTS))
def test_each_entrypoint_serves_the_rollout_its_loop_shape_implies(entrypoint):
    """The offset differs per driver and is not a naive +1, and getting it wrong misattributes silently."""
    tree = ast.parse((_REPO_ROOT / entrypoint).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update_weights"
    ]
    assert calls, f"{entrypoint} pushes no weights; this guard would silently pass"

    served = {
        ast.unparse(keyword.value)
        for call in calls
        for keyword in call.keywords
        if keyword.arg == "num_trained_rollouts"
    }

    assert served == _EXPECTED_NUM_TRAINED_ROLLOUTS[entrypoint]
