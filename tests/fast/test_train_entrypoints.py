import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The rollout each entrypoint's weight push will serve, as source text. Sync
# trains rollout N then pushes the weights rollout N+1 generates with; ordinary
# async has one extra rollout in flight, so its push serves N+2; multi-LoRA
# reconciles and pushes before generating rollout N, so it serves N. Every
# entrypoint also pushes once before its loop, serving start_rollout_id.
_EXPECTED_WEIGHT_ROLLOUT_IDS = {
    "train.py": {"args.start_rollout_id", "rollout_id + 1"},
    "train_async.py": {"args.start_rollout_id", "rollout_id + 2"},
    "train_multi_lora_async.py": {"rollout_id"},
}


@pytest.mark.parametrize("entrypoint", sorted(_EXPECTED_WEIGHT_ROLLOUT_IDS))
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
        if keyword.arg == "weight_rollout_id"
    }

    assert served == _EXPECTED_WEIGHT_ROLLOUT_IDS[entrypoint]
