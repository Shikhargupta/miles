import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENTRYPOINTS = ("train.py", "train_async.py", "train_multi_lora_async.py")

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


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_every_update_weights_call_states_which_rollout_its_weights_serve(entrypoint):
    """A weight push that does not say which rollout it serves cannot be attributed at all."""
    for call in _update_weights_calls(entrypoint):
        assert any(
            keyword.arg == "weight_rollout_id" for keyword in call.keywords
        ), f"{entrypoint}:{call.lineno} calls update_weights without weight_rollout_id"


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_each_entrypoint_serves_the_rollout_its_loop_shape_implies(entrypoint):
    """The offset differs per driver and is not a naive +1, so pin the exact expressions."""
    served = {
        ast.unparse(keyword.value)
        for call in _update_weights_calls(entrypoint)
        for keyword in call.keywords
        if keyword.arg == "weight_rollout_id"
    }

    assert served == _EXPECTED_WEIGHT_ROLLOUT_IDS[entrypoint]


@pytest.mark.parametrize("entrypoint", _ENTRYPOINTS)
def test_weights_are_pushed_before_the_rollout_they_serve_is_generated(entrypoint):
    """A push that lands after generate would stamp the samples with weights they never saw."""
    tree = ast.parse((_REPO_ROOT / entrypoint).read_text())
    generates = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("generate.remote")
    ]

    assert generates, f"{entrypoint} never generates; this guard would silently pass"
    assert min(call.lineno for call in _update_weights_calls(entrypoint)) < min(generates)


def _update_weights_calls(entrypoint: str) -> list[ast.Call]:
    tree = ast.parse((_REPO_ROOT / entrypoint).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update_weights"
    ]

    assert calls, f"{entrypoint} pushes no weights; this guard would silently pass"
    return calls
