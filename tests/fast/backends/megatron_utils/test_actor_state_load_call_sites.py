import ast

from tests.fast.source_scan import FRAMEWORK_ROOT

# reading the source, not the module: importing it pulls in torch_memory_saver, which a cpu shard
# does not have, and a collection error there takes down every other test in the same shard
_SOURCE = FRAMEWORK_ROOT / "backends" / "megatron_utils" / "actor.py"
_CORE_METHOD = "_load_state_core"
_LOAD_FUNCTION = "load_model_state"
_WEIGHT_UPDATER = "weight_updater"


def _functions_calling(name: str, *, as_method: bool) -> list[str]:
    tree = ast.parse(_SOURCE.read_text())
    calling: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            called = inner.func.attr if as_method and isinstance(inner.func, ast.Attribute) else None
            if not as_method and isinstance(inner.func, ast.Name):
                called = inner.func.id
            if called == name:
                calling.append(node.name)
    return sorted(set(calling))


def _init_function() -> ast.FunctionDef:
    tree = ast.parse(_SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "init":
            return node
    raise AssertionError("actor.py no longer defines an init method")


def _state_load_call_lines(init: ast.FunctionDef) -> list[int]:
    return [
        inner.lineno
        for inner in ast.walk(init)
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == _CORE_METHOD
    ]


def _weight_updater_assign_lines(init: ast.FunctionDef) -> list[int]:
    lines: list[int] = []
    for inner in ast.walk(init):
        if not isinstance(inner, ast.Assign):
            continue
        for target in inner.targets:
            if isinstance(target, ast.Attribute) and target.attr == _WEIGHT_UPDATER:
                lines.append(inner.lineno)
    return lines


class TestWhereTheTrainerLoadsItsState:
    def test_the_state_load_is_reached_from_one_function_only(self):
        """A second call site would let a reload drift from what init does, which is what this op exists to stop."""
        assert _functions_calling(_LOAD_FUNCTION, as_method=False) == [_CORE_METHOD]

    def test_init_reaches_its_state_load_through_that_function(self):
        """The invariant is worth nothing if init stopped going through the function everything else goes through."""
        assert "init" in _functions_calling(_CORE_METHOD, as_method=True)


class TestWhenTheTrainerBuildsItsWeightUpdater:
    def test_the_weight_updater_is_built_after_the_state_load(self):
        """Built before the load it snapshots expert_bias as bf16, and peer ranks then size their cross-PP receive buffer from that stale dtype."""
        init = _init_function()
        assigns = _weight_updater_assign_lines(init)
        loads = _state_load_call_lines(init)

        assert assigns, "init no longer assigns self.weight_updater"
        assert loads, f"init no longer calls self.{_CORE_METHOD}"
        assert min(assigns) > max(loads)
