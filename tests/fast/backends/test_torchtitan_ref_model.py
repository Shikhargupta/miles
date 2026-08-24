"""build_ref_runner contract, with the titan build behind a fake.

These run without torchtitan installed: everything the engine would do with
titan is monkeypatched, and only the miles-side contract is under test --
the ref checkpoint is mandatory, PP is rejected, the second build is
CPU-offloaded, and the returned runner drives a frozen model.
"""

from types import SimpleNamespace

import pytest

from miles.backends.torchtitan_utils.engine import TitanEngine


class _FakePart:
    def __init__(self):
        self.frozen = None
        self.evaluated = False

    def eval(self):
        self.evaluated = True

    def requires_grad_(self, flag):
        self.frozen = not flag


def _bare_engine(pp_enabled=False):
    engine = TitanEngine.__new__(TitanEngine)
    engine.parallel_dims = SimpleNamespace(pp_enabled=pp_enabled)
    return engine


def test_ref_runner_requires_a_checkpoint_to_load():
    with pytest.raises(ValueError, match="--ref-load"):
        _bare_engine().build_ref_runner(None)


def test_ref_runner_is_rejected_under_pp():
    with pytest.raises(NotImplementedError, match="pipeline"):
        _bare_engine(pp_enabled=True).build_ref_runner("/ckpt")


def test_ref_model_is_built_offloaded_then_frozen(monkeypatch):
    engine = _bare_engine()
    part = _FakePart()
    seen = {}

    def fake_build_parts(*, cpu_offload):
        seen["cpu_offload"] = cpu_offload
        return [part], None, True, True

    monkeypatch.setattr(engine, "_build_parts", fake_build_parts)
    monkeypatch.setattr(engine, "load_hf", lambda path, parts: seen.setdefault("loaded_from", path))

    runner = engine.build_ref_runner("/ckpt")

    assert seen == {"cpu_offload": True, "loaded_from": "/ckpt"}
    assert part.evaluated is True
    assert part.frozen is True
    assert hasattr(runner, "forward_only_step")


def test_ref_runner_forwards_through_the_ref_parts_not_the_actor(monkeypatch):
    """The runner must be bound to the second (frozen) copy; running the actor
    model would make ref log probs equal to actor log probs and silently zero
    the KL term."""
    engine = _bare_engine()
    ref_part = _FakePart()

    monkeypatch.setattr(engine, "_build_parts", lambda *, cpu_offload: ([ref_part], None, True, True))
    monkeypatch.setattr(engine, "load_hf", lambda path, parts: None)
    calls = []
    monkeypatch.setattr(
        engine, "_forward", lambda batch, module=None: calls.append(module) or f"logits-{batch['i']}"
    )

    runner = engine.build_ref_runner("/ckpt")
    results = runner.forward_only_step([{"i": 0}, {"i": 1}], lambda logits, batch: (logits, batch["i"]))

    assert calls == [ref_part, ref_part]
    assert results == [("logits-0", 0), ("logits-1", 1)]
