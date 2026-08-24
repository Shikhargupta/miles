"""The engine-side weight-push protocol, exercised without a GPU or a real engine.

These pin the two behaviors that had drifted between the backends' private
copies before the handshake was shared: honoring --pause-generation-mode, and
not flushing the prefix cache under in_place pausing.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.training_utils.weight_sync import weight_push_session, weight_update_selector


class _FakeEngine:
    """Records the order of calls a rollout engine receives."""

    def __init__(self, calls: list):
        self._calls = calls

    def __getattr__(self, name):
        def method(**kwargs):
            self._calls.append((name, kwargs))
            return f"ref:{name}"

        return MagicMock(side_effect=None, remote=method)


def _args(**overrides) -> Namespace:
    return Namespace(**{"pause_generation_mode": "retract", **overrides})


def _run_session(args, *, announce=True):
    calls: list = []
    engines = [_FakeEngine(calls)]
    with (
        patch("miles.backends.training_utils.weight_sync.dist") as dist,
        patch("miles.backends.training_utils.weight_sync.ray.get", lambda x: x),
        patch("miles.backends.training_utils.weight_sync.get_gloo_group", lambda: None),
    ):
        dist.get_rank.return_value = 0
        with weight_push_session(args, engines, announce=announce):
            calls.append(("<<stream>>", {}))
    return [name for name, _ in calls]


def test_session_order_is_pause_flush_begin_stream_end_continue():
    assert _run_session(_args()) == [
        "pause_generation",
        "flush_cache",
        "begin_weight_update",
        "<<stream>>",
        "end_weight_update",
        "continue_generation",
    ]


def test_in_place_pausing_does_not_flush_the_cache():
    """in_place resumes requests against their existing KV cache; flushing here
    would discard exactly what the mode preserves."""
    assert "flush_cache" not in _run_session(_args(pause_generation_mode="in_place"))


@pytest.mark.parametrize("mode", ["abort", "retract"])
def test_other_modes_still_flush(mode):
    assert "flush_cache" in _run_session(_args(pause_generation_mode=mode))


def test_pause_mode_is_forwarded_to_the_engine():
    calls: list = []
    engines = [_FakeEngine(calls)]
    args = _args(pause_generation_mode="abort")
    with (
        patch("miles.backends.training_utils.weight_sync.dist") as dist,
        patch("miles.backends.training_utils.weight_sync.ray.get", lambda x: x),
        patch("miles.backends.training_utils.weight_sync.get_gloo_group", lambda: None),
    ):
        dist.get_rank.return_value = 0
        with weight_push_session(args, engines):
            pass
    pause_kwargs = next(kwargs for name, kwargs in calls if name == "pause_generation")
    assert pause_kwargs == {"mode": "abort"}


def test_announce_false_skips_the_session_markers():
    """The LoRA-only push sends no fresh base weights, so the engine must not run
    its post-load step."""
    names = _run_session(_args(), announce=False)
    assert "begin_weight_update" not in names
    assert "end_weight_update" not in names
    assert "continue_generation" in names


def test_non_driver_ranks_issue_no_engine_rpcs():
    calls: list = []
    engines = [_FakeEngine(calls)]
    with (
        patch("miles.backends.training_utils.weight_sync.dist") as dist,
        patch("miles.backends.training_utils.weight_sync.ray.get", lambda x: x),
        patch("miles.backends.training_utils.weight_sync.get_gloo_group", lambda: None),
    ):
        dist.get_rank.return_value = 1
        with weight_push_session(_args(), engines):
            pass
    assert calls == []


def test_selector_excludes_draft_only_without_an_mtp_block():
    speculative = dict(sglang_speculative_algorithm="EAGLE", megatron_to_hf_mode="raw")
    assert weight_update_selector(Namespace(mtp_num_layers=None, **speculative)) == "target"
    assert weight_update_selector(Namespace(mtp_num_layers=1, **speculative)) == "all"
    assert weight_update_selector(Namespace(sglang_speculative_algorithm=None)) == "all"
