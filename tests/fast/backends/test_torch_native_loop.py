"""Orchestration of the shared torch-native loops.

Checks call counts and ordering rather than numerics: the bugs this code is prone
to are structural (a missing zero_grad between optimizer steps, a step applied
per microbatch instead of per step), and those are invisible in a loss curve.
"""

from argparse import Namespace
from unittest.mock import patch

import pytest
import torch

from miles.backends.training_utils import torch_native_loop as tnl


class _Profiler:
    """Stands in for TrainProfiler, which only wraps the iterators."""

    def iterate_train_actor(self, it):
        return it

    def iterate_train_log_probs(self, it):
        return it


class _DataIterator:
    def __init__(self):
        self.resets = 0
        self.fetches = 0

    def reset(self):
        self.resets += 1
        return self


def _args() -> Namespace:
    return Namespace(
        data_pad_size_multiplier=1,
        qkv_format="thd",
        ci_test=False,
        clip_grad=1.0,
    )


@pytest.fixture
def loop_env():
    """Patch out everything that needs real tensors, keeping the control flow."""
    calls: list[str] = []

    def fake_get_batch(data_iterator, keys, *a, **kw):
        data_iterator.fetches += 1
        calls.append("batch")
        return {"unconcat_tokens": None, "total_lengths": [1], "response_lengths": [1]}

    def fake_loss(args, batch, num_microbatches, logits, apply_megatron_loss_scaling):
        calls.append("loss")
        assert apply_megatron_loss_scaling is False, "the shared loop never goes through a PP schedule"
        return logits.sum(), 1, {}

    with (
        patch.object(tnl, "get_batch", fake_get_batch),
        patch.object(tnl, "loss_function", fake_loss),
        patch.object(tnl, "aggregate_train_losses", lambda x: {}),
        patch.object(tnl, "log_train_step", lambda **kw: calls.append(f"log:{kw['step_id']}")),
        patch.object(tnl, "check_grad_norm", lambda **kw: calls.append("check_grad_norm")),
        patch.object(tnl, "aggregate_forward_results", lambda store, *a, **kw: {"n": len(store)}),
        patch.object(tnl, "get_log_probs_and_entropy", lambda **kw: {"log_probs": 1, "entropy": 2}),
        patch.object(tnl.dist, "get_rank", lambda: 0),
    ):
        yield calls


def _run_steps(calls, num_microbatches):
    data_iterator = _DataIterator()
    steps: list[str] = []

    def forward(batch):
        calls.append("forward")
        return torch.zeros(1, requires_grad=True)

    tnl.run_optimizer_steps(
        _args(),
        rollout_id=0,
        data_iterator=data_iterator,
        num_microbatches=num_microbatches,
        forward_fn=forward,
        zero_grad_fn=lambda: (calls.append("zero_grad"), steps.append("z"))[0],
        step_fn=lambda: (calls.append("step"), tnl.StepMetrics(grad_norm=0.5))[1],
        profiler=_Profiler(),
    )
    return data_iterator, steps


def test_gradients_are_cleared_once_per_optimizer_step(loop_env):
    """Without this the gradients of step N-1 leak into step N."""
    _run_steps(loop_env, [2, 3])
    assert loop_env.count("zero_grad") == 2


def test_zero_grad_precedes_the_microbatches_of_its_step(loop_env):
    _run_steps(loop_env, [2])
    assert loop_env.index("zero_grad") < loop_env.index("forward")


def test_optimizer_steps_once_per_step_not_per_microbatch(loop_env):
    _run_steps(loop_env, [4])
    assert loop_env.count("forward") == 4
    assert loop_env.count("step") == 1


def test_each_step_logs_exactly_once(loop_env):
    _run_steps(loop_env, [1, 1, 1])
    assert [c for c in loop_env if c.startswith("log:")] == ["log:0", "log:1", "log:2"]


def test_grad_norm_check_only_under_ci_test(loop_env):
    _run_steps(loop_env, [1])
    assert "check_grad_norm" not in loop_env


def test_the_iterator_is_rewound_before_the_loop(loop_env):
    data_iterator, _ = _run_steps(loop_env, [2, 2])
    assert data_iterator.resets == 1
    assert data_iterator.fetches == 4


def test_log_probs_collects_one_entry_per_microbatch(loop_env):
    data_iterator = _DataIterator()
    result = tnl.run_log_probs(
        _args(),
        data_iterator,
        [2, 3],
        lambda batch: torch.zeros(1),
        profiler=_Profiler(),
    )
    assert result == {"n": 5}
    assert data_iterator.resets == 1


def test_log_probs_skips_entropy_for_a_prefixed_pass(loop_env):
    """Only the actor pass feeds entropy to the loss hub; ref/teacher passes do not."""
    seen = {}

    def spy(**kwargs):
        seen["with_entropy"] = kwargs["with_entropy"]
        return {"log_probs": 1}

    with patch.object(tnl, "get_log_probs_and_entropy", spy):
        tnl.run_log_probs(
            _args(), _DataIterator(), [1], lambda b: torch.zeros(1), profiler=_Profiler(), store_prefix="ref_"
        )
    assert seen["with_entropy"] is False
