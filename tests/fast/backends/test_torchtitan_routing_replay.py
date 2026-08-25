"""The routing queues must line up with the microbatches, one entry each.

Nothing in the replay mechanism enforces that alignment: a queue serves whoever
asks next, so one unexpected forward shifts every later lookup by a microbatch.
That is not a crash and not obviously wrong in the metrics either -- adjacent
microbatches of a GRPO group share a prompt, so a shifted replay reproduces the
prompt's routing exactly and only diverges over the response.

The unexpected forward is real: a pipeline schedule infers the shapes its stages
exchange by running one forward per stage over microbatch 0, and repeats that
whenever the pass changes direction (in RL, every log-prob-then-train pair).
"""

from tests.ci.ci_register import register_cuda_ci

# Needs a GPU only because the shared replay queue hands its entries out on the
# current CUDA device; the alignment being tested is device-independent.
register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["torchtitan", "routing-replay"])

import pytest
import torch
import torch.nn as nn

from torchtitan.models.common.moe import TokenChoiceTopKRouter

from miles.backends.torchtitan_utils import routing_replay
from miles.utils.replay_base import routing_replay_manager


class _Gate(nn.Module):
    def forward(self, x):
        return x


class _Router(TokenChoiceTopKRouter):
    """A real router with the attributes its forward reads, and nothing else.

    Subclassing the real class matters: install() finds routers by type and
    rebinds torchtitan's own forward, so the queue is exercised through the code
    that actually runs in training.
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.gate = _Gate()
        self.score_func = "softmax"
        self.num_expert_groups = None
        self.top_k = 2
        self.route_norm = False
        self.route_scale = 1.0


class _Part(nn.Module):
    """A model part with one router, like a pipeline stage's submodule."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({"0": _Router()})

    def forward(self, scores):
        _, expert_ids, _ = self.layers["0"](scores)
        return expert_ids


@pytest.fixture
def part():
    routing_replay_manager.enabled = True
    routing_replay_manager.enable_check_replay_result = False
    routing_replay_manager.stage = routing_replay.REPLAY_FORWARD
    routing_replay_manager.replays = []
    part = _Part()
    routing_replay.install([part])
    yield part
    routing_replay_manager.enabled = False
    routing_replay_manager.replays = []
    routing_replay_manager.stage = routing_replay.FALLTHROUGH


def _queue_microbatches(count):
    """Queue one entry per microbatch, entry k naming expert k."""
    replay = routing_replay_manager.replays[0]
    for k in range(count):
        replay.record(torch.full((4, 2), k, dtype=torch.long))
    return replay


def test_each_microbatch_replays_its_own_entry(part):
    _queue_microbatches(3)
    scores = torch.rand(1, 4, 8, device="cuda")

    for expected in range(3):
        picked = part(scores)
        assert picked.unique().tolist() == [expected]

    routing_replay.check_consumption(3)


def test_the_schedules_metadata_forward_does_not_consume(part):
    _queue_microbatches(2)
    scores = torch.rand(1, 4, 8, device="cuda")

    routing_replay.bypass_next_forward([part])
    inferred = part(scores)
    # Bypassed: the router chose for itself rather than replaying entry 0.
    assert inferred.unique().tolist() != [0]

    for expected in range(2):
        assert part(scores).unique().tolist() == [expected]

    routing_replay.check_consumption(2)


def test_bypass_covers_exactly_one_forward(part):
    _queue_microbatches(2)
    scores = torch.rand(1, 4, 8, device="cuda")

    routing_replay.bypass_next_forward([part])
    part(scores)
    part(scores)
    # The bypass is spent, so the second forward replayed entry 0 -- and the
    # pass is short one microbatch.
    with pytest.raises(RuntimeError, match="no longer line up"):
        routing_replay.check_consumption(2)


def test_a_stray_forward_is_reported(part):
    _queue_microbatches(3)
    scores = torch.rand(1, 4, 8, device="cuda")

    for _ in range(3):
        part(scores)

    with pytest.raises(RuntimeError, match="advanced 3 times over a pass of 2"):
        routing_replay.check_consumption(2)
