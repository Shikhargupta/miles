"""Rollout routing replay (R3) for the torchtitan backend.

R3 makes training re-select the experts the rollout engine chose, so the
training forward is a faithful replay of what generated the tokens. The
mechanism is the shared ``routing_replay_manager``: a per-MoE-layer queue of
expert ids, keyed by decoder-layer index -- the axis the rollout's
``[tokens, layers, topk]`` tensor uses.

What is torchtitan-specific is the install step. Every titan MoE model routes
through one class, ``torchtitan.models.common.moe.TokenChoiceTopKRouter``, so a single
rebound forward covers qwen3, qwen3_5, deepseek_v3 and the rest; the FSDP
backend needs a per-architecture adapter table instead because it trains stock
HF modeling, where each family has its own router module.

The rebound forward is upstream's, with one substitution: the expert-selection
``torch.topk`` becomes the manager's topk. Everything above it (gate, score
function, expert bias, node-limited routing) and below it (score gather,
normalization, scaling) is unchanged, so replayed routing stays differentiable
through the gate exactly as recorded routing is.
"""

import contextlib
import functools
import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from miles.backends.training_utils.replay_data import fill_replay_data, register_replay_list_sequential
from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)

FALLTHROUGH = "fallthrough"
RECORD = "record"
REPLAY_FORWARD = "replay_forward"
REPLAY_BACKWARD = "replay_backward"


def uses_rollout_replay(args) -> bool:
    """True when routing comes from the rollout rather than from a recording pass."""
    return bool(getattr(args, "use_rollout_routing_replay", False))


def enable(args) -> bool:
    """Settle manager state before the model is built, and report whether R3 is on."""
    routing_replay_manager.enabled = bool(getattr(args, "use_routing_replay", False))
    routing_replay_manager.enable_check_replay_result = routing_replay_manager.enabled and args.ci_test
    routing_replay_manager.register_replay_list_func = register_replay_list_sequential
    return routing_replay_manager.enabled


def _token_router_forward(self, x_BLD: torch.Tensor, expert_bias_E: torch.Tensor | None = None):
    """torchtitan TokenChoiceTopKRouter.forward with the expert-selection topk replaced.

    titan routes on ``(B, L, E)`` while the manager speaks ``(tokens, experts)``,
    so the scores are flattened for the call and the ids restored after.
    """
    with torch.autocast(device_type=x_BLD.device.type, dtype=torch.float32):
        scores_BLE = self.gate(x_BLD)

    if self.score_func == "sigmoid":
        scores_BLE = torch.sigmoid(scores_BLE)
    elif self.score_func == "softmax":
        scores_BLE = F.softmax(scores_BLE, dim=-1)
    else:
        raise NotImplementedError(f"Unknown score function {self.score_func}")

    scores_for_choice_BLE = scores_BLE if expert_bias_E is None else scores_BLE + expert_bias_E
    if self.num_expert_groups is not None:
        scores_for_choice_BLE = self._get_node_limited_routing_scores(scores_for_choice_BLE)

    b, seq_len, _ = scores_for_choice_BLE.shape
    topk_expert_ids_BLK = self._miles_replay_topk(
        scores_for_choice_BLE.reshape(b * seq_len, -1), self.top_k
    ).reshape(b, seq_len, self.top_k)

    # The gating values come from the model's own scores, so a replayed run
    # never reuses the rollout's probabilities -- only its expert choice.
    topk_scores_BLK = scores_BLE.gather(dim=-1, index=topk_expert_ids_BLK)

    if self.route_norm:
        denominator = topk_scores_BLK.sum(dim=-1, keepdim=True) + 1e-20
        topk_scores_BLK = topk_scores_BLK / denominator
    topk_scores_BLK = topk_scores_BLK * self.route_scale

    return topk_scores_BLK, topk_expert_ids_BLK, scores_BLE


_BYPASS_ATTR = "_miles_replay_bypass_next"
_INSTALLED_ATTR = "_miles_replay_installed"


def install(model_parts: list[nn.Module]) -> int:
    """Install R3 on every TokenChoiceTopKRouter and return the number of streams.

    Returns 0 without touching the model when R3 is off. Call for the actor
    only: a second registration would double the manager's stream list and
    invalidate every ``stream_idx``. Streams are keyed by the router's
    decoder-layer index, taken from the module path -- a pipeline stage's
    ``layers`` keys keep their global indices, so PP needs no special case.
    """
    if not routing_replay_manager.enabled:
        return 0

    from torchtitan.models.common.moe import TokenChoiceTopKRouter

    routers: list[tuple[int, nn.Module]] = []
    for part in model_parts:
        for name, module in part.named_modules():
            if not isinstance(module, TokenChoiceTopKRouter):
                continue
            layer_key = next((p for p in name.split(".") if p.isdigit()), None)
            if layer_key is None:
                raise ValueError(f"cannot derive a decoder-layer index from router path {name!r}")
            routers.append((int(layer_key), module))

    if not routers:
        raise ValueError(
            "routing replay is enabled but this model has no torchtitan TokenChoiceTopKRouter; "
            "R3 applies to MoE models only"
        )

    for part in model_parts:
        _bracket_real_forward(part)
        setattr(part, _INSTALLED_ATTR, True)

    for layer_idx, router in sorted(routers, key=lambda pair: pair[0]):
        router._miles_replay_topk = routing_replay_manager.get_topk_fn(
            lambda scores, k: torch.topk(scores, k, dim=-1, sorted=False)[1], return_probs=False
        )
        router.forward = types.MethodType(_token_router_forward, router)
        routing_replay_manager.register_to_module(router, "routing_replay", stream_idx=layer_idx)

    indices = sorted(idx for idx, _ in routers)
    logger.info(
        f"[titan routing_replay] registered {len(routers)} MoE layers "
        f"(global indices {indices[0]}..{indices[-1]})"
    )
    return len(routers)


def _is_installed(model_parts: list[nn.Module]) -> bool:
    """Whether these parts are the ones replaying.

    Only the actor's model gets routers rebound; the reference model runs its
    own routing. Both are TitanTrainers driving the same shared manager, so the
    replay hooks have to tell them apart.
    """
    return routing_replay_manager.enabled and all(getattr(part, _INSTALLED_ATTR, False) for part in model_parts)


def bypass_next_forward(model_parts: list[nn.Module]) -> None:
    """Let each part's next forward compute its own routing instead of replaying.

    The pipeline schedule infers its send/recv metadata by running one real
    forward per stage over microbatch 0's shapes, discarding everything but the
    output shapes. That forward reaches the routers, and a queue is consumed by
    whoever asks -- so it would take microbatch 0's routing and leave every real
    microbatch reading its predecessor's. Adjacent microbatches inside a GRPO
    group share a prompt, which is why the damage looked like "the prompt
    replays correctly and the response does not".

    Torch reruns the inference on every eval-to-train switch, so this is not a
    once-per-job event: an RL step does a log-prob pass and then a training
    pass, and each one re-infers.
    """
    if not _is_installed(model_parts):
        return
    for part in model_parts:
        setattr(part, _BYPASS_ATTR, True)


@contextlib.contextmanager
def consumption_guard(model_parts: list[nn.Module], expected: int):
    """Assert the pass read exactly one queue entry per microbatch.

    The replay is only correct if entry k is read by microbatch k, and nothing
    in the mechanism enforces it: a stray forward shifts every later lookup
    silently. This turns that shift into a failure in the pass that caused it.

    The queues are filled once per rollout and read across its optimizer steps,
    so what a single pass can be held to is the *advance*, not the position.
    """
    if not _is_installed(model_parts):
        yield
        return
    before = {id(replay): replay.forward_index for replay in routing_replay_manager.replays}
    yield
    for replay in routing_replay_manager.replays:
        advance = replay.forward_index - before[id(replay)]
        if advance != expected:
            raise RuntimeError(
                f"routing replay stream {replay.stream_idx} advanced {advance} times over a pass "
                f"of {expected} microbatches; the queues no longer line up with the microbatches"
            )


def _bracket_real_forward(part: nn.Module) -> None:
    """Make a model part's own forward draw from the forward cursor.

    The manager keeps two cursors over the same recorded routing: a training
    step runs under ``replay_backward`` so that activation-checkpoint recompute
    -- which re-runs each *block* during backward -- has its own cursor, while
    the step's real forward must still read the forward cursor. Bracketing the
    top-level forward separates them: recompute happens inside backward, i.e.
    outside this call, and keeps the backward cursor. Without this both draw
    from the backward cursor and it advances twice per microbatch.

    Only ``replay_backward`` is promoted; ``fallthrough`` (the reference model)
    and ``record`` must reach the routers unchanged.

    ``functools.wraps`` is load-bearing, not cosmetic: callers introspect the
    model's forward signature to decide which family kwargs to pass (qwen3_5
    dereferences ``special_tokens`` unconditionally), and a bare
    ``*args, **kwargs`` wrapper hides those parameters.
    """
    inner = part.forward

    @functools.wraps(inner)
    def forward(*args, **kwargs):
        if getattr(part, _BYPASS_ATTR, False):
            setattr(part, _BYPASS_ATTR, False)
            with stage(FALLTHROUGH):
                return inner(*args, **kwargs)
        if routing_replay_manager.stage == REPLAY_BACKWARD:
            with stage(REPLAY_FORWARD):
                return inner(*args, **kwargs)
        return inner(*args, **kwargs)

    part.forward = forward


def fill(args, model_parts, data_iterators, num_microbatches, rollout_data, pad_to: int | None = None) -> None:
    """Load the rollout's routing into the per-layer replay queues.

    Takes the iterator list rather than a single iterator: ``fill_replay_data``
    resets every element and reads through element 0, so call before the caller
    unwraps it.

    ``pad_to`` extends every queued entry to that many tokens. Pipeline
    parallelism needs one shape for the whole run, so the trainer pads each
    microbatch to the configured sequence length; the routing queues are filled
    from the rollout at the microbatch's own length and would otherwise be
    shorter than the scores the routers see. The extension is -1, which the
    replay manager already treats as padding (it substitutes arange so the
    lookup stays in range) and which the loss never reads.
    """
    if not uses_rollout_replay(args):
        return

    fill_replay_data(
        args=args,
        models=model_parts,
        data_iterator=data_iterators,
        num_microbatches=num_microbatches,
        rollout_data=rollout_data,
        data_key=routing_replay_manager.data_key,
        replay_list=routing_replay_manager.replays,
        register_replay_list_func=routing_replay_manager.register_replay_list_func,
        if_sp_region=routing_replay_manager.if_sp_region,
        indices_are_token_positions=routing_replay_manager.replay_indices_are_token_positions,
    )

    if pad_to is None:
        return
    for replay in routing_replay_manager.replays:
        for i, entry in enumerate(replay.top_indices_list):
            missing = pad_to - entry.shape[0]
            if missing < 0:
                raise ValueError(
                    f"routing queue entry has {entry.shape[0]} tokens, more than the "
                    f"{pad_to}-token shape the pipeline schedule exchanges"
                )
            if missing:
                pad = torch.full((missing, *entry.shape[1:]), -1, dtype=entry.dtype)
                replay.top_indices_list[i] = torch.cat([entry, pad], dim=0)


def log_prob_stage(args) -> str:
    """Stage for the actor log-prob pass.

    Rollout replay consumes the queues filled from the rollout; the
    record-then-replay variant has nothing to consume yet and records instead.
    """
    if not routing_replay_manager.enabled:
        return FALLTHROUGH
    return REPLAY_FORWARD if uses_rollout_replay(args) else RECORD


class stage:
    """Run a block with the replay manager in ``name``, restoring it after.

    Nesting a ``replay_forward`` forward inside a ``replay_backward`` step is
    what lets activation-checkpoint recompute draw from the independent
    backward cursor.
    """

    def __init__(self, name: str):
        self.name = name
        self._previous: str | None = None

    def __enter__(self):
        self._previous = routing_replay_manager.stage
        routing_replay_manager.stage = self.name
        return self

    def __exit__(self, *exc):
        routing_replay_manager.stage = self._previous
        return False


def rewind() -> None:
    """Return the forward cursors to the head of their queues."""
    routing_replay_manager.clear_all_forward()


def reset() -> None:
    """Drop the recorded routing once the rollout is done training."""
    routing_replay_manager.clear_all()
