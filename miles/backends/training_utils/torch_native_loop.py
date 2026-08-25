"""The optimizer-step loops shared by the torch-native training backends.

The unit a backend implements is one optimizer step's worth of microbatches,
not one forward: under pipeline parallelism the schedule owns the microbatch
ordering, so a per-microbatch hook cannot exist there (torchtitan settled on
the same seam for its trainer in pytorch/torchtitan#3856). Backends supply a
*step runner*:

``forward_only_step(batches, compute) -> list``
    No-grad pass; calls ``compute(logits, batch)`` per microbatch and returns
    the results (under PP, only where logits exist).

``forward_backward_step(batches, loss_closure) -> list[dict]``
    Forward+backward for one optimizer step; ``loss_closure(logits, batch) ->
    (loss, log_dict)``. Returns the log dicts.

``zero_grad()`` / ``apply_step() -> StepMetrics``
    Bracket the step: clear grads before, then clip + optimizer + LR after.

``batches`` is passed as an iterator so a linear runner can keep fetch and
compute interleaved (a schedule-based runner drains it up front instead).
``LinearStepRunner`` is the no-schedule implementation FSDP uses and
schedule-owning backends fall back to when their schedule is off. Megatron does
not appear here on purpose: its whole step lives inside
``get_forward_backward_func``.
"""

import logging
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.training_utils.ci_utils import check_grad_norm
from miles.backends.training_utils.data import DataIterator, get_batch
from miles.backends.training_utils.log_utils import (
    aggregate_forward_results,
    aggregate_train_losses,
    log_train_step,
)
from miles.backends.training_utils.loss import get_log_probs_and_entropy, loss_function
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.timer import timer

logger = logging.getLogger(__name__)

FORWARD_ONLY_KEYS = [
    "tokens",
    "loss_masks",
    "multimodal_train_inputs",
    "total_lengths",
    "response_lengths",
    "max_seq_lens",
]
TRAIN_KEYS = FORWARD_ONLY_KEYS + [
    "log_probs",
    "advantages",
    "returns",
    "ref_log_probs",
    "rollout_log_probs",
]


@dataclass
class StepMetrics:
    """What a backend reports back after applying one optimizer step."""

    grad_norm: float
    extra_metrics: dict[str, float] = field(default_factory=dict)


class LinearStepRunner:
    """The step runner for backends without a schedule: microbatches one after
    another, ``loss.backward()`` accumulating grads across them."""

    def __init__(
        self,
        forward_fn: Callable[[dict], torch.Tensor],
        zero_grad_fn: Callable[[], None] | None = None,
        step_fn: Callable[[], StepMetrics] | None = None,
    ):
        self._forward = forward_fn
        self._zero_grad = zero_grad_fn
        self._step = step_fn

    def forward_only_step(self, batches, compute: Callable) -> list:
        return [compute(self._forward(batch), batch) for batch in batches]

    def forward_backward_step(self, batches, loss_closure: Callable) -> list[dict]:
        logs = []
        for batch in batches:
            loss, log_dict = loss_closure(self._forward(batch), batch)
            loss.backward()
            logs.append(log_dict)
        return logs

    def zero_grad(self) -> None:
        self._zero_grad()

    def apply_step(self) -> StepMetrics:
        return self._step()


def _fetch(args: Namespace, data_iterator: DataIterator, keys: list[str]) -> dict:
    return get_batch(
        data_iterator,
        keys,
        args.data_pad_size_multiplier,
        args.qkv_format,
        get_position_ids=True,
    )


@torch.no_grad()
def run_log_probs(
    args: Namespace,
    data_iterator: DataIterator,
    num_microbatches: list[int],
    runner,
    *,
    profiler,
    store_prefix: str = "",
) -> dict[str, list[torch.Tensor]]:
    """No-grad pass over the rollout, collecting token log probs (and entropy).

    Entropy is only collected for the actor pass (``store_prefix == ""``), which
    is what the loss hub consumes; the reference/teacher passes only need log
    probs.
    """
    forward_store: list[dict] = []
    data_iterator.reset()

    def compute(logits: torch.Tensor, batch: dict) -> dict:
        result = get_log_probs_and_entropy(
            logits=logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            with_entropy=(store_prefix == ""),
            max_seq_lens=batch.get("max_seq_lens"),
        )
        entry = {f"{store_prefix}log_probs": result["log_probs"]}
        if store_prefix == "" and "entropy" in result:
            entry["entropy"] = result["entropy"]
        return entry

    with timer(f"{store_prefix}log_probs"):
        for step_id in range(len(num_microbatches)):
            iterator = tqdm(
                range(num_microbatches[step_id]),
                desc=f"{store_prefix}log_probs",
                disable=dist.get_rank() != 0,
            )

            def batches():
                for _ in profiler.iterate_train_log_probs(iterator):
                    yield _fetch(args, data_iterator, FORWARD_ONLY_KEYS)

            forward_store.extend(runner.forward_only_step(batches(), compute))

    return aggregate_forward_results(forward_store, data_iterator, args, store_prefix)


def run_optimizer_steps(
    args: Namespace,
    rollout_id: int,
    data_iterator: DataIterator,
    num_microbatches: list[int],
    runner,
    *,
    profiler,
    role: str = "actor",
) -> None:
    """Run one optimizer step per entry in ``num_microbatches``.

    ``apply_megatron_loss_scaling`` is False throughout: that scaling exists to
    cancel the reduction Megatron's pipeline schedule applies, and the loss
    normalization here is already whole-step (``num_microbatches``-aware) so a
    schedule-based runner needs no extra scaling either.
    """
    data_iterator.reset()
    num_steps = len(num_microbatches)

    for step_id in range(num_steps):
        runner.zero_grad()
        iterator = tqdm(range(num_microbatches[step_id]), desc=f"{role}_train", disable=dist.get_rank() != 0)

        def loss_closure(logits: torch.Tensor, batch: dict, step_id: int = step_id) -> tuple:
            loss, _normalizer, log_dict = loss_function(
                args=args,
                batch=batch,
                num_microbatches=num_microbatches[step_id],
                logits=logits,
                apply_megatron_loss_scaling=False,
            )
            return loss, log_dict

        def batches():
            for _ in profiler.iterate_train_actor(iterator):
                yield _fetch(args, data_iterator, TRAIN_KEYS)

        losses_reduced = runner.forward_backward_step(batches(), loss_closure)

        metrics = runner.apply_step()

        if args.ci_test:
            check_grad_norm(
                args=args,
                grad_norm=metrics.grad_norm,
                rollout_id=rollout_id,
                step_id=step_id,
                role=role,
                rank=get_parallel_state().intra_dp_cp.rank,
            )

        # log_train_step defaults to global rank 0, which under pipeline
        # parallelism is a FIRST-stage rank holding no loss -- it would report
        # grad_norm and the LR and drop every loss metric. The metrics live on
        # the same rank experiment tracking was initialized on: the last stage,
        # tp rank 0, dp-cp rank 0 (the predicate Megatron encodes as
        # is_first_replica_megatron_main_rank).
        state = get_parallel_state()
        log_train_step(
            args=args,
            loss_dict=aggregate_train_losses(losses_reduced),
            grad_norm=metrics.grad_norm,
            rollout_id=rollout_id,
            step_id=step_id,
            num_steps_per_rollout=num_steps,
            role=role,
            extra_metrics=metrics.extra_metrics,
            should_log=(
                state.effective_dp_cp.rank == 0 and state.tp.rank == 0 and state.is_pp_last_stage
            ),
        )


def clip_and_report(parameters, clip_grad: float) -> float:
    """Grad-norm clip shared by the torch-native backends.

    DTensor-sharded models return a DTensor norm that has to be materialized
    before it can be logged.
    """
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
    if hasattr(grad_norm, "full_tensor"):
        grad_norm = grad_norm.full_tensor()
    return float(grad_norm.item())
