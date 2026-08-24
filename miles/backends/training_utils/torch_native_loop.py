"""The microbatch loops shared by the torch-native training backends.

FSDP and torchtitan both drive their own microbatch loop: fetch a microbatch,
call the model, hand the logits to miles' loss hub, and step. Megatron does not
appear here on purpose -- its loop lives inside the pipeline schedule
(``get_forward_backward_func``), which owns the microbatch ordering under
1F1B/interleaving, so the smallest thing it can expose is a whole optimizer step.

Backends supply two callables and keep everything backend-specific inside them:

``forward_fn(batch) -> logits``
    Runs the model. FSDP wraps its precision context, routing-replay stage, and
    multimodal inputs in here; torchtitan passes positions through to titan's
    attention. The loop never sees any of it.

``step_fn() -> StepMetrics``
    Clips gradients, steps the optimizer and LR scheduler, and reports the
    grad norm plus any per-group LR metrics.
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
    forward_fn: Callable[[dict], torch.Tensor],
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

    with timer(f"{store_prefix}log_probs"):
        for step_id in range(len(num_microbatches)):
            iterator = tqdm(
                range(num_microbatches[step_id]),
                desc=f"{store_prefix}log_probs",
                disable=dist.get_rank() != 0,
            )
            for _ in profiler.iterate_train_log_probs(iterator):
                batch = _fetch(args, data_iterator, FORWARD_ONLY_KEYS)
                result = get_log_probs_and_entropy(
                    logits=forward_fn(batch),
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
                forward_store.append(entry)

    return aggregate_forward_results(forward_store, data_iterator, args, store_prefix)


def run_optimizer_steps(
    args: Namespace,
    rollout_id: int,
    data_iterator: DataIterator,
    num_microbatches: list[int],
    forward_fn: Callable[[dict], torch.Tensor],
    step_fn: Callable[[], StepMetrics],
    *,
    profiler,
    role: str = "actor",
) -> None:
    """Run one optimizer step per entry in ``num_microbatches``.

    ``apply_megatron_loss_scaling`` is False throughout: that scaling exists to
    cancel the reduction Megatron's pipeline schedule applies, and nothing here
    goes through a schedule.
    """
    data_iterator.reset()
    num_steps = len(num_microbatches)

    for step_id in range(num_steps):
        losses_reduced = []
        iterator = tqdm(range(num_microbatches[step_id]), desc=f"{role}_train", disable=dist.get_rank() != 0)
        for _ in profiler.iterate_train_actor(iterator):
            batch = _fetch(args, data_iterator, TRAIN_KEYS)
            loss, _normalizer, log_dict = loss_function(
                args=args,
                batch=batch,
                num_microbatches=num_microbatches[step_id],
                logits=forward_fn(batch),
                apply_megatron_loss_scaling=False,
            )
            loss.backward()
            losses_reduced.append(log_dict)

        metrics = step_fn()

        if args.ci_test:
            check_grad_norm(
                args=args,
                grad_norm=metrics.grad_norm,
                rollout_id=rollout_id,
                step_id=step_id,
                role=role,
                rank=get_parallel_state().intra_dp_cp.rank,
            )

        log_train_step(
            args=args,
            loss_dict=aggregate_train_losses(losses_reduced),
            grad_norm=metrics.grad_norm,
            rollout_id=rollout_id,
            step_id=step_id,
            num_steps_per_rollout=num_steps,
            role=role,
            extra_metrics=metrics.extra_metrics,
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
