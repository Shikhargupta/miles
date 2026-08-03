"""Eval that consumes exported HF checkpoint snapshots.

Two eval postures exist: against the live training engines (shared, pinned by
blocking call order) or against a checkpoint file (pinned by the file itself).
``CheckpointEvalFn`` is the contract for the second posture — implement it to take
a snapshot directory and return results, however you like. The in-job eval fleet is
not one of these: it delivers weights and leaves generation to the eval fn (see
``miles/ray/rollout/eval_fleet.py``).

Backends never join training weight updates; weights reach them only through a
snapshot exported for a specific rollout_id.
"""

import abc
import copy
import inspect
import logging
from argparse import Namespace

from miles.rollout.base_types import RolloutFnEvalInput, RolloutFnEvalOutput, RolloutFnInput
from miles.utils.misc import load_function

__all__ = [
    "retarget_args",
    "EvalSkip",
    "CheckpointEvalFn",
    "is_checkpoint_eval_fn",
]

logger = logging.getLogger(__name__)


def retarget_args(args: Namespace, router_ip, router_port, num_gpus: int, num_gpus_per_engine: int) -> Namespace:
    """Shallow-copy ``args`` with the router address and GPU sizing swapped for eval.

    Generate functions read the router from ``args`` and ``GenerateState`` sizes its
    semaphore off the GPU counts, so a retargeted copy runs the standard eval path
    against a different set of engines unchanged.
    """
    eval_args = copy.copy(args)
    eval_args.sglang_router_ip = router_ip
    eval_args.sglang_router_port = router_port
    eval_args.rollout_num_gpus = num_gpus
    eval_args.rollout_num_gpus_per_engine = num_gpus_per_engine
    return eval_args


class EvalSkip(Exception):
    """Raise from a ``CheckpointEvalFn`` to skip this eval point with an attributable
    reason (logged as ``eval/skipped_{reason}``) instead of counting as a crash."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class CheckpointEvalFn(abc.ABC):
    """Contract for eval backends that consume HF checkpoint snapshots.

    ``__init__`` prepares everything (launch or attach to your backend); each call
    then receives a snapshot dir + eval info and returns the eval results. The
    trainer owns the rest: per-point snapshot export, async dispatch, overflow
    policy, logging at the snapshot's step, and snapshot GC.

    Subclass and implement ``evaluate_checkpoint``; raise ``EvalSkip(reason)`` to
    skip a point with proper accounting. Point ``--eval-function-path`` at the
    subclass (requires ``train_async.py`` and a snapshot source: ``--eval-hf-dir``
    or ``--save-hf``). See ``examples/fully_async/external_eval_fn.py`` for a full
    implementation against an external sglang server.

    The in-job eval fleet is not one of these: it delivers weights and leaves
    generation to the eval fn, so it is mutually exclusive with this contract.
    """

    @abc.abstractmethod
    async def evaluate_checkpoint(self, checkpoint_dir: str, input: RolloutFnEvalInput) -> RolloutFnEvalOutput: ...

    async def __call__(self, input: RolloutFnInput) -> RolloutFnEvalOutput:
        assert input.evaluation, "CheckpointEvalFn only serves eval; keep the train fn on --rollout-function-path"
        assert input.hf_dir is not None, (
            "no snapshot was dispatched — checkpoint eval fns require train_async.py "
            "and a snapshot source (--eval-hf-dir or --save-hf)"
        )
        return await self.evaluate_checkpoint(input.hf_dir, input)

    def dispose(self) -> None:  # noqa: B027 — optional hook, deliberately a no-op default
        """Tear down anything launched in ``__init__``. Called by RolloutManager.dispose()."""


def is_checkpoint_eval_fn(eval_function_path: str | None) -> bool:
    """Whether ``--eval-function-path`` points at a black-box checkpoint backend."""
    eval_fn = load_function(eval_function_path)
    return inspect.isclass(eval_fn) and issubclass(eval_fn, CheckpointEvalFn)
