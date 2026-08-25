"""Bridges between released torch and the torch-nightly APIs torchtitan tracks.

torchtitan pins nightly torch; miles pins the torch that sglang's kernels link
against (2.13.0 as of sglang v0.5.18), so the backend permanently straddles the
two. Each shim here backports a small upstream torch fix and is a no-op once
the running torch already carries the upstream form -- so bumping torch
retires shims without code changes here.

The schedule-API gap (nightly ``step(arg_mbs=...)`` vs 2.13's whole-batch
``step()``) is bridged in the engine instead, where the schedule object lives.
"""

import inspect
import logging

logger = logging.getLogger(__name__)


def install() -> None:
    _patch_fsdp2_grad_accumulation_attr_error()
    _patch_pipeline_schedule_microbatch_api()


def _patch_pipeline_schedule_microbatch_api() -> None:
    """Backport nightly's per-microbatch-list schedule API to released torch.

    torchtitan's trainer and validator drive the pipeline schedule with
    ``step(arg_mbs=..., kwarg_mbs=..., target_mbs=..., losses=...,
    loss_kwargs=...)`` -- pre-split microbatch lists, which is the only shape
    that works for packed variable-length batches. Released torch 2.13 only
    has the whole-batch ``step(*args, **kwargs)`` (it would try to re-split
    the lists as model inputs); its internal ``_step_microbatches`` is the
    same code nightly's step() calls, so this patch adds the nightly signature
    on top of it, replicating step()'s per-iteration bookkeeping.
    """
    import torch

    # The schedule classes through torchtitan's surface: its pipeline module
    # re-exports the torch schedules it builds, and that is the same object the
    # trainer will call step() on.
    from torchtitan.distributed.pipeline_parallel import PipelineScheduleMulti, PipelineScheduleSingle

    if "arg_mbs" in inspect.signature(PipelineScheduleSingle.step).parameters:
        return  # this torch already has the microbatch-list API

    def _make_step(original_step):
        def step(self, *args, arg_mbs=None, kwarg_mbs=None, target_mbs=None, target=None,
                 losses=None, return_outputs=True, loss_kwargs=None, **kwargs):
            if arg_mbs is None and kwarg_mbs is None and target_mbs is None:
                return original_step(
                    self, *args, target=target, losses=losses,
                    return_outputs=return_outputs, loss_kwargs=loss_kwargs, **kwargs,
                )
            if (
                self._has_backward
                and getattr(self, "_backward_requires_autograd", True)
                and not torch.is_grad_enabled()
            ):
                raise RuntimeError(
                    "step() requires gradients to be enabled for backward computation; "
                    "call eval() under torch.no_grad() instead."
                )
            stages = getattr(self, "_stages", None) or [self._stage]
            for stage in stages:
                stage.has_backward = self._has_backward
                stage.clear_runtime_states()
            return self._step_microbatches(
                arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs, loss_kwargs=loss_kwargs
            )

        return step

    PipelineScheduleSingle.step = _make_step(PipelineScheduleSingle.step)
    PipelineScheduleMulti.step = _make_step(PipelineScheduleMulti.step)
    logger.info("Patched pipeline schedules with nightly's microbatch-list step API")


def _patch_fsdp2_grad_accumulation_attr_error() -> None:
    """Backport pytorch/pytorch's getattr guard in ``to_accumulated_grad_if_needed``.

    torch 2.13.0's body dereferences ``self._unsharded_param`` directly, but the
    attribute only exists between ``init_unsharded_param`` and
    ``free_unsharded_param``: a parameter that was resharded (or never
    all-gathered) before a second backward has no unsharded gradient to upcast,
    and upstream's fixed body returns early for it. A pipeline schedule's
    back-to-back backwards on non-last stages hit exactly that window and crash
    with AttributeError on the unpatched body.

    This is the one deliberate torch-internal import in the backend: the patch
    target is torch itself (torchtitan never touches FSDPParam), so there is no
    torchtitan surface to reach it through.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

    if "getattr" in inspect.getsource(FSDPParam.to_accumulated_grad_if_needed):
        return  # this torch already has the upstream fix

    def to_accumulated_grad_if_needed(self) -> None:
        unsharded_param = getattr(self, "_unsharded_param", None)
        if (
            self.reduce_dtype is None
            or unsharded_param is None
            or unsharded_param.grad is None
            or unsharded_param.grad.dtype == self.reduce_dtype
        ):
            return
        unsharded_grad = unsharded_param.grad
        unsharded_param.grad = None
        self.unsharded_accumulated_grad = unsharded_grad.to(self.reduce_dtype)

    FSDPParam.to_accumulated_grad_if_needed = to_accumulated_grad_if_needed
    logger.info("Patched FSDPParam.to_accumulated_grad_if_needed with the upstream getattr guard")
