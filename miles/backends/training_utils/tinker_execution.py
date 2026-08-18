"""Compatibility imports for the renamed training-operation execution seam.

Tinker is a protocol adapter, while these optimizer commands are shared
execution semantics. New code should import :mod:`operation_execution`.
"""

from miles.backends.training_utils.operation_execution import (
    ADAM_PARAM_DEFAULTS,
    ParameterExecutor,
    StepRequest,
    reset_grad_metadata_keep_grads,
    resolve_adam_params,
    run_optim_controls,
)
from miles.utils.tinker_backend import BatchExecutionLease, BindingT

__all__ = [
    "ADAM_PARAM_DEFAULTS",
    "BatchExecutionLease",
    "BindingT",
    "ParameterExecutor",
    "StepRequest",
    "reset_grad_metadata_keep_grads",
    "resolve_adam_params",
    "run_optim_controls",
]
