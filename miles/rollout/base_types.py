from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.inference_rollout.inference_rollout_common import GenerateState


@dataclass(frozen=True)
class RolloutFnConstructorInput:
    args: Namespace
    # TODO may refactor DataSource API
    data_source: DataSource


@dataclass(frozen=True)
class RolloutFnBaseInput:
    rollout_id: int

    @property
    def evaluation(self):
        raise NotImplementedError


# subclassing for different data in the future
@dataclass(frozen=True)
class RolloutFnTrainInput(RolloutFnBaseInput):
    # engine weight version, None before the first weight update
    weight_version: int | None = None

    @property
    def evaluation(self):
        return False


@dataclass(frozen=True)
class RolloutFnEvalInput(RolloutFnBaseInput):
    generate_state: GenerateState | None = None
    weight_version: str | None = None
    hf_dir: str | None = None

    @property
    def evaluation(self):
        return True


@dataclass(frozen=True)
class RolloutFnHandoff:
    """Opaque fn-to-driver sidecar of one train batch (same species as
    RolloutPostprocessOptions: the fn declares, the manager forwards). The fn
    fills ``driver_metadata`` with whatever its driver needs to finalize the
    batch (e.g. claimed operation ids plus a dispatch lease); the manager
    copies it onto the returned pack verbatim and never inspects a key.

    The same object is the abort token: when the manager's downstream phase
    (save/log/convert/split/store) fails AFTER the fn handed its output over,
    the manager gives the handoff back through the fn's optional
    ``abort_handoff`` capability so the fn can terminalize the claimed work it
    can no longer retry — without it, the failure would orphan state only the
    fn knows about (external review 0813 §4.1)."""

    driver_metadata: dict[str, Any]


class RolloutFnHandoffAborter(Protocol):
    """Optional rollout-fn capability: terminalize the work behind a handoff
    when the downstream phase fails after the output receipt. Must be safe to
    repeat (the manager may race a retry against teardown)."""

    async def abort_handoff(self, handoff: RolloutFnHandoff, error: BaseException) -> None: ...


@dataclass(frozen=True)
class RolloutPostprocessOptions:
    """Postprocess policy the rollout fn declares for its own output, so the
    generic manager never has to recognize fn-specific metadata keys.

    pad_to_dp: zero-weight pad the flat sample list up to the DP grid instead
    of trimming — for whole-batch selections (e.g. tinker client operations)
    where dropping samples would corrupt the result plane.
    """

    pad_to_dp: bool = False


# TODO make it frozen
@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None
    # Fn-internal control plane (e.g. the tinker child's per-operation info);
    # the rollout manager does not read it.
    metadata: dict[str, Any] | None = None
    # Conversion-metadata contribution: the rollout manager merges this dict
    # verbatim into the postprocess metadata handed to train-data conversion
    # (e.g. the tinker adapter ships its BatchPlan already converted), never
    # interpreting individual keys.
    conversion_metadata: dict[str, Any] | None = None
    # How the manager postprocesses samples before conversion.
    postprocess: RolloutPostprocessOptions = field(default_factory=RolloutPostprocessOptions)
    # Opaque driver-facing sidecar (dispatch identity + abort token); the
    # manager forwards it to the driver and hands it back to the fn's
    # abort_handoff on a downstream failure. None for fns with no
    # driver-visible dispatch state.
    handoff: RolloutFnHandoff | None = None


# TODO make it frozen
@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


RolloutFnInput = RolloutFnTrainInput | RolloutFnEvalInput
RolloutFnOutput = RolloutFnTrainOutput | RolloutFnEvalOutput


@dataclass(frozen=True)
class GenerateFnInput:
    state: GenerateState
    sample: Sample
    sampling_params: dict[str, Any]
    evaluation: bool

    @property
    def args(self) -> Namespace:
        return self.state.args


@dataclass(frozen=True)
class GenerateFnOutput:
    # One generate may lead to multiple samples, such as multi-agent, tree-like exploration, or
    # multi-turn with removing thinking tokens.
    samples: Sample | list[Sample]


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    """Legacy rollout function call interface. Used when MILES_USE_LEGACY_ROLLOUT_V1 is enabled."""
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output
