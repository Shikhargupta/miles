"""Role-separated construction of the rollout plane
(codex-rollout-fullparameter-design-0810 §4.3/§4.8).

Consumer-facing names are fixed NOW to the roles PR #1842 will ship —
``inference_controller`` (engine/router/weight-update ownership) and
``rollout_executor`` (rollout-fn execution/conversion) — while the current
concretes are ``Legacy...Adapter`` views over ONE combined RolloutManager
actor. When the split lands, only ``create_rollout_components`` changes:
construct the real InferenceController and RolloutExecutor (behind a thin
adapter if their invocation shape differs), and every call site keeps its
role variable. Deliberately not named ``InferenceController``/
``RolloutExecutor`` (the future classes must not collide) and not ``_tbd``
(Legacy states what the object actually is and when it dies).

The ports carry only what the tinker driver needs — no copy of the full
future public surface, and sampling/scoring never enters the executor."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class InferenceEndpoint:
    """Where sampling requests go (the SGLang router)."""

    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class InferenceControllerPort(Protocol):
    async def get_inference_endpoint(self) -> InferenceEndpoint: ...


class RolloutExecutorPort(Protocol):
    async def generate(self, rollout_id: int): ...


class RolloutLifecyclePort(Protocol):
    async def dispose_once(self) -> None: ...


class LegacyInferenceControllerAdapter:
    """Inference-owner role view over the combined RolloutManager. ``manager``
    stays reachable for the engine/weight-update plumbing that still wires the
    raw actor handle into the training actors (create_training_models);
    PR #1842's controller will own that wiring itself."""

    def __init__(self, manager) -> None:
        self.manager = manager

    async def get_inference_endpoint(self) -> InferenceEndpoint:
        host, port = await self.manager.get_router_address.remote()
        return InferenceEndpoint(host=host, port=port)


class LegacyRolloutExecutorAdapter:
    """Execution role view over the same combined RolloutManager."""

    def __init__(self, manager) -> None:
        self._manager = manager

    async def generate(self, rollout_id: int):
        return await self._manager.generate.remote(rollout_id)


class LegacyRolloutLifecycle:
    """Exactly-once disposal of the SHARED underlying actor: two role views
    must never each dispose the same manager."""

    def __init__(self, manager) -> None:
        self._manager = manager
        self._disposed = False

    async def dispose_once(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        await self._manager.dispose.remote()


@dataclass
class RolloutComponents:
    inference_controller: InferenceControllerPort
    rollout_executor: RolloutExecutorPort
    lifecycle: RolloutLifecyclePort
    num_rollout_per_epoch: int | None

    async def dispose(self) -> None:
        await self.lifecycle.dispose_once()


def create_rollout_components(args, pg) -> RolloutComponents:
    """The one construction seam: today it builds one RolloutManager and two
    role views over it; after PR #1842 it builds the real controller/executor
    pair — call sites never change."""
    from miles.ray.placement_group import create_rollout_manager

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pg)
    return RolloutComponents(
        inference_controller=LegacyInferenceControllerAdapter(rollout_manager),
        rollout_executor=LegacyRolloutExecutorAdapter(rollout_manager),
        lifecycle=LegacyRolloutLifecycle(rollout_manager),
        num_rollout_per_epoch=num_rollout_per_epoch,
    )
