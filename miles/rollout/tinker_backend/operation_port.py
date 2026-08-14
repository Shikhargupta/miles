"""Operation-queue and residency transports for the tinker rollout adapter
(codex-rollout-fullparameter-design-0810 §4.5).

The adapter's scheduling logic (RR, coalesce, kind lock, whole-batch
selection) talks to these narrow ports; ONLY the Ray concretes below know
``get_tinker_controller()``, ``.remote()`` and ``ray.get`` — a future
RolloutExecutor injects its own transports and the adapter's policy code
never changes, and unit tests drive the scheduler with fakes instead of a
Ray cluster."""

import asyncio
from typing import Protocol

import ray

from miles.utils.tinker_backend import BindingT, RegistrationKey


class TransientOperationPortError(RuntimeError):
    """A port call failed in a way KNOWN not to have mutated the operation
    ledger (e.g. the controller lookup failed before any RPC was sent, or the
    remote method is read-only/non-mutating by contract). Safe to retry after
    a backoff. A failure that MAY have mutated the ledger (a claim RPC whose
    response was lost) must NOT be wrapped in this type: retrying such a
    stream would find an already-CLAIMED head and poll forever while hiding
    the orphan (external review 0813 §4.3)."""


class StaleBindingError(RuntimeError):
    """The controller executed the batch-lease acquisition and REFUSED it: at
    least one claimed operation's registration no longer owns its execution
    binding (deregistered/re-registered after the claim). Authoritative and
    terminal for the refused receipt — never retried; the exact stale claims
    are terminal-failed instead (external review 0813 §4.6)."""


class OperationQueuePort(Protocol[BindingT]):
    """Claims against the backend's operation ledger.

    ``ready_streams`` lists the current READY registration streams (keyed by
    name, valued by the controller's run views) — these are streams, not
    unclaimed operation candidates: a stream's head kind is unknown until
    claimed. ``claim_data`` is claim-and-bind in ONE backend actor call: the
    exact READY binding resolves first, only then does the ledger turn the
    head CLAIMED, and the returned claim carries the binding; a missing
    binding leaves the head QUEUED."""

    async def ready_streams(self) -> dict: ...

    async def claim_data(self, key: RegistrationKey) -> dict | None: ...

    async def fail(self, operation_id: str, error: str, category: str) -> None: ...


class BatchResidencyPort(Protocol[BindingT]):
    """Selection-side view of the trainer-residency facade: after RR/coalesce
    picks a selection, acquire ONE immutable dispatch receipt for its
    already-claimed bindings. (The synchronous port lives controller-side —
    miles/utils/tinker_backend.TrainerResidencyPort; this is its async
    transport face.)"""

    async def acquire_batch(self, bindings_by_operation: list) -> object: ...


class BatchAbortPort(Protocol):
    """Abnormal-outcome finalizer for claimed operations that will never reach
    the trainer: terminal-fail the still-CLAIMED operations typed server and
    release the batch lease (``lease_metadata=None`` when no lease was
    acquired yet). One idempotent controller boundary — the same
    ``fail_tinker_batch`` the driver's train finalizer uses: it fails only
    still-CLAIMED operations and releases the lease in ``finally``, so
    repeating it (or racing it against a commit) can never overwrite a landed
    terminal result."""

    async def abort_batch(self, operation_ids: list[str], error: str, lease_metadata: dict | None) -> None: ...


class RayTinkerOperationQueue:
    """Only this class (and its residency sibling) knows get_tinker_controller(),
    .remote(), and ray.get."""

    async def ready_streams(self) -> dict:
        from miles.ray.tinker_backend.controller import get_tinker_controller

        snapshot = await asyncio.to_thread(ray.get, get_tinker_controller().snapshot.remote())
        return snapshot["ready"]

    async def claim_data(self, key: RegistrationKey) -> dict | None:
        from miles.ray.tinker_backend.controller import get_tinker_controller

        name, registration_id = key
        try:
            controller = get_tinker_controller()
        except Exception as e:
            # The actor lookup never reached the controller: provably no
            # ledger mutation, so the child may retry after a backoff.
            raise TransientOperationPortError(f"tinker controller unavailable: {e}") from e
        # A failure of the claim RPC itself is left UNCLASSIFIED on purpose:
        # claim-and-bind mutates the ledger, and a lost response cannot be
        # disambiguated locally (the head may already be CLAIMED). The
        # runtime quarantines (FAILED) until reconciliation/deregistration;
        # a controller-side idempotent-claim query is the future fix.
        return await asyncio.to_thread(ray.get, controller.claim_data_operation.remote(name, registration_id))

    async def fail(self, operation_id: str, error: str, category: str) -> None:
        from miles.ray.tinker_backend.controller import get_tinker_controller

        await asyncio.to_thread(ray.get, get_tinker_controller().fail_operation.remote(operation_id, error, category))


class RayTrainerResidencyPort:
    """Thin async proxy to the backend-owned FixedSlotResidency."""

    async def acquire_batch(self, bindings_by_operation: list) -> object:
        from miles.ray.tinker_backend.controller import get_tinker_controller

        try:
            return await asyncio.to_thread(
                ray.get, get_tinker_controller().acquire_batch_lease.remote(list(bindings_by_operation))
            )
        except ray.exceptions.RayTaskError as e:
            # The controller EXECUTED and raised: acquire_batch_lease is a
            # pure validate+mint (it never mutates), so an application error
            # is an authoritative refusal of these bindings, not a transport
            # blip. Anything else (actor lookup/transport) propagates raw and
            # is retryable for the same non-mutating reason.
            raise StaleBindingError(str(e.as_instanceof_cause())) from e


class RayTinkerBatchAbort:
    """BatchAbortPort concrete over the controller's idempotent
    ``fail_tinker_batch`` boundary."""

    async def abort_batch(self, operation_ids: list[str], error: str, lease_metadata: dict | None) -> None:
        from miles.ray.tinker_backend.controller import get_tinker_controller

        await asyncio.to_thread(
            ray.get,
            get_tinker_controller().fail_tinker_batch.remote(list(operation_ids), error, lease_metadata),
        )
