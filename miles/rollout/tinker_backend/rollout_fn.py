"""Tinker rollout frontend: one child per registration, each child turning one
claimed client operation into one complete batch. The wrapper selects whole
child batches with a persistent round-robin under a KIND LOCK — a selection is
all forward_backward or all forward, never mixed — and the BatchPlan, shipped
already converted as the output's conversion-metadata contribution, is the
only rollout-to-train control plane.

Nothing here generates: data operations arrive fully tokenized from the
client, and sampling happens against the router directly.
"""

import asyncio
import copy
import logging
import time
from collections import deque
from typing import Any

from miles.ray.tinker_backend.config import AdapterRun
from miles.ray.tinker_backend.residency import lease_to_metadata
from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnHandoff,
    RolloutFnInput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
    RolloutPostprocessOptions,
)
from miles.rollout.tinker_backend.operation_port import (
    BatchAbortPort,
    BatchResidencyPort,
    OperationQueuePort,
    RayTinkerBatchAbort,
    RayTinkerOperationQueue,
    RayTrainerResidencyPort,
    StaleBindingError,
    TransientOperationPortError,
)
from miles.utils.tinker_backend import EmptyBatchTimeoutError
from miles.utils.types import AdapterRef, Sample

logger = logging.getLogger(__name__)


def batch_plan_to_metadata(batch_plan: list[dict], lease) -> dict[str, Any]:
    """Distill one tinker selection's BatchPlan into conversion metadata.
    Selections are homogeneous: exactly one data-operation kind — mixed
    forward/forward_backward batches are structurally impossible, which is
    what keeps forward operations gradient-free without loss surgery.

    Correlation is batch-local (codex-rollout-fullparameter-design-0810 §3.3):
    each selected operation gets a small integer ``lane`` (its position in the
    selection), and the loss/result plane is keyed by lane — never by trainer
    slot, so operation identity survives any parameterization.

    The batch's ``BatchExecutionLease`` is the single binding truth (§5.3):
    it ships plain-encoded, and the conversion derives ``adapter_slots`` by
    joining ``operation_by_lane`` through it — the plan never stores a second
    copy of the binding."""
    kinds = {entry["operation_kind"] for entry in batch_plan}
    if len(kinds) != 1 or not kinds <= {"forward_backward", "forward"}:
        raise ValueError(f"tinker selection must be one homogeneous data kind, got {sorted(kinds)}")
    metadata: dict[str, Any] = {
        "batch_kind": "tinker",
        # Per-sample lanes in selection order (each entry's rows are contiguous).
        "tinker_operation_lanes": [
            lane for lane, entry in enumerate(batch_plan) for _ in range(entry["sample_count"])
        ],
        "tinker_loss_by_lane": {lane: entry.get("loss_spec") or {} for lane, entry in enumerate(batch_plan)},
        # The trainer completes these operations after the batch lands.
        "operation_by_lane": {lane: entry["operation_id"] for lane, entry in enumerate(batch_plan)},
        # Exact registration per lane: the batch commit dirties these streams,
        # never a trainer-reported name list.
        "registration_by_lane": {
            lane: (entry["name"], entry["registration_id"]) for lane, entry in enumerate(batch_plan)
        },
        # The lease is mandatory: a batch without its dispatch receipt is one
        # the trainer must reject, so the optional path may not exist here.
        "batch_execution_lease": lease_to_metadata(lease),
    }
    if kinds == {"forward"}:
        metadata["tinker_forward_only"] = True
    return metadata


_CLAIM_POLL_S = 0.5

# Known-transient child failures (TransientOperationPortError: provably no
# ledger mutation) return the runtime to IDLE with this capped exponential
# backoff instead of quarantining it.
_CHILD_BACKOFF_BASE_S = 0.5
_CHILD_BACKOFF_CAP_S = 30.0

# Batch-lease acquisition never mutates controller state, so transient
# transport failures are retried in-adapter (bounded) before propagating.
_ACQUIRE_ATTEMPTS = 4
_ACQUIRE_BACKOFF_BASE_S = 0.2
_ACQUIRE_BACKOFF_CAP_S = 2.0

# A refused batch receipt terminal-fails the exact stale claims and reselects
# the survivors; bounded so racing registry churn cannot loop forever.
_MAX_STALE_RESELECTS = 3

Tenant = tuple[str, str]

DATA_OPERATION_KINDS = ("forward_backward", "forward")


class TinkerOperationSource:
    """Per-registration stand-in for a data source: tinker adapters have no
    dataset, so this only carries the child args and the current run view used
    for stamping serving identity."""

    def __init__(self, args, run: AdapterRun):
        self.args = copy.copy(args)
        self.run = run

    def refresh(self, run: AdapterRun) -> None:
        """Serving version advances between batches; identity stays fixed."""
        self.run = run

    def stamp(self, groups: list[list[Sample]]) -> list[list[Sample]]:
        run = self.run
        ref = AdapterRef(
            name=run.name,
            registration_id=run.registration_id,
            serving_version=run.version,
            slot=run.slot,
        )
        for group in groups:
            for sample in group:
                sample.adapter = ref
                sample.metadata = {**run.config.metadata, **sample.metadata}
        return groups

    def save(self, rollout_id) -> None:
        pass

    def load(self, rollout_id=None) -> None:
        pass


class TinkerNullDataSource:
    """The manager-level data source slot for tinker runs. Tinker has no
    dataset — every child pulls from the operation queue — so this only
    satisfies the manager's save/load/close surface."""

    dataset = ()

    def __init__(self, args):
        self.args = args

    def get_samples(self, num_samples: int):
        raise RuntimeError("tinker runs have no dataset; data arrives as client operations")

    def add_samples(self, samples) -> None:
        pass

    def save(self, rollout_id) -> None:
        pass

    def load(self, rollout_id=None) -> None:
        pass


class QueueChildRolloutFn:
    """Awaits the registration's next data-bearing operation and returns it as
    one complete batch. Blocking while the client queue is idle is normal: the
    runtime simply stays IN_FLIGHT and other adapters keep training. Claims go
    through the injected OperationQueuePort — this class knows no Ray."""

    def __init__(self, input: RolloutFnConstructorInput, operations: OperationQueuePort | None = None):
        assert isinstance(input.data_source, TinkerOperationSource)
        self.source: TinkerOperationSource = input.data_source
        self.operations = operations if operations is not None else RayTinkerOperationQueue()

    async def __call__(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        key = (self.source.run.name, self.source.run.registration_id)
        while True:
            operation = await self.operations.claim_data(key)
            if operation is None:
                await asyncio.sleep(_CLAIM_POLL_S)
                continue
            try:
                return self._batch_from_operation(operation)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - a bad payload fails its op, not the adapter
                logger.exception(f"[tinker] ({key[0]}) operation '{operation['operation_id']}' rejected: {e}")
                await self.operations.fail(operation["operation_id"], f"invalid operation payload: {e}", "user")

    def _batch_from_operation(self, operation: dict) -> RolloutFnTrainOutput:
        if operation["kind"] not in DATA_OPERATION_KINDS:
            raise ValueError(f"operation kind '{operation['kind']}' is not a data operation")
        payload = operation.get("payload") or {}
        raw_samples = payload.get("samples")
        if not raw_samples:
            raise ValueError(f"{operation['kind']} payload carries no samples")
        groups: list[list[Sample]] = []
        for i, raw in enumerate(raw_samples):
            raw = dict(raw)
            raw.setdefault("status", Sample.Status.COMPLETED.value)
            # Row identity within the operation is server-owned: the result
            # plane returns per-datum logprobs in this order, and a negative
            # index is the DP-padding sentinel — a client-supplied value could
            # alias it (rows silently dropped) or collide in the collector.
            raw["index"] = i
            groups.append([Sample.from_dict(raw)])
        return RolloutFnTrainOutput(
            samples=self.source.stamp(groups),
            metadata=dict(
                operation_id=operation["operation_id"],
                operation_kind=operation["kind"],
                loss_spec=payload.get("loss"),
                # Fixed binding resolved atomically with the claim (claim-and-
                # bind); the long-lived runtime's AdapterRun.slot is never the
                # dispatch truth.
                binding=operation["binding"],
            ),
        )


class AdapterRolloutRuntime:
    """One per registration: at most one in-flight child call and one ready
    output."""

    IDLE = "IDLE"
    IN_FLIGHT = "IN_FLIGHT"
    READY = "READY"
    SELECTED = "SELECTED"
    FAILED = "FAILED"

    def __init__(self, args, run: AdapterRun, operations: OperationQueuePort | None = None):
        self.run = run
        self.data_source = TinkerOperationSource(args, run)
        child_input = RolloutFnConstructorInput(args=self.data_source.args, data_source=self.data_source)
        self.child_fn = QueueChildRolloutFn(child_input, operations)
        self.state = self.IDLE
        self.ready_output: RolloutFnTrainOutput | None = None
        self.task: asyncio.Task | None = None
        # Known-transient failure recovery: consecutive-failure count and the
        # monotonic deadline before which an IDLE runtime is not relaunched.
        self.transient_failures = 0
        self.retry_at = 0.0

    @property
    def ready_kind(self) -> str | None:
        if self.ready_output is None:
            return None
        return self.ready_output.metadata["operation_kind"]

    def refresh(self, run: AdapterRun) -> None:
        self.run = run
        self.data_source.refresh(run)

    async def aclose(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - teardown must not raise
                pass
        self.task = None


class TinkerOperationBatchAdapter:
    """Operation-to-batch adapter (codex-rollout-fullparameter-design-0810
    §4.5): turns claimed client operations into whole training batches —
    persistent round-robin, homogeneous kind lock, coalesce timeout,
    registration fencing. Transports are injected ports (OperationQueuePort,
    BatchResidencyPort), so a future RolloutExecutor loads this adapter
    unchanged and unit tests need no Ray — "unchanged" is the executor/Ray
    boundary only. The adapter is NOT parameterization-neutral: its runtimes
    build ``TinkerOperationSource``/``AdapterRun`` views and stamp samples
    with ``AdapterRef``, so a full-parameter deployment reuses the operation/
    result semantics but still needs a small sample-stamping extraction here
    (external review 0811: soften, do not pre-build the hook).

    The adapter never samples prompts, never generates, never scores, never
    builds Datums, and never touches residency policy — it only claims,
    selects, and converts."""

    def __init__(
        self,
        input: RolloutFnConstructorInput,
        operations: OperationQueuePort | None = None,
        residency: BatchResidencyPort | None = None,
        abort: BatchAbortPort | None = None,
    ):
        self.args = input.args
        self.operations = operations if operations is not None else RayTinkerOperationQueue()
        self.residency = residency if residency is not None else RayTrainerResidencyPort()
        self.abort = abort if abort is not None else RayTinkerBatchAbort()
        self.runtimes: dict[Tenant, AdapterRolloutRuntime] = {}
        self.rotation: deque[Tenant] = deque()
        self._ready = asyncio.Event()
        self._closed = False

    # ------------------------------ lifecycle ------------------------------

    async def __call__(self, input: RolloutFnInput) -> RolloutFnTrainOutput:
        if input.evaluation:
            raise ValueError(
                "TinkerOperationBatchAdapter does not serve eval; tinker runs have no server-side eval loop"
            )
        if self._closed:
            raise RuntimeError("TinkerOperationBatchAdapter is closed; no new claim work may start")
        # READY streams only: a retiring registration's queued operations are
        # fenced terminal, so a child claim would never return for it.
        adapters = await self.operations.ready_streams()
        await self._reconcile(adapters)
        refusal: StaleBindingError | None = None
        for _ in range(_MAX_STALE_RESELECTS):
            self._launch_idle_children(input.rollout_id)
            selected = await self._select()
            try:
                return await self._merge(selected)
            except StaleBindingError as e:
                # The exact stale claims were terminal-failed inside _merge;
                # the surviving READY batches reselect immediately.
                refusal = e
                continue
        raise refusal

    async def aclose(self) -> None:
        """Claim-safe shutdown (external review 0813 §4.7): stop new claim
        work, then per runtime cancel-and-await its child FIRST — a claim can
        land during the cancellation race — and terminal-fail any claim it
        still holds: a READY output IS a CLAIMED operation with no lease yet,
        so dropping it silently would block its stream forever. Ambiguous
        in-flight claim RPCs (cancelled before any response) cannot be
        reconciled locally; registration fencing/recovery owns those, and a
        controller-side idempotent-claim query is the future fix. Teardown
        never raises."""
        self._closed = True
        for tenant, runtime in list(self.runtimes.items()):
            await runtime.aclose()
            output = runtime.ready_output
            if output is None:
                continue
            operation_id = output.metadata["operation_id"]
            try:
                await self.abort.abort_batch(
                    [operation_id],
                    "rollout adapter closed before the claimed operation could dispatch — "
                    "resubmit it as a new operation",
                    None,  # no batch lease was acquired for an undispatched claim
                )
                logger.info(f"[tinker] terminal-failed undispatched claim '{operation_id}' for '{tenant[0]}' at close")
            except Exception:
                logger.exception(f"[tinker] failed to terminal-fail claim '{operation_id}' at close")
            runtime.ready_output = None
        self.runtimes.clear()
        self.rotation.clear()

    async def abort_handoff(self, handoff: RolloutFnHandoff, error: BaseException) -> None:
        """RolloutFnHandoffAborter capability: the manager's downstream phase
        failed after this adapter handed over a leased selection, so the
        driver will never see the dispatch receipt. Terminal-fail the exact
        claimed operations and release the exact lease through the one
        idempotent controller boundary (``fail_tinker_batch`` fails only
        still-CLAIMED operations and releases the lease in ``finally``, so a
        repeat can never overwrite a landed result). The failed
        forward_backwards poison their gradient windows exactly as a failed
        train dispatch does; retry ownership stays with the client."""
        await self.abort.abort_batch(
            list(handoff.driver_metadata["operation_ids"]),
            f"rollout postprocessing failed before trainer dispatch: {error}; the batch never "
            "reached the trainer and its gradient window is poisoned — resubmit the batch and "
            "optim_step again",
            handoff.driver_metadata["lease"],
        )

    # ------------------------------ runtimes ------------------------------

    async def _reconcile(self, adapters: dict[str, AdapterRun]) -> None:
        live = {(name, run.registration_id) for name, run in adapters.items()}
        for tenant in [t for t in self.runtimes if t not in live]:
            # Deregistered or re-registered: close the old tenant's runtime;
            # its late results are dropped with it (registration fencing).
            await self.runtimes.pop(tenant).aclose()
            logger.info(f"[tinker] closed child runtime for '{tenant[0]}' ({tenant[1][:8]})")
        for name, run in adapters.items():
            tenant = (name, run.registration_id)
            if tenant in self.runtimes:
                self.runtimes[tenant].refresh(run)
                continue
            self.runtimes[tenant] = AdapterRolloutRuntime(self.args, run, self.operations)
            logger.info(f"[tinker] created child runtime for '{name}' ({run.registration_id[:8]})")
        self._sync_rotation()

    def _sync_rotation(self) -> None:
        in_queue = set()
        kept: deque[Tenant] = deque()
        while self.rotation:
            if (tenant := self.rotation.popleft()) in self.runtimes and tenant not in in_queue:
                kept.append(tenant)
                in_queue.add(tenant)
        for tenant in self.runtimes:
            if tenant not in in_queue:
                kept.append(tenant)
        self.rotation = kept

    def _launch_idle_children(self, rollout_id: int) -> None:
        if self._closed:
            return
        now = time.monotonic()
        for runtime in self.runtimes.values():
            if runtime.state != AdapterRolloutRuntime.IDLE:
                continue
            if now < runtime.retry_at:
                # Transient-failure backoff: the runtime relaunches on a later
                # cycle (bounded by the empty-batch deadline, after which the
                # driver yields to its control phase and calls again).
                continue
            runtime.state = AdapterRolloutRuntime.IN_FLIGHT
            runtime.task = asyncio.create_task(self._run_child(runtime, rollout_id))

    async def _run_child(self, runtime: AdapterRolloutRuntime, rollout_id: int) -> None:
        try:
            output = await runtime.child_fn(RolloutFnTrainInput(rollout_id=rollout_id))
            if not output.samples:
                raise ValueError(f"child for '{runtime.run.name}' returned an empty batch")
            runtime.transient_failures = 0
            runtime.retry_at = 0.0
            runtime.ready_output = output
            runtime.state = AdapterRolloutRuntime.READY
        except asyncio.CancelledError:
            runtime.state = AdapterRolloutRuntime.IDLE
            raise
        except TransientOperationPortError as e:
            # Provably no ledger mutation happened: the registration stays
            # runnable, with a capped exponential backoff so a flapping
            # controller is not hammered (external review 0813 §4.3).
            runtime.transient_failures += 1
            backoff = min(_CHILD_BACKOFF_CAP_S, _CHILD_BACKOFF_BASE_S * 2 ** (runtime.transient_failures - 1))
            runtime.retry_at = time.monotonic() + backoff
            runtime.state = AdapterRolloutRuntime.IDLE
            logger.warning(
                f"[tinker] child for '{runtime.run.name}' hit a transient port failure "
                f"(consecutive #{runtime.transient_failures}, relaunching in {backoff:.1f}s): {e}"
            )
        except Exception as e:
            # Ambiguous failure (e.g. a claim RPC whose response was lost may
            # already have turned the stream head CLAIMED): quarantine this
            # runtime rather than retry into a possible orphan. FAILED is
            # terminal until deregistration/re-registration removes the
            # runtime; a controller-side idempotent-claim reconciliation is
            # the future recovery path. Other adapters keep going.
            logger.exception(f"[tinker] child for '{runtime.run.name}' failed (quarantined): {e}")
            runtime.state = AdapterRolloutRuntime.FAILED
        finally:
            self._ready.set()

    # ------------------------------ selection ------------------------------

    async def _select(self) -> list[AdapterRolloutRuntime]:
        """Collect READY child batches under the kind lock. The first selected
        operation locks the selection's kind (D11 homogeneity); other-kind
        READY batches stay READY for the next call. Two clocks: the empty-batch
        deadline before anything is selected, the coalesce window after."""
        soft_target = self.args.rollout_batch_size * self.args.n_samples_per_prompt
        coalesce_wait = self.args.tinker_max_coalesce_wait_s
        empty_deadline = time.monotonic() + self.args.tinker_max_empty_wait_s
        selected: list[AdapterRolloutRuntime] = []
        kind_lock: str | None = None
        collected = 0
        coalesce_deadline: float | None = None

        while True:
            # Defensive ordering: clear BEFORE the authoritative state scan.
            # A completion then either lands before the scan (found in state)
            # or after it (leaves the event set, so the wait returns at
            # once). The scan-to-wait block below has no await point today —
            # the reviewed lost-wakeup interleaving was not reachable — but
            # clear-after-scan would silently turn any future await added in
            # between into a full-timeout latency bubble.
            self._ready.clear()
            runtime = self._pop_next_ready(kind_lock)
            if runtime is not None:
                selected.append(runtime)
                # Leave READY immediately or the round-robin would re-select
                # the same batch until the target is met (duplicated samples).
                runtime.state = AdapterRolloutRuntime.SELECTED
                kind_lock = runtime.ready_kind
                collected += sum(len(group) for group in runtime.ready_output.samples)
                if coalesce_deadline is None:
                    coalesce_deadline = time.monotonic() + coalesce_wait
                # Whole batches only: overshoot past the soft target is allowed,
                # trimming is not.
                if collected >= soft_target or len(selected) >= len(self.runtimes):
                    break
                continue

            now = time.monotonic()
            if selected:
                if now >= coalesce_deadline:
                    break
                timeout = coalesce_deadline - now
            else:
                if now >= empty_deadline:
                    raise EmptyBatchTimeoutError(
                        "no adapter produced a batch within "
                        f"--tinker-max-empty-wait-s ({self.args.tinker_max_empty_wait_s}s)"
                    )
                timeout = empty_deadline - now
            try:
                await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            except TimeoutError:
                continue
        return selected

    def _pop_next_ready(self, kind_lock: str | None) -> AdapterRolloutRuntime | None:
        """Persistent round-robin over READY runtimes matching the kind lock:
        the cursor survives across selections so fast adapters cannot starve
        slow ones."""
        for _ in range(len(self.rotation)):
            tenant = self.rotation.popleft()
            self.rotation.append(tenant)
            runtime = self.runtimes.get(tenant)
            if runtime is None or runtime.state != AdapterRolloutRuntime.READY:
                continue
            if kind_lock is not None and runtime.ready_kind != kind_lock:
                continue
            return runtime
        return None

    # ------------------------------ merge ------------------------------

    async def _merge(self, selected: list[AdapterRolloutRuntime]) -> RolloutFnTrainOutput:
        data: list[list[Sample]] = []
        batch_plan: list[dict] = []
        metrics: dict = {}
        # Read-only pass: build the merged data and plan WITHOUT touching the
        # runtimes, so a failure anywhere up to and including lease
        # acquisition leaves every selected runtime READY with its output
        # intact (the claimed operation stays retryable at the next selection
        # instead of orphaning the only in-memory copy of an already-CLAIMED
        # output).
        try:
            for runtime in selected:
                output = runtime.ready_output
                run = runtime.run
                data.extend(output.samples)
                # The claim's binding is the dispatch truth (resolved
                # atomically with the claim); the runtime's AdapterRun view
                # only names the metrics stream.
                binding = output.metadata["binding"]
                name, registration_id = binding.registration_key
                batch_plan.append(
                    dict(
                        name=name,
                        registration_id=registration_id,
                        operation_id=output.metadata["operation_id"],
                        operation_kind=output.metadata["operation_kind"],
                        loss_spec=output.metadata.get("loss_spec"),
                        sample_count=sum(len(group) for group in output.samples),
                        binding=binding,
                    )
                )
                metrics[f"{run.name}/operation_samples"] = sum(len(group) for group in output.samples)
            # One immutable dispatch receipt for the whole selection: the
            # controller re-validates exact slot ownership before issuing it.
            lease = await self._acquire_batch_with_retry(batch_plan)
        except StaleBindingError:
            # Authoritative refusal: terminal-fail exactly the stale claims,
            # keep the still-valid ones READY, and let __call__ reselect.
            await self._terminalize_stale_claims(selected)
            raise
        except BaseException:
            for runtime in selected:
                runtime.state = AdapterRolloutRuntime.READY
            raise
        # Acquisition succeeded: NOW consume the outputs.
        for runtime in selected:
            runtime.ready_output = None
            runtime.state = AdapterRolloutRuntime.IDLE  # relaunches at the NEXT generate call
        return self._build_selection_output(data, batch_plan, metrics, lease)

    async def _acquire_batch_with_retry(self, batch_plan: list[dict]):
        """Acquire the selection's dispatch receipt with bounded in-adapter
        retries. ``acquire_batch`` never mutates controller state (pure
        validate + mint), so ANY transport failure is safe to retry — without
        this, one transient controller blip re-raised out of generate() and
        killed the driver service (external review 0813 §4.6). An
        executed-and-refused acquisition arrives typed (StaleBindingError)
        and is never retried here."""
        bindings = [(entry["operation_id"], entry["binding"]) for entry in batch_plan]
        attempt = 1
        while True:
            try:
                return await self.residency.acquire_batch(bindings)
            except (StaleBindingError, asyncio.CancelledError):
                raise
            except Exception as e:
                if attempt >= _ACQUIRE_ATTEMPTS:
                    raise
                backoff = min(_ACQUIRE_BACKOFF_CAP_S, _ACQUIRE_BACKOFF_BASE_S * 2 ** (attempt - 1))
                logger.warning(
                    f"[tinker] batch lease acquisition failed transiently "
                    f"(attempt {attempt}/{_ACQUIRE_ATTEMPTS}, retrying in {backoff:.1f}s): {e}"
                )
                attempt += 1
                await asyncio.sleep(backoff)

    async def _terminalize_stale_claims(self, selected: list[AdapterRolloutRuntime]) -> None:
        """The batch receipt was refused, so at least one claimed binding is
        stale. Probe each selected claim individually so ONLY the stale
        operations terminal-fail — a coalesced selection spans adapters, and
        adapter B's valid claim must never be poisoned by adapter A's
        deregistration. Survivors return to READY for the reselection; probe
        receipts are discarded (fixed residency reserves nothing — a paged
        residency will need a release verb on this path)."""
        for runtime in selected:
            metadata = runtime.ready_output.metadata
            operation_id = metadata["operation_id"]
            try:
                await self.residency.acquire_batch([(operation_id, metadata["binding"])])
            except StaleBindingError as probe:
                try:
                    await self.abort.abort_batch(
                        [operation_id],
                        f"execution binding went stale before dispatch: {probe}; the claim can "
                        "never execute — resubmit it as a new operation",
                        None,  # refused before any batch lease existed
                    )
                except Exception:
                    # Keep the claim retryable: the next selection re-refuses
                    # and re-attempts this terminalization.
                    logger.exception(
                        f"[tinker] failed to terminal-fail stale claim '{operation_id}'; keeping it for retry"
                    )
                    runtime.state = AdapterRolloutRuntime.READY
                    continue
                logger.warning(f"[tinker] terminal-failed stale claim '{operation_id}' for '{runtime.run.name}'")
                runtime.ready_output = None
                runtime.state = AdapterRolloutRuntime.IDLE
            except Exception:
                # Transient probe failure: undecided, stays READY for retry.
                runtime.state = AdapterRolloutRuntime.READY
            else:
                runtime.state = AdapterRolloutRuntime.READY

    def _build_selection_output(self, data, batch_plan, metrics, lease) -> RolloutFnTrainOutput:
        return RolloutFnTrainOutput(
            samples=data,
            metrics=metrics,
            # Converted HERE, not in the manager: the generic rollout plane
            # never recognizes tinker keys.
            conversion_metadata=batch_plan_to_metadata(batch_plan, lease),
            # Whole client batches: zero-weight pads round the selection up to
            # the DP grid so the multi-LoRA dynamic-GBS branch sizes the step
            # to the batch instead of trimming it.
            postprocess=RolloutPostprocessOptions(pad_to_dp=True),
            # Dispatch identity minted ONCE, here, where it is exactly known —
            # never reconstructed from converted tensors. The driver's
            # abnormal-outcome finalizer and the manager's downstream abort
            # both consume this same opaque receipt.
            handoff=RolloutFnHandoff(
                driver_metadata={
                    "operation_ids": [entry["operation_id"] for entry in batch_plan],
                    "lease": lease_to_metadata(lease),
                }
            ),
        )


# Stable import path: --rollout-function-path defaults keep working, and the
# historical name survives as an alias of the adapter it always was.
TinkerRolloutFn = TinkerOperationBatchAdapter
