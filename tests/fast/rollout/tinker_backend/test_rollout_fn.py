"""Tinker operation-to-batch adapter: one claimed operation becomes one
stamped batch, bad payloads fail their own operation, and the selection loop
enforces the homogeneous kind lock with persistent round-robin fairness — all
driven through FAKE OperationQueuePort/BatchResidencyPort transports (no Ray
import, per codex-rollout-fullparameter-design-0810 §8.2)."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import pytest

from miles.ray.tinker_backend.config import AdapterRun, AdapterRunConfig
from miles.ray.tinker_backend.residency import ResidentBinding
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.compatibility import call_rollout_function_async
from miles.rollout.tinker_backend.operation_port import StaleBindingError, TransientOperationPortError
from miles.rollout.tinker_backend.rollout_fn import (
    AdapterRolloutRuntime,
    QueueChildRolloutFn,
    TinkerOperationBatchAdapter,
    TinkerOperationSource,
    TinkerRolloutFn,
)
from miles.utils.tinker_backend import BatchExecutionLease, EmptyBatchTimeoutError


def make_run(name="X", reg="rx", slot=3, version=2) -> AdapterRun:
    config = AdapterRunConfig(rank=8, alpha=16, metadata={"team": "t1"})
    return AdapterRun(name=name, config=config, slot=slot, version=version, registration_id=reg)


def make_child(run: AdapterRun, operations) -> QueueChildRolloutFn:
    source = TinkerOperationSource(SimpleNamespace(), run)
    return QueueChildRolloutFn(RolloutFnConstructorInput(args=source.args, data_source=source), operations)


def sample_payload(n=2) -> dict:
    return {
        "batch_id": "batch-7",
        "samples": [
            {"prompt": "p", "tokens": [1, 2, 3, 4], "response_length": 2, "loss_mask": [1, 1]} for _ in range(n)
        ],
        "loss": {"loss_fn": "cross_entropy"},
    }


class FakeOperationQueue:
    """Scripted OperationQueuePort: claims pop in order, failures record."""

    def __init__(self, claims=(), ready=None):
        self._claims = list(claims)
        self._ready = ready or {}
        self.failed: list[tuple] = []

    async def ready_streams(self) -> dict:
        return self._ready

    async def claim_data(self, key):
        return self._claims.pop(0) if self._claims else None

    async def fail(self, operation_id, error, category):
        self.failed.append((operation_id, error, category))


class FakeResidency:
    """Scripted BatchResidencyPort: mints deterministic leases."""

    def __init__(self):
        self.leases: list[tuple] = []

    async def acquire_batch(self, bindings_by_operation):
        self.leases.append(tuple(bindings_by_operation))
        return BatchExecutionLease(dispatch_id="lease-1", bindings_by_operation=tuple(bindings_by_operation))


class FakeBatchAbort:
    """Recording BatchAbortPort: every abnormal-outcome finalization lands here."""

    def __init__(self):
        self.aborts: list[tuple] = []

    async def abort_batch(self, operation_ids, error, lease_metadata):
        self.aborts.append((list(operation_ids), error, lease_metadata))


@pytest.fixture()
def fast_poll(monkeypatch):
    import miles.rollout.tinker_backend.rollout_fn as rollout_module

    monkeypatch.setattr(rollout_module, "_CLAIM_POLL_S", 0.01)


@pytest.fixture()
def fast_backoff(monkeypatch):
    import miles.rollout.tinker_backend.rollout_fn as rollout_module

    monkeypatch.setattr(rollout_module, "_ACQUIRE_BACKOFF_BASE_S", 0.001)
    monkeypatch.setattr(rollout_module, "_CHILD_BACKOFF_BASE_S", 0.01)


def op(op_id="op1", kind="forward_backward", payload=None, slot=3):
    # A claim always carries its fixed binding (claim-and-bind).
    return dict(
        operation_id=op_id,
        name="X",
        registration_id="rx",
        kind=kind,
        payload=sample_payload() if payload is None else payload,
        state="CLAIMED",
        binding=ResidentBinding(registration_key=("X", "rx"), training_slot=slot),
    )


class TestQueueChild:
    def test_one_operation_becomes_one_stamped_batch(self):
        output = asyncio.run(make_child(make_run(), FakeOperationQueue([op()]))(RolloutFnTrainInput(rollout_id=0)))

        assert len(output.samples) == 2 and all(len(group) == 1 for group in output.samples)
        stamped = output.samples[0][0]
        assert (stamped.adapter.name, stamped.adapter.registration_id) == ("X", "rx")
        assert stamped.adapter.serving_version == 2 and stamped.adapter.slot == 3
        assert stamped.metadata["team"] == "t1"  # run metadata merged in
        assert stamped.status == stamped.Status.COMPLETED
        assert [group[0].index for group in output.samples] == [0, 1]  # result-plane row identity
        assert output.metadata == dict(
            operation_id="op1",
            operation_kind="forward_backward",
            batch_id="batch-7",
            loss_spec={"loss_fn": "cross_entropy"},
            binding=ResidentBinding(registration_key=("X", "rx"), training_slot=3),
        )

    def test_client_supplied_row_index_is_overwritten(self):
        # index is server-owned: a client -1 would alias the DP-padding
        # sentinel (row silently dropped from the result plane) and duplicates
        # would collide in the (lane, row) logprob collector.
        payload = sample_payload()
        payload["samples"][0]["index"] = -1
        payload["samples"][1]["index"] = 0
        queue = FakeOperationQueue([op(payload=payload)])
        output = asyncio.run(make_child(make_run(), queue)(RolloutFnTrainInput(rollout_id=0)))
        assert [group[0].index for group in output.samples] == [0, 1]

    def test_child_waits_for_a_claim(self, fast_poll):
        queue = FakeOperationQueue([None, None, op()])
        output = asyncio.run(make_child(make_run(), queue)(RolloutFnTrainInput(rollout_id=0)))
        assert output.metadata["operation_id"] == "op1"

    def test_bad_payload_fails_its_operation_and_the_child_continues(self):
        queue = FakeOperationQueue([op("bad", payload={"samples": []}), op("good")])
        output = asyncio.run(make_child(make_run(), queue)(RolloutFnTrainInput(rollout_id=0)))

        assert output.metadata["operation_id"] == "good"
        [(failed_id, error, category)] = queue.failed
        assert failed_id == "bad" and category == "user" and "no samples" in error

    def test_forward_operations_build_batches_too(self):
        payload = {"samples": [{"prompt": "p", "tokens": [1, 2], "response_length": 1, "loss_mask": [1]}]}
        queue = FakeOperationQueue([op("fwd", kind="forward", payload=payload)])
        output = asyncio.run(make_child(make_run(), queue)(RolloutFnTrainInput(rollout_id=0)))
        assert output.metadata["operation_kind"] == "forward"
        assert output.metadata["loss_spec"] is None
        assert queue.failed == []


def ready_runtime(fn: TinkerOperationBatchAdapter, name: str, slot: int, kind: str) -> AdapterRolloutRuntime:
    # The runtime's stamped slot (9) is deliberately stale: the claim's
    # binding, not the long-lived AdapterRun view, is the dispatch truth.
    run = make_run(name=name, reg=f"r-{name}", slot=9)
    runtime = AdapterRolloutRuntime(fn.args, run)
    runtime.state = AdapterRolloutRuntime.READY
    runtime.ready_output = RolloutFnTrainOutput(
        samples=[[SimpleNamespace(adapter=None, metadata={})]],
        metadata=dict(
            operation_id=f"op-{name}",
            operation_kind=kind,
            loss_spec=None,
            binding=ResidentBinding(registration_key=(name, f"r-{name}"), training_slot=slot),
        ),
    )
    fn.runtimes[runtime.tenant] = runtime
    fn._sync_rotation()
    return runtime


def merge(fn: TinkerOperationBatchAdapter, selected) -> RolloutFnTrainOutput:
    return asyncio.run(fn._merge(selected))


def make_fn(soft_target=100) -> TinkerOperationBatchAdapter:
    args = SimpleNamespace(
        rollout_batch_size=soft_target,
        n_samples_per_prompt=1,
        tinker_max_coalesce_wait_s=0.05,
        tinker_max_empty_wait_s=0.05,
    )
    return TinkerOperationBatchAdapter(
        RolloutFnConstructorInput(args=args, data_source=None),
        operations=FakeOperationQueue(),
        residency=FakeResidency(),
        abort=FakeBatchAbort(),
    )


def test_the_historical_import_path_is_an_alias():
    assert TinkerRolloutFn is TinkerOperationBatchAdapter


class TestSelectionKindLock:
    def test_first_ready_locks_the_kind(self):
        fn = make_fn()
        ready_runtime(fn, "A", 0, "forward_backward")
        other = ready_runtime(fn, "B", 1, "forward")
        ready_runtime(fn, "C", 2, "forward_backward")

        selected = asyncio.run(fn._select())
        assert sorted(r.run.name for r in selected) == ["A", "C"]
        # The other-kind batch is untouched and stays READY for the next call.
        assert other.state == AdapterRolloutRuntime.READY

    def test_all_forward_selection_is_fine(self):
        fn = make_fn()
        ready_runtime(fn, "A", 0, "forward")
        ready_runtime(fn, "B", 1, "forward")
        selected = asyncio.run(fn._select())
        assert {r.ready_kind for r in selected} == {"forward"}

    def test_soft_target_stops_collection_but_never_trims(self):
        fn = make_fn(soft_target=1)
        ready_runtime(fn, "A", 0, "forward_backward")
        ready_runtime(fn, "B", 1, "forward_backward")
        selected = asyncio.run(fn._select())
        assert len(selected) == 1  # whole batches; B waits for the next call

    def test_empty_selection_times_out(self):
        fn = make_fn()
        with pytest.raises(EmptyBatchTimeoutError):
            asyncio.run(fn._select())

    def test_merge_ships_the_converted_plan_and_pad_policy(self):
        """Correlation is batch-local (§3.3): the selected operation gets lane
        0, the loss/result maps key by lane, and the exact registration rides
        along for the commit. The claim's binding is the single binding truth
        — it flows into the batch lease (§5.3) and the routing helper; the
        runtime's stale stamped slot (9) appears nowhere."""
        fn = make_fn()
        first = ready_runtime(fn, "A", 0, "forward_backward")
        selected = asyncio.run(fn._select())
        output = merge(fn, selected)
        assert output.conversion_metadata == {
            "batch_kind": "tinker",
            "tinker_operation_lanes": [0],
            "tinker_loss_by_lane": {0: {}},
            "operation_by_lane": {0: "op-A"},
            "registration_by_lane": {0: ("A", "r-A")},
            "batch_execution_lease": {
                "dispatch_id": "lease-1",
                "bindings_by_operation": [["op-A", ["A", "r-A", 0]]],
            },
        }
        assert output.postprocess.pad_to_dp is True
        assert first.state == AdapterRolloutRuntime.IDLE and first.ready_output is None

    def test_transient_lease_failure_retries_in_adapter(self, fast_backoff):
        """External review 0813 §4.6: ``acquire_batch`` never mutates, so a
        transient transport failure retries INSIDE the adapter — one blip must
        not propagate out of generate() and kill the driver service."""

        class RefusingOnceResidency(FakeResidency):
            def __init__(self):
                super().__init__()
                self.refusals_left = 1
                self.attempts = 0

            async def acquire_batch(self, bindings_by_operation):
                self.attempts += 1
                if self.refusals_left:
                    self.refusals_left -= 1
                    raise ConnectionError("controller transport blip")
                return await super().acquire_batch(bindings_by_operation)

        fn = make_fn()
        fn.residency = RefusingOnceResidency()
        runtime = ready_runtime(fn, "A", 0, "forward_backward")
        selected = asyncio.run(fn._select())

        output = merge(fn, selected)
        assert fn.residency.attempts == 2  # retried in-adapter, same call
        assert output.conversion_metadata["operation_by_lane"] == {0: "op-A"}
        assert runtime.state == AdapterRolloutRuntime.IDLE and runtime.ready_output is None

    def test_exhausted_transient_lease_failures_keep_claimed_output_retryable(self, fast_backoff, monkeypatch):
        """When the bounded retries exhaust, the failure must still not orphan
        the only in-memory copy of an already-CLAIMED output — the selected
        runtimes return to READY with their outputs intact, and the next
        selection retries them."""
        import miles.rollout.tinker_backend.rollout_fn as rollout_module

        monkeypatch.setattr(rollout_module, "_ACQUIRE_ATTEMPTS", 2)

        class AlwaysRefusingResidency(FakeResidency):
            async def acquire_batch(self, bindings_by_operation):
                raise ConnectionError("controller unreachable")

        fn = make_fn()
        fn.residency = AlwaysRefusingResidency()
        runtime = ready_runtime(fn, "A", 0, "forward_backward")
        selected = asyncio.run(fn._select())

        with pytest.raises(ConnectionError, match="unreachable"):
            merge(fn, selected)
        assert runtime.state == AdapterRolloutRuntime.READY
        assert runtime.ready_output is not None

        # The SAME claimed output dispatches once the controller is back.
        fn.residency = FakeResidency()
        selected = asyncio.run(fn._select())
        output = merge(fn, selected)
        assert output.conversion_metadata["operation_by_lane"] == {0: "op-A"}
        assert runtime.state == AdapterRolloutRuntime.IDLE and runtime.ready_output is None

    def test_merge_of_a_forward_selection_marks_forward_only(self):
        """Forward kind: the same composition with ``tinker_forward_only``
        set — the flag that keeps forward operations gradient-free must
        survive the lane re-keying."""
        fn = make_fn()
        ready_runtime(fn, "A", 0, "forward")
        ready_runtime(fn, "B", 1, "forward")
        selected = asyncio.run(fn._select())
        output = merge(fn, selected)
        assert output.conversion_metadata["tinker_forward_only"] is True
        assert output.conversion_metadata["operation_by_lane"] == {0: "op-A", 1: "op-B"}
        assert output.conversion_metadata["tinker_operation_lanes"] == [0, 1]
        assert output.postprocess.pad_to_dp is True

    def test_lanes_are_selection_local_and_independent_of_slots(self):
        """Two operations on HIGH slots (7, 2) still get lanes 0 and 1 in
        selection order: identity never rides the physical slot, so a future
        parameterization (or slot reuse across operations) cannot collide in
        the collector/result plane."""
        fn = make_fn()
        ready_runtime(fn, "A", 7, "forward_backward")
        ready_runtime(fn, "B", 2, "forward_backward")
        selected = asyncio.run(fn._select())
        output = merge(fn, selected)
        assert output.conversion_metadata["tinker_operation_lanes"] == [0, 1]
        assert output.conversion_metadata["registration_by_lane"] == {0: ("A", "r-A"), 1: ("B", "r-B")}
        lease = output.conversion_metadata["batch_execution_lease"]
        assert lease["bindings_by_operation"] == [["op-A", ["A", "r-A", 7]], ["op-B", ["B", "r-B", 2]]]


class TestDriverHandoff:
    """The dispatch receipt (operation ids + encoded lease) is minted ONCE, in
    ``_merge`` where it is exactly known — the generic manager forwards it
    opaquely and the driver finalizes with it. Reconstruction from converted
    train data no longer exists (external review 0813 §4.8/§6.1)."""

    def test_merge_mints_the_handoff_with_exact_ids_and_lease(self):
        fn = make_fn()
        ready_runtime(fn, "A", 7, "forward_backward")
        ready_runtime(fn, "B", 2, "forward_backward")
        selected = asyncio.run(fn._select())
        output = merge(fn, selected)
        assert output.handoff.driver_metadata["operation_ids"] == ["op-A", "op-B"]
        # One binding truth: the handoff's lease IS the conversion plane's
        # lease — the same encoded receipt, never a second copy of anything.
        assert output.handoff.driver_metadata["lease"] == output.conversion_metadata["batch_execution_lease"]

    def test_abort_handoff_terminal_fails_the_exact_batch(self):
        """RolloutFnHandoffAborter capability: a downstream failure after the
        output receipt fails exactly the handoff's operations and releases
        exactly its lease through the one idempotent batch-abort boundary
        (external review 0813 §4.1/§6.2)."""
        fn = make_fn()
        ready_runtime(fn, "A", 0, "forward_backward")
        selected = asyncio.run(fn._select())
        output = merge(fn, selected)

        asyncio.run(fn.abort_handoff(output.handoff, OSError("simulated object-store placement failure")))

        [(operation_ids, error, lease_metadata)] = fn.abort.aborts
        assert operation_ids == ["op-A"]
        assert lease_metadata == output.handoff.driver_metadata["lease"]
        # Retry ownership is explicit in the message: the client resubmits,
        # and the poisoned gradient window discards on the next optim_step.
        assert "placement failure" in error and "poisoned" in error and "resubmit" in error


class StaleSetResidency(FakeResidency):
    """Refuses any receipt containing a configured stale operation id — the
    per-operation probe then isolates exactly those."""

    def __init__(self, stale_ids):
        super().__init__()
        self.stale_ids = set(stale_ids)

    async def acquire_batch(self, bindings_by_operation):
        bindings = list(bindings_by_operation)
        stale = [op_id for op_id, _binding in bindings if op_id in self.stale_ids]
        if stale:
            raise StaleBindingError(f"registration no longer owns trainer slot for {sorted(stale)}")
        return await super().acquire_batch(bindings)


class TestStaleBindingTerminalization:
    """External review 0813 §4.6: an authoritative stale-binding refusal must
    terminal-fail the EXACT stale claims (never infinite-retry them) while a
    coalesced selection's still-valid claims survive and dispatch."""

    def test_stale_claim_terminal_fails_and_survivor_stays_ready(self):
        fn = make_fn()
        fn.residency = StaleSetResidency(["op-A"])
        stale = ready_runtime(fn, "A", 0, "forward_backward")
        survivor = ready_runtime(fn, "B", 1, "forward_backward")
        selected = asyncio.run(fn._select())

        with pytest.raises(StaleBindingError):
            merge(fn, selected)

        [(operation_ids, error, lease_metadata)] = fn.abort.aborts
        assert operation_ids == ["op-A"] and lease_metadata is None  # no lease existed yet
        assert "stale" in error and "resubmit" in error
        assert stale.state == AdapterRolloutRuntime.IDLE and stale.ready_output is None
        assert survivor.state == AdapterRolloutRuntime.READY and survivor.ready_output is not None

        # The survivor dispatches alone on the reselection.
        selected = asyncio.run(fn._select())
        output = merge(fn, selected)
        assert output.conversion_metadata["operation_by_lane"] == {0: "op-B"}

    def test_call_reselects_survivors_after_a_stale_refusal(self, fast_poll):
        """End-to-end through __call__: the stale claim terminal-fails, the
        valid claim reselects and returns in the SAME generate call."""

        class KeyedQueue:
            def __init__(self, operations_by_name):
                self.operations_by_name = dict(operations_by_name)
                self.runs = {name: make_run(name=name, reg=f"r-{name}", slot=i) for i, name in enumerate(["A", "B"])}

            async def ready_streams(self):
                return self.runs

            async def claim_data(self, key):
                return self.operations_by_name.pop(key[0], None)

            async def fail(self, operation_id, error, category):
                raise AssertionError("no payload failure expected")

        def keyed_op(name, slot):
            operation = op(op_id=f"op-{name}", slot=slot)
            operation["name"] = name
            operation["registration_id"] = f"r-{name}"
            operation["binding"] = ResidentBinding(registration_key=(name, f"r-{name}"), training_slot=slot)
            return operation

        args = SimpleNamespace(
            rollout_batch_size=100,
            n_samples_per_prompt=1,
            tinker_max_coalesce_wait_s=0.05,
            tinker_max_empty_wait_s=2.0,
        )
        fn = TinkerOperationBatchAdapter(
            RolloutFnConstructorInput(args=args, data_source=None),
            operations=KeyedQueue({"A": keyed_op("A", 0), "B": keyed_op("B", 1)}),
            residency=StaleSetResidency(["op-A"]),
            abort=FakeBatchAbort(),
        )

        output = asyncio.run(fn(RolloutFnTrainInput(rollout_id=0)))

        assert output.conversion_metadata["operation_by_lane"] == {0: "op-B"}
        [(operation_ids, _error, lease_metadata)] = fn.abort.aborts
        assert operation_ids == ["op-A"] and lease_metadata is None


class TestTransientChildRecovery:
    """External review 0813 §4.3: a KNOWN-transient claim failure (provably no
    ledger mutation) keeps the registration runnable — IDLE with a capped
    exponential backoff — while ambiguous failures still quarantine."""

    def test_transient_claim_failure_backs_off_and_recovers(self, fast_poll, fast_backoff):
        class FlakyOnceQueue(FakeOperationQueue):
            def __init__(self):
                super().__init__(claims=[op()], ready={"X": make_run()})
                self.transient_left = 1

            async def claim_data(self, key):
                if self.transient_left:
                    self.transient_left -= 1
                    raise TransientOperationPortError("controller unavailable")
                return await super().claim_data(key)

        args = SimpleNamespace(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            tinker_max_coalesce_wait_s=0.02,
            tinker_max_empty_wait_s=0.15,
        )
        fn = TinkerOperationBatchAdapter(
            RolloutFnConstructorInput(args=args, data_source=None),
            operations=FlakyOnceQueue(),
            residency=FakeResidency(),
            abort=FakeBatchAbort(),
        )

        async def scenario():
            # First call: the transient failure lands the runtime back in
            # IDLE with a backoff; nothing is READY, so the call yields the
            # empty-batch timeout (the driver's control-phase yield).
            with pytest.raises(EmptyBatchTimeoutError):
                await fn(RolloutFnTrainInput(rollout_id=0))
            runtime = next(iter(fn.runtimes.values()))
            assert runtime.state == AdapterRolloutRuntime.IDLE
            assert runtime.transient_failures == 1 and runtime.retry_at > 0
            # Next call (after the backoff): the SAME registration relaunches
            # and its claim dispatches; the failure counter resets.
            await asyncio.sleep(0.02)
            output = await fn(RolloutFnTrainInput(rollout_id=1))
            assert output.conversion_metadata["operation_by_lane"] == {0: "op1"}
            assert runtime.transient_failures == 0
            return runtime

        asyncio.run(scenario())

    def test_ambiguous_child_failure_stays_quarantined(self):
        """Characterization (documented, not a bug): a failure that MAY have
        mutated the ledger — a claim RPC whose response was lost — must NOT
        be retried (the stream head may already be CLAIMED; blind retries
        would poll forever while hiding the orphan). The runtime quarantines
        as FAILED until deregistration/re-registration removes it; the future
        recovery is a controller-side idempotent-claim reconciliation."""
        fn = make_fn()
        run = make_run(name="A", reg="rid-A")
        asyncio.run(fn._reconcile({"A": run}))
        runtime = fn.runtimes[("A", "rid-A")]

        class FailsOnce:
            calls = 0

            async def __call__(self, _input):
                type(self).calls += 1
                raise RuntimeError("claim RPC response lost")

        runtime.child_fn = FailsOnce()
        runtime.state = AdapterRolloutRuntime.IN_FLIGHT
        asyncio.run(fn._run_child(runtime, rollout_id=0))
        assert runtime.state == AdapterRolloutRuntime.FAILED

        async def cycles():
            for cycle in range(3):
                await fn._reconcile({"A": run})
                fn._launch_idle_children(rollout_id=1 + cycle)

        asyncio.run(cycles())
        assert fn.runtimes[("A", "rid-A")] is runtime
        assert runtime.state == AdapterRolloutRuntime.FAILED
        assert FailsOnce.calls == 1


class TestClaimSafeClose:
    """External review 0813 §4.7: closing the adapter terminal-fails every
    claim it still holds — a READY output IS a CLAIMED operation with no
    lease yet — and refuses new claim work afterwards."""

    def _adapter_with_ready_claim(self):
        args = SimpleNamespace(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            tinker_max_coalesce_wait_s=0.02,
            tinker_max_empty_wait_s=1.0,
        )
        queue = FakeOperationQueue(claims=[op()], ready={"X": make_run()})
        fn = TinkerOperationBatchAdapter(
            RolloutFnConstructorInput(args=args, data_source=None),
            operations=queue,
            residency=FakeResidency(),
            abort=FakeBatchAbort(),
        )
        return fn

    def test_close_terminal_fails_ready_claims(self):
        async def scenario():
            fn = self._adapter_with_ready_claim()
            await fn._reconcile(await fn.operations.ready_streams())
            fn._launch_idle_children(rollout_id=0)
            for _ in range(200):
                if any(r.state == AdapterRolloutRuntime.READY for r in fn.runtimes.values()):
                    break
                await asyncio.sleep(0.01)

            await fn.aclose()

            [(operation_ids, error, lease_metadata)] = fn.abort.aborts
            assert operation_ids == ["op1"] and lease_metadata is None
            assert "closed" in error and "resubmit" in error
            assert fn.runtimes == {} and len(fn.rotation) == 0

            with pytest.raises(RuntimeError, match="closed"):
                await fn(RolloutFnTrainInput(rollout_id=1))

        asyncio.run(scenario())

    def test_close_without_claims_aborts_nothing(self):
        async def scenario():
            fn = self._adapter_with_ready_claim()
            await fn.aclose()
            assert fn.abort.aborts == []

        asyncio.run(scenario())

    def test_close_cancels_inflight_children_without_false_aborts(self):
        """An IN_FLIGHT child blocked in its claim holds NO known claim: close
        cancels and awaits it, and must not invent an abort for an operation
        that was never claimed. (An RPC cancelled before any response is the
        documented ambiguity — registration fencing owns it.)"""

        class BlockedQueue(FakeOperationQueue):
            def __init__(self):
                super().__init__(ready={"X": make_run()})
                self.entered = asyncio.Event()

            async def claim_data(self, key):
                self.entered.set()
                await asyncio.sleep(3600)

        args = SimpleNamespace(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            tinker_max_coalesce_wait_s=0.02,
            tinker_max_empty_wait_s=1.0,
        )
        queue = BlockedQueue()
        fn = TinkerOperationBatchAdapter(
            RolloutFnConstructorInput(args=args, data_source=None),
            operations=queue,
            residency=FakeResidency(),
            abort=FakeBatchAbort(),
        )

        async def scenario():
            await fn._reconcile(await fn.operations.ready_streams())
            fn._launch_idle_children(rollout_id=0)
            await asyncio.wait_for(queue.entered.wait(), timeout=2.0)
            await fn.aclose()
            assert fn.abort.aborts == []  # nothing claimed, nothing aborted
            assert fn.runtimes == {}

        asyncio.run(scenario())


class TestCallerCancellation:
    """External review 0813 §4.2: the manager awaits the adapter DIRECTLY, so
    cancelling the caller cancels the adapter coroutine — the abandoned
    selection can no longer claim an operation into a dead future and take a
    lease nobody will release."""

    def test_cancelling_the_caller_leaves_claims_recoverable(self, fast_poll):
        gate = asyncio.Event()

        class GatedQueue(FakeOperationQueue):
            def __init__(self):
                super().__init__(claims=[op()], ready={"X": make_run()})
                self.entered = asyncio.Event()

            async def claim_data(self, key):
                self.entered.set()
                await gate.wait()
                return await super().claim_data(key)

        args = SimpleNamespace(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            tinker_max_coalesce_wait_s=0.02,
            tinker_max_empty_wait_s=30.0,
        )
        queue = GatedQueue()
        fn = TinkerOperationBatchAdapter(
            RolloutFnConstructorInput(args=args, data_source=None),
            operations=queue,
            residency=FakeResidency(),
            abort=FakeBatchAbort(),
        )

        async def scenario():
            task = asyncio.create_task(call_rollout_function_async(fn, RolloutFnTrainInput(rollout_id=0)))
            await asyncio.wait_for(queue.entered.wait(), timeout=2.0)
            task.cancel()
            # Direct await: the cancellation reaches the adapter coroutine
            # immediately — no 30s empty-wait runs on after the caller died.
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
            assert fn.residency.leases == []  # nothing leased after death

            # The child task keeps its claim BY DESIGN: the result lands in
            # ADAPTER STATE (READY), recoverable by the next generate call —
            # never consumed into a dead future.
            gate.set()
            runtime = next(iter(fn.runtimes.values()))
            for _ in range(200):
                if runtime.state == AdapterRolloutRuntime.READY:
                    break
                await asyncio.sleep(0.01)
            assert runtime.state == AdapterRolloutRuntime.READY
            assert runtime.ready_output is not None
            assert fn.residency.leases == []

            # And teardown terminal-fails that recovered claim (§4.7).
            await fn.aclose()
            [(operation_ids, _error, lease_metadata)] = fn.abort.aborts
            assert operation_ids == ["op1"] and lease_metadata is None

        asyncio.run(scenario())


class TestSelectionWakeup:
    """External review 0813 §4.5 (REFUTED, defensive): with clear-before-scan,
    a completion landing between the state scan and the wait leaves the event
    set, so the selector wakes immediately instead of sleeping out the full
    empty-batch timeout."""

    def test_completion_in_the_scan_gap_is_not_lost(self):
        args = SimpleNamespace(
            rollout_batch_size=1,
            n_samples_per_prompt=1,
            tinker_max_coalesce_wait_s=0.02,
            tinker_max_empty_wait_s=5.0,
        )
        fn = TinkerOperationBatchAdapter(
            RolloutFnConstructorInput(args=args, data_source=None),
            operations=FakeOperationQueue(),
            residency=FakeResidency(),
            abort=FakeBatchAbort(),
        )
        runtime = ready_runtime(fn, "A", 0, "forward_backward")
        runtime.state = AdapterRolloutRuntime.IN_FLIGHT  # not yet visible to the scan

        real_pop = fn._pop_next_ready
        fired = {"done": False}

        def pop_with_completion_in_the_gap(kind_lock):
            found = real_pop(kind_lock)
            if found is None and not fired["done"]:
                fired["done"] = True
                # The child completes AFTER the scan missed it: state flips
                # READY and the event is set — exactly the reviewed schedule.
                runtime.state = AdapterRolloutRuntime.READY
                fn._ready.set()
            return found

        fn._pop_next_ready = pop_with_completion_in_the_gap

        async def scenario():
            import time as time_module

            start = time_module.monotonic()
            selected = await fn._select()
            elapsed = time_module.monotonic() - start
            assert selected == [runtime]
            # Well under the 5s empty-batch timeout the lost wakeup would cost.
            assert elapsed < 1.0

        asyncio.run(scenario())
