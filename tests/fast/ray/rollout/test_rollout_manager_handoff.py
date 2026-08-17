"""RolloutManager's cleanup-safe downstream phase (external review 0813 §4.1/
§6.2): once a rollout fn hands its output over, EVERY failure between that
receipt and ``generate()`` returning must give the fn's opaque handoff back
through ``abort_handoff`` before the error propagates — otherwise claimed
state only the fn knows about (e.g. a tinker operation + its execution lease)
would be orphaned with the exception.

Driven end-to-end through the production manager ``generate()`` implementation
(the raw class behind ``@ray.remote``, in-process so monkeypatch reaches its
dependencies) with a REAL TinkerOperationBatchAdapter on fake ports — no Ray.
"""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import pytest

import miles.ray.rollout.rollout_manager as rollout_manager_module
from miles.ray.tinker_backend.config import AdapterRun, AdapterRunConfig
from miles.ray.tinker_backend.residency import ResidentBinding
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput
from miles.rollout.tinker_backend.rollout_fn import TinkerOperationBatchAdapter
from miles.utils import object_store
from miles.utils.tinker_backend import BatchExecutionLease


def make_run(name="A", registration_id="rid-A", slot=0) -> AdapterRun:
    return AdapterRun(
        name=name,
        registration_id=registration_id,
        slot=slot,
        version=0,
        config=AdapterRunConfig(rank=8, alpha=16),
    )


def make_args(**overrides) -> SimpleNamespace:
    values = dict(
        # adapter selection clocks
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        tinker_max_coalesce_wait_s=0.02,
        tinker_max_empty_wait_s=1.0,
        # postprocess/conversion plane
        multi_lora=True,
        multi_lora_n_adapters=4,
        use_dynamic_global_batch_size=True,
        disable_rollout_trim_samples=False,
        global_batch_size=1,
        balance_data=False,
        # manager generate() surface
        ci_test=False,
        use_fault_tolerance=False,
        load_debug_rollout_data=False,
        save_debug_rollout_data=None,
        delay_split_train_data_by_dp=False,
        ci_inject_rollout_data_path=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class OneShotQueue:
    """Scripted OperationQueuePort holding one claimable operation."""

    def __init__(self, operation):
        self.operation = operation
        self.state = "QUEUED"
        self.failed: list[tuple] = []

    async def ready_streams(self) -> dict:
        return {"A": make_run()}

    async def claim_data(self, key):
        if self.state != "QUEUED":
            return None
        self.state = "CLAIMED"
        return self.operation

    async def fail(self, operation_id, error, category):
        self.failed.append((operation_id, error, category))


class RecordingResidency:
    def __init__(self):
        self.acquired: list[tuple] = []

    async def acquire_batch(self, bindings_by_operation):
        bindings = tuple(bindings_by_operation)
        self.acquired.append(bindings)
        return BatchExecutionLease(dispatch_id="lease-handoff", bindings_by_operation=bindings)


class RecordingBatchAbort:
    def __init__(self, boom: Exception | None = None):
        self.aborts: list[tuple] = []
        self.boom = boom

    async def abort_batch(self, operation_ids, error, lease_metadata):
        self.aborts.append((list(operation_ids), error, lease_metadata))
        if self.boom is not None:
            raise self.boom


def valid_operation(loss_mask=(1, 1)):
    return {
        "operation_id": "op-A",
        "name": "A",
        "registration_id": "rid-A",
        "kind": "forward_backward",
        "state": "QUEUED",
        "binding": ResidentBinding(("A", "rid-A"), 0),
        "payload": {
            "samples": [
                {
                    "prompt": "p",
                    "tokens": [1, 2, 3, 4],
                    "response_length": 2,
                    "loss_mask": list(loss_mask),
                    "loss_weights": [1.0, 1.0],
                }
            ],
            "loss": {"loss_fn": "cross_entropy"},
        },
    }


class FakeObjectStore:
    def __init__(self):
        self.puts: list = []

    def put(self, value, value_spec):
        self.puts.append(value)
        return ("ref", len(self.puts) - 1)


@pytest.fixture()
def fake_store(monkeypatch):
    store = FakeObjectStore()
    monkeypatch.setattr(object_store, "get_instance", lambda: store)
    return store


@pytest.fixture()
def quiet_manager_io(monkeypatch):
    monkeypatch.setattr(rollout_manager_module.dashboard_hooks, "register_engines", lambda _servers: None)
    monkeypatch.setattr(rollout_manager_module, "log_rollout_data", lambda *a, **k: None)


def make_manager(args, rollout_fn) -> object:
    """Production RolloutManager instance without __init__ (no servers, no
    tracking): exactly the attributes ``generate()`` touches."""
    manager = object.__new__(rollout_manager_module.RolloutManager.__ray_actor_class__)
    manager.args = args
    manager.servers = {}
    manager.rollout_id = -1
    manager.weight_version = None
    manager.train_parallel_config = {"dp_size": 1}
    manager.use_legacy_rollout_v1 = False
    manager.generate_rollout = rollout_fn
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = None
    manager.data_source = SimpleNamespace()
    manager._health_monitoring_resume = lambda: None
    return manager


def make_adapter(args, operation, abort=None):
    queue = OneShotQueue(operation)
    adapter = TinkerOperationBatchAdapter(
        RolloutFnConstructorInput(args=args, data_source=None),
        operations=queue,
        residency=RecordingResidency(),
        abort=abort if abort is not None else RecordingBatchAbort(),
    )
    return adapter, queue


class TestDownstreamFailuresAbortTheHandoff:
    """The orphan window the 0813 review reproduced: disk, logger, converter,
    DP split, and object-store failures all live between the fn's output
    receipt and ``generate()`` returning. Each one must invoke the fn's abort
    with the exact dispatch identity before re-raising."""

    def _assert_aborted_exactly(self, adapter):
        [(operation_ids, error, lease_metadata)] = adapter.abort.aborts
        assert operation_ids == ["op-A"]
        assert lease_metadata["dispatch_id"] == "lease-handoff"
        assert lease_metadata["bindings_by_operation"] == [["op-A", ["A", "rid-A", 0]]]
        return error

    @pytest.mark.asyncio
    async def test_debug_save_failure_aborts_the_exact_operations(self, monkeypatch, quiet_manager_io):
        args = make_args()
        adapter, queue = make_adapter(args, valid_operation())
        manager = make_manager(args, adapter)

        def fail_debug_save(*_a, **_k):
            raise OSError("simulated debug save filesystem failure")

        monkeypatch.setattr(rollout_manager_module, "save_debug_rollout_data", fail_debug_save)

        with pytest.raises(OSError, match="filesystem failure"):
            await manager.generate(rollout_id=1)

        assert queue.state == "CLAIMED"  # the claim itself is untouched...
        error = self._assert_aborted_exactly(adapter)  # ...but terminal-failed via the abort port
        assert "filesystem failure" in error and "resubmit" in error

    @pytest.mark.asyncio
    async def test_conversion_failure_after_lease_aborts(self, monkeypatch, quiet_manager_io):
        """A preflight-shaped payload that only conversion rejects (loss mask
        shorter than the response) used to leave the operation CLAIMED with
        the lease unreleased and the stream blocked forever."""
        args = make_args()
        adapter, _queue = make_adapter(args, valid_operation(loss_mask=(1,)))
        manager = make_manager(args, adapter)
        monkeypatch.setattr(rollout_manager_module, "save_debug_rollout_data", lambda *a, **k: None)

        with pytest.raises(AssertionError, match="loss mask length 1 != response length 2"):
            await manager.generate(rollout_id=2)

        error = self._assert_aborted_exactly(adapter)
        assert "loss mask length" in error

    @pytest.mark.asyncio
    async def test_dp_split_failure_aborts(self, monkeypatch, quiet_manager_io):
        args = make_args()
        adapter, _queue = make_adapter(args, valid_operation())
        manager = make_manager(args, adapter)
        monkeypatch.setattr(rollout_manager_module, "save_debug_rollout_data", lambda *a, **k: None)

        def fail_split(*_a, **_k):
            raise OSError("simulated object-store placement failure")

        monkeypatch.setattr(rollout_manager_module, "split_train_data_by_dp", fail_split)

        with pytest.raises(OSError, match="placement failure"):
            await manager.generate(rollout_id=3)

        self._assert_aborted_exactly(adapter)

    @pytest.mark.asyncio
    async def test_postprocess_failure_aborts(self, monkeypatch, quiet_manager_io):
        """The window opens at the OUTPUT RECEIPT, not at conversion:
        a postprocess failure inside ``_get_rollout_data`` aborts too."""
        args = make_args()
        adapter, _queue = make_adapter(args, valid_operation())
        manager = make_manager(args, adapter)

        def fail_postprocess(*_a, **_k):
            raise ValueError("simulated postprocess failure")

        monkeypatch.setattr(rollout_manager_module, "postprocess_rollout_data", fail_postprocess)

        with pytest.raises(ValueError, match="postprocess failure"):
            await manager.generate(rollout_id=4)

        self._assert_aborted_exactly(adapter)

    @pytest.mark.asyncio
    async def test_abort_failure_never_masks_the_original_error(self, monkeypatch, quiet_manager_io):
        """If the abort itself fails, the ORIGINAL downstream failure still
        propagates (the abort failure is logged, never raised in its place)."""
        args = make_args()
        adapter, _queue = make_adapter(
            args, valid_operation(), abort=RecordingBatchAbort(boom=RuntimeError("controller unreachable"))
        )
        manager = make_manager(args, adapter)

        def fail_debug_save(*_a, **_k):
            raise OSError("original downstream failure")

        monkeypatch.setattr(rollout_manager_module, "save_debug_rollout_data", fail_debug_save)

        with pytest.raises(OSError, match="original downstream failure"):
            await manager.generate(rollout_id=5)

        assert len(adapter.abort.aborts) == 1  # the abort was attempted

    @pytest.mark.asyncio
    async def test_downstream_abort_is_safe_to_repeat(self, quiet_manager_io):
        """Duplicate finalization (a manager abort racing the driver's train
        finalizer) goes through the same idempotent boundary; the port sees
        each attempt, the ledger keeps the first terminal result (witnessed by
        ``TestFailTinkerBatch::test_duplicate_finalization_is_idempotent``)."""
        args = make_args()
        adapter, _queue = make_adapter(args, valid_operation())
        output = await adapter(RolloutFnTrainInput(rollout_id=6))
        error = OSError("downstream failure")

        await adapter.abort_handoff(output.handoff, error)
        await adapter.abort_handoff(output.handoff, error)

        assert len(adapter.abort.aborts) == 2
        assert adapter.abort.aborts[0][0] == adapter.abort.aborts[1][0] == ["op-A"]


class TestSuccessPathForwardsTheHandoff:
    """Regression 8: the opaque handoff survives postprocess, conversion, the
    DP split, and the delayed object-store path — the driver receives it
    verbatim as ``rollout_fn_metadata`` and the manager interprets nothing."""

    @pytest.mark.asyncio
    async def test_split_path(self, monkeypatch, quiet_manager_io, fake_store):
        args = make_args()
        adapter, queue = make_adapter(args, valid_operation())
        manager = make_manager(args, adapter)
        monkeypatch.setattr(rollout_manager_module, "save_debug_rollout_data", lambda *a, **k: None)

        pack = await manager.generate(rollout_id=7)

        assert queue.state == "CLAIMED" and adapter.abort.aborts == []
        assert pack["rollout_fn_metadata"]["operation_ids"] == ["op-A"]
        assert pack["rollout_fn_metadata"]["lease"]["dispatch_id"] == "lease-handoff"
        # The trainer-facing correlation plane still rides the train data.
        [shard] = fake_store.puts
        assert shard["operation_by_lane"] == {0: "op-A"}
        assert shard["batch_execution_lease"] == pack["rollout_fn_metadata"]["lease"]

    @pytest.mark.asyncio
    async def test_delayed_split_path(self, monkeypatch, quiet_manager_io, fake_store):
        args = make_args(delay_split_train_data_by_dp=True)
        adapter, _queue = make_adapter(args, valid_operation())
        manager = make_manager(args, adapter)
        monkeypatch.setattr(rollout_manager_module, "save_debug_rollout_data", lambda *a, **k: None)

        pack = await manager.generate(rollout_id=8)

        assert pack["rollout_fn_metadata"]["operation_ids"] == ["op-A"]
        [train_data] = fake_store.puts
        assert train_data["batch_execution_lease"] == pack["rollout_fn_metadata"]["lease"]


class TestDisposeClosesTheRolloutFn:
    """External review 0813 §4.7: disposal must invoke the train rollout fn's
    async lifecycle hook — a claim-holding fn (the tinker adapter) terminal-
    fails the claims it still holds before its runtimes are dropped."""

    @pytest.mark.asyncio
    async def test_dispose_awaits_aclose_and_claims_are_terminal_failed(self, monkeypatch, quiet_manager_io):
        args = make_args()
        adapter, queue = make_adapter(args, valid_operation())
        manager = make_manager(args, adapter)
        manager._metric_checker = None
        manager._health_monitors = []
        manager.eval_generate_rollout = adapter  # shared instance, as in production tinker runs
        monkeypatch.setattr(rollout_manager_module.event_analyzer, "run_analysis_from_args", lambda _args: None)

        # Park a real claimed-but-undispatched batch in the adapter.
        await adapter._reconcile(await queue.ready_streams())
        adapter._launch_idle_children()
        for _ in range(200):
            if any(r.ready_output is not None for r in adapter.runtimes.values()):
                break
            await asyncio.sleep(0.01)
        assert queue.state == "CLAIMED"

        await manager.dispose()

        [(operation_ids, error, lease_metadata)] = adapter.abort.aborts
        assert operation_ids == ["op-A"] and lease_metadata is None
        assert "closed" in error
        assert adapter.runtimes == {}


def test_the_manager_owns_no_tinker_identity():
    """Regression 7 (§4.8/§6.3): the generic manager neither imports nor
    reconstructs fn-specific dispatch identity — no tinker name reaches this
    module, and the deleted ``tinker_dispatch_summary`` reconstruction must
    not come back."""
    import inspect

    assert not any("tinker" in name.lower() for name in dir(rollout_manager_module))
    source = inspect.getsource(rollout_manager_module)
    assert "tinker_dispatch_summary" not in source
