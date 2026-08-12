"""Tinker rollout frontend: one claimed operation becomes one stamped batch,
bad payloads fail their own operation, and the selection loop enforces the
homogeneous kind lock with persistent round-robin fairness."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import pytest

import miles.rollout.tinker_backend.rollout_fn as rollout_module
from miles.ray.tinker_backend.config import AdapterRun, AdapterRunConfig
from miles.rollout.base_types import RolloutFnConstructorInput, RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.tinker_backend.rollout_fn import (
    AdapterRolloutRuntime,
    QueueChildRolloutFn,
    TinkerOperationSource,
    TinkerRolloutFn,
)
from miles.utils.tinker_backend import EmptyBatchTimeoutError


def make_run(name="X", reg="rx", slot=3, version=2) -> AdapterRun:
    config = AdapterRunConfig(rank=8, alpha=16, metadata={"team": "t1"})
    return AdapterRun(name=name, config=config, slot=slot, version=version, registration_id=reg)


def make_child(run: AdapterRun) -> QueueChildRolloutFn:
    source = TinkerOperationSource(SimpleNamespace(), run)
    return QueueChildRolloutFn(RolloutFnConstructorInput(args=source.args, data_source=source))


def sample_payload(n=2) -> dict:
    return {
        "batch_id": "batch-7",
        "samples": [
            {"prompt": "p", "tokens": [1, 2, 3, 4], "response_length": 2, "loss_mask": [1, 1]} for _ in range(n)
        ],
        "loss": {"loss_fn": "cross_entropy"},
    }


class _FakeController:
    """Scripted claim results; records failures."""

    def __init__(self, claims):
        self._claims = list(claims)
        self.failed: list[tuple] = []
        self.claim_data_operation = SimpleNamespace(remote=lambda name, reg: self._next_claim())
        self.fail_operation = SimpleNamespace(remote=lambda *args: self.failed.append(args))

    def _next_claim(self):
        return self._claims.pop(0) if self._claims else None


@pytest.fixture()
def fake_ray(monkeypatch):
    monkeypatch.setattr(rollout_module, "ray", SimpleNamespace(get=lambda ref: ref))
    monkeypatch.setattr(rollout_module, "_CLAIM_POLL_S", 0.01)

    def install(controller):
        monkeypatch.setattr(rollout_module, "get_tinker_controller", lambda: controller)

    return install


def op(op_id="op1", kind="forward_backward", payload=None):
    return dict(
        operation_id=op_id,
        name="X",
        registration_id="rx",
        kind=kind,
        payload=sample_payload() if payload is None else payload,
        state="CLAIMED",
    )


class TestQueueChild:
    def test_one_operation_becomes_one_stamped_batch(self, fake_ray):
        fake_ray(_FakeController([op()]))
        output = asyncio.run(make_child(make_run())(RolloutFnTrainInput(rollout_id=0)))

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
        )

    def test_client_supplied_row_index_is_overwritten(self, fake_ray):
        # index is server-owned: a client -1 would alias the DP-padding
        # sentinel (row silently dropped from the result plane) and duplicates
        # would collide in the (slot, row) logprob collector.
        payload = sample_payload()
        payload["samples"][0]["index"] = -1
        payload["samples"][1]["index"] = 0
        fake_ray(_FakeController([op(payload=payload)]))
        output = asyncio.run(make_child(make_run())(RolloutFnTrainInput(rollout_id=0)))
        assert [group[0].index for group in output.samples] == [0, 1]

    def test_child_waits_for_a_claim(self, fake_ray):
        fake_ray(_FakeController([None, None, op()]))
        output = asyncio.run(make_child(make_run())(RolloutFnTrainInput(rollout_id=0)))
        assert output.metadata["operation_id"] == "op1"

    def test_bad_payload_fails_its_operation_and_the_child_continues(self, fake_ray):
        controller = _FakeController([op("bad", payload={"samples": []}), op("good")])
        fake_ray(controller)
        output = asyncio.run(make_child(make_run())(RolloutFnTrainInput(rollout_id=0)))

        assert output.metadata["operation_id"] == "good"
        [(failed_id, error, category)] = controller.failed
        assert failed_id == "bad" and category == "user" and "no samples" in error

    def test_forward_operations_build_batches_too(self, fake_ray):
        payload = {"samples": [{"prompt": "p", "tokens": [1, 2], "response_length": 1, "loss_mask": [1]}]}
        controller = _FakeController([op("fwd", kind="forward", payload=payload)])
        fake_ray(controller)
        output = asyncio.run(make_child(make_run())(RolloutFnTrainInput(rollout_id=0)))
        assert output.metadata["operation_kind"] == "forward"
        assert output.metadata["loss_spec"] is None
        assert controller.failed == []


def ready_runtime(fn: TinkerRolloutFn, name: str, slot: int, kind: str) -> AdapterRolloutRuntime:
    run = make_run(name=name, reg=f"r-{name}", slot=slot)
    runtime = AdapterRolloutRuntime(fn.args, run)
    runtime.state = AdapterRolloutRuntime.READY
    runtime.ready_output = RolloutFnTrainOutput(
        samples=[[SimpleNamespace(adapter=None, metadata={})]],
        metadata=dict(operation_id=f"op-{name}", operation_kind=kind, loss_spec=None),
    )
    fn.runtimes[runtime.tenant] = runtime
    fn._sync_rotation()
    return runtime


def make_fn(soft_target=100) -> TinkerRolloutFn:
    args = SimpleNamespace(
        rollout_batch_size=soft_target,
        n_samples_per_prompt=1,
        tinker_max_coalesce_wait_s=0.05,
        tinker_max_empty_wait_s=0.05,
    )
    return TinkerRolloutFn(RolloutFnConstructorInput(args=args, data_source=None))


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
        along for the commit. The physical slot appears ONLY in the Multi-LoRA
        compatibility helper ``adapter_name_by_slot`` — model routing, never
        operation identity."""
        fn = make_fn()
        first = ready_runtime(fn, "A", 0, "forward_backward")
        selected = asyncio.run(fn._select())
        output = fn._merge(selected)
        assert output.conversion_metadata == {
            "batch_kind": "tinker",
            "tinker_operation_lanes": [0],
            "tinker_loss_by_lane": {0: {}},
            "operation_by_lane": {0: "op-A"},
            "registration_by_lane": {0: ("A", "r-A")},
            "adapter_name_by_slot": {0: "A"},
        }
        assert output.postprocess.pad_to_dp is True
        assert first.state == AdapterRolloutRuntime.IDLE and first.ready_output is None

    def test_merge_of_a_forward_selection_marks_forward_only(self):
        """Forward kind: the same composition with ``tinker_forward_only``
        set — the flag that keeps forward operations gradient-free must
        survive the lane re-keying."""
        fn = make_fn()
        ready_runtime(fn, "A", 0, "forward")
        ready_runtime(fn, "B", 1, "forward")
        selected = asyncio.run(fn._select())
        output = fn._merge(selected)
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
        output = fn._merge(selected)
        assert output.conversion_metadata["tinker_operation_lanes"] == [0, 1]
        assert output.conversion_metadata["registration_by_lane"] == {0: ("A", "r-A"), 1: ("B", "r-B")}
        assert output.conversion_metadata["adapter_name_by_slot"] == {7: "A", 2: "B"}
