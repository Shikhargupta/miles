from __future__ import annotations

import asyncio
import logging
import time
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from miles.ray import hot_restart as hot_restart_module
from miles.ray.hot_restart import (
    _abort_inflight_rollouts,
    init_or_load_trainer,
    init_or_reset_inference_controller,
    quiesce_and_claim_trainers,
    wait_until_rollout_executor_is_free,
)
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.train.group import TrainerController
from miles.utils.context_lock import ContextLock
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_spec import NamedHostAndPorts

_TRAINER_ID = "policy_a-actor"
_OTHER_TRAINER_ID = "policy_b-actor"
_STALLED_SECONDS = 5.0
_SHORT_BUDGET_SECONDS = 0.05


class _FakeTrainer:
    def __init__(self, *, initialized: bool, idle_seconds: float = 0.0, load_seconds: float = 0.0) -> None:
        self.initialized = initialized
        self.idle_seconds = idle_seconds
        self.load_seconds = load_seconds
        self.calls: list[str] = []
        self.idle_timeouts: list[float] = []

    async def is_initialized(self) -> bool:
        self.calls.append("is_initialized")
        return self.initialized

    async def init(self, model_args: Namespace) -> list[Any]:
        self.calls.append("init")
        return [7]

    async def load_state(self) -> list[Any]:
        self.calls.append("load_state")
        await asyncio.sleep(self.load_seconds)
        return [3]

    async def wait_idle(self, *, timeout: float) -> None:
        self.calls.append("wait_idle")
        self.idle_timeouts.append(timeout)
        await asyncio.sleep(self.idle_seconds)

    async def claim_driver_epoch(self) -> None:
        self.calls.append("claim_driver_epoch")


class _FakeInferenceController:
    def __init__(
        self,
        *,
        initialized: bool,
        broadcast_lock_held: bool = False,
        busy: bool = False,
        wedged: bool = False,
        abort_error: Exception | None = None,
        fleet_incomplete: bool = False,
    ) -> None:
        self.initialized = initialized
        self.broadcast_lock_held = broadcast_lock_held
        self.busy = busy
        self.wedged = wedged
        self.abort_error = abort_error
        self.fleet_incomplete = fleet_incomplete
        self.calls: list[str] = []
        self.idle_timeouts: list[float] = []

    async def is_initialized(self) -> bool:
        return self.initialized

    async def init(self) -> None:
        self.calls.append("init")

    async def wait_idle(self, *, timeout: float) -> None:
        self.calls.append("wait_idle")
        self.idle_timeouts.append(timeout)
        if self.busy:
            raise TimeoutError("InferenceController was still busy")

    async def reset_broadcast_lock(self) -> bool:
        self.calls.append("reset_broadcast_lock")
        await self._maybe_hang()
        return self.broadcast_lock_held

    async def abort_all(self) -> None:
        self.calls.append("abort_all")
        await self._maybe_hang()
        if self.abort_error is not None:
            raise self.abort_error

    async def wait_expected_num_cells(self, timeout: float) -> None:
        self.calls.append("wait_expected_num_cells")
        if self.fleet_incomplete:
            raise TimeoutError("the fleet is short of engines")

    async def claim_driver_epoch(self) -> None:
        self.calls.append("claim_driver_epoch")

    async def _maybe_hang(self) -> None:
        if self.wedged:
            await asyncio.sleep(_STALLED_SECONDS)


class _FakeExecutor:
    def __init__(self, answers: list[bool | Exception], *, ready_error: Exception | None = None) -> None:
        self._answers = list(answers)
        self._ready_error = ready_error
        self.ready_calls = 0

    async def is_initialized(self) -> bool:
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def wait_ready(self, *, timeout: float) -> None:
        self.ready_calls += 1
        if self._ready_error is not None:
            raise self._ready_error


@pytest.fixture
def fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hot_restart_module, "_EXECUTOR_POLL_INTERVAL_SECONDS", 0.01)


@pytest.fixture
def short_take_over_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hot_restart_module, "TAKE_OVER_GATE_TIMEOUT_SECONDS", _SHORT_BUDGET_SECONDS)


def _freeze_clock(monkeypatch: pytest.MonkeyPatch, readings: list[float]) -> None:
    pending = list(readings)

    def _monotonic() -> float:
        return pending.pop(0) if len(pending) > 1 else pending[0]

    monkeypatch.setattr(hot_restart_module, "time", SimpleNamespace(monotonic=_monotonic))


class TestGateOneTheTrainersAreClaimed:
    async def test_a_trainer_that_never_ran_is_not_waited_for_but_is_still_claimed(self):
        """A cold start must be untouched by the resume protocol, and still fence out any older driver."""
        trainer = _FakeTrainer(initialized=False)

        await quiesce_and_claim_trainers({_TRAINER_ID: trainer})

        assert trainer.calls == ["claim_driver_epoch", "is_initialized"]

    async def test_a_surviving_trainer_is_fenced_before_it_is_waited_out(self):
        """Fencing first is what makes the drain converge: the zombie script's new calls are refused meanwhile."""
        trainer = _FakeTrainer(initialized=True)

        await quiesce_and_claim_trainers({_TRAINER_ID: trainer})

        assert trainer.calls == ["claim_driver_epoch", "is_initialized", "wait_idle"]

    async def test_the_wait_carries_what_is_left_of_the_gate_budget(self, monkeypatch: pytest.MonkeyPatch):
        """One budget covers the whole gate, so the wait must be handed the budget minus what the gate already spent."""
        _freeze_clock(monkeypatch, [0.0, 1.5])
        trainer = _FakeTrainer(initialized=True)

        await quiesce_and_claim_trainers({_TRAINER_ID: trainer})

        assert trainer.idle_timeouts == [pytest.approx(hot_restart_module.TAKE_OVER_GATE_TIMEOUT_SECONDS - 1.5)]

    async def test_a_second_trainer_inherits_what_the_first_one_left_of_the_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The audit's failure was serial 600s waits, one per policy, adding up to an unbounded take-over."""
        _freeze_clock(monkeypatch, [0.0, 0.0, 0.2])
        first = _FakeTrainer(initialized=True)
        second = _FakeTrainer(initialized=True)

        await quiesce_and_claim_trainers({_TRAINER_ID: first, _OTHER_TRAINER_ID: second})

        assert first.idle_timeouts == [pytest.approx(hot_restart_module.TAKE_OVER_GATE_TIMEOUT_SECONDS)]
        assert second.idle_timeouts == [pytest.approx(hot_restart_module.TAKE_OVER_GATE_TIMEOUT_SECONDS - 0.2)]

    async def test_a_take_over_that_reaches_no_trainer_at_all_is_refused(self):
        """A gate that claims nothing would let the previous script keep driving every trainer of the run."""
        with pytest.raises(AssertionError, match="at least one trainer"):
            await quiesce_and_claim_trainers({})

    async def test_a_trainer_that_never_goes_idle_fails_loud(self, short_take_over_budget: None):
        """Taking a trainer over mid-step is exactly the corruption these gates exist to refuse."""
        trainer = _FakeTrainer(initialized=True, idle_seconds=_STALLED_SECONDS)

        started = time.monotonic()
        with pytest.raises(TimeoutError, match="reloading a checkpoint into a model a train step is still writing"):
            await quiesce_and_claim_trainers({_TRAINER_ID: trainer})

        assert time.monotonic() - started < _STALLED_SECONDS
        assert trainer.calls == ["claim_driver_epoch", "is_initialized", "wait_idle"]

    async def test_trainers_that_disagree_about_being_initialized_stop_the_run(self):
        """A mixed fleet must stop here, before gate 2 aborts generations and resets a lock for a run it cannot resume."""
        with pytest.raises(AssertionError, match="disagree about whether"):
            await quiesce_and_claim_trainers(
                {_TRAINER_ID: _FakeTrainer(initialized=True), _OTHER_TRAINER_ID: _FakeTrainer(initialized=False)}
            )

    async def test_every_trainer_of_the_run_is_claimed(self):
        """A trainer left unclaimed still answers the previous script, which then drives it alongside this one."""
        first = _FakeTrainer(initialized=False)
        second = _FakeTrainer(initialized=False)

        await quiesce_and_claim_trainers({_TRAINER_ID: first, _OTHER_TRAINER_ID: second})

        assert first.calls[0] == "claim_driver_epoch" and second.calls[0] == "claim_driver_epoch"


class TestTheTrainerStateIsRolledBack:
    async def test_a_cold_trainer_is_initialized_and_a_resumed_one_only_reloads(self):
        """Init rebuilds a trainer; a survivor must only be rolled back to its checkpoint."""
        cold = _FakeTrainer(initialized=False)
        warm = _FakeTrainer(initialized=True)

        assert await init_or_load_trainer(cold, Namespace(), trainer_id=_TRAINER_ID, resumed=False) == [7]
        assert await init_or_load_trainer(warm, Namespace(), trainer_id=_TRAINER_ID, resumed=True) == [3]
        assert cold.calls == ["init"] and warm.calls == ["load_state"]

    async def test_a_reload_that_never_returns_fails_loud(self, short_take_over_budget: None):
        """A trainer wedged inside load_state would otherwise leave the run waiting on it forever."""
        trainer = _FakeTrainer(initialized=True, load_seconds=_STALLED_SECONDS)

        started = time.monotonic()
        with pytest.raises(TimeoutError, match="back to its checkpoint"):
            await init_or_load_trainer(trainer, Namespace(), trainer_id=_TRAINER_ID, resumed=True)

        assert time.monotonic() - started < _STALLED_SECONDS

    async def test_each_trainer_gets_a_reload_budget_of_its_own(self, monkeypatch: pytest.MonkeyPatch):
        """A reload is bounded per trainer, so one slow policy cannot starve the policy reloaded after it."""
        monkeypatch.setattr(hot_restart_module, "TAKE_OVER_GATE_TIMEOUT_SECONDS", 0.3)
        first = _FakeTrainer(initialized=True, load_seconds=0.2)
        second = _FakeTrainer(initialized=True, load_seconds=0.2)

        assert await init_or_load_trainer(first, Namespace(), trainer_id=_TRAINER_ID, resumed=True) == [3]
        assert await init_or_load_trainer(second, Namespace(), trainer_id=_OTHER_TRAINER_ID, resumed=True) == [3]


class TestGateTwoTheInferenceSideIsReset:
    async def test_a_fresh_controller_is_initialized_and_claimed(self):
        """A cold start initializes the inference side as it always did, and then owns it."""
        controller = _FakeInferenceController(initialized=False)

        await init_or_reset_inference_controller(controller)

        assert controller.calls == ["init", "claim_driver_epoch"]

    async def test_the_abort_runs_only_once_the_whole_expected_fleet_is_present(self):
        """A cell that was away during the abort would rejoin still generating the previous run's requests."""
        controller = _FakeInferenceController(initialized=True)

        await init_or_reset_inference_controller(controller)

        assert controller.calls == [
            "wait_idle",
            "reset_broadcast_lock",
            "wait_expected_num_cells",
            "abort_all",
            "claim_driver_epoch",
        ]

    async def test_a_broadcast_lock_left_held_is_announced(self, caplog):
        """An operator reading the log has to know the previous script died inside a weight update."""
        controller = _FakeInferenceController(initialized=True, broadcast_lock_held=True)

        with caplog.at_level(logging.WARNING):
            await init_or_reset_inference_controller(controller)

        assert "broadcast lock" in caplog.text

    async def test_a_call_of_the_previous_script_that_never_ends_fails_loud(self):
        """The script that died inside start_update_weights is exactly the case this wait exists for."""
        controller = _FakeInferenceController(initialized=True, busy=True)

        with pytest.raises(TimeoutError, match="calls of the previous orchestration script to end"):
            await init_or_reset_inference_controller(controller)

        assert "abort_all" not in controller.calls

    async def test_the_gate_shares_one_budget_across_all_of_its_steps(self):
        """Four operations of 600s each is 40 minutes, which is not the bounded gate the design promises."""
        controller = _FakeInferenceController(initialized=True)

        await init_or_reset_inference_controller(controller)

        assert controller.idle_timeouts[0] <= hot_restart_module.TAKE_OVER_GATE_TIMEOUT_SECONDS

    async def test_a_controller_that_never_answers_fails_loud(self, short_take_over_budget: None):
        """Hanging here would leave the operator with a silent hot restart that never starts training."""
        controller = _FakeInferenceController(initialized=True, wedged=True)

        started = time.monotonic()
        with pytest.raises(TimeoutError, match="reset the broadcast lock"):
            await init_or_reset_inference_controller(controller)

        assert time.monotonic() - started < _STALLED_SECONDS

    async def test_a_take_over_waits_for_the_whole_fleet_just_as_a_cold_start_does(self):
        """Generating on half a fleet because an engine was being rescheduled is not what the command asked for."""
        controller = _FakeInferenceController(initialized=True, fleet_incomplete=True)

        with pytest.raises(TimeoutError, match="every engine this run expects"):
            await init_or_reset_inference_controller(controller)

    async def test_a_cell_that_refused_the_abort_fails_the_take_over(self):
        """The whole fleet was already there, so a refusal is a sick engine that may still be generating."""
        controller = _FakeInferenceController(
            initialized=True, abort_error=RuntimeError("west-engine-0-0-0 refused the abort")
        )

        with pytest.raises(RuntimeError, match="west-engine-0-0-0"):
            await init_or_reset_inference_controller(controller)

        assert "claim_driver_epoch" not in controller.calls


class TestAbortInflightRollouts:
    @staticmethod
    def _expires_at(seconds: float = 30.0) -> float:
        return time.monotonic() + seconds

    async def test_a_refusing_cell_stops_the_run_instead_of_being_logged_past(self):
        """A cell that kept generating pollutes this run's data, so the take-over cannot continue over it."""
        controller = _FakeInferenceController(initialized=True, abort_error=RuntimeError("the cell refused"))

        with pytest.raises(RuntimeError, match="the cell refused"):
            await _abort_inflight_rollouts(controller, expires_at=self._expires_at())

    async def test_a_fleet_that_answered_every_abort_is_announced_quiet(self, caplog):
        """The ordinary take-over says so, and an operator reads that line as a fleet with no request left on it."""
        controller = _FakeInferenceController(initialized=True)

        with caplog.at_level(logging.INFO):
            await _abort_inflight_rollouts(controller, expires_at=self._expires_at())

        assert "quiet inference fleet" in caplog.text


class _AbortingCell:
    def __init__(self, *, cell_id: str, failure: Exception | None = None) -> None:
        self.meta = SimpleNamespace(cell_id=cell_id, needs_offload=False, num_gpus_per_engine=1, gpu_offset=0)
        self.is_pending_weights_or_serving = True
        self.failure = failure
        self.aborted = False

    async def abort_all(self) -> None:
        self.aborted = True
        if self.failure is not None:
            raise self.failure


class _UnaddressedEngineProvider:
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        raise AssertionError(f"aborting a fleet never addresses a cell ({worker_name=})")


class TestEveryCellOfAServerIsAborted:
    @staticmethod
    def _server(cells: list[_AbortingCell]) -> RolloutServer:
        return RolloutServer(
            server_cells={cell.meta.cell_id: cell for cell in cells},
            args=SimpleNamespace(colocate=True),
            context_lock=ContextLock("InferenceController"),
            engine_provider=_UnaddressedEngineProvider(),
        )

    async def test_a_fleet_that_answered_every_abort_reports_no_refusal(self):
        """The ordinary take-over aborts every cell, and the gate above it has nothing to act on."""
        cells = [_AbortingCell(cell_id="west-0"), _AbortingCell(cell_id="west-1")]
        server = self._server(cells)

        async with server.context_lock:
            assert await server.abort_all() is None

        assert all(cell.aborted for cell in cells)

    async def test_every_refusing_cell_is_logged_before_the_run_stops(self, caplog):
        """An operator has to see every sick engine, not only the one the gather happened to order first."""
        cells = [
            _AbortingCell(cell_id="west-0", failure=RuntimeError("west-0 refused")),
            _AbortingCell(cell_id="west-1", failure=RuntimeError("west-1 refused")),
        ]
        server = self._server(cells)

        with caplog.at_level(logging.ERROR):
            async with server.context_lock:
                with pytest.raises(RuntimeError, match="west-0 refused"):
                    await server.abort_all()

        assert "west-0" in caplog.text and "west-1" in caplog.text
        assert all(cell.aborted for cell in cells)


class TestEveryServerOfTheFleetIsAborted:
    @staticmethod
    def _controller(cells: dict[str, _AbortingCell]) -> InferenceController:
        context_lock = ContextLock("InferenceController")
        controller = InferenceController.__new__(InferenceController)
        controller.context_lock = context_lock
        controller.servers = {
            model_name: RolloutServer(
                server_cells={cell.meta.cell_id: cell},
                args=SimpleNamespace(colocate=True),
                context_lock=context_lock,
                engine_provider=_UnaddressedEngineProvider(),
                model_name=model_name,
            )
            for model_name, cell in cells.items()
        }
        return controller

    async def test_a_fleet_whose_every_server_answered_reports_no_refusal(self):
        """One abort per model is the ordinary take-over, and the gate above it has nothing to act on."""
        cells = {"actor": _AbortingCell(cell_id="actor-0"), "ref": _AbortingCell(cell_id="ref-0")}

        assert await self._controller(cells).abort_all() is None

        assert all(cell.aborted for cell in cells.values())

    async def test_a_refusing_cell_of_every_server_is_logged_before_the_run_stops(self, caplog):
        """One raising server must not hide the sick engines of the servers beside it, nor orphan their failures."""
        cells = {
            "actor": _AbortingCell(cell_id="actor-0", failure=RuntimeError("actor-0 refused")),
            "ref": _AbortingCell(cell_id="ref-0", failure=RuntimeError("ref-0 refused")),
        }

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="refused"):
                await self._controller(cells).abort_all()

        assert "actor-0" in caplog.text and "ref-0" in caplog.text
        assert all(cell.aborted for cell in cells.values())


class TestWaitUntilRolloutExecutorIsFree:
    async def test_a_fresh_executor_is_accepted_at_once(self):
        """The normal case must not pay a polling delay."""
        executor = _FakeExecutor([False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

    async def test_an_executor_of_the_previous_script_is_waited_out(self, fast_polling: None):
        """Initializing the old executor a second time would drive the process that is about to die."""
        executor = _FakeExecutor([True, True, False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

    async def test_the_poll_interval_paces_the_wait(self, monkeypatch: pytest.MonkeyPatch):
        """A busy loop against an executor that answers at once would hammer it for the whole budget."""
        monkeypatch.setattr(hot_restart_module, "_EXECUTOR_POLL_INTERVAL_SECONDS", 0.2)
        executor = _FakeExecutor([True, False])

        started = time.monotonic()
        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

        assert time.monotonic() - started >= 0.2

    async def test_an_executor_being_replaced_mid_wait_is_re_baselined(self, fast_polling: None):
        """The executor is expected to restart during this wait, so its boot uuid change is not a violation."""
        executor = _FakeExecutor([True, ServerRestartedError("replaced"), False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

        assert executor.ready_calls == 1

    async def test_an_executor_that_never_frees_up_times_out(self, fast_polling: None):
        """A new script must not silently share a run with the executor of its predecessor."""
        executor = _FakeExecutor([True] * 100)

        with pytest.raises(TimeoutError, match="not a fresh process"):
            await wait_until_rollout_executor_is_free(executor, timeout=_SHORT_BUDGET_SECONDS)

    async def test_an_executor_pod_being_recreated_is_waited_out(self, fast_polling: None):
        """Replacing the pod is the whole point, and it is unreachable for as long as that takes."""
        executor = _FakeExecutor([True, WorkerUnreachableError("pod is gone"), False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

    async def test_a_transport_error_is_waited_out_too(self, fast_polling: None):
        """A connection refused while kubernetes reschedules the pod is the expected state, not a failure."""
        executor = _FakeExecutor([httpx.ConnectError("refused"), False])

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)

    async def test_an_executor_that_stays_unreachable_reports_a_timeout(self, fast_polling: None):
        """The message has to say the executor never came back, not leak a transport error."""
        executor = _FakeExecutor([WorkerUnreachableError("pod is gone")] * 100)

        with pytest.raises(TimeoutError, match="not a fresh process"):
            await wait_until_rollout_executor_is_free(executor, timeout=_SHORT_BUDGET_SECONDS)

    async def test_a_readiness_wait_that_fails_is_retried_rather_than_fatal(self, fast_polling: None):
        """wait_ready itself throws while the replacement pod is still being scheduled."""
        executor = _FakeExecutor([ServerRestartedError("replaced"), False], ready_error=WorkerUnreachableError("no"))

        await wait_until_rollout_executor_is_free(executor, timeout=5.0)


class TestTheTakeOverSurfaceCrossesTheWire:
    @pytest.mark.parametrize(
        "worker_cls, methods",
        [
            (TrainerController, {"is_initialized", "load_state"}),
            (RolloutExecutor, {"is_initialized"}),
            (InferenceController, {"is_initialized", "reset_broadcast_lock", "abort_all", "wait_expected_num_cells"}),
        ],
    )
    def test_the_take_over_surface_is_exposed_over_rpc(self, worker_cls: type, methods: set[str]):
        """A restarted orchestration script drives the whole take-over through exactly these rpc methods."""
        assert methods <= set(collect_rpc_method_specs(worker_cls))
