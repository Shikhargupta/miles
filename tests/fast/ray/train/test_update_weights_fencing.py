import asyncio
import threading
from unittest.mock import MagicMock

import pytest
from tests.fast.fixtures.controller_fixtures import make_trainer_controller
from tests.fast.ray.train.conftest import make_cell

from miles.ray.train import cell as cell_module
from miles.ray.train.cell import TrainerCell
from miles.ray.train.group import TrainerController, _counts_as_broadcasting
from miles.ray.train.update_weights_liveness import UpdateWeightsLiveness, marks_update_weights_in_flight
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError

_WAIT_TIMEOUT_SECONDS = 5.0


class _FakeBroadcastingActor:
    def __init__(self) -> None:
        self._update_weights_liveness = UpdateWeightsLiveness()
        self.entered = threading.Event()
        self.may_return = threading.Event()

    @marks_update_weights_in_flight
    def update_weights(self) -> None:
        self.entered.set()
        self.may_return.wait(timeout=_WAIT_TIMEOUT_SECONDS)

    @marks_update_weights_in_flight
    def failing_update_weights(self) -> None:
        raise RuntimeError("the broadcast died")

    def is_update_weights_in_flight(self) -> bool:
        return self._update_weights_liveness.is_in_flight()


class _FakeTrainerCell:
    def __init__(self, cell_id: str) -> None:
        self.cell_id = cell_id
        self.cell_index = 0
        self.is_alive = True
        self.is_allocated = True
        self.broadcast_certainly_stopped = False
        self.death_confirmation_attempts = 0
        self.per_worker_answers: list[bool | BaseException] = [False]
        self.broadcast_started = asyncio.Event()

    async def is_update_weights_in_flight_per_worker(self) -> list[bool | BaseException]:
        return list(self.per_worker_answers)

    async def confirm_workers_dead(self) -> bool:
        self.death_confirmation_attempts += 1
        return self.broadcast_certainly_stopped

    async def execute(self, fn_name: str, *, kill_on_failure: bool = True, **kwargs: object) -> list:
        self.per_worker_answers = [True]
        self.broadcast_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _make_controller(cell: _FakeTrainerCell) -> TrainerController:
    return make_trainer_controller(_cells_by_id={cell.cell_id: cell})


async def test_an_errored_cell_that_cannot_answer_keeps_the_window_open() -> None:
    """The dangerous state is errored-but-not-yet-confirmed-dead, where a rank may still be writing weights."""
    cell = _FakeTrainerCell("trainer-engine-actor-0")
    cell.is_alive = False
    cell.per_worker_answers = [WorkerUnreachableError("the worker is gone")]
    controller = _make_controller(cell)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(controller.wait_update_weights_finished(window_id=11), timeout=0.05)


async def test_a_worker_that_cannot_answer_never_hides_a_worker_that_is_still_broadcasting() -> None:
    """One dead worker must not discard the answer of a worker that says it is still writing weights."""
    cell = _FakeTrainerCell("trainer-engine-actor-0")
    cell.per_worker_answers = [True, WorkerUnreachableError("the worker is gone")]
    controller = _make_controller(cell)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(controller.wait_update_weights_finished(window_id=11), timeout=0.05)


async def test_a_cell_whose_workers_are_confirmed_dead_no_longer_counts_as_broadcasting() -> None:
    """Confirmed dead processes cannot be writing, so that is the only thing that releases the window."""
    cell = _FakeTrainerCell("trainer-engine-actor-0")
    cell.is_alive = False
    cell.broadcast_certainly_stopped = True
    cell.per_worker_answers = [WorkerUnreachableError("the worker is gone")]
    controller = _make_controller(cell)

    assert await asyncio.wait_for(controller.wait_update_weights_finished(window_id=11), timeout=_WAIT_TIMEOUT_SECONDS)


async def test_a_cell_that_cannot_answer_is_asked_to_confirm_its_death_again() -> None:
    """Reading a confirmation that was taken once would strand the window on a death confirmed later."""
    cell = _FakeTrainerCell("trainer-engine-actor-0")
    cell.is_alive = False
    cell.broadcast_certainly_stopped = True
    cell.per_worker_answers = [WorkerUnreachableError("the worker is gone")]
    controller = _make_controller(cell)

    assert await asyncio.wait_for(controller.wait_update_weights_finished(window_id=11), timeout=_WAIT_TIMEOUT_SECONDS)
    assert cell.death_confirmation_attempts == 1


async def test_a_cell_whose_death_check_raises_counts_as_broadcasting() -> None:
    """A liveness read that blew up is not an answer, so releasing the window on it releases it on nothing."""

    class _RaisingCell(_FakeTrainerCell):
        async def confirm_workers_dead(self) -> bool:
            raise RuntimeError("the provider refused to answer")

    cell = _RaisingCell("trainer-engine-actor-0")
    cell.is_alive = False
    cell.per_worker_answers = [WorkerUnreachableError("the worker is gone")]
    controller = _make_controller(cell)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(controller.wait_update_weights_finished(window_id=11), timeout=0.05)


def test_a_failed_liveness_read_counts_as_broadcasting_by_decision() -> None:
    """Counting a failure as broadcasting is the deliberate answer, not an exception happening to be truthy."""
    cell = _FakeTrainerCell("trainer-engine-actor-0")

    assert _counts_as_broadcasting(cell=cell, verdict=RuntimeError("the provider refused to answer")) is True
    assert _counts_as_broadcasting(cell=cell, verdict=False) is False


async def test_a_cell_that_answers_for_no_worker_counts_as_broadcasting() -> None:
    """Only every worker denying it releases the window, so an empty answer proves nothing."""
    cell = _FakeTrainerCell("trainer-engine-actor-0")
    cell.per_worker_answers = []
    controller = _make_controller(cell)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(controller.wait_update_weights_finished(window_id=11), timeout=0.05)


async def test_a_cancelled_controller_call_never_reports_the_broadcast_as_finished() -> None:
    """Cancelling the controller coroutine does not cancel the cell's synchronous broadcast body."""
    cell = _FakeTrainerCell("trainer-engine-actor-0")
    controller = _make_controller(cell)

    task = asyncio.create_task(controller.update_weights(info=MagicMock(name="updatable_engines"), rollout_id=3))
    await cell.broadcast_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(controller.wait_update_weights_finished(window_id=11), timeout=0.05)


async def test_the_wait_returns_once_the_cell_reports_the_broadcast_ended() -> None:
    """The wait is answered by the cell, so it ends exactly when the cell stops broadcasting."""
    cell = _FakeTrainerCell("trainer-engine-actor-0")
    cell.per_worker_answers = [True]
    controller = _make_controller(cell)

    waiting = asyncio.create_task(controller.wait_update_weights_finished(window_id=11))
    await asyncio.sleep(0)
    assert not waiting.done()

    cell.per_worker_answers = [False]

    assert await asyncio.wait_for(waiting, timeout=_WAIT_TIMEOUT_SECONDS)


def test_a_running_broadcast_body_marks_itself_in_flight() -> None:
    """Liveness is the synchronous body's own state, which is what an abort has to wait out."""
    actor = _FakeBroadcastingActor()
    thread = threading.Thread(target=actor.update_weights)

    thread.start()
    try:
        actor.entered.wait(timeout=_WAIT_TIMEOUT_SECONDS)
        assert actor.is_update_weights_in_flight()
    finally:
        actor.may_return.set()
        thread.join(timeout=_WAIT_TIMEOUT_SECONDS)

    assert not actor.is_update_weights_in_flight()


def test_a_broadcast_body_that_raises_clears_its_in_flight_mark() -> None:
    """A broadcast that died is finished, and a mark left behind would stall every later abort."""
    actor = _FakeBroadcastingActor()

    with pytest.raises(RuntimeError, match="the broadcast died"):
        actor.failing_update_weights()

    assert not actor.is_update_weights_in_flight()


class _FakeWorkerHandle(BaseWorkerHandle):
    def __init__(self, *, is_dead: bool) -> None:
        self.is_dead = is_dead
        self.kill_self_call_count: int = 0

    async def kill_self(self) -> None:
        self.kill_self_call_count += 1

    async def wait_ready(self, *, timeout: float) -> None:
        raise NotImplementedError

    async def _probe_is_dead(self) -> bool:
        return self.is_dead

    async def is_update_weights_in_flight(self) -> bool:
        raise WorkerUnreachableError("the worker is gone")


class _BroadcastingWorkerHandle(_FakeWorkerHandle):
    async def is_update_weights_in_flight(self) -> bool:
        return True


def _make_cell_with_handles(handles: list[_FakeWorkerHandle], monkeypatch: pytest.MonkeyPatch) -> TrainerCell:
    cell = make_cell(0)
    monkeypatch.setattr(cell, "_get_worker_handles", lambda: handles)
    return cell


async def test_marking_a_cell_errored_does_not_confirm_that_its_broadcast_stopped() -> None:
    """The errored mark is set before the kill even starts, so it can never stand in for confirmed death."""
    cell = make_cell(0)

    cell._mark_as_errored()

    assert not cell.is_alive
    assert not cell.broadcast_certainly_stopped


async def test_a_death_probe_that_times_out_does_not_confirm_that_the_broadcast_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that never answers the death probe may still be writing weights, so it stays unconfirmed."""
    monkeypatch.setattr(cell_module, "CONFIRM_DEAD_TIMEOUT_S", 0.0)
    cell = _make_cell_with_handles([_FakeWorkerHandle(is_dead=True), _FakeWorkerHandle(is_dead=False)], monkeypatch)

    await cell._kill_workers_and_confirm_dead()

    assert not cell.broadcast_certainly_stopped


async def test_every_worker_confirmed_dead_confirms_that_the_broadcast_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dead processes cannot write weights, which is the one state that lets the window close."""
    monkeypatch.setattr(cell_module, "CONFIRM_DEAD_TIMEOUT_S", 0.0)
    cell = _make_cell_with_handles([_FakeWorkerHandle(is_dead=True), _FakeWorkerHandle(is_dead=True)], monkeypatch)

    await cell._kill_workers_and_confirm_dead()

    assert cell.broadcast_certainly_stopped


async def test_the_liveness_query_never_kills_the_cell_it_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The query is a probe, so it must not be the thing that tears the broadcasting cell down."""
    handle = _FakeWorkerHandle(is_dead=False)
    cell = _make_cell_with_handles([handle], monkeypatch)

    answers = await cell.is_update_weights_in_flight_per_worker()

    assert [type(answer) for answer in answers] == [WorkerUnreachableError]
    assert handle.kill_self_call_count == 0
    assert not cell.is_errored
    assert not cell.broadcast_certainly_stopped


async def test_a_worker_that_cannot_answer_never_discards_the_answer_of_one_that_can(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cell answers per worker, so the rank that says it is still writing survives its unreachable neighbour."""
    broadcasting = _BroadcastingWorkerHandle(is_dead=False)
    cell = _make_cell_with_handles([broadcasting, _FakeWorkerHandle(is_dead=False)], monkeypatch)

    answers = await cell.is_update_weights_in_flight_per_worker()

    assert [answers[0], type(answers[1])] == [True, WorkerUnreachableError]


async def test_a_death_that_could_not_be_confirmed_at_kill_time_can_be_confirmed_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-shot confirmation would keep the window's lock and its paused health checking for the whole run."""
    monkeypatch.setattr(cell_module, "CONFIRM_DEAD_TIMEOUT_S", 0.0)
    monkeypatch.setattr(cell_module, "RECONFIRM_DEAD_TIMEOUT_S", 0.0)
    handle = _FakeWorkerHandle(is_dead=False)
    cell = _make_cell_with_handles([handle], monkeypatch)
    await cell._kill_workers_and_confirm_dead()
    assert not cell.broadcast_certainly_stopped

    handle.is_dead = True

    assert await cell.confirm_workers_dead() is True
    assert cell.broadcast_certainly_stopped


async def test_a_confirmed_death_is_never_withdrawn_by_a_later_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The window is released on this confirmation, so withdrawing it would strand a window nothing can close."""
    monkeypatch.setattr(cell_module, "CONFIRM_DEAD_TIMEOUT_S", 0.0)
    monkeypatch.setattr(cell_module, "RECONFIRM_DEAD_TIMEOUT_S", 0.0)
    handle = _FakeWorkerHandle(is_dead=True)
    cell = _make_cell_with_handles([handle], monkeypatch)
    await cell._kill_workers_and_confirm_dead()
    assert cell.broadcast_certainly_stopped

    handle.is_dead = False
    await cell._kill_workers_and_confirm_dead()

    assert cell.broadcast_certainly_stopped
    assert await cell.confirm_workers_dead() is True


async def test_a_worker_that_stays_alive_is_never_confirmed_dead_by_a_later_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-probing may only ever turn unconfirmed into confirmed, never wave a living rank through."""
    monkeypatch.setattr(cell_module, "CONFIRM_DEAD_TIMEOUT_S", 0.0)
    monkeypatch.setattr(cell_module, "RECONFIRM_DEAD_TIMEOUT_S", 0.0)
    cell = _make_cell_with_handles([_FakeWorkerHandle(is_dead=True), _FakeWorkerHandle(is_dead=False)], monkeypatch)
    await cell._kill_workers_and_confirm_dead()

    assert await cell.confirm_workers_dead() is False
    assert not cell.broadcast_certainly_stopped
