from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from miles.ray.train.group import RayTrainGroup
from miles.utils.ft_utils.api_server.handles import (
    _ActorCellHandler,
    _CellHandler,
    _RolloutCellHandler,
    compute_engine_cell_ids,
)
from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.test_utils.fault_injector import FailureMode

from .conftest import (
    MockInferenceController,
    MockRayTrainCell,
    MockRemoteCall,
    MockWorkerManager,
    make_cell_summaries,
    make_mock_group,
)


class TestActorCellHandler:
    async def test_every_cell_of_the_group_is_listed(self) -> None:
        """The api server addresses trainer cells by their index in the group."""
        handler = _ActorCellHandler(group=make_mock_group([MockRayTrainCell(), MockRayTrainCell()]))
        assert await handler.list_cell_keys() == ["0", "1"]

    def test_cell_type(self) -> None:
        handler = _ActorCellHandler(group=make_mock_group([MockRayTrainCell()]))
        assert handler.cell_type == "actor"
        assert handler.compute_cell_name("0") == "actor-0"

    @pytest.mark.asyncio
    async def test_get_cell_returns_full_cell_structure(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        handler = _ActorCellHandler(group=group)
        cell = await handler.get_cell("0")

        assert cell.model_dump() == {
            "apiVersion": "miles.io/v1",
            "kind": "Cell",
            "metadata": {
                "name": "actor-0",
                "labels": {
                    "miles.io/cell-type": "actor",
                    "miles.io/cell-index": "0",
                },
            },
            "spec": {"suspend": False},
            "status": {
                "phase": "Running",
                "conditions": [
                    {
                        "type": "Allocated",
                        "status": "True",
                        "reason": None,
                        "message": None,
                        "lastTransitionTime": None,
                    },
                    {"type": "Healthy", "status": "True", "reason": None, "message": None, "lastTransitionTime": None},
                ],
            },
        }

    @pytest.mark.asyncio
    async def test_get_cell_suspended(self) -> None:
        group = make_mock_group(
            [
                MockRayTrainCell(
                    phase="Suspended",
                    conditions=[
                        {"type": "Allocated", "status": "False"},
                        {"type": "Healthy", "status": "False"},
                    ],
                    is_stopped=True,
                )
            ]
        )
        handler = _ActorCellHandler(group=group)
        cell = await handler.get_cell("0")

        assert cell.spec.suspend is True
        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_suspend_delegates_to_group(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        group.stop_cell = MagicMock()
        handler = _ActorCellHandler(group=group)
        await handler.suspend("2")
        group.stop_cell.assert_called_once_with(2)

    @pytest.mark.asyncio
    async def test_resume_delegates_to_group(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        group.start_cell = MagicMock()
        handler = _ActorCellHandler(group=group)
        await handler.resume("1")
        group.start_cell.assert_called_once_with(1)


ENGINE_CELL_ID = "inference-engine-0-0-0"


def _make_rollout_handler(
    *,
    cell_id: str = ENGINE_CELL_ID,
    suspended: bool = False,
    health: TriState | None = TriState.TRUE,
) -> tuple[_RolloutCellHandler, MockWorkerManager, MockInferenceController]:
    manager = MockWorkerManager(make_cell_summaries(cell_id, suspended=suspended))
    controller = MockInferenceController({cell_id: health} if health is not None else {})
    handler = _RolloutCellHandler(worker_manager=manager, inference_controller=controller)
    return handler, manager, controller


class TestRolloutCellHandler:
    @pytest.mark.asyncio
    async def test_a_healthy_cell_is_reported_running(self) -> None:
        """A serving engine that answers its probe is what the heal loop must leave alone."""
        handler, _manager, _controller = _make_rollout_handler()

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.metadata.name == "rollout-inference-engine-0-0-0"
        assert cell.metadata.labels["miles.io/cell-type"] == "rollout"
        assert cell.status.phase == "Running"
        assert cell.spec.suspend is False
        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.TRUE),
        ]

    @pytest.mark.asyncio
    async def test_a_failing_probe_is_reported_unhealthy(self) -> None:
        """This is the signal the mini ft controller heals on."""
        handler, _manager, _controller = _make_rollout_handler(health=TriState.FALSE)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Running"
        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.FALSE),
        ]

    @pytest.mark.asyncio
    async def test_suspension_comes_from_the_worker_manager(self) -> None:
        """The manager owns the processes, so it alone knows a cell was suspended."""
        handler, _manager, _controller = _make_rollout_handler(suspended=True)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.spec.suspend is True
        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_a_suspended_cell_reports_no_health(self) -> None:
        """Its engine is gone, so any health the controller still remembers is stale."""
        handler, _manager, _controller = _make_rollout_handler(suspended=True, health=TriState.TRUE)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert [(c.type, c.status) for c in cell.status.conditions] == [("Allocated", TriState.FALSE)]

    @pytest.mark.asyncio
    async def test_a_cell_the_controller_does_not_track_yet_is_unknown(self) -> None:
        """A cell exists in the manager before reconcile hands it to the controller."""
        handler, _manager, _controller = _make_rollout_handler(health=None)

        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert [(c.type, c.status) for c in cell.status.conditions] == [
            ("Allocated", TriState.TRUE),
            ("Healthy", TriState.UNKNOWN),
        ]

    @pytest.mark.asyncio
    async def test_health_is_read_without_awaiting_the_controller(self) -> None:
        """The api server serves from its own event loop, so the controller is read synchronously."""
        handler, _manager, controller = _make_rollout_handler()

        await handler.get_cell(ENGINE_CELL_ID)

        assert controller.health_status_calls == 1

    def test_cell_type_is_rollout(self) -> None:
        handler, _manager, _controller = _make_rollout_handler(cell_id="inference-engine-0-0-3")
        assert handler.cell_type == "rollout"
        assert handler.compute_cell_name("inference-engine-0-0-3") == "rollout-inference-engine-0-0-3"

    async def test_only_engine_cells_are_listed(self) -> None:
        """Routers and session servers are cells of the manager too, but not rollout cells."""
        manager = MockWorkerManager(
            {
                **make_cell_summaries("inference-engine-0-0-0"),
                **make_cell_summaries("miles-router-0", engine=False),
            }
        )
        handler = _RolloutCellHandler(worker_manager=manager, inference_controller=MockInferenceController())

        assert await handler.list_cell_keys() == ["inference-engine-0-0-0"]

    async def test_a_suspended_cell_is_still_listed(self) -> None:
        """A suspended cell that vanished from the listing could never be resumed."""
        handler, _manager, _controller = _make_rollout_handler(suspended=True)

        assert await handler.list_cell_keys() == [ENGINE_CELL_ID]

    async def test_listing_reads_its_sources_once_for_all_cells(self) -> None:
        """This listing is polled for the life of the run, so it must not scale in round trips."""
        manager = MockWorkerManager(make_cell_summaries("engine-a", "engine-b", "engine-c"))
        controller = MockInferenceController()
        handler = _RolloutCellHandler(worker_manager=manager, inference_controller=controller)

        cells = await handler.list_cells()

        assert len(cells) == 3
        assert controller.health_status_calls == 1

    @pytest.mark.asyncio
    async def test_suspend_stops_the_cell_in_the_worker_manager(self) -> None:
        """The manager owns the processes, so healing goes through it, not the controller."""
        handler, manager, _controller = _make_rollout_handler()

        await handler.suspend(ENGINE_CELL_ID)

        assert manager.stopped_cells == [[ENGINE_CELL_ID]]

    @pytest.mark.asyncio
    async def test_resume_starts_the_cell_in_the_worker_manager(self) -> None:
        """Resume relaunches the cell, which reconcile then observes as a new generation."""
        handler, manager, _controller = _make_rollout_handler(suspended=True)

        await handler.resume(ENGINE_CELL_ID)

        assert manager.started_cells == [[ENGINE_CELL_ID]]

    @pytest.mark.asyncio
    async def test_suspending_only_touches_the_named_cell(self) -> None:
        """Healing one engine must leave its siblings serving."""
        manager = MockWorkerManager(make_cell_summaries("engine-a", "engine-b"))
        handler = _RolloutCellHandler(worker_manager=manager, inference_controller=MockInferenceController())

        await handler.suspend("engine-a")

        assert manager.stopped_cells == [["engine-a"]]

    @pytest.mark.asyncio
    async def test_a_suspended_cell_reports_suspended_afterwards(self) -> None:
        """The heal loop reads back the status it just asked for."""
        handler, _manager, _controller = _make_rollout_handler()

        await handler.suspend(ENGINE_CELL_ID)
        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_a_resumed_cell_reports_running_afterwards(self) -> None:
        """A resumed cell must leave the state the heal loop is waiting on."""
        handler, _manager, _controller = _make_rollout_handler(suspended=True)

        await handler.resume(ENGINE_CELL_ID)
        cell = await handler.get_cell(ENGINE_CELL_ID)

        assert cell.status.phase == "Running"


class TestComputeEngineCellIds:
    def test_only_cells_carrying_a_model_are_engine_cells(self) -> None:
        """Routers and session servers are cells too, but healing them is not rollout ft."""
        summaries = {
            **make_cell_summaries("inference-engine-0-0-1", "inference-engine-0-0-0"),
            **make_cell_summaries("miles-router-0", engine=False),
        }

        assert compute_engine_cell_ids(summaries) == ["inference-engine-0-0-0", "inference-engine-0-0-1"]

    def test_nothing_matches_without_engine_cells(self) -> None:
        """A train-only deployment exposes no rollout cells."""
        assert compute_engine_cell_ids(make_cell_summaries("miles-router-0", engine=False)) == []


class _FakeRemoteMethod:
    def __init__(self) -> None:
        self.remote_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args: object, **kwargs: object) -> None:
        self.remote_calls.append((args, kwargs))


class _FakeActor:
    def __init__(self) -> None:
        self.inject_fault = _FakeRemoteMethod()


class _FakeInjectCell:
    def __init__(self, *, is_alive: bool = True, num_actors: int = 2) -> None:
        self._is_alive = is_alive
        self._actor = _FakeActor()
        self._num_actors = num_actors

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    def _get_actor_handles(self) -> list[_FakeActor]:
        return [self._actor for _ in range(self._num_actors)]


def _make_inject_group(cell: _FakeInjectCell) -> object:
    group = object.__new__(RayTrainGroup)
    group._cells = [cell]
    return group


class _ConcreteCellHandler(_CellHandler):
    @property
    def cell_type(self) -> str:
        return "fake"

    async def list_cell_keys(self) -> list[str]:
        return ["0"]

    async def get_cell(self, cell_key: str) -> object:
        raise NotImplementedError

    async def suspend(self, cell_key: str) -> None:
        raise NotImplementedError

    async def resume(self, cell_key: str) -> None:
        raise NotImplementedError


class TestActorCellHandlerInjectFault:
    @pytest.mark.asyncio
    async def test_inject_fault_calls_actor_with_mode_value(self) -> None:
        """inject_fault forwards mode.value to the selected actor's remote handle."""
        cell = _FakeInjectCell(is_alive=True, num_actors=2)
        group = _make_inject_group(cell)
        handler = _ActorCellHandler(group=group)

        await handler.inject_fault("0", mode=FailureMode.SIGKILL, sub_index=1)

        assert cell._actor.inject_fault.remote_calls == [(("sigkill",), {})]

    @pytest.mark.asyncio
    async def test_inject_fault_raises_when_cell_not_alive(self) -> None:
        """inject_fault raises RuntimeError when the target cell is not alive."""
        cell = _FakeInjectCell(is_alive=False, num_actors=2)
        group = _make_inject_group(cell)
        handler = _ActorCellHandler(group=group)

        with pytest.raises(RuntimeError, match="not alive"):
            await handler.inject_fault("0", mode=FailureMode.SIGKILL, sub_index=0)

        assert cell._actor.inject_fault.remote_calls == []

    @pytest.mark.asyncio
    async def test_inject_fault_raises_index_error_when_sub_index_out_of_range(self) -> None:
        """inject_fault raises IndexError when sub_index exceeds the actor count."""
        cell = _FakeInjectCell(is_alive=True, num_actors=2)
        group = _make_inject_group(cell)
        handler = _ActorCellHandler(group=group)

        with pytest.raises(IndexError, match="out of range"):
            await handler.inject_fault("0", mode=FailureMode.SIGKILL, sub_index=2)

        assert cell._actor.inject_fault.remote_calls == []

    @pytest.mark.asyncio
    async def test_inject_fault_raises_index_error_when_sub_index_negative(self) -> None:
        """inject_fault raises IndexError when sub_index is negative."""
        cell = _FakeInjectCell(is_alive=True, num_actors=2)
        group = _make_inject_group(cell)
        handler = _ActorCellHandler(group=group)

        with pytest.raises(IndexError, match="out of range"):
            await handler.inject_fault("0", mode=FailureMode.SIGKILL, sub_index=-1)


class TestBaseCellHandlerInjectFault:
    @pytest.mark.asyncio
    async def test_base_inject_fault_raises_not_implemented(self) -> None:
        """The base handler names the subclass that cannot inject faults."""
        handler = _ConcreteCellHandler()

        with pytest.raises(NotImplementedError, match="_ConcreteCellHandler does not support fault injection"):
            await handler.inject_fault("0", mode=FailureMode.SIGKILL, sub_index=0)


class TestRolloutCellHandlerInjectFault:
    @pytest.mark.asyncio
    async def test_injection_is_forwarded_to_the_worker_manager(self) -> None:
        """The manager owns the actors, so it is the one that can crash them."""
        manager = MockWorkerManager(make_cell_summaries(ENGINE_CELL_ID))
        manager.inject_fault = MockRemoteCall(None)
        handler = _RolloutCellHandler(worker_manager=manager, inference_controller=MockInferenceController())

        await handler.inject_fault(ENGINE_CELL_ID, mode=FailureMode.SIGKILL, sub_index=1)

        assert manager.inject_fault.calls == [
            ((ENGINE_CELL_ID,), {"mode": "sigkill", "worker_in_cell_index": 1}),
        ]
