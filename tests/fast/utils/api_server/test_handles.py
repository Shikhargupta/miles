from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from miles.ray.train.group import RayTrainGroup
from miles.utils.ft_utils.api_server.handles import _ActorCellHandle, _CellHandle, _RolloutCellHandle
from miles.utils.test_utils.fault_injector import FailureMode

from .conftest import MockInferenceController, MockRayTrainCell, MockWorkerManager, make_mock_group


class TestActorCellHandle:
    def test_cell_id_and_type(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        handle = _ActorCellHandle(group=group, cell_index=0)
        assert handle.cell_id == "actor-0"
        assert handle.cell_type == "actor"

    @pytest.mark.asyncio
    async def test_get_cell_returns_full_cell_structure(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        handle = _ActorCellHandle(group=group, cell_index=0)
        cell = await handle.get_cell()

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
        handle = _ActorCellHandle(group=group, cell_index=0)
        cell = await handle.get_cell()

        assert cell.spec.suspend is True
        assert cell.status.phase == "Suspended"

    @pytest.mark.asyncio
    async def test_suspend_delegates_to_group(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        group.stop_cell = MagicMock()
        handle = _ActorCellHandle(group=group, cell_index=2)
        await handle.suspend()
        group.stop_cell.assert_called_once_with(2)

    @pytest.mark.asyncio
    async def test_resume_delegates_to_group(self) -> None:
        group = make_mock_group([MockRayTrainCell()])
        group.start_cell = MagicMock()
        handle = _ActorCellHandle(group=group, cell_index=1)
        await handle.resume()
        group.start_cell.assert_called_once_with(1)


class TestRolloutCellStatus:
    """compute_cell_status is what the api server reports for a rollout cell."""

    def _controller(self, *, spec, cell):
        from tests.fast.ray.rollout.conftest import make_args

        from miles.ray.rollout.inference_controller import InferenceController
        from miles.ray.rollout.rollout_server import RolloutServer

        args = make_args(num_gpus_per_node=8)
        srv = RolloutServer(
            cell_specs={spec.cell_id: spec},
            args=args,
            server_cells={} if cell is None else {spec.cell_id: cell},
        )
        with patch("miles.ray.rollout.inference_controller.Lock", MagicMock()):
            controller = InferenceController(args, servers={"default": srv}, provider=None)
        return controller

    def _spec_and_cell(self, *, attached: bool, alive: bool):
        from tests.fast.ray.rollout.conftest import fake_actor_handle, make_args, make_cell_spec

        from miles.ray.rollout.server_cell import ServerCell
        from miles.utils.workers.worker_provider.base import CellInfo, CellMember
        from miles.utils.workers.worker_spec import WorkerPlacement

        args = make_args(num_gpus_per_node=8)
        spec = make_cell_spec(args=args)
        if not attached:
            return spec, None

        cell = ServerCell.attach(
            args=args,
            spec=spec,
            update_weights=True,
            cell_info=CellInfo(
                cell_id=spec.cell_id,
                members=[
                    CellMember(
                        handle=fake_actor_handle(),
                        payload={"host": "10.0.0.1", "port": 30000},
                        placement=WorkerPlacement(local_index=0, global_rank=0, base_gpu_id=0),
                    )
                ],
            ),
        )
        if alive:
            cell.mark_alive()
        return spec, cell

    def test_a_detached_cell_reports_suspended(self) -> None:
        """Nothing is attached, so ops sees the cell as suspended rather than unhealthy."""
        spec, cell = self._spec_and_cell(attached=False, alive=False)
        status = self._controller(spec=spec, cell=cell).compute_cell_status(spec.cell_id)
        assert status.phase == "Suspended"
        assert [(c.type, c.status) for c in status.conditions] == [("Allocated", "False")]

    def test_an_attached_but_not_alive_cell_reports_pending_health(self) -> None:
        """The workers exist but the cell has not been taken into service yet."""
        spec, cell = self._spec_and_cell(attached=True, alive=False)
        status = self._controller(spec=spec, cell=cell).compute_cell_status(spec.cell_id)
        assert status.phase == "Running"
        assert [(c.type, c.status) for c in status.conditions] == [("Allocated", "True"), ("Healthy", "Unknown")]
        assert status.conditions[1].reason == "AttachPending"

    def test_an_alive_cell_reports_healthy(self) -> None:
        spec, cell = self._spec_and_cell(attached=True, alive=True)
        status = self._controller(spec=spec, cell=cell).compute_cell_status(spec.cell_id)
        assert status.phase == "Running"
        assert [(c.type, c.status) for c in status.conditions] == [("Allocated", "True"), ("Healthy", "True")]


class TestRolloutCellHandle:
    @pytest.mark.asyncio
    async def test_get_cell_reads_the_status_from_the_controller(self) -> None:
        """Cell health is what the consumer observes, so it comes from the controller."""
        handle = _RolloutCellHandle(
            inference_controller=MockInferenceController(),
            worker_manager=MockWorkerManager(),
            rollout_cell_id="actor-0",
        )
        cell = await handle.get_cell()

        assert cell.metadata.name == "rollout-actor-0"
        assert cell.metadata.labels["miles.io/cell-type"] == "rollout"
        assert cell.status.phase == "Running"

    @pytest.mark.asyncio
    async def test_get_cell_reads_the_suspend_state_from_the_worker_manager(self) -> None:
        """Suspension is desired state, which only the manager owns."""
        handle = _RolloutCellHandle(
            inference_controller=MockInferenceController(),
            worker_manager=MockWorkerManager(has_workers=False),
            rollout_cell_id="actor-0",
        )
        assert (await handle.get_cell()).spec.suspend is True

    @pytest.mark.asyncio
    async def test_suspend_stops_the_cell_through_the_worker_manager(self) -> None:
        """The ops boundary commands the infrastructure layer, never the consumer."""
        worker_manager = MockWorkerManager()
        handle = _RolloutCellHandle(
            inference_controller=MockInferenceController(),
            worker_manager=worker_manager,
            rollout_cell_id="actor-0",
        )
        await handle.suspend()
        assert worker_manager.stopped_cells == ["actor-0"]

    @pytest.mark.asyncio
    async def test_resume_starts_the_cell_through_the_worker_manager(self) -> None:
        worker_manager = MockWorkerManager(has_workers=False)
        handle = _RolloutCellHandle(
            inference_controller=MockInferenceController(),
            worker_manager=worker_manager,
            rollout_cell_id="actor-0",
        )
        await handle.resume()
        assert worker_manager.started_cells == ["actor-0"]

    @pytest.mark.asyncio
    async def test_suspending_an_already_suspended_cell_does_nothing(self) -> None:
        """The manager is strict about double stops, so the ops layer is idempotent."""
        worker_manager = MockWorkerManager(has_workers=False)
        handle = _RolloutCellHandle(
            inference_controller=MockInferenceController(),
            worker_manager=worker_manager,
            rollout_cell_id="actor-0",
        )
        await handle.suspend()
        assert worker_manager.stopped_cells == []

    @pytest.mark.asyncio
    async def test_resuming_a_live_cell_does_nothing(self) -> None:
        """Starting over live workers would leak them."""
        worker_manager = MockWorkerManager()
        handle = _RolloutCellHandle(
            inference_controller=MockInferenceController(),
            worker_manager=worker_manager,
            rollout_cell_id="actor-0",
        )
        await handle.resume()
        assert worker_manager.started_cells == []

    def test_cell_type_is_rollout(self) -> None:
        handle = _RolloutCellHandle(inference_controller=object(), worker_manager=object(), rollout_cell_id="actor-0")
        assert handle.cell_type == "rollout"
        assert handle.cell_id == "rollout-actor-0"


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


class _ConcreteCellHandle(_CellHandle):
    @property
    def cell_type(self) -> str:
        return "fake"

    @property
    def cell_key(self) -> str:
        return "0"

    async def get_cell(self) -> object:
        raise NotImplementedError

    async def suspend(self) -> None:
        raise NotImplementedError

    async def resume(self) -> None:
        raise NotImplementedError


class TestActorCellHandleInjectFault:
    @pytest.mark.asyncio
    async def test_inject_fault_calls_actor_with_mode_value(self) -> None:
        """inject_fault forwards mode.value to the selected actor's remote handle."""
        cell = _FakeInjectCell(is_alive=True, num_actors=2)
        group = _make_inject_group(cell)
        handle = _ActorCellHandle(group=group, cell_index=0)

        await handle.inject_fault(mode=FailureMode.SIGKILL, sub_index=1)

        assert cell._actor.inject_fault.remote_calls == [(("sigkill",), {})]

    @pytest.mark.asyncio
    async def test_inject_fault_raises_when_cell_not_alive(self) -> None:
        """inject_fault raises RuntimeError when the target cell is not alive."""
        cell = _FakeInjectCell(is_alive=False, num_actors=2)
        group = _make_inject_group(cell)
        handle = _ActorCellHandle(group=group, cell_index=0)

        with pytest.raises(RuntimeError, match="not alive"):
            await handle.inject_fault(mode=FailureMode.SIGKILL, sub_index=0)

        assert cell._actor.inject_fault.remote_calls == []

    @pytest.mark.asyncio
    async def test_inject_fault_raises_index_error_when_sub_index_out_of_range(self) -> None:
        """inject_fault raises IndexError when sub_index exceeds the actor count."""
        cell = _FakeInjectCell(is_alive=True, num_actors=2)
        group = _make_inject_group(cell)
        handle = _ActorCellHandle(group=group, cell_index=0)

        with pytest.raises(IndexError, match="out of range"):
            await handle.inject_fault(mode=FailureMode.SIGKILL, sub_index=2)

        assert cell._actor.inject_fault.remote_calls == []

    @pytest.mark.asyncio
    async def test_inject_fault_raises_index_error_when_sub_index_negative(self) -> None:
        """inject_fault raises IndexError when sub_index is negative."""
        cell = _FakeInjectCell(is_alive=True, num_actors=2)
        group = _make_inject_group(cell)
        handle = _ActorCellHandle(group=group, cell_index=0)

        with pytest.raises(IndexError, match="out of range"):
            await handle.inject_fault(mode=FailureMode.SIGKILL, sub_index=-1)


class TestBaseCellHandleInjectFault:
    @pytest.mark.asyncio
    async def test_base_inject_fault_raises_not_implemented(self) -> None:
        """The base _CellHandle.inject_fault raises NotImplementedError naming the subclass."""
        handle = _ConcreteCellHandle()

        with pytest.raises(NotImplementedError, match="_ConcreteCellHandle does not support fault injection"):
            await handle.inject_fault(mode=FailureMode.SIGKILL, sub_index=0)
