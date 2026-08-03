from __future__ import annotations

import abc
import asyncio

import ray

from miles.ray.train.group import RayTrainGroup
from miles.utils.ft_utils.api_server.models import Cell, CellCondition, CellMetadata, CellSpec, CellStatus, TriState
from miles.utils.test_utils.fault_injector import FailureMode


class _CellHandler(abc.ABC):
    """Owns every cell of one kind, and is asked which of them exist right now.

    Cells come and go (reconcile, healing, elastic scaling), so the set is resolved per
    request instead of captured once at startup.
    """

    @property
    @abc.abstractmethod
    def cell_type(self) -> str: ...

    @abc.abstractmethod
    async def list_cell_keys(self) -> list[str]: ...

    @abc.abstractmethod
    async def get_cell(self, cell_key: str) -> Cell: ...

    @abc.abstractmethod
    async def suspend(self, cell_key: str) -> None: ...

    @abc.abstractmethod
    async def resume(self, cell_key: str) -> None: ...

    async def inject_fault(self, cell_key: str, *, mode: FailureMode, sub_index: int) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support fault injection")

    async def list_cells(self) -> list[Cell]:
        cell_keys = await self.list_cell_keys()
        return list(await asyncio.gather(*(self.get_cell(cell_key) for cell_key in cell_keys)))

    def compute_cell_name(self, cell_key: str) -> str:
        return f"{self.cell_type}-{cell_key}"

    def parse_cell_name(self, cell_name: str) -> str | None:
        prefix = f"{self.cell_type}-"
        return cell_name[len(prefix) :] if cell_name.startswith(prefix) else None

    def _compute_metadata(self, cell_key: str) -> CellMetadata:
        return CellMetadata(
            name=self.compute_cell_name(cell_key),
            labels={
                "miles.io/cell-type": self.cell_type,
                "miles.io/cell-index": cell_key,
            },
        )


class _ActorCellHandler(_CellHandler):
    def __init__(self, *, group: RayTrainGroup) -> None:
        self._group = group

    @property
    def cell_type(self) -> str:
        return "actor"

    async def list_cell_keys(self) -> list[str]:
        return [str(cell_index) for cell_index in range(len(self._group._cells))]

    async def get_cell(self, cell_key: str) -> Cell:
        cell = self._find_cell(cell_key)
        return Cell(
            metadata=self._compute_metadata(cell_key),
            spec=CellSpec(suspend=cell.is_stopped),
            status=cell.cell_status(),
        )

    async def suspend(self, cell_key: str) -> None:
        self._group.stop_cell(int(cell_key))

    async def resume(self, cell_key: str) -> None:
        self._group.start_cell(int(cell_key))

    async def inject_fault(self, cell_key: str, *, mode: FailureMode, sub_index: int) -> None:
        """Inject a fault into a specific actor of this cell. Fire-and-forget."""
        cell = self._find_cell(cell_key)
        if not cell.is_alive:
            raise RuntimeError(f"Cell {cell_key} is not alive, cannot inject fault")
        actors = cell._get_actor_handles()
        if sub_index < 0 or sub_index >= len(actors):
            raise IndexError(f"sub_index {sub_index} out of range for cell {cell_key} (has {len(actors)} actors)")
        actors[sub_index].inject_fault.remote(mode.value)

    def _find_cell(self, cell_key: str):
        return self._group._cells[int(cell_key)]


class _RolloutCellHandler(_CellHandler):
    """Two sources: the worker manager owns the processes, the controller owns their health."""

    def __init__(self, *, worker_manager: ray.actor.ActorHandle, inference_controller: object) -> None:
        self._worker_manager = worker_manager
        self._inference_controller = inference_controller

    @property
    def cell_type(self) -> str:
        return "rollout"

    async def list_cell_keys(self) -> list[str]:
        return compute_engine_cell_ids(await self._worker_manager.get_cell_summaries.remote())

    async def list_cells(self) -> list[Cell]:
        # one round trip for the whole listing: this endpoint is polled for the life of the run
        summaries = await self._worker_manager.get_cell_summaries.remote()
        health_statuses = self._inference_controller.get_cell_health_statuses()
        return [
            self._compute_cell(cell_key, summaries=summaries, health_statuses=health_statuses)
            for cell_key in compute_engine_cell_ids(summaries)
        ]

    async def get_cell(self, cell_key: str) -> Cell:
        return self._compute_cell(
            cell_key,
            summaries=await self._worker_manager.get_cell_summaries.remote(),
            health_statuses=self._inference_controller.get_cell_health_statuses(),
        )

    def _compute_cell(self, cell_key: str, *, summaries: dict, health_statuses: dict) -> Cell:
        suspended = summaries[cell_key].suspended
        return Cell(
            metadata=self._compute_metadata(cell_key),
            spec=CellSpec(suspend=suspended),
            status=_compute_rollout_cell_status(
                suspended=suspended,
                health_checker_status=health_statuses.get(cell_key, TriState.UNKNOWN),
            ),
        )

    async def suspend(self, cell_key: str) -> None:
        await self._worker_manager.stop_cells.remote([cell_key])

    async def resume(self, cell_key: str) -> None:
        await self._worker_manager.start_cells.remote([cell_key])

    async def inject_fault(self, cell_key: str, *, mode: FailureMode, sub_index: int) -> None:
        await self._worker_manager.inject_fault.remote(cell_key, mode=mode.value, worker_in_cell_index=sub_index)


def compute_engine_cell_ids(summaries: dict) -> list[str]:
    """Engine cells are the ones carrying a model, unlike routers and session servers."""
    return sorted(cell_id for cell_id, summary in summaries.items() if "model_id" in summary.meta)


def _compute_rollout_cell_status(*, suspended: bool, health_checker_status: TriState) -> CellStatus:
    if suspended:
        return CellStatus(phase="Suspended", conditions=[CellCondition.allocated(TriState.FALSE)])
    return CellStatus(
        phase="Running",
        conditions=[
            CellCondition.allocated(TriState.TRUE),
            CellCondition.from_health_checker_status(health_checker_status),
        ],
    )
