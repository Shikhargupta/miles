import asyncio
import logging
import time

import ray

from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.ray_worker_manager.launcher import CellLauncher
from miles.utils.workers.ray_worker_manager.placement import SpecPlacement
from miles.utils.workers.ray_worker_manager.state import CellState, WorkerState
from miles.utils.workers.worker_provider.ray import RayWorkerInfo
from miles.utils.workers.worker_spec import BaseWorkerSpec

logger = logging.getLogger(__name__)

_ACTOR_GONE_TIMEOUT_SECONDS = 30.0


class RayWorkerManager:
    def __init__(self) -> None:
        self._launcher: CellLauncher | None = None
        self._cells: dict[str, CellState] = {}
        self._workers: dict[str, WorkerState] = {}
        self._cell_lifecycle_lock = asyncio.Lock()

    async def init(self, *, worker_specs: list[BaseWorkerSpec], placements: dict[str, SpecPlacement]) -> None:
        assert self._launcher is None, "RayWorkerManager.init() must be called exactly once"
        _validate_specs(worker_specs=worker_specs, placements=placements)
        self._launcher = CellLauncher(placements=placements)

        cells = []
        for spec in worker_specs:
            for cell_index in range(spec.scheduling.num_cells):
                cell = CellState(
                    spec=spec,
                    cell_id=compute_cell_id(spec_name=spec.name, cell_index=cell_index),
                    cell_index=cell_index,
                    generation=0,
                )
                self._cells[cell.cell_id] = cell
                cells.append(cell)

        await self._launcher.bring_up_cells(cells=cells, register_worker=self._register_worker)

    async def get_worker_infos(self, *, spec_name: str) -> list[RayWorkerInfo]:
        return [
            RayWorkerInfo(name=w.name, cell_id=w.cell.cell_id, generation=w.cell.generation, url=w.url)
            for w in self._workers.values()
            if w.cell.spec.name == spec_name
        ]

    async def start_cell(self, cell_id: str) -> None:
        async with self._cell_lifecycle_lock:
            await self._start_cell_locked(cell_id)

    async def restart_cell(self, cell_id: str) -> None:
        async with self._cell_lifecycle_lock:
            cell = self._cells[cell_id]
            if self._cell_is_alive(cell):
                await self._stop_cell_locked(cell_id)
            await self._start_cell_locked(cell_id)

    async def stop_cell(self, cell_id: str) -> None:
        async with self._cell_lifecycle_lock:
            await self._stop_cell_locked(cell_id)

    async def _start_cell_locked(self, cell_id: str) -> None:
        assert self._launcher is not None
        cell = self._cells[cell_id]
        assert not self._cell_is_alive(cell), f"{cell_id=} must be stopped before starting"
        cell.generation += 1

        try:
            await self._launcher.bring_up_cells(cells=[cell], register_worker=self._register_worker)
        except Exception:
            logger.exception(f"Bringing up cell {cell_id} failed; rolling its workers back")
            await self._kill_cell_workers(cell)
            raise

    async def _stop_cell_locked(self, cell_id: str) -> None:
        cell = self._cells[cell_id]
        assert self._cell_is_alive(cell), f"{cell_id=} must be alive before stopping"
        await self._kill_cell_workers(cell)

    async def _kill_cell_workers(self, cell: CellState) -> None:
        workers = [worker for worker in self._workers.values() if worker.cell is cell]
        for worker in workers:
            ray.kill(worker.actor, no_restart=True)
            del self._workers[worker.name]
        await _wait_actors_gone([worker.name for worker in workers])

    def _cell_is_alive(self, cell: CellState) -> bool:
        return any(worker.cell is cell for worker in self._workers.values())

    def _register_worker(self, worker: WorkerState) -> None:
        self._workers[worker.name] = worker


def _validate_specs(*, worker_specs: list[BaseWorkerSpec], placements: dict[str, SpecPlacement]) -> None:
    spec_names = [spec.name for spec in worker_specs]
    assert len(spec_names) == len(set(spec_names)), f"{spec_names=} must be unique"
    assert set(placements) <= set(spec_names), f"{sorted(placements)=} must be a subset of {spec_names=}"

    for spec in worker_specs:
        url_port_names = [p.name for p in spec.port_infos if p.url_scheme is not None]
        assert len(url_port_names) <= 1, f"{spec.name=} may declare at most one url port, got {url_port_names=}"

        placement = placements.get(spec.name)
        num_workers = spec.scheduling.num_cells * spec.scheduling.num_workers_per_cell
        assert (
            placement is None or len(placement.bundle_indices) == num_workers
        ), f"{spec.name=} needs one bundle per worker: {num_workers=} vs {len(placement.bundle_indices)=}"


async def _wait_actors_gone(names: list[str]) -> None:
    deadline = time.monotonic() + _ACTOR_GONE_TIMEOUT_SECONDS
    remaining = set(names)
    while True:
        remaining = {name for name in remaining if _actor_exists(name)}
        if not remaining:
            return
        assert (
            time.monotonic() < deadline
        ), f"actors {sorted(remaining)} still resolvable after {_ACTOR_GONE_TIMEOUT_SECONDS}s"
        await asyncio.sleep(0.1)


def _actor_exists(name: str) -> bool:
    try:
        ray.get_actor(name)
    except ValueError:
        return False
    return True
