import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

import ray

from miles.utils.misc import NodeProbeMixin, load_function
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.ray_worker_manager.addressing import compute_worker_addressings
from miles.utils.workers.ray_worker_manager.placement import SpecPlacement
from miles.utils.workers.ray_worker_manager.ports import PortAllocator
from miles.utils.workers.ray_worker_manager.resources import resolve_actor_options
from miles.utils.workers.ray_worker_manager.state import CellState, WorkerState
from miles.utils.workers.worker_provider.ray import RayWorkerInfo
from miles.utils.workers.worker_spec import BaseWorkerSpec, CommandWorkerSpec, ServeWorkerSpec

logger = logging.getLogger(__name__)

_ACTOR_GONE_TIMEOUT_SECONDS = 30.0


class RayWorkerManager:
    def __init__(self) -> None:
        self._placements: dict[str, SpecPlacement] = {}
        self._cells: dict[str, CellState] = {}
        self._workers: dict[str, WorkerState] = {}
        self._serve_actor_classes: dict[str, Any] = {}
        self._port_allocator = PortAllocator()
        self._cell_lifecycle_lock = asyncio.Lock()

    async def init(self, *, worker_specs: list[BaseWorkerSpec], placements: dict[str, SpecPlacement]) -> None:
        assert not self._cells, "RayWorkerManager.init() must be called exactly once"
        _validate_specs(worker_specs=worker_specs, placements=placements)
        self._placements = dict(placements)

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

        await self._bring_up_cells(cells)

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
        cell = self._cells[cell_id]
        assert not self._cell_is_alive(cell), f"{cell_id=} must be stopped before starting"
        cell.generation += 1

        try:
            await self._bring_up_cells([cell])
        except Exception:
            logger.exception(f"Bringing up cell {cell_id} failed; rolling its workers back")
            await self._rollback_cell_workers(cell)
            raise

    async def _stop_cell_locked(self, cell_id: str) -> None:
        cell = self._cells[cell_id]
        assert self._cell_is_alive(cell), f"{cell_id=} must be alive before stopping"

        workers = [worker for worker in self._workers.values() if worker.cell is cell]
        for worker in workers:
            ray.kill(worker.actor, no_restart=True)
            del self._workers[worker.name]
        await _wait_actors_gone([worker.name for worker in workers])

    async def _rollback_cell_workers(self, cell: CellState) -> None:
        workers = [worker for worker in self._workers.values() if worker.cell is cell]
        for worker in workers:
            ray.kill(worker.actor, no_restart=True)
            del self._workers[worker.name]
        await _wait_actors_gone([worker.name for worker in workers])

    def _cell_is_alive(self, cell: CellState) -> bool:
        return any(worker.cell is cell for worker in self._workers.values())

    async def _bring_up_cells(self, cells: list[CellState]) -> None:
        env_vars_by_cell_id = {cell.cell_id: cell.spec.env_var() for cell in cells}
        workers_by_cell_id = {
            cell.cell_id: self._launch_cell_actors(cell=cell, env_vars=env_vars_by_cell_id[cell.cell_id])
            for cell in cells
        }
        all_workers = [worker for workers in workers_by_cell_id.values() for worker in workers]

        node_ips = await asyncio.gather(*[worker.actor._get_node_ip.remote() for worker in all_workers])
        for worker, node_ip in zip(all_workers, node_ips, strict=True):
            worker.node_ip = node_ip
        await asyncio.gather(
            *[
                self._port_allocator.collect_worker_ports(worker=worker, is_master=worker is workers[0])
                for workers in workers_by_cell_id.values()
                for worker in workers
            ]
        )

        await asyncio.gather(
            *[
                self._activate_cell(workers=workers, env_vars=env_vars_by_cell_id[cell_id])
                for cell_id, workers in workers_by_cell_id.items()
            ]
        )

    def _launch_cell_actors(self, *, cell: CellState, env_vars: dict[str, str]) -> list[WorkerState]:
        workers = []
        for worker_index in range(cell.spec.scheduling.num_workers_per_cell):
            name = compute_worker_name(spec_name=cell.spec.name, cell_index=cell.cell_index, worker_index=worker_index)
            actor = self._launch_actor(cell=cell, worker_index=worker_index, name=name, env_vars=env_vars)
            worker = WorkerState(name=name, cell=cell, actor=actor)
            self._workers[name] = worker
            workers.append(worker)
        return workers

    def _launch_actor(
        self, *, cell: CellState, worker_index: int, name: str, env_vars: dict[str, str]
    ) -> ray.actor.ActorHandle:
        spec = cell.spec
        placement = self._placements.get(spec.name)
        options = resolve_actor_options(
            scheduling=spec.scheduling,
            placement=placement,
            flat_worker_index=cell.cell_index * spec.scheduling.num_workers_per_cell + worker_index,
        )

        if isinstance(spec, ServeWorkerSpec):
            return (
                self._serve_actor_class(spec)
                .options(
                    name=name,
                    num_cpus=options.num_cpus,
                    num_gpus=options.num_gpus,
                    max_restarts=0,
                    scheduling_strategy=options.scheduling_strategy,
                    runtime_env={"env_vars": env_vars},
                )
                .remote(ctor_kwargs_fn=spec.ctor_kwargs, cell_index=cell.cell_index, worker_index=worker_index)
            )

        assert isinstance(spec, CommandWorkerSpec), f"unsupported worker spec type: {type(spec)=}"
        return (
            ray.remote(CommandActor)
            .options(
                name=name,
                num_cpus=options.num_cpus,
                num_gpus=options.num_gpus,
                max_restarts=0,
                scheduling_strategy=options.scheduling_strategy,
            )
            .remote()
        )

    def _serve_actor_class(self, spec: ServeWorkerSpec) -> Any:
        if spec.name not in self._serve_actor_classes:
            wrapped_cls = _make_wrapped_worker_cls(load_function(spec.worker_class))
            placement = self._placements.get(spec.name)
            remote_kwargs: dict[str, Any] = {}
            if placement is not None and placement.concurrency_groups is not None:
                remote_kwargs["concurrency_groups"] = placement.concurrency_groups
            self._serve_actor_classes[spec.name] = (
                ray.remote(**remote_kwargs)(wrapped_cls) if remote_kwargs else ray.remote(wrapped_cls)
            )
        return self._serve_actor_classes[spec.name]

    async def _activate_cell(self, *, workers: list[WorkerState], env_vars: dict[str, str]) -> None:
        spec = workers[0].cell.spec
        addressings = compute_worker_addressings(spec=spec, workers=workers)
        for worker in workers:
            worker.url = addressings[worker.name].url

        if isinstance(spec, ServeWorkerSpec):
            if spec.port_infos:
                await asyncio.gather(
                    *[
                        worker.actor.configure_addrs_and_ports.remote(**addressings[worker.name].addr_port_kwargs)
                        for worker in workers
                    ]
                )
            return

        assert isinstance(spec, CommandWorkerSpec), f"unsupported worker spec type: {type(spec)=}"
        for worker in workers:
            command = spec.launch_command.format(**addressings[worker.name].addr_port_kwargs)
            worker.actor.run.remote(cmd=command, envs=env_vars)


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


def _make_wrapped_worker_cls(worker_cls: type) -> type:
    class _WrappedWorker(worker_cls, NodeProbeMixin):
        def __init__(
            self, *, ctor_kwargs_fn: Callable[[int, int], dict[str, Any]], cell_index: int, worker_index: int
        ) -> None:
            super().__init__(**ctor_kwargs_fn(cell_index, worker_index))

    _WrappedWorker.__name__ = worker_cls.__name__
    _WrappedWorker.__qualname__ = worker_cls.__qualname__
    return _WrappedWorker


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
