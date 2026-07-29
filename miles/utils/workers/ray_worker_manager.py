import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.misc import NodeProbeMixin, load_function
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_provider.ray import RayWorkerInfo
from miles.utils.workers.worker_spec import BaseWorkerSpec, CommandWorkerSpec, ServeWorkerSpec

logger = logging.getLogger(__name__)

_DYNAMIC_PORT_START = 15000
_ACTOR_GONE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class SpecPlacement:
    placement_group: PlacementGroup
    bundle_indices: list[int]
    num_gpus_per_worker: float | None = None
    num_cpus_per_worker: float | None = None
    concurrency_groups: dict[str, int] | None = None


@dataclass
class _CellState:
    spec: BaseWorkerSpec
    cell_id: str
    cell_index: int
    generation: int


@dataclass
class _WorkerState:
    name: str
    cell: _CellState
    actor: ray.actor.ActorHandle
    node_ip: str = ""
    owned_ports: dict[str, int] = field(default_factory=dict)
    url: str | None = None


class RayWorkerManager:
    def __init__(self) -> None:
        self._placements: dict[str, SpecPlacement] = {}
        self._cells: dict[str, _CellState] = {}
        self._workers: dict[str, _WorkerState] = {}
        self._serve_actor_classes: dict[str, Any] = {}
        self._port_cursors = _NodePortCursors()
        self._cell_lifecycle_lock = asyncio.Lock()

    async def init(self, *, worker_specs: list[BaseWorkerSpec], placements: dict[str, SpecPlacement]) -> None:
        assert not self._cells, "RayWorkerManager.init() must be called exactly once"
        _validate_specs(worker_specs=worker_specs, placements=placements)
        self._placements = dict(placements)

        cells = []
        for spec in worker_specs:
            for cell_index in range(spec.scheduling.num_cells):
                cell = _CellState(
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

    async def _rollback_cell_workers(self, cell: _CellState) -> None:
        workers = [worker for worker in self._workers.values() if worker.cell is cell]
        for worker in workers:
            ray.kill(worker.actor, no_restart=True)
            del self._workers[worker.name]
        await _wait_actors_gone([worker.name for worker in workers])

    def _cell_is_alive(self, cell: _CellState) -> bool:
        return any(worker.cell is cell for worker in self._workers.values())

    async def _bring_up_cells(self, cells: list[_CellState]) -> None:
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
                self._collect_worker_ports(worker=worker, is_master=worker is workers[0])
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

    def _launch_cell_actors(self, *, cell: _CellState, env_vars: dict[str, str]) -> list[_WorkerState]:
        workers = []
        for worker_index in range(cell.spec.scheduling.num_workers_per_cell):
            name = compute_worker_name(spec_name=cell.spec.name, cell_index=cell.cell_index, worker_index=worker_index)
            actor = self._launch_actor(cell=cell, worker_index=worker_index, name=name, env_vars=env_vars)
            worker = _WorkerState(name=name, cell=cell, actor=actor)
            self._workers[name] = worker
            workers.append(worker)
        return workers

    def _launch_actor(
        self, *, cell: _CellState, worker_index: int, name: str, env_vars: dict[str, str]
    ) -> ray.actor.ActorHandle:
        spec = cell.spec
        scheduling = spec.scheduling
        placement = self._placements.get(spec.name)

        scheduling_strategy: PlacementGroupSchedulingStrategy | None = None
        num_gpus = scheduling.num_gpus_per_worker
        num_cpus = scheduling.num_cpus_per_worker
        if placement is not None:
            flat_index = cell.cell_index * scheduling.num_workers_per_cell + worker_index
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement.placement_group,
                placement_group_bundle_index=placement.bundle_indices[flat_index],
            )
            if placement.num_gpus_per_worker is not None:
                num_gpus = placement.num_gpus_per_worker
            if placement.num_cpus_per_worker is not None:
                num_cpus = placement.num_cpus_per_worker

        if isinstance(spec, ServeWorkerSpec):
            return (
                self._serve_actor_class(spec)
                .options(
                    name=name,
                    num_cpus=num_cpus,
                    num_gpus=num_gpus,
                    max_restarts=0,
                    scheduling_strategy=scheduling_strategy,
                    runtime_env={"env_vars": env_vars},
                )
                .remote(ctor_kwargs_fn=spec.ctor_kwargs, cell_index=cell.cell_index, worker_index=worker_index)
            )

        assert isinstance(spec, CommandWorkerSpec), f"unsupported worker spec type: {type(spec)=}"
        return (
            ray.remote(CommandActor)
            .options(
                name=name,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                max_restarts=0,
                scheduling_strategy=scheduling_strategy,
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

    async def _collect_worker_ports(self, *, worker: _WorkerState, is_master: bool) -> None:
        owned_port_infos = [p for p in worker.cell.spec.port_infos if p.mode == "per_worker" or is_master]

        dynamic_port_infos = [p for p in owned_port_infos if p.allow_dynamic]
        if dynamic_port_infos:
            first_port = await self._port_cursors.allocate(
                actor=worker.actor,
                node_ip=worker.node_ip,
                count=sum(p.num_consecutive for p in dynamic_port_infos),
            )
            next_port = first_port
            for port_info in dynamic_port_infos:
                worker.owned_ports[port_info.name] = next_port
                next_port += port_info.num_consecutive

        for port_info in owned_port_infos:
            if not port_info.allow_dynamic:
                worker.owned_ports[port_info.name] = port_info.static_port

    async def _activate_cell(self, *, workers: list[_WorkerState], env_vars: dict[str, str]) -> None:
        spec = workers[0].cell.spec
        master = workers[0]

        addr_port_kwargs_by_worker: dict[str, dict[str, str | int]] = {}
        for worker in workers:
            kwargs: dict[str, str | int] = {}
            for port_info in spec.port_infos:
                owner = master if port_info.mode == "master" else worker
                port = owner.owned_ports[port_info.name]
                kwargs[f"{port_info.name}_addr"] = owner.node_ip
                kwargs[f"{port_info.name}_port"] = port
                if port_info.url_scheme is not None:
                    worker.url = f"{port_info.url_scheme}://{owner.node_ip}:{port}"
            addr_port_kwargs_by_worker[worker.name] = kwargs

        if isinstance(spec, ServeWorkerSpec):
            if spec.port_infos:
                await asyncio.gather(
                    *[
                        worker.actor.configure_addrs_and_ports.remote(**addr_port_kwargs_by_worker[worker.name])
                        for worker in workers
                    ]
                )
            return

        assert isinstance(spec, CommandWorkerSpec), f"unsupported worker spec type: {type(spec)=}"
        for worker in workers:
            command = spec.launch_command.format(**addr_port_kwargs_by_worker[worker.name])
            worker.actor.run.remote(cmd=command, envs=env_vars)


class _NodePortCursors:
    def __init__(self) -> None:
        self._cursors: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def allocate(self, *, actor: ray.actor.ActorHandle, node_ip: str, count: int) -> int:
        async with self._lock:
            start_port = self._cursors.get(node_ip, _DYNAMIC_PORT_START)
            first_port = await actor._get_free_port_block.remote(start_port=start_port, count=count)
            self._cursors[node_ip] = first_port + count
            return first_port


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
