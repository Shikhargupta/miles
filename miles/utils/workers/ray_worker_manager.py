import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.misc import get_current_node_ip, get_free_port, load_function
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.worker_provider.ray import RayWorkerInfo
from miles.utils.workers.worker_spec import BaseWorkerSpec, CommandWorkerSpec, ServeWorkerSpec

logger = logging.getLogger(__name__)

RAY_WORKER_MANAGER_ACTOR_NAME = "miles_ray_worker_manager"

_DYNAMIC_PORT_START = 15000
_ACTOR_GONE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class SpecPlacement:
    placement_group: PlacementGroup
    bundle_indices: list[int]


class _AddrPort(NamedTuple):
    addr: str
    port: int


@dataclass
class _CellState:
    spec: BaseWorkerSpec
    placement: SpecPlacement | None
    cell_id: str
    cell_index: int
    generation: int
    alive: bool


@dataclass
class _WorkerState:
    name: str
    cell: _CellState
    worker_index: int
    actor: ray.actor.ActorHandle
    node_ip: str = ""
    owned_ports: dict[str, int] = field(default_factory=dict)
    addr_ports: dict[str, _AddrPort] = field(default_factory=dict)

    @property
    def url(self) -> str | None:
        addr_port = self.addr_ports.get("http")
        if addr_port is None:
            return None
        return f"http://{addr_port.addr}:{addr_port.port}"


class RayWorkerManager:
    def __init__(self) -> None:
        self._cells: dict[str, _CellState] = {}
        self._workers: dict[str, _WorkerState] = {}
        self._port_cursors: dict[str, int] = {}

    async def init(self, *, worker_specs: list[BaseWorkerSpec], placements: dict[str, SpecPlacement]) -> None:
        assert not self._cells, "RayWorkerManager.init() must be called exactly once"
        spec_names = [spec.name for spec in worker_specs]
        assert len(spec_names) == len(set(spec_names)), f"{spec_names=} must be unique"
        assert set(placements) <= set(spec_names), f"{sorted(placements)=} must be a subset of {spec_names=}"
        for spec in worker_specs:
            placement = placements.get(spec.name)
            num_workers = spec.scheduling.num_cells * spec.scheduling.num_workers_per_cell
            assert (
                placement is None or len(placement.bundle_indices) == num_workers
            ), f"{spec.name=} needs one bundle per worker: {num_workers=} vs {len(placement.bundle_indices)=}"

        cells = []
        for spec in worker_specs:
            for cell_index in range(spec.scheduling.num_cells):
                cell = _CellState(
                    spec=spec,
                    placement=placements.get(spec.name),
                    cell_id=f"{spec.name}-{cell_index}",
                    cell_index=cell_index,
                    generation=0,
                    alive=True,
                )
                self._cells[cell.cell_id] = cell
                cells.append(cell)

        workers_by_cell_id = {cell.cell_id: self._launch_cell_actors(cell) for cell in cells}
        for cell in cells:
            await self._collect_cell_ports(workers=workers_by_cell_id[cell.cell_id])
        await asyncio.gather(
            *[self._activate_cell(cell=cell, workers=workers_by_cell_id[cell.cell_id]) for cell in cells]
        )

    async def get_worker_infos(self, *, spec_name: str) -> list[RayWorkerInfo]:
        return [
            RayWorkerInfo(name=w.name, cell_id=w.cell.cell_id, generation=w.cell.generation, url=w.url)
            for w in self._workers.values()
            if w.cell.spec.name == spec_name
        ]

    async def start_cell(self, cell_id: str) -> None:
        cell = self._cells[cell_id]
        assert not cell.alive, f"{cell_id=} must be stopped before starting"
        cell.generation += 1
        cell.alive = True

        workers = self._launch_cell_actors(cell)
        await self._collect_cell_ports(workers=workers)
        await self._activate_cell(cell=cell, workers=workers)

    async def stop_cell(self, cell_id: str) -> None:
        cell = self._cells[cell_id]
        assert cell.alive, f"{cell_id=} must be alive before stopping"
        cell.alive = False

        workers = [worker for worker in self._workers.values() if worker.cell is cell]
        for worker in workers:
            ray.kill(worker.actor, no_restart=True)
            del self._workers[worker.name]
        await asyncio.gather(*[_wait_actor_gone(worker.name) for worker in workers])

    def _launch_cell_actors(self, cell: _CellState) -> list[_WorkerState]:
        workers = []
        for worker_index in range(cell.spec.scheduling.num_workers_per_cell):
            name = f"{cell.cell_id}-{worker_index}"
            actor = self._launch_actor(cell=cell, worker_index=worker_index, name=name)
            worker = _WorkerState(name=name, cell=cell, worker_index=worker_index, actor=actor)
            self._workers[name] = worker
            workers.append(worker)
        return workers

    def _launch_actor(self, *, cell: _CellState, worker_index: int, name: str) -> ray.actor.ActorHandle:
        spec = cell.spec
        num_gpus = spec.scheduling.num_gpus_per_worker
        options: dict = dict(name=name, num_cpus=num_gpus or 1, num_gpus=num_gpus, max_restarts=0)
        if cell.placement is not None:
            flat_index = cell.cell_index * spec.scheduling.num_workers_per_cell + worker_index
            options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=cell.placement.placement_group,
                placement_group_bundle_index=cell.placement.bundle_indices[flat_index],
            )

        if isinstance(spec, ServeWorkerSpec):
            options["runtime_env"] = {"env_vars": spec.env_var()}
            wrapped_cls = _make_wrapped_worker_cls(load_function(spec.worker_class))
            return ray.remote(wrapped_cls).options(**options).remote(ctor_kwargs_fn=spec.ctor_kwargs)

        assert isinstance(spec, CommandWorkerSpec), f"unsupported worker spec type: {type(spec)=}"
        return ray.remote(CommandActor).options(**options).remote()

    async def _collect_cell_ports(self, *, workers: list[_WorkerState]) -> None:
        for worker in workers:
            worker.node_ip = await worker.actor._get_node_ip.remote()

            owned_port_infos = [
                p for p in worker.cell.spec.port_infos if p.mode == "per_worker" or worker.worker_index == 0
            ]
            dynamic_port_infos = [p for p in owned_port_infos if p.allow_dynamic]
            if dynamic_port_infos:
                start_port = self._port_cursors.get(worker.node_ip, _DYNAMIC_PORT_START)
                first_port = await worker.actor._get_free_consecutive_ports.remote(
                    start_port=start_port, consecutive=len(dynamic_port_infos)
                )
                self._port_cursors[worker.node_ip] = first_port + len(dynamic_port_infos)
                for offset, port_info in enumerate(dynamic_port_infos):
                    worker.owned_ports[port_info.name] = first_port + offset

            for port_info in owned_port_infos:
                if not port_info.allow_dynamic:
                    worker.owned_ports[port_info.name] = port_info.static_port

    async def _activate_cell(self, *, cell: _CellState, workers: list[_WorkerState]) -> None:
        master = workers[0]
        for worker in workers:
            for port_info in cell.spec.port_infos:
                owner = master if port_info.mode == "master" else worker
                worker.addr_ports[port_info.name] = _AddrPort(
                    addr=owner.node_ip, port=owner.owned_ports[port_info.name]
                )

        if isinstance(cell.spec, ServeWorkerSpec):
            if cell.spec.port_infos:
                await asyncio.gather(
                    *[
                        worker.actor.configure_addrs_and_ports.remote(**_flatten_addr_ports(worker.addr_ports))
                        for worker in workers
                    ]
                )
            return

        assert isinstance(cell.spec, CommandWorkerSpec), f"unsupported worker spec type: {type(cell.spec)=}"
        envs = cell.spec.env_var()
        for worker in workers:
            command = cell.spec.launch_command.format(**_flatten_addr_ports(worker.addr_ports))
            worker.actor.run.remote(cmd=command, envs=envs)


def _make_wrapped_worker_cls(worker_cls: type) -> type:
    class _WrappedWorker(worker_cls):
        def __init__(self, *, ctor_kwargs_fn) -> None:
            super().__init__(**ctor_kwargs_fn())

        @staticmethod
        def _get_node_ip() -> str:
            return get_current_node_ip()

        @staticmethod
        def _get_free_consecutive_ports(*, start_port: int, consecutive: int) -> int:
            return get_free_port(start_port=start_port, consecutive=consecutive)

    _WrappedWorker.__name__ = worker_cls.__name__
    _WrappedWorker.__qualname__ = worker_cls.__qualname__
    return _WrappedWorker


def _flatten_addr_ports(addr_ports: dict[str, _AddrPort]) -> dict[str, str | int]:
    result: dict[str, str | int] = {}
    for name, addr_port in addr_ports.items():
        result[f"{name}_addr"] = addr_port.addr
        result[f"{name}_port"] = addr_port.port
    return result


async def _wait_actor_gone(name: str) -> None:
    deadline = time.monotonic() + _ACTOR_GONE_TIMEOUT_SECONDS
    while True:
        try:
            ray.get_actor(name)
        except ValueError:
            return
        assert time.monotonic() < deadline, f"actor {name=} still resolvable after {_ACTOR_GONE_TIMEOUT_SECONDS}s"
        await asyncio.sleep(0.1)
