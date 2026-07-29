import asyncio
from collections.abc import Callable

import ray

from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.ray_worker_manager.addressing import compute_worker_addressings
from miles.utils.workers.ray_worker_manager.kinds import WorkerKind, make_worker_kinds
from miles.utils.workers.ray_worker_manager.placement import SpecPlacement
from miles.utils.workers.ray_worker_manager.ports import PortAllocator
from miles.utils.workers.ray_worker_manager.resources import resolve_actor_options
from miles.utils.workers.ray_worker_manager.state import ActorState, CellLaunch
from miles.utils.workers.worker_spec import BaseWorkerSpec


class CellLauncher:
    def __init__(self, *, placements: dict[str, SpecPlacement]) -> None:
        self._placements = dict(placements)
        self._worker_kinds = make_worker_kinds()
        self._port_allocator = PortAllocator()

    async def bring_up_cells(self, *, cells: list[CellLaunch], register_worker: Callable[[ActorState], None]) -> None:
        env_vars_by_cell_id = {cell.cell_id: cell.spec.env_var() for cell in cells}

        workers_by_cell_id = {}
        for cell in cells:
            workers = self._launch_cell_actors(cell=cell, env_vars=env_vars_by_cell_id[cell.cell_id])
            for worker in workers:
                register_worker(worker)
            workers_by_cell_id[cell.cell_id] = workers
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

    def _launch_cell_actors(self, *, cell: CellLaunch, env_vars: dict[str, str]) -> list[ActorState]:
        workers = []
        for worker_index in range(cell.spec.scheduling.num_workers_per_cell):
            name = compute_worker_name(spec_name=cell.spec.name, cell_index=cell.cell_index, worker_index=worker_index)
            actor = self._launch_actor(cell=cell, worker_index=worker_index, name=name, env_vars=env_vars)
            workers.append(
                ActorState(name=name, spec=cell.spec, cell_id=cell.cell_id, generation=cell.generation, actor=actor)
            )
        return workers

    def _launch_actor(
        self, *, cell: CellLaunch, worker_index: int, name: str, env_vars: dict[str, str]
    ) -> ray.actor.ActorHandle:
        spec = cell.spec
        placement = self._placements.get(spec.name)
        options = resolve_actor_options(
            scheduling=spec.scheduling,
            placement=placement,
            flat_worker_index=cell.cell_index * spec.scheduling.num_workers_per_cell + worker_index,
        )
        return self._kind_for(spec).create_actor(
            cell=cell, worker_index=worker_index, name=name, env_vars=env_vars, options=options, placement=placement
        )

    async def _activate_cell(self, *, workers: list[ActorState], env_vars: dict[str, str]) -> None:
        spec = workers[0].spec
        addressings = compute_worker_addressings(spec=spec, workers=workers)
        for worker in workers:
            worker.url = addressings[worker.name].url

        await self._kind_for(spec).activate_workers(workers=workers, addressings=addressings, env_vars=env_vars)

    def _kind_for(self, spec: BaseWorkerSpec) -> WorkerKind:
        kind = self._worker_kinds.get(type(spec))
        assert kind is not None, f"unsupported worker spec type: {type(spec)=}"
        return kind
