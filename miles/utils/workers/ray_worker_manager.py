from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.http_utils import _wrap_ipv6
from miles.utils.ray_utils import compute_ray_pin_head_options
from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.naming import compute_worker_name
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_spec import (
    BaseWorkerSpec,
    CommandWorkerSpec,
    HostAndPort,
    LaunchCommandContext,
    NamedHostAndPorts,
    StartupProbeContext,
    WorkerMetaContext,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from miles.ray.placement_group import PlacementGroupInfo

# TODO: unique name, maybe with args.run_uuid
_ACTOR_NAME = "ray_worker_manager"


@dataclass(kw_only=True)
class WorkerInfo:
    name: str
    generation: int
    self_addrs: NamedHostAndPorts
    gpu_ids: list[int]
    actor_handle: ray.actor.ActorHandle


class RayWorkerManager:
    def __init__(self):
        self.port_allocator = PortAllocator()

    @staticmethod
    def launch(specs: list[BaseWorkerSpec], pgs: dict[str, PlacementGroupInfo]):
        obj = ray.remote(RayWorkerManager).options(name=_ACTOR_NAME).remote()
        ray.get(obj.init.remote(specs, pgs))
        return obj

    @staticmethod
    def get_handle() -> ray.actor.ActorHandle:
        return ray.get_actor(_ACTOR_NAME)

    async def init(self, specs: list[BaseWorkerSpec], pgs: dict[str, PlacementGroupInfo]):
        self.pgs = pgs
        self._group_infos = {spec.name: _GroupManager.initial(spec, self) for spec in specs}
        assert len(self._group_infos) == len(specs)

        await self._for_all_cells(lambda a: a.launch_actors())
        await self._for_all_cells(lambda a: a.alloc_ports())
        await self._for_all_cells(lambda a: a.post_setup())

    def get_worker_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return self._find_actor(worker_name).self_addrs

    def get_addrs(self) -> dict[str, list[NamedHostAndPorts]]:
        return {name: [a.self_addrs for c in g.cells for a in c.actors] for name, g in self._group_infos.items()}

    def get_worker_infos(self, spec_name: str, cell_index: int) -> list[WorkerInfo]:
        cell = self._group_infos[spec_name].cells[cell_index]
        return [
            WorkerInfo(
                name=actor.name,
                generation=actor.generation,
                self_addrs=actor.self_addrs,
                gpu_ids=actor.gpu_ids,
                actor_handle=actor.actor_handle,
            )
            for actor in cell.actors
        ]

    def get_cell_infos(self) -> dict[str, CellInfo]:
        # TODO: about `get_worker_infos` (which is only used by dashboard)
        infos = [c.get_info() for _, g in self._group_infos.items() for c in g.cells if c.alive and c.started]
        return {info.cell_id: info for info in infos}

    def _find_actor(self, worker_name: str) -> _BaseActorManager:
        matches = [a for g in self._group_infos.values() for c in g.cells for a in c.actors if a.name == worker_name]
        assert len(matches) == 1, f"{matches=}"
        return matches[0]

    async def _for_all_cells(self, fn: Callable[[_CellManager], Any]):
        await asyncio.gather(*[fn(c) for g in self._group_infos.values() for c in g.cells])


@dataclass(kw_only=True)
class _GroupManager:
    spec: BaseWorkerSpec
    cells: list[_CellManager]

    @classmethod
    def initial(cls, spec: BaseWorkerSpec, manager: RayWorkerManager) -> _GroupManager:
        return cls(
            spec=spec,
            cells=[
                _CellManager(
                    manager=manager,
                    cell_index=cell_index,
                    spec=spec,
                    actors=None,
                )
                for cell_index in range(spec.scheduling.num_cells)
            ],
        )


SpecT = TypeVar("SpecT", bound=BaseWorkerSpec)


@dataclass(kw_only=True)
class _CellManager(Generic[SpecT]):
    manager: RayWorkerManager
    cell_index: int
    spec: SpecT
    actors: list[_BaseActorManager] | None
    generation: int = 0
    startup_prober: _CellManagerStartupProber | None = None

    async def launch_actors(self):
        assert self.actors is None
        self.generation += 1
        scheduling = self.spec.scheduling
        self.actors = [
            # TODO support Serve mode
            _CommandActorManager(
                manager=self.manager,
                parent=self,
                worker_in_cell_index=worker_in_cell_index,
                spec=self.spec,
                actor_handle=None,
                gpu_slot_index=(
                    scheduling.pg_slot_offset
                    + (self.cell_index * scheduling.num_workers_per_cell + worker_in_cell_index)
                    * scheduling.num_gpu_slots_per_worker
                    if scheduling.pg_name is not None
                    else None
                ),
            )
            for worker_in_cell_index in range(scheduling.num_workers_per_cell)
        ]
        await self._for_all_actors(lambda a: a.launch_actor())

    async def alloc_ports(self) -> None:
        await self._for_all_actors(lambda a: a.alloc_ports())

    async def post_setup(self) -> None:
        await self._for_all_actors(lambda a: a.post_setup())
        self.startup_prober = _CellManagerStartupProber.start(cell=self)

    async def stop(self) -> None:
        if self.startup_prober is not None:
            self.startup_prober.cancel()
            self.startup_prober = None
        await self._for_all_actors(lambda a: a.stop())
        self.actors = None

    async def _for_all_actors(self, fn: Callable[[_BaseActorManager], Any]):
        await asyncio.gather(*[fn(a) for a in self.actors])

    def get_info(self) -> CellInfo:
        return CellInfo(
            cell_id=self.cell_id,
            worker_names=[a.name for a in self.actors],
            workers_hash=f"pseudo-hash-{self.generation}",
            meta=f(WorkerMetaContext(cell_index=self.cell_index)) if (f := self.spec.meta) is not None else {},
        )

    @property
    def cell_id(self) -> str:
        return f"{self.spec.name}-{self.cell_index}"

    @property
    def alive(self) -> bool:
        return self.actors is not None

    @property
    def started(self) -> bool:
        return self.startup_prober.started


@dataclass(kw_only=True)
class _CellManagerStartupProber:
    cell: _CellManager
    started: bool = False
    _task: asyncio.Task | None = None

    @classmethod
    def start(cls, cell: _CellManager) -> _CellManagerStartupProber:
        prober = cls(cell=cell)
        prober._task = asyncio.create_task(prober._poll())
        return prober

    def cancel(self) -> None:
        self._task.cancel()

    async def _poll(self) -> None:
        probe = self.cell.spec.startup_probe or _always_started_probe
        ctx = StartupProbeContext(addrs=self.cell.actors[0].self_addrs)
        attempt = 0
        while not await self._check_once(probe, ctx):
            attempt += 1
            if attempt % _STARTUP_PROBE_LOG_EVERY_ATTEMPTS == 0:
                logger.info(f"Cell {self.cell.cell_id} is still waiting for its startup probe ({attempt=})")
            await asyncio.sleep(_STARTUP_PROBE_INTERVAL_SECONDS)
        self.started = True
        logger.info(f"Cell {self.cell.cell_id} completed its startup probe")

    async def _check_once(
        self, probe: Callable[[StartupProbeContext], Awaitable[bool]], ctx: StartupProbeContext
    ) -> bool:
        try:
            return await probe(ctx)
        except Exception:
            logger.exception(f"Startup probe of cell {self.cell.cell_id} raised; treating as not started")
            return False


async def _always_started_probe(ctx: StartupProbeContext) -> bool:
    return True


_SHUTDOWN_TIMEOUT = 30
_STARTUP_PROBE_INTERVAL_SECONDS = 2
_STARTUP_PROBE_LOG_EVERY_ATTEMPTS = 15


@dataclass(kw_only=True)
class _BaseActorManager(Generic[SpecT]):
    manager: RayWorkerManager
    parent: _CellManager
    worker_in_cell_index: int
    spec: SpecT
    actor_handle: ray.actor.ActorHandle | None
    self_addrs: NamedHostAndPorts | None = None
    gpu_slot_index: int | None

    async def launch_actor(self) -> None:
        raise NotImplementedError

    async def alloc_ports(self) -> None:
        raise NotImplementedError

    async def post_setup(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        try:
            ray.get(self.actor_handle.shutdown.remote(), timeout=_SHUTDOWN_TIMEOUT)
        except Exception as e:
            logger.warning(f"Graceful shutdown of {self=} failed ({e})")

        try:
            ray.kill(self.actor_handle)
            logger.info(f"Killed actor at {self=}")
        except Exception as e:
            logger.warning(f"Failed to kill actor at {self=} ({e})")

    @property
    def name(self) -> str:
        return compute_worker_name(
            spec_name=self.spec.name,
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
        )

    @property
    def generation(self) -> int:
        return self.parent.generation

    @property
    def gpu_ids(self) -> list[int]:
        if (pg_name := self.spec.scheduling.pg_name) is None:
            return []
        pg = self.manager.pgs[pg_name]
        base_gpu_id = int(pg.pg_reordered_gpu_ids[self.gpu_slot_index])
        return list(range(base_gpu_id, base_gpu_id + self.spec.scheduling.num_gpu_slots_per_worker))

    @property
    def master_mode_addrs(self) -> NamedHostAndPorts:
        return {info.name: self.self_addrs[info.name] for info in self.spec.port_infos if info.mode == "master"}


@dataclass
class _CommandActorManager(_BaseActorManager[CommandWorkerSpec]):
    async def launch_actor(self) -> None:
        scheduling_strategy = None
        if (pg_name := self.spec.scheduling.pg_name) is not None:
            pg = self.manager.pgs[pg_name]
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg.pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=pg.pg_reordered_bundle_indices[self.gpu_slot_index],
            )

        self.actor_handle = (
            ray.remote(CommandActor)
            .options(
                # TODO generalize
                num_cpus=0.2,
                num_gpus=self.spec.scheduling.num_gpus_per_worker,
                **(dict(scheduling_strategy=s) if (s := scheduling_strategy) is not None else {}),
                runtime_env={"env_vars": self.spec.env_var()},
                **(compute_ray_pin_head_options() if self.spec.scheduling.pin_to_head else {}),
            )
            .remote()
        )

    async def alloc_ports(self) -> None:
        self.self_addrs = {}

        node_ip = await self.actor_handle._get_node_ip.remote()
        for port_info in self.spec.port_infos:
            if self.worker_in_cell_index != 0 and port_info.mode == "master":
                continue
            port = (
                self.manager.port_allocator.alloc(
                    self.actor_handle, node_ip=node_ip, consecutive=port_info.num_consecutive
                )
                if port_info.allow_dynamic
                else port_info.static_port
            )
            self.self_addrs[port_info.name] = HostAndPort(host=_wrap_ipv6(node_ip), port=port)

    async def post_setup(self) -> None:
        ctx = LaunchCommandContext(
            cell_index=self.parent.cell_index,
            worker_in_cell_index=self.worker_in_cell_index,
            self_addrs={
                **self.self_addrs,
                **self.parent.actors[0].master_mode_addrs,
            },
            spec_addrs=self.manager.get_addrs(),
            gpu_ids=self.gpu_ids,
        )
        launch_cmd = self.spec.launch_command(ctx)
        self.actor_handle.run.remote(cmd=launch_cmd, envs={})
