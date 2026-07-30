import asyncio
import functools
import logging
from dataclasses import dataclass, field
from typing import Any

import ray

from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.base import BaseWorkerManager
from miles.utils.workers.cell_launch import (
    allocate_cell_ports,
    cell_worker_placements,
    create_cell_worker_actors,
    probe_node_ips,
)
from miles.utils.workers.worker_spec import BaseCellSpec, CommandWorkerSpec, WorkerPlacement

logger = logging.getLogger(__name__)

SHUTDOWN_TIMEOUT = 30


@dataclass
class ActorState:
    actor: ray.actor.ActorHandle
    payload: dict
    placement: WorkerPlacement


@dataclass
class _CellRecord:
    workers: list[ActorState] = field(default_factory=list)
    ready: bool = False


@dataclass
class _CellInfo:
    spec: BaseCellSpec | None = None
    record: _CellRecord | None = None


@dataclass
class RayWorkerManager(BaseWorkerManager):
    """Owns the ray actors of every cell it started.

    It launches a cell from that cell's spec alone, so it stays free of any
    knowledge about what the workers are: the spec says how to turn a placement
    and an addressing into each worker's command, and when the cell is ready.
    """

    pg: Any
    _port_allocator: PortAllocator = field(default_factory=PortAllocator)
    _infos: dict[str, _CellInfo] = field(default_factory=dict)

    async def register_cells(self, specs: list[BaseCellSpec]) -> None:
        """Record the cells and bring them up: a registered spec is a running cell."""
        for spec in specs:
            assert spec.cell_id not in self._infos, f"cell {spec.cell_id} is already registered"
            self._infos[spec.cell_id] = _CellInfo(spec=spec)
        await asyncio.gather(*[self.start_cell(spec.cell_id) for spec in specs])

    def registered_cell_ids(self) -> list[str]:
        return sorted(cell_id for cell_id, info in self._infos.items() if info.spec is not None)

    async def start_cell(self, cell_id: str) -> None:
        info = self._infos[cell_id]
        assert info.record is None, f"cell {cell_id} already has live workers"
        spec = info.spec
        worker = spec.worker
        assert isinstance(worker, CommandWorkerSpec), f"{worker=} does not say how to launch by command"

        record = _CellRecord()
        info.record = record

        try:
            placements = cell_worker_placements(spec=spec, pg=self.pg)
            actor_handles = create_cell_worker_actors(spec=spec, pg=self.pg)
            record.workers = [
                ActorState(actor=actor, payload={}, placement=placement)
                for actor, placement in zip(actor_handles, placements, strict=True)
            ]

            addressing = allocate_cell_ports(
                port_allocator=self._port_allocator,
                port_infos=worker.port_infos,
                actors=actor_handles,
                node_ips=await probe_node_ips(actor_handles),
            )
            payloads = worker.build_member_payloads(addressing)
            record.workers = [
                ActorState(actor=actor, payload=payload, placement=placement)
                for actor, payload, placement in zip(actor_handles, payloads, placements, strict=True)
            ]

            if worker.prepare_workers is not None:
                await worker.prepare_workers(placements, actor_handles)

            plans = [worker.build_launch_plan(placement, addressing) for placement in placements]
            await asyncio.gather(
                *[
                    actor.run.remote(cmd=plan.cmd, envs=plan.envs)
                    for actor, plan in zip(actor_handles, plans, strict=True)
                ]
            )

            await worker.wait_cell_ready(addressing, functools.partial(_worker_is_alive, actor_handles[0]))
        except BaseException:
            logger.warning(f"Cell {cell_id} failed to come up; tearing its workers down")
            await self.stop_cell(cell_id)
            raise

        record.ready = True

    async def stop_cell(self, cell_id: str) -> None:
        info = self._infos.get(cell_id)
        record = info.record if info is not None else None
        if record is None:
            return
        info.record = None

        try:
            await asyncio.gather(
                *[
                    _shutdown_worker(cell_id=cell_id, local_index=local_index, worker=worker)
                    for local_index, worker in enumerate(record.workers)
                ]
            )
        finally:
            for local_index, worker in enumerate(record.workers):
                try:
                    ray.kill(worker.actor)
                    logger.info(f"Cell {cell_id}: killed worker at cell-local index {local_index}")
                except Exception as e:
                    logger.warning(f"Cell {cell_id}: fail to kill worker at cell-local index {local_index} ({e})")

    def cell_ids(self) -> list[str]:
        """The cells a consumer may use: only those that finished coming up."""
        return sorted(
            cell_id for cell_id, info in self._infos.items() if info.record is not None and info.record.ready
        )

    def cell_workers(self, cell_id: str) -> list[ActorState]:
        """The workers a consumer may use: only those of a cell that finished coming up."""
        info = self._infos.get(cell_id)
        record = info.record if info is not None else None
        return record.workers if record is not None and record.ready else []


def _worker_is_alive(actor_handle: ray.actor.ActorHandle) -> bool:
    try:
        ray.get(actor_handle._get_node_ip.remote(), timeout=30)
        return True
    except Exception:
        return False


async def _shutdown_worker(*, cell_id: str, local_index: int, worker: ActorState) -> None:
    logger.info(f"Cell {cell_id}: shutting down worker at cell-local index {local_index}")
    try:
        await asyncio.wait_for(_resolve(worker.actor.shutdown.remote()), timeout=SHUTDOWN_TIMEOUT)
    except Exception as e:
        logger.warning(
            f"Cell {cell_id}: graceful shutdown of worker at cell-local index {local_index} "
            f"failed, killing anyway ({e})"
        )


async def _resolve(object_ref: ray.ObjectRef) -> object:
    """Wrap a ray object ref in a coroutine, which is what asyncio timeouts expect."""
    return await object_ref
