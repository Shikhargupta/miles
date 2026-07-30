import asyncio
import functools
import importlib

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.worker_spec import (
    BaseCellSpec,
    BaseWorkerSpec,
    CellAddressing,
    CommandWorkerSpec,
    PortInfo,
    ServeWorkerSpec,
    WorkerPlacement,
)


def cell_worker_placements(*, spec: BaseCellSpec, pg) -> list[WorkerPlacement]:
    """Where each worker of the cell goes; the pg index math lives only here."""
    _, _, reordered_gpu_ids = pg
    num_gpus_per_worker = int(spec.worker.scheduling.num_gpus_per_worker)
    return [
        WorkerPlacement(
            local_index=local_index,
            global_rank=spec.rank_offset + local_index,
            base_gpu_id=int(reordered_gpu_ids[spec.gpu_offset + local_index * num_gpus_per_worker]),
        )
        for local_index in range(spec.worker.scheduling.num_workers_per_cell)
    ]


def create_cell_worker_actors(*, spec: BaseCellSpec, pg) -> list[ray.actor.ActorHandle]:
    """Create every worker of one cell, from the spec alone."""
    pg_handle, reordered_bundle_indices, _ = pg
    num_gpus_per_worker = int(spec.worker.scheduling.num_gpus_per_worker)

    return [
        create_cell_worker_actor(
            worker=spec.worker,
            placement=placement,
            pg_handle=pg_handle,
            bundle_index=reordered_bundle_indices[spec.gpu_offset + placement.local_index * num_gpus_per_worker],
        )
        for placement in cell_worker_placements(spec=spec, pg=pg)
    ]


def create_cell_worker_actor(
    *,
    worker: BaseWorkerSpec,
    placement: WorkerPlacement,
    pg_handle: object,
    bundle_index: int,
) -> ray.actor.ActorHandle:
    if isinstance(worker, CommandWorkerSpec):
        worker_cls: type = CommandActor
        ctor_kwargs: dict = {}
    else:
        assert isinstance(worker, ServeWorkerSpec), f"{worker=} does not say how to bring a worker up"
        worker_cls = _resolve_worker_class(worker.worker_class)
        ctor_kwargs = worker.ctor_kwargs(placement)

    return create_pg_worker_actor(
        worker_cls=worker_cls,
        pg_handle=pg_handle,
        bundle_index=bundle_index,
        env_vars=worker.env_var(placement),
        num_cpus=worker.ray_options.num_cpus,
        num_gpus=worker.ray_options.num_gpus,
        ctor_kwargs=ctor_kwargs,
    )


def _resolve_worker_class(worker_class: str) -> type:
    module_path, _, class_name = worker_class.rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def create_pg_worker_actor(
    *,
    worker_cls: type,
    pg_handle: object,
    bundle_index: int,
    env_vars: dict[str, str],
    num_cpus: float,
    num_gpus: float,
    ctor_kwargs: dict,
) -> ray.actor.ActorHandle:
    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg_handle,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=bundle_index,
    )
    actor_cls = ray.remote(worker_cls)
    return actor_cls.options(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        scheduling_strategy=scheduling_strategy,
        runtime_env={
            "env_vars": env_vars,
        },
    ).remote(**ctor_kwargs)


async def probe_node_ips(actor_handles: list[ray.actor.ActorHandle]) -> list[str]:
    return list(await asyncio.gather(*[actor._get_node_ip.remote() for actor in actor_handles]))


def allocate_cell_ports(
    *,
    port_allocator: PortAllocator,
    port_infos: list[PortInfo],
    actors: list[ray.actor.ActorHandle],
    node_ips: list[str],
) -> CellAddressing:
    master_ports: dict[str, int] = {}
    per_worker_ports: list[dict[str, int]] = []

    for local_index, (actor, node_ip) in enumerate(zip(actors, node_ips, strict=True)):
        alloc = functools.partial(port_allocator.alloc, engine=actor, node_ip=node_ip)

        if local_index == 0:
            master_ports = {
                info.name: alloc(consecutive=info.num_consecutive) for info in port_infos if info.mode == "master"
            }
        per_worker_ports.append(
            {info.name: alloc(consecutive=info.num_consecutive) for info in port_infos if info.mode == "per_worker"}
        )

    return CellAddressing(node_ips=node_ips, master_ports=master_ports, per_worker_ports=per_worker_ports)
