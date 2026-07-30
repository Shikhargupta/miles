import asyncio
import functools
from dataclasses import dataclass

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.workers.addr_allocator import PortAllocator
from miles.utils.workers.worker_spec import PortInfo


@dataclass(frozen=True)
class CellAddressing:
    node_ips: list[str]
    master_ports: dict[str, int]
    per_worker_ports: list[dict[str, int]]


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
