import asyncio

import ray

from miles.utils.workers.ray_worker_manager.state import WorkerState

_DYNAMIC_PORT_START = 15000


class PortAllocator:
    def __init__(self) -> None:
        self._cursors: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def collect_worker_ports(self, *, worker: WorkerState, is_master: bool) -> None:
        owned_port_infos = [p for p in worker.cell.spec.port_infos if p.mode == "per_worker" or is_master]

        dynamic_port_infos = [p for p in owned_port_infos if p.allow_dynamic]
        if dynamic_port_infos:
            first_port = await self._allocate_block(
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

    async def _allocate_block(self, *, actor: ray.actor.ActorHandle, node_ip: str, count: int) -> int:
        async with self._lock:
            start_port = self._cursors.get(node_ip, _DYNAMIC_PORT_START)
            first_port = await actor._get_free_port_block.remote(start_port=start_port, count=count)
            self._cursors[node_ip] = first_port + count
            return first_port
