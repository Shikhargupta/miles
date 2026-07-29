import asyncio

import ray

_DYNAMIC_PORT_START = 15000


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
