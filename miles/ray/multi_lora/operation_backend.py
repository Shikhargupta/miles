"""Operation-queue-owning backend; mount via --multi-lora-backend-path miles.ray.multi_lora.operation_backend.MultiLoRAOperationBackend."""

from typing import Any

from miles.ray.multi_lora.backend import MultiLoRABackend
from miles.ray.multi_lora.operations import OperationQueue


class MultiLoRAOperationBackend(MultiLoRABackend):
    """Owns one OperationQueue per live registration; retirement fences the queue."""

    def __init__(self, args: Any, router_url: str) -> None:
        super().__init__(args, router_url)
        self._operation_queues: dict[str, OperationQueue] = {}

    def operation_queue(self, name: str) -> OperationQueue:
        return self._operation_queues.setdefault(name, OperationQueue())

    async def register(self, name: str, config: Any) -> dict:
        result = await super().register(name, config)
        # A fresh queue per registration life: a re-registered name never sees its predecessor's ops.
        self._operation_queues[name] = OperationQueue()
        return result

    async def deregister(self, name: str) -> None:
        queue = self._operation_queues.get(name)
        if queue is not None:
            queue.fence("registration retired before the operation ran")
        await super().deregister(name)
