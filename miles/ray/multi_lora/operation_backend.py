"""Operation-queue-owning backend; mount via --multi-lora-backend-path miles.ray.multi_lora.operation_backend.MultiLoRAOperationBackend."""

from collections import deque
from typing import Any

from miles.ray.multi_lora.backend import MultiLoRABackend
from miles.ray.multi_lora.operations import CONTROL_KINDS, OperationQueue
from miles.ray.multi_lora.registry import AdapterState


class MultiLoRAOperationBackend(MultiLoRABackend):
    """Owns one OperationQueue per live registration; retirement fences the queue."""

    def __init__(self, args: Any, router_url: str) -> None:
        super().__init__(args, router_url)
        self._operation_queues: dict[str, OperationQueue] = {}
        self._rotation: deque[str] = deque()

    def operation_queue(self, name: str) -> OperationQueue:
        return self._operation_queues.setdefault(name, OperationQueue())

    async def register(self, name: str, config: Any) -> dict:
        result = await super().register(name, config)
        # A fresh queue per registration life: a re-registered name never sees its predecessor's ops.
        self._operation_queues[name] = OperationQueue()
        if name not in self._rotation:
            self._rotation.append(name)
        return result

    async def deregister(self, name: str) -> None:
        queue = self._operation_queues.get(name)
        if queue is not None:
            queue.fence("registration retired before the operation ran")
        await super().deregister(name)

    # ------------------------------ operation rounds ------------------------------

    def collect_operation_round(self) -> dict:
        """One claim per ACTIVE registration in rotation order: data prefixes co-batch, control ops route to RPCs."""
        data_ops: list[dict] = []
        control_ops: list[dict] = []
        for name in self._rotation_pass():
            record = self.registry.find(name)
            if record is None or record.state is not AdapterState.ACTIVE:
                continue
            queue = self._operation_queues.get(name)
            if queue is None:
                continue
            claimed = queue.claim_next_runnable_ops()
            bucket = control_ops if (claimed and claimed[0].kind in CONTROL_KINDS) else data_ops
            for rec in claimed:
                bucket.append(
                    {
                        "name": name,
                        "slot": record.slot,
                        "ordinal": rec.ordinal,
                        "request_id": rec.request_id,
                        "kind": rec.kind,
                        "payload": rec.payload,
                    }
                )
        return {"data_ops": data_ops, "control_ops": control_ops}

    def _rotation_pass(self) -> list[str]:
        """Snapshot in rotation order (pruning names whose queue is gone), then advance the head."""
        live = [name for name in self._rotation if name in self._operation_queues]
        self._rotation = deque(live)
        if self._rotation:
            self._rotation.rotate(-1)
        return live

    def complete_operations(self, results: list[dict]) -> None:
        """Apply driver outcomes; first-terminal-wins in the queue makes fence races safe."""
        for outcome in results:
            queue = self._operation_queues.get(outcome["name"])
            if queue is None:
                continue
            if outcome.get("ok", False):
                queue.complete(outcome["ordinal"], outcome.get("result"))
            else:
                queue.fail(
                    outcome["ordinal"], outcome.get("error", "operation failed"), outcome.get("category", "server")
                )

    def operation_queue_depths(self) -> dict[str, int]:
        return {name: queue.open_count() for name, queue in self._operation_queues.items()}
