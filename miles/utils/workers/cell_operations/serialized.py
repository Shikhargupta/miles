from __future__ import annotations

from typing import Protocol

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.worker_provider.base import CellInfo


class SuspendGate(Protocol):
    async def stop_cell_between_weight_updates(self, cell_id: str) -> None: ...


class SerializedCellOperations(BaseCellOperations):
    def __init__(self, inner: BaseCellOperations, *, gate: SuspendGate) -> None:
        self._inner = inner
        self._gate = gate

    async def cell_infos(self, *, pool_ids: list[str]) -> dict[str, CellInfo]:
        return await self._inner.cell_infos(pool_ids=pool_ids)

    async def suspend(self, *, cell_id: str) -> None:
        await self._gate.stop_cell_between_weight_updates(cell_id=cell_id)

    async def resume(self, *, cell_id: str) -> None:
        await self._inner.resume(cell_id=cell_id)

    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None:
        await self._inner.inject_fault(cell_id=cell_id, mode=mode, sub_index=sub_index)
