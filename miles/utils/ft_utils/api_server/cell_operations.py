from __future__ import annotations

from typing import Any, Protocol

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.worker_provider.base import CellInfo


class BaseCellOperations(Protocol):
    async def cell_infos(self, spec_names: list[str]) -> dict[str, CellInfo]: ...

    async def suspend(self, cell_id: str) -> None: ...

    async def resume(self, cell_id: str) -> None: ...

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None: ...


class RayCellOperations:
    def __init__(self, worker_manager: Any) -> None:
        self._worker_manager = worker_manager

    async def cell_infos(self, spec_names: list[str]) -> dict[str, CellInfo]:
        return await self._worker_manager.get_cell_infos.remote(spec_names=spec_names)

    async def suspend(self, cell_id: str) -> None:
        await self._worker_manager.stop_cells.remote([cell_id])

    async def resume(self, cell_id: str) -> None:
        await self._worker_manager.start_cells.remote([cell_id])

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        await self._worker_manager.inject_fault.remote(cell_id, mode=mode.value, worker_in_cell_index=sub_index)


class K8sCellOperations:
    def __init__(self, *, provider: Any, delete_pods: Any, colocated_with: Any = None) -> None:
        self._provider = provider
        self._delete_pods = delete_pods
        self._colocated_with = colocated_with

    async def cell_infos(self, spec_names: list[str]) -> dict[str, CellInfo]:
        wanted = set(spec_names)
        infos = (self._provider.cell_info(cell_id) for cell_id in self._provider.cell_ids())
        return {info.cell_id: info for info in infos if info is not None and info.spec_name in wanted}

    async def suspend(self, cell_id: str) -> None:
        pods = list(self._provider.pod_names(cell_id))
        assert pods, f"cannot suspend {cell_id}, which has no pods"

        if self._colocated_with is not None:
            pods += self._colocated_with(cell_id)
        await self._delete_pods(pods)

    async def resume(self, cell_id: str) -> None:
        return None

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        raise NotImplementedError("fault injection reaches into a worker process, which needs the rpc layer")
