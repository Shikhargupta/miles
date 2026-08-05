import abc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts


@dataclass(frozen=True)
class CellInfo:
    cell_id: str
    spec_name: str
    alive: bool
    worker_names: list[str]
    workers_hash: str
    meta: dict[str, Any]  # TODO: in k8s native mode, may be provided from pod annotations


# args: (cell_id, CellInfo)
ReconcileFn = Callable[[str, CellInfo | None], Awaitable[None]]
StopWatchFn = Callable[[], Awaitable[None]]


class BaseWorkerProvider(abc.ABC):
    @abc.abstractmethod
    async def get_addr(self, worker_name: str) -> HostAndPort: ...

    @abc.abstractmethod
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts: ...

    @abc.abstractmethod
    async def watch_cells(self, reconcile: ReconcileFn, *, spec_names: list[str]) -> StopWatchFn: ...

    @abc.abstractmethod
    def get_worker_infos(self, *, cell_id: str) -> list[WorkerInfo]: ...

    @abc.abstractmethod
    def get_worker_infos_of(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]: ...

    def get_handle(self, worker_name: str, *, cell_id: str) -> BaseWorkerHandle:
        matches = [info for info in self.get_worker_infos(cell_id=cell_id) if info.name == worker_name]
        assert len(matches) == 1, f"{worker_name=} {cell_id=} matched {[info.name for info in matches]}"
        return matches[0].handle
