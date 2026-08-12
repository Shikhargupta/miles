import abc
import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from miles.utils.workers.naming import compute_cell_id, parse_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_spec import NamedHostAndPorts


_observation_sequence = itertools.count(1)


def allocate_observation_seq() -> int:
    return next(_observation_sequence)


class ObservationSupersededError(Exception):
    pass


@dataclass(frozen=True)
class CellInfo:
    cell_id: str
    pool_id: str
    alive: bool
    worker_names: list[str]
    workers_hash: str
    meta: dict[str, Any]  # TODO: in k8s native mode, may be provided from pod annotations
    observation_seq: int = field(default_factory=allocate_observation_seq, compare=False)


def restamp_observation(info: CellInfo) -> CellInfo:
    return replace(info, observation_seq=allocate_observation_seq())


# args: (cell_id, CellInfo)
ReconcileFn = Callable[[str, CellInfo | None], Awaitable[None]]
StopWatchFn = Callable[[], Awaitable[None]]


class BaseWorkerProvider(abc.ABC):
    async def init(self) -> None:
        return None

    @abc.abstractmethod
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts: ...

    @abc.abstractmethod
    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]: ...

    async def watch_cells(self, reconcile: ReconcileFn) -> StopWatchFn:
        raise NotImplementedError(f"{type(self).__name__} answers addresses, it does not observe cells")

    def expected_num_cells(self, *, model_id: str) -> int | None:
        return None

    def extra_expected_num_cells(self, *, model_id: str) -> int:
        """How many cells of ``model_id`` this provider expects on top of the ones this deployment launches."""
        return 0

    @abc.abstractmethod
    async def invalidate_cell(self, cell_id: str) -> None:
        """Drop ``cell_id``, so a provider that watches a platform announces it again once it observes it anew.

        Callers hold their own context lock across this call, so an implementation must return without
        doing any networked call: it may only mark its own state and let its watch loop observe anew.
        """

    def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        (infos,) = self.get_worker_infos(cell_ids=[cell_id_of_worker(worker_name)])
        return select_handle(worker_name=worker_name, infos=infos)

    async def get_handle_async(self, worker_name: str) -> BaseWorkerHandle:
        return self.get_handle(worker_name)


def cell_id_of_worker(worker_name: str) -> str:
    pool_id, cell_index, _worker_in_cell_index = parse_worker_name(worker_name)
    return compute_cell_id(pool_id=pool_id, cell_index=cell_index)


def select_handle(*, worker_name: str, infos: list[WorkerInfo]) -> BaseWorkerHandle:
    matches = [info for info in infos if info.name == worker_name]
    assert len(matches) == 1, f"{worker_name=} matched {[info.name for info in matches]}"
    handle = matches[0].handle
    assert handle is not None, f"{worker_name} has no worker class, so its rpc methods are unknown"
    return handle
