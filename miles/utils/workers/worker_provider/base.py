import abc
from collections.abc import Awaitable, Callable

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.worker_handle import BaseWorkerHandle


class CellInfo(FrozenStrictBaseModel):
    cell_id: str
    members_hash: str
    member_urls: list[str]


ReconcileFn = Callable[[str, CellInfo | None], Awaitable[None]]

StopWatchFn = Callable[[], Awaitable[None]]


class BaseWorkerProvider(abc.ABC):
    @abc.abstractmethod
    def get_handle(self, worker_name: str) -> BaseWorkerHandle: ...

    @abc.abstractmethod
    def get_url(self, worker_name: str) -> str: ...

    @abc.abstractmethod
    async def watch_cells(self, reconcile_fn: ReconcileFn) -> StopWatchFn: ...
