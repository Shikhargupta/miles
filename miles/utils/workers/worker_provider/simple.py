from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn


class SimpleWorkerProvider(BaseWorkerProvider):
    def __init__(self, *, worker_urls: dict[str, str], cells: list[CellInfo]) -> None:
        cell_ids = [cell.cell_id for cell in cells]
        assert len(cell_ids) == len(set(cell_ids)), f"{cell_ids=} must be unique"

        self._worker_urls = dict(worker_urls)
        self._cells = list(cells)

    async def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        raise NotImplementedError("SimpleWorkerProvider.get_handle requires the rpc worker handle layer")

    async def get_url(self, worker_name: str) -> str:
        url = self._worker_urls.get(worker_name)
        assert url is not None, f"{worker_name=} not found in {sorted(self._worker_urls)=}"
        return url

    async def watch_cells(self, reconcile_fn: ReconcileFn) -> StopWatchFn:
        for cell in self._cells:
            await reconcile_fn(cell.cell_id, cell)
        return _noop_stop_watch


async def _noop_stop_watch() -> None:
    pass
