import asyncio
import logging

from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellMember, ReconcileFn, StopWatchFn

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5.0


class RayWorkerProvider(BaseWorkerProvider):
    def __init__(self, *, worker_manager: RayWorkerManager, poll_interval_seconds: float = POLL_INTERVAL_SECONDS):
        self._worker_manager = worker_manager
        self._poll_interval_seconds = poll_interval_seconds

    async def list_cells(self) -> dict[str, CellInfo]:
        cells: dict[str, CellInfo] = {}
        for cell_id in self._worker_manager.cell_ids():
            workers = self._worker_manager.cell_workers(cell_id)
            cells[cell_id] = CellInfo(
                cell_id=cell_id,
                members=[
                    CellMember(handle=worker.actor, payload=worker.payload, placement=worker.placement)
                    for worker in workers
                ],
            )
        return cells

    async def watch_cells(self, reconcile: ReconcileFn, seen: dict[str, CellInfo] | None = None) -> StopWatchFn:
        task = asyncio.create_task(self._watch_loop(reconcile, dict(seen) if seen is not None else {}))

        async def _stop() -> None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        return _stop

    async def _watch_loop(self, reconcile: ReconcileFn, seen: dict[str, CellInfo]) -> None:
        while True:
            try:
                observed = await self.list_cells()
                for cell_id in sorted(set(seen) | set(observed)):
                    current = observed.get(cell_id)
                    if seen.get(cell_id) == current:
                        continue
                    await reconcile(cell_id, current)
                seen = observed
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Worker provider poll failed; retrying")
            await asyncio.sleep(self._poll_interval_seconds)
