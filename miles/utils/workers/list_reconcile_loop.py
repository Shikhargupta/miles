import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

from miles.utils.test_utils.clock import Clock, RealClock
from miles.utils.workers.worker_provider.base import CellInfo, ReconcileFn

logger = logging.getLogger(__name__)

ListCellsFn = Callable[[], Awaitable[list[CellInfo]]]


class ListBasedReconcileLoop:
    def __init__(
        self,
        *,
        list_cells_fn: ListCellsFn,
        reconcile_fn: ReconcileFn,
        poll_interval_seconds: float,
        clock: Clock | None = None,
    ) -> None:
        assert poll_interval_seconds > 0, f"{poll_interval_seconds=} must be positive"

        self._list_cells_fn = list_cells_fn
        self._reconcile_fn = reconcile_fn
        self._poll_interval_seconds = poll_interval_seconds
        self._clock = clock or RealClock()

        self._snapshot: dict[str, CellInfo] = {}
        self._task: asyncio.Task[None] | None = None
        self._started = False

    async def start(self) -> None:
        assert not self._started, "ListBasedReconcileLoop.start() must be called exactly once"
        self._started = True

        await self._poll_once()
        self._task = asyncio.create_task(self._poll_forever())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _poll_forever(self) -> None:
        while True:
            await self._clock.sleep(self._poll_interval_seconds)
            try:
                await self._poll_once()
            except Exception:
                logger.exception("List-based reconcile poll failed; will retry after the poll interval")

    async def _poll_once(self) -> None:
        observed_cells = await self._list_cells_fn()
        observed = {cell.cell_id: cell for cell in observed_cells}
        assert len(observed) == len(observed_cells), f"cell ids must be unique, got {sorted(observed)=}"

        for cell_id, cell in observed.items():
            if self._snapshot.get(cell_id) != cell:
                await self._reconcile_fn(cell_id, cell)
        for cell_id in self._snapshot:
            if cell_id not in observed:
                await self._reconcile_fn(cell_id, None)

        self._snapshot = observed
