import hashlib
import json

import ray

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.clock import Clock
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.list_reconcile_loop import ListBasedReconcileLoop
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn


class RayWorkerInfo(FrozenStrictBaseModel):
    name: str
    cell_id: str
    generation: int
    url: str | None


class RayWorkerProvider(BaseWorkerProvider):
    def __init__(
        self,
        *,
        manager: ray.actor.ActorHandle,
        spec_name: str,
        poll_interval_seconds: float,
        clock: Clock | None = None,
    ) -> None:
        self._manager = manager
        self._spec_name = spec_name
        self._poll_interval_seconds = poll_interval_seconds
        self._clock = clock

    async def get_handle(self, worker_name: str) -> BaseWorkerHandle:
        return RayWorkerHandle(ray.get_actor(worker_name))

    async def get_url(self, worker_name: str) -> str:
        worker_info = self._find_worker_info(worker_infos=await self._fetch_worker_infos(), worker_name=worker_name)
        assert worker_info.url is not None, f"{worker_name=} has no url"
        return worker_info.url

    async def watch_cells(self, reconcile_fn: ReconcileFn) -> StopWatchFn:
        loop = ListBasedReconcileLoop(
            list_cells_fn=self._list_cells,
            reconcile_fn=reconcile_fn,
            poll_interval_seconds=self._poll_interval_seconds,
            clock=self._clock,
        )
        await loop.start()
        return loop.stop

    async def _list_cells(self) -> list[CellInfo]:
        members_by_cell_id: dict[str, list[RayWorkerInfo]] = {}
        for worker_info in await self._fetch_worker_infos():
            members_by_cell_id.setdefault(worker_info.cell_id, []).append(worker_info)

        return [
            CellInfo(
                cell_id=cell_id,
                members_hash=_compute_members_hash(members),
                member_urls=[member.url for member in members if member.url is not None],
            )
            for cell_id, members in members_by_cell_id.items()
        ]

    async def _fetch_worker_infos(self) -> list[RayWorkerInfo]:
        return await self._manager.get_worker_infos.remote(spec_name=self._spec_name)

    def _find_worker_info(self, *, worker_infos: list[RayWorkerInfo], worker_name: str) -> RayWorkerInfo:
        matches = [worker_info for worker_info in worker_infos if worker_info.name == worker_name]
        names = sorted(worker_info.name for worker_info in worker_infos)
        assert len(matches) == 1, f"{worker_name=} must name exactly one worker of {self._spec_name=}, got {names=}"
        return matches[0]


def _compute_members_hash(members: list[RayWorkerInfo]) -> str:
    payload = json.dumps([[member.name, member.generation] for member in members])
    return hashlib.sha256(payload.encode()).hexdigest()
