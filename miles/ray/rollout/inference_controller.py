import asyncio
import contextlib
import logging
from dataclasses import dataclass

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.rollout_server import RolloutServer, list_cell_ids
from miles.ray.utils import Lock
from miles.utils.ft_utils.api_server.models import CellCondition, CellStatus, TriState
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, StopWatchFn

logger = logging.getLogger(__name__)


class InferenceController:
    @staticmethod
    async def create(
        args, *, servers: dict[str, "RolloutServer"], provider: BaseWorkerProvider | None
    ) -> "InferenceController":
        controller = InferenceController(args, servers=servers, provider=provider)
        if provider is not None:
            observed = await provider.list_cells()
            for cell_id in controller.list_cell_ids():
                await controller._reconcile(cell_id, observed.get(cell_id))
            controller._watcher_disposers.append(await provider.watch_cells(controller._reconcile, seen=observed))
        return controller

    def __init__(self, args, *, servers: dict[str, "RolloutServer"], provider: BaseWorkerProvider | None):
        self.args = args
        self.servers = servers
        self.provider = provider
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self.rollout_id = -1
        self._reconcile_gate = _ReconcileGate()
        self._cell_ops_lock = asyncio.Lock()
        self._watcher_disposers: list[StopWatchFn] = []

    # -------------------------- rollout lifecycle hooks -----------------------------

    async def prepare_rollout(self, rollout_id):
        self.rollout_id = rollout_id
        await self.health_monitoring_resume()
        if self.args.ci_test and self.args.use_fault_tolerance and rollout_id >= 2:
            await self._try_ci_fault_injection()
        dashboard_hooks.register_engines(self.servers)

    async def prepare_eval(self):
        await self.health_monitoring_resume()

    async def dispose(self):
        for disposer in self._watcher_disposers:
            await disposer()
        self._watcher_disposers = []

    # -------------------------- offload/onload -----------------------------

    # TODO may parallelly execute offload/onload across services
    async def offload(self, tags: list[str] | None = None):
        await self.health_monitoring_pause()
        for srv in self.servers.values():
            await srv.offload(tags=tags)

    async def onload(self, tags: list[str] | None = None):
        for srv in self.servers.values():
            await srv.onload(tags)

    async def onload_weights(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    async def onload_kv(self):
        await self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    # -------------------------- engine management -----------------------------

    async def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates."""
        srv = self._get_updatable_server()
        if not srv:
            return EnginesAndLock(
                rollout_engines=[],
                rollout_engine_lock=self.rollout_engine_lock,
                has_new_engines=False,
                engine_gpu_counts=[],
                engine_gpu_offsets=[],
            )

        await srv.wait_all_engines_alive()
        return EnginesAndLock(
            rollout_engines=srv.api_clients,
            rollout_engine_lock=self.rollout_engine_lock,
            has_new_engines=srv.has_new_engines,
            engine_gpu_counts=srv.engine_gpu_counts,
            engine_gpu_offsets=srv.engine_gpu_offsets,
        )

    async def clear_updatable_has_new_engines(self):
        # when fault tolerance is not enabled, we need to manually clear has_new_engines after update_weights
        srv = self._get_updatable_server()
        if srv:
            srv.clear_has_new_engines()

    def _get_updatable_server(self) -> RolloutServer | None:
        updatable = [srv for srv in self.servers.values() if srv.update_weights]
        match updatable:
            case []:
                return None
            case [srv]:
                return srv
            case _:
                raise ValueError(
                    f"Multiple servers have update_weights=True: {[srv.model_name for srv in updatable]}. "
                    f"Only one updatable server is supported."
                )

    # -------------------------- external observation -----------------------------

    def list_cell_ids(self) -> list[str]:
        return list_cell_ids(self.servers)

    def compute_cell_status(self, cell_id: str) -> CellStatus:
        cell = self._server_of(cell_id).server_cells.get(cell_id)
        if cell is None:
            return CellStatus(phase="Suspended", conditions=[CellCondition.allocated(TriState.FALSE)])
        if not cell.is_alive:
            return CellStatus(
                phase="Running",
                conditions=[
                    CellCondition.allocated(TriState.TRUE),
                    CellCondition.healthy(TriState.UNKNOWN, reason="AttachPending"),
                ],
            )
        return CellStatus(
            phase="Running",
            conditions=[CellCondition.allocated(TriState.TRUE), CellCondition.healthy(TriState.TRUE)],
        )

    def _server_of(self, cell_id: str) -> RolloutServer:
        srv = self._find_server_of(cell_id)
        assert srv is not None, f"{cell_id=} does not name a cell of any rollout server"
        return srv

    def _find_server_of(self, cell_id: str) -> RolloutServer | None:
        owners = [srv for srv in self.servers.values() if cell_id in srv.cell_specs]
        assert len(owners) <= 1, f"{cell_id=} must name at most one cell, but {len(owners)} servers hold it"
        return owners[0] if owners else None

    # -------------------------- misc APIs -----------------------------

    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        # Only the updatable model is re-synced; a frozen model would always mismatch.
        srv = self._get_updatable_server()
        if srv is None:
            return []
        return await srv.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )

    # -------------------------- reconcile -----------------------------

    async def _reconcile(self, cell_id: str, cell_info: CellInfo | None) -> None:
        async with self._reconcile_gate.operate(), self._cell_ops_lock:
            srv = self._find_server_of(cell_id)
            if srv is None:
                return
            attached = srv.server_cells.get(cell_id)

            if cell_info is None:
                if attached is not None:
                    logger.info(f"Reconcile removes cell {cell_id}")
                    await srv.reconcile_detach(cell_id)
                return

            if len(cell_info.members) < srv.cell_specs[cell_id].worker.scheduling.num_workers_per_cell:
                return

            if attached is not None:
                if [member.payload for member in attached.attached_members] == [
                    member.payload for member in cell_info.members
                ]:
                    return
                logger.info(f"Cell {cell_id} changed members; replacing it")
                await srv.reconcile_detach(cell_id)

            logger.info(f"Reconcile adds cell {cell_id}")
            await srv.reconcile_attach(cell_info, release_memory=self.rollout_id >= 0)

    # -------------------------- utils -----------------------------

    async def health_monitoring_pause(self) -> None:
        await self._reconcile_gate.pause()

    async def health_monitoring_resume(self) -> None:
        await self._reconcile_gate.resume()

    @property
    def _server(self) -> RolloutServer | None:
        """Default server (first model).  For backward compatibility."""
        if not self.servers:
            return None
        return next(iter(self.servers.values()))

    async def _try_ci_fault_injection(self):
        raise NotImplementedError("rollout fault injection is being rebuilt with rollout fault tolerance")


class _ReconcileGate:
    """Blocks reconcile add/remove while weight updates or offload snapshots run.

    ``pause`` waits for in-flight reconcile operations to drain, so the set of
    attached cells is stable from pause until resume."""

    def __init__(self) -> None:
        self._paused = False
        self._num_active = 0
        self._condition = asyncio.Condition()

    async def pause(self) -> None:
        async with self._condition:
            self._paused = True
            await self._condition.wait_for(lambda: self._num_active == 0)

    async def resume(self) -> None:
        async with self._condition:
            self._paused = False
            self._condition.notify_all()

    @contextlib.asynccontextmanager
    async def operate(self):
        async with self._condition:
            await self._condition.wait_for(lambda: not self._paused)
            self._num_active += 1
        try:
            yield
        finally:
            async with self._condition:
                self._num_active -= 1
                self._condition.notify_all()


@dataclass(frozen=True)
class EnginesAndLock:
    rollout_engines: list[SGLangApiClient]
    rollout_engine_lock: ray.actor.ActorHandle
    has_new_engines: bool
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
