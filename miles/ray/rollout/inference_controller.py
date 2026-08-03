import asyncio
import logging
from dataclasses import dataclass

import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.rollout_server import RolloutServer, create_rollout_servers
from miles.ray.rollout.router_manager import start_session_server
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.ray.utils import Lock
from miles.utils.context_lock import (
    ContextLock,
    acquires_lock,
    enforce_lock_discipline,
    lock_exempt,
    releases_lock,
    requires_lock,
    with_lock,
)
from miles.utils.ft_utils.api_server.models import TriState
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

logger = logging.getLogger(__name__)


@enforce_lock_discipline
class InferenceController:
    @staticmethod
    @lock_exempt
    async def create(args) -> "InferenceController":
        controller = InferenceController(args)
        if not args.debug_train_only:
            controller.servers = await create_rollout_servers(args, context_lock=controller.context_lock)

            # TODO: may change to InferenceController.init(engine_provider, ...) later
            provider: BaseWorkerProvider = RayWorkerProvider.create()  # TODO inject instance
            controller._watcher_disposers.append(await provider.watch_cells(controller._reconcile))

            dashboard_hooks.register_router(args)
            await start_session_server(args)

            await asyncio.gather(*[srv.wait_expected_num_cells() for srv in controller.servers.values()])

        return controller

    @lock_exempt
    def __init__(self, args):
        self.args = args
        self.context_lock = ContextLock("InferenceController")
        self.servers: dict[str, RolloutServer] = {}
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()
        self._watcher_disposers: list[StopWatchFn] = []

    # -------------------------- rollout lifecycle hooks -----------------------------

    @with_lock
    async def prepare_rollout(self, rollout_id):
        await self._health_monitoring_resume()
        dashboard_hooks.register_engines(self.servers)

    @with_lock
    async def prepare_eval(self):
        await self._health_monitoring_resume()

    @with_lock
    async def dispose(self):
        for disposer in self._watcher_disposers:
            await disposer()
        self._watcher_disposers = []

    # -------------------------- offload/onload -----------------------------

    # TODO may parallelly execute offload/onload across services
    @with_lock
    async def offload(self, tags: list[str] | None = None):
        await self._health_monitoring_pause()
        for srv in self.servers.values():
            await srv.offload(tags=tags)

    @with_lock
    async def onload(self, tags: list[str] | None = None):
        await self._onload(tags=tags)

    @with_lock
    async def onload_weights(self):
        await self._onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    @with_lock
    async def onload_kv(self):
        await self._onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    @requires_lock
    async def _onload(self, tags: list[str] | None):
        for srv in self.servers.values():
            await srv.onload(tags)

    # -------------------------- engine management -----------------------------

    @acquires_lock
    async def start_update_weights(self) -> "EnginesAndLock":
        """Return engines eligible for weight updates."""
        await self._health_monitoring_pause()

        srv = self._get_updatable_server()
        if not srv:
            return EnginesAndLock(
                rollout_engines=[],
                rollout_engine_lock=self.rollout_engine_lock,
                engine_gpu_counts=[],
                engine_gpu_offsets=[],
                snapshot_cell_id_to_hashes={},
            )

        return EnginesAndLock(
            rollout_engines=srv.api_clients,
            rollout_engine_lock=self.rollout_engine_lock,
            engine_gpu_counts=srv.engine_gpu_counts,
            engine_gpu_offsets=srv.engine_gpu_offsets,
            snapshot_cell_id_to_hashes={cell_id: cell.meta.workers_hash for cell_id, cell in srv.server_cells.items()},
        )

    @releases_lock
    async def end_update_weights(self, snapshot_cell_id_to_hashes: dict[str, str]):
        await asyncio.gather(
            *[
                cell.mark_weights_ready()
                for srv in self.servers.values()
                for cell_id, cell in srv.server_cells.items()
                if cell_id in snapshot_cell_id_to_hashes
                and snapshot_cell_id_to_hashes[cell_id] == cell.meta.workers_hash
                and cell.is_pending_weights
            ]
        )

    @requires_lock
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

    # -------------------------- misc APIs -----------------------------

    @lock_exempt
    def get_cell_health_statuses(self) -> dict[str, TriState]:
        """Snapshot for the api server, which serves from its own thread and event loop.

        Deliberately lock-free and synchronous: taking the controller lock from there would
        block on whatever rollout step currently holds it, and awaiting into the controller's
        loop is not possible from the server's.
        """
        return {
            cell_id: cell.health_checker.status
            for srv in list(self.servers.values())
            for cell_id, cell in list(srv.server_cells.items())
        }

    @with_lock
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

    @with_lock
    async def _reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        # the provider reports every cell (routers, session servers, ...); only engine cells carry our meta
        if observed is not None and "model_id" not in observed.meta:
            return

        observed_cell_meta: ServerCellMetadata | None = (
            _compute_server_cell_meta_from_info(observed) if observed is not None else None
        )

        actual_srv: RolloutServer | None = None
        actual_cell: ServerCell | None = None
        for srv in self.servers.values():
            if (c := srv.server_cells.get(cell_id)) is not None:
                actual_srv, actual_cell = srv, c
                break

        if observed is not None and actual_srv is None:
            await self.servers[observed_cell_meta.model_id].add_cell(observed_cell_meta)
        elif observed is None and actual_srv is not None:
            await actual_srv.remove_cell(cell_id)
        elif (
            observed is not None
            and actual_srv is not None
            and observed_cell_meta.workers_hash != actual_cell.meta.workers_hash
        ):
            await actual_srv.remove_cell(cell_id)
            await actual_srv.add_cell(observed_cell_meta)

    # -------------------------- utils -----------------------------

    @requires_lock
    async def _health_monitoring_pause(self) -> None:
        for srv in self.servers.values():
            srv.health_checking_pause()

    @requires_lock
    async def _health_monitoring_resume(self) -> None:
        for srv in self.servers.values():
            srv.health_checking_resume()


@dataclass(frozen=True)
class EnginesAndLock:
    rollout_engines: list[SGLangApiClient]
    rollout_engine_lock: ray.actor.ActorHandle
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
    snapshot_cell_id_to_hashes: dict[str, str]


# TODO may move and generalize later
def _compute_server_cell_meta_from_info(info: CellInfo) -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id=info.meta["model_id"],
        worker_type=info.meta["worker_type"],
        cell_id=info.cell_id,
        num_gpus_per_engine=info.meta["num_gpus_per_engine"],
        gpu_offset=info.meta["gpu_offset"],
        worker_name=info.worker_names[0],
        needs_offload=info.meta["needs_offload"],
        update_weights=info.meta["update_weights"],
        workers_hash=info.workers_hash,
    )
