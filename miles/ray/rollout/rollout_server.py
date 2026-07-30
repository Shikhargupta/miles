import asyncio
import dataclasses
import logging
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.router_manager import start_router
from miles.ray.rollout.server_cell import ServerCell
from miles.ray.specs.inference import InferenceModelSpec, compute_inference_model_specs
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_provider.base import CellInfo

logger = logging.getLogger(__name__)


async def start_rollout_servers(args, worker_manager: RayWorkerManager) -> dict[str, "RolloutServer"]:
    """Start the routers and bring up every rollout worker.

    The cells stay detached: the reconcile loop attaches them to whatever the
    worker manager reports. Returns a dict mapping model name -> ``RolloutServer``.
    """
    model_specs = compute_inference_model_specs(args)

    servers: dict[str, RolloutServer] = {}

    for model_idx, model_spec in enumerate(model_specs):
        router_ip, router_port = start_router(
            args, has_pd_disaggregation=model_spec.has_pd_disaggregation, force_new=(model_idx > 0)
        )

        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        servers[model_spec.name] = RolloutServer(
            server_cells=_build_server_cells(args, model_spec=model_spec),
            args=args,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_spec.name,
            update_weights=model_spec.update_weights,
        )

    if not args.debug_train_only:
        await asyncio.gather(
            *[worker_manager.start_cell(cell.spec) for srv in servers.values() for cell in srv.server_cells.values()]
        )

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


def _build_server_cells(args, *, model_spec: InferenceModelSpec) -> dict[str, ServerCell]:
    return {
        cell.cell_id: ServerCell(args=args, spec=cell, update_weights=model_spec.update_weights)
        for cell in model_spec.cells
    }


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, as a dict of cell id -> cell.

    Each RolloutServer represents one model deployed behind a single router.
    """

    server_cells: dict[str, ServerCell]
    args: Any
    # NOTE: this may have risk when recovering engines parallelly; may use source of truth (cells) later
    has_new_engines: bool = False
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True

    async def reconcile_attach(self, cell_info: CellInfo, *, release_memory: bool) -> None:
        """Attach the observed workers to their cell and register it with the router."""
        cell = self.server_cells[cell_info.cell_id]
        cell.attach(cell_info)

        try:
            if release_memory and cell.needs_offload:
                await asyncio.wait_for(cell.release_offloaded_memory(), timeout=_ATTACH_STEP_TIMEOUT_SECONDS)

            cell.mark_alive()
            await asyncio.wait_for(cell.register(self.router_api_client), timeout=_ATTACH_STEP_TIMEOUT_SECONDS)
        except Exception:
            logger.warning(f"Attaching cell {cell.cell_id} failed; detaching it so the next poll retries")
            cell.detach()
            raise

        if self.update_weights:
            self.has_new_engines = True

    async def reconcile_detach(self, cell_id: str) -> None:
        """Detach the cell whose workers vanished, unregistering it from the router first."""
        cell = self.server_cells[cell_id]
        if cell.is_alive:
            try:
                await asyncio.wait_for(cell.unregister(self.router_api_client), timeout=_UNREGISTER_TIMEOUT_SECONDS)
            except Exception:
                logger.warning(f"Unregistering cell {cell_id} from the router failed; detaching anyway")
        cell.detach()
        if self.update_weights:
            self.has_new_engines = True

    @property
    def api_clients(self) -> list[SGLangApiClient]:
        """One client per cell, talking to its primary (node-0) engine."""
        return [cell.api_client for cell in self.server_cells.values()]

    def clear_has_new_engines(self):
        self.has_new_engines = False

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [cell.num_gpus_per_engine for cell in self.server_cells.values()]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        return [cell.gpu_offset for cell in self.server_cells.values()]

    async def offload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.offload(tags=tags) for cell in self._allocated_cells_of() if cell.needs_offload]
        )

    async def onload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.onload(tags=tags) for cell in self._allocated_cells_of() if cell.needs_offload]
        )

    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        return await asyncio.gather(
            *[
                cell.check_weights(
                    action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
                )
                for cell in self._allocated_cells_of()
            ]
        )

    async def wait_all_engines_alive(self, timeout: float = 600):
        # TODO: 600s default is hardcoded; make it configurable (e.g. via args) once we have a clearer
        # picture of init/recovery upper bounds across model sizes
        sleep_time = 2
        for _ in range(int(timeout // sleep_time)):
            if all(cell.is_alive for cell in self.server_cells.values()):
                return
            await asyncio.sleep(sleep_time)
            logger.info("wait_all_engines_alive looping...")
        raise TimeoutError(f"Timed out after {timeout}s waiting for engines to become ready")

    def _allocated_cells_of(self, cell_ids: list[str] | None = None) -> list[ServerCell]:
        if cell_ids is None:
            cell_ids = list(self.server_cells)
        return [self.server_cells[cell_id] for cell_id in cell_ids if self.server_cells[cell_id].is_allocated]

    @property
    def router_api_client(self) -> SGLangRouterApiClient:
        return SGLangRouterApiClient(router_url=f"http://{self.router_ip}:{self.router_port}")


def list_cell_ids(servers: dict[str, "RolloutServer"]) -> list[str]:
    return [cell_id for model_id in sorted(servers) for cell_id in servers[model_id].server_cells]


_UNREGISTER_TIMEOUT_SECONDS = 30

_ATTACH_STEP_TIMEOUT_SECONDS = 60
