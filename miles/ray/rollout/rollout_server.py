import asyncio
import dataclasses
import logging
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.router_manager import start_router
from miles.ray.rollout.server_cell import ServerCell
from miles.ray.specs.inference import InferenceCellSpec, compute_inference_model_specs
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_provider.base import CellInfo

logger = logging.getLogger(__name__)


async def start_rollout_servers(args, worker_manager: RayWorkerManager) -> dict[str, "RolloutServer"]:
    """Start the routers and bring up every rollout worker.

    No cell object exists yet: the reconcile loop creates one per cell the worker
    manager reports. Returns a dict mapping model name -> ``RolloutServer``.
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
            cell_specs={cell.cell_id: cell for cell in model_spec.cells},
            args=args,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_spec.name,
            update_weights=model_spec.update_weights,
        )

    if not args.debug_train_only:
        specs = [spec for srv in servers.values() for spec in srv.cell_specs.values()]
        worker_manager.register_cells(specs)
        await asyncio.gather(*[worker_manager.start_cell(spec.cell_id) for spec in specs])

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router.

    ``cell_specs`` is the static layout of the model; ``server_cells`` holds
    only the cells whose workers currently exist, so it grows and shrinks as
    the reconcile loop observes the infrastructure layer.
    """

    cell_specs: dict[str, InferenceCellSpec]
    args: Any
    server_cells: dict[str, ServerCell] = dataclasses.field(default_factory=dict)
    # NOTE: this may have risk when recovering engines parallelly; may use source of truth (cells) later
    has_new_engines: bool = False
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True

    async def reconcile_attach(self, cell_info: CellInfo, *, release_memory: bool) -> None:
        """Create the cell for the workers just observed and register it with the router."""
        assert cell_info.cell_id not in self.server_cells, f"cell {cell_info.cell_id} is already attached"

        cell = ServerCell.attach(
            args=self.args,
            spec=self.cell_specs[cell_info.cell_id],
            update_weights=self.update_weights,
            cell_info=cell_info,
        )
        self.server_cells[cell_info.cell_id] = cell

        try:
            if release_memory and cell.needs_offload:
                await asyncio.wait_for(cell.release_offloaded_memory(), timeout=_ATTACH_STEP_TIMEOUT_SECONDS)

            cell.mark_alive()
            await asyncio.wait_for(cell.register(self.router_api_client), timeout=_ATTACH_STEP_TIMEOUT_SECONDS)
        except Exception:
            logger.warning(f"Attaching cell {cell.cell_id} failed; removing it so the next poll retries")
            del self.server_cells[cell.cell_id]
            raise

        if self.update_weights:
            self.has_new_engines = True

    async def reconcile_detach(self, cell_id: str) -> None:
        """Drop the cell whose workers vanished, unregistering it from the router first."""
        cell = self.server_cells[cell_id]
        if cell.is_alive:
            try:
                await asyncio.wait_for(cell.unregister(self.router_api_client), timeout=_UNREGISTER_TIMEOUT_SECONDS)
            except Exception:
                logger.warning(f"Unregistering cell {cell_id} from the router failed; removing anyway")
        del self.server_cells[cell_id]
        if self.update_weights:
            self.has_new_engines = True

    @property
    def api_clients(self) -> list[SGLangApiClient]:
        """One client per configured cell, in layout order, talking to its node-0 engine.

        The trainer's engine indices are positions in the configured layout, so this
        refuses to hand out a shorter list rather than silently renumbering the engines.
        """
        assert self._all_configured_cells_are_alive(), (
            f"the updatable model has {len(self.server_cells)} of {len(self.cell_specs)} cells attached; "
            f"weight updates index engines by their place in the configured layout"
        )
        return [self.server_cells[cell_id].api_client for cell_id in self.cell_specs]

    def clear_has_new_engines(self):
        self.has_new_engines = False

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count in configured-layout order, parallel to ``api_clients``."""
        return [spec.worker.num_gpus_per_engine for spec in self.cell_specs.values()]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        return [spec.gpu_offset for spec in self.cell_specs.values()]

    async def offload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.offload(tags=tags) for cell in self.server_cells.values() if cell.needs_offload]
        )

    async def onload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.onload(tags=tags) for cell in self.server_cells.values() if cell.needs_offload]
        )

    async def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        return await asyncio.gather(
            *[
                cell.check_weights(
                    action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
                )
                for cell in self.server_cells.values()
            ]
        )

    async def wait_all_engines_alive(self, timeout: float = 600):
        """Wait until every configured cell is attached and alive.

        The weight-update path indexes its engine list by the cell's position in the
        configured layout, so it must not run against a partial population.
        """
        # TODO: 600s default is hardcoded; make it configurable (e.g. via args) once we have a clearer
        # picture of init/recovery upper bounds across model sizes
        sleep_time = 2
        for _ in range(int(timeout // sleep_time)):
            if self._all_configured_cells_are_alive():
                return
            await asyncio.sleep(sleep_time)
            logger.info("wait_all_engines_alive looping...")
        missing = sorted(set(self.cell_specs) - {cell.cell_id for cell in self.server_cells.values() if cell.is_alive})
        raise TimeoutError(f"Timed out after {timeout}s waiting for engines to become ready ({missing=})")

    def _all_configured_cells_are_alive(self) -> bool:
        return len(self.server_cells) == len(self.cell_specs) and all(
            cell.is_alive for cell in self.server_cells.values()
        )

    @property
    def router_api_client(self) -> SGLangRouterApiClient:
        return SGLangRouterApiClient(router_url=f"http://{self.router_ip}:{self.router_port}")


def list_cell_ids(servers: dict[str, "RolloutServer"]) -> list[str]:
    return [cell_id for model_id in sorted(servers) for cell_id in servers[model_id].cell_specs]


_UNREGISTER_TIMEOUT_SECONDS = 30

_ATTACH_STEP_TIMEOUT_SECONDS = 60
