import asyncio
import dataclasses
import logging
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.router_manager import wait_router_ready
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock, enforce_lock_discipline, lock_exempt, requires_lock

logger = logging.getLogger(__name__)


async def create_rollout_servers(args, context_lock: ContextLock) -> dict[str, "RolloutServer"]:
    """Create rollout servers: one per model, each with its own router."""
    assert args.sglang_router_ip is None and args.sglang_router_port is None, (
        "external router mode was removed: miles always starts its own routers "
        "(expected to return with the k8s-native mode)"
    )

    config = resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}

    for model_idx, model_cfg in enumerate(config.models):
        router_addr = await wait_router_ready(model_idx=model_idx)

        if model_idx == 0:
            args.sglang_router_ip = router_addr.host
            args.sglang_router_port = router_addr.port

        servers[model_cfg.name] = RolloutServer(
            server_cells={},
            args=args,
            context_lock=context_lock,
            router_ip=router_addr.host,
            router_port=router_addr.port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
            expected_num_cells=sum(
                group_cfg.num_gpus // group_cfg.num_gpus_per_engine
                for group_cfg in model_cfg.server_groups
                if group_cfg.worker_type != "placeholder"
            ),
        )

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


@dataclasses.dataclass
@enforce_lock_discipline
class RolloutServer:
    """A model served behind a shared router, as a dict of cell id -> cell.

    Each RolloutServer represents one model deployed behind a single router.
    """

    server_cells: dict[str, ServerCell]
    args: Any
    context_lock: ContextLock
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True
    expected_num_cells: int = 0

    @property
    @requires_lock
    def api_clients(self) -> list[SGLangApiClient]:
        """One client per cell, talking to its primary (node-0) engine."""
        return [cell.api_client for cell in self._cells_by_gpu_offset()]

    @property
    @requires_lock
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for all node-0 engines, parallel to ``engines``."""
        return [cell.meta.num_gpus_per_engine for cell in self._cells_by_gpu_offset()]

    @property
    @requires_lock
    def engine_gpu_offsets(self) -> list[int]:
        return [cell.meta.gpu_offset for cell in self._cells_by_gpu_offset()]

    @requires_lock
    def _cells_by_gpu_offset(self) -> list[ServerCell]:
        return sorted(self.server_cells.values(), key=lambda cell: cell.meta.gpu_offset)

    @requires_lock
    async def add_cell(self, cell_meta: ServerCellMetadata):
        cell_id = cell_meta.cell_id
        assert cell_id not in self.server_cells
        cell = ServerCell(args=self.args, router_api_client=self._router_api_client, meta=cell_meta)
        await cell.add()
        self.server_cells[cell_id] = cell

    @requires_lock
    async def remove_cell(self, cell_id: str):
        logger.info(f"Killing server {cell_id=}...")
        await self.server_cells[cell_id].dispose()
        del self.server_cells[cell_id]

    @requires_lock
    async def offload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.offload(tags=tags) for cell in self.server_cells.values() if cell.meta.needs_offload]
        )

    @requires_lock
    async def onload(self, tags: list[str] | None = None):
        return await asyncio.gather(
            *[cell.onload(tags=tags) for cell in self.server_cells.values() if cell.meta.needs_offload]
        )

    @requires_lock
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

    @lock_exempt
    async def wait_expected_num_cells(self, timeout: float = 3600):
        sleep_time = 2
        for _ in range(int(timeout // sleep_time)):
            if len(self.server_cells) == self.expected_num_cells:
                return
            await asyncio.sleep(sleep_time)
            logger.info(
                f"wait_expected_num_cells looping ({len(self.server_cells)}/{self.expected_num_cells} cells)..."
            )
        raise TimeoutError(f"Timed out after {timeout}s waiting for {self.expected_num_cells} cells to appear")

    @requires_lock
    async def wait_all_engines_alive(self, timeout: float = 600):
        # TODO: 600s default is hardcoded; make it configurable (e.g. via args) once we have a clearer
        # picture of init/recovery upper bounds across model sizes
        sleep_time = 2
        for _ in range(int(timeout // sleep_time)):
            if all(cell.is_pending_weights_or_serving for cell in self.server_cells.values()):
                return
            await asyncio.sleep(sleep_time)
            logger.info("wait_all_engines_alive looping...")
        raise TimeoutError(f"Timed out after {timeout}s waiting for engines to become ready")

    @property
    @requires_lock
    def _router_api_client(self) -> SGLangRouterApiClient:
        return SGLangRouterApiClient(router_url=f"http://{self.router_ip}:{self.router_port}")
