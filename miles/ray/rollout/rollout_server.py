import asyncio
import dataclasses
import logging
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.router_manager import start_router
from miles.ray.rollout.server_cell import ServerCell
from miles.ray.specs.inference import compute_megatron_num_gpus, compute_rollout_offset, setup_engine_group

logger = logging.getLogger(__name__)


async def start_rollout_servers(args, pg) -> dict[str, "RolloutServer"]:
    """Start rollout servers: one per model, each with its own router.

    Returns a dict mapping model name -> ``RolloutServer``.
    """
    config = resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    gpu_offset = 0
    engine_offset = 0
    port_allocator = PortAllocator()

    rollout_pg_offset = compute_rollout_offset(args)
    megatron_num_gpus = compute_megatron_num_gpus(args)

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_cells: dict[str, ServerCell] = {}

        for group_cfg in model_cfg.server_groups:
            setup = setup_engine_group(
                args,
                model_cfg=model_cfg,
                group_cfg=group_cfg,
                pg=pg,
                cell_index_offset=len(server_cells),
                engine_offset=engine_offset,
                gpu_offset=gpu_offset,
                rollout_pg_offset=rollout_pg_offset,
                megatron_num_gpus=megatron_num_gpus,
            )
            server_cells.update(setup.server_cells)
            engine_offset = setup.engine_offset
            gpu_offset = setup.gpu_offset

        servers[model_cfg.name] = RolloutServer(
            server_cells=server_cells,
            args=args,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )

    await asyncio.gather(*[srv.start_all_cells(port_allocator) for srv in servers.values()])

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


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
    _port_allocator: PortAllocator = dataclasses.field(default_factory=PortAllocator)

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

    async def start_all_cells(self, port_allocator: PortAllocator):
        if self.args.debug_train_only:
            return

        self._port_allocator = port_allocator
        cell_ids = [cell_id for cell_id, cell in self.server_cells.items() if not cell.is_allocated]
        await asyncio.gather(
            *[self.server_cells[cell_id].start(port_allocator, self._router_api_client) for cell_id in cell_ids]
        )
        self.has_new_engines |= bool(cell_ids)

    async def recover(self, cell_ids: list[str] | None = None):
        """Recover dead cells, overlapping init across cells.

        Reuses the startup allocator so its per-node cursors still sit past the
        ports the live engines hold, instead of rescanning from the base port.
        """
        port_allocator = self._port_allocator
        if cell_ids is None:
            cell_ids = list(self.server_cells)
        cell_ids = [cell_id for cell_id in cell_ids if not self.server_cells[cell_id].is_allocated]

        await asyncio.gather(
            *[
                self.server_cells[cell_id].start(port_allocator, self._router_api_client, recover=True)
                for cell_id in cell_ids
            ]
        )
        self.has_new_engines |= bool(cell_ids)

        logger.info(f"Recovered {len(cell_ids)} dead rollout cells")

    async def stop_cells(self, cell_ids: list[str]):
        logger.info(f"Killing server {cell_ids=}...")
        for cell_id in sorted(set(cell_ids)):
            await self.server_cells[cell_id].stop(self._router_api_client)

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
    def _router_api_client(self) -> SGLangRouterApiClient:
        return SGLangRouterApiClient(router_url=f"http://{self.router_ip}:{self.router_port}")


def format_cell_id(*, server_id: str, index: int) -> str:
    return f"{server_id}-{index}"


def list_cell_ids(servers: dict[str, "RolloutServer"]) -> list[str]:
    return [cell_id for model_id in sorted(servers) for cell_id in servers[model_id].server_cells]
