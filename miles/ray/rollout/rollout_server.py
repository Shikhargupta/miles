import asyncio
import dataclasses
import logging
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.router_manager import start_router
from miles.ray.rollout.server_cell import ServerCell
from miles.ray.specs.inference import InferenceDeployment
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

logger = logging.getLogger(__name__)


async def start_rollout_servers(
    args,
    *,
    deployments: list[InferenceDeployment],
    provider: BaseWorkerProvider | None,
    worker_cell_control: Any,
) -> dict[str, "RolloutServer"]:
    """Start rollout servers: one per model, each with its own router.

    Returns a dict mapping model name -> ``RolloutServer``.
    """
    if args.rollout_external:
        raise NotImplementedError("external rollout is being rebuilt on top of worker providers")

    model_names: list[str] = []
    for deployment in deployments:
        if deployment.model_name not in model_names:
            model_names.append(deployment.model_name)

    servers: dict[str, RolloutServer] = {}
    for model_index, model_name in enumerate(model_names):
        model_deployments = [d for d in deployments if d.model_name == model_name]
        has_pd = any(d.worker_type in ("prefill", "decode") for d in model_deployments)
        router_ip, router_port = start_router(args, has_pd_disaggregation=has_pd, force_new=(model_index > 0))

        if model_index == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_cells = build_server_cells(
            args, deployments=model_deployments, provider=provider, worker_cell_control=worker_cell_control
        )

        servers[model_name] = RolloutServer(
            server_cells=server_cells,
            args=args,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_name,
            update_weights=model_deployments[0].update_weights,
        )

    await asyncio.gather(*[srv.start_all_cells() for srv in servers.values()])

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


def build_server_cells(
    args,
    *,
    deployments: list[InferenceDeployment],
    provider: BaseWorkerProvider | None,
    worker_cell_control: Any,
) -> dict[str, ServerCell]:
    server_cells: dict[str, ServerCell] = {}
    for deployment in deployments:
        for cell_index in range(deployment.spec.scheduling.num_cells):
            cell_id = compute_cell_id(spec_name=deployment.spec.name, cell_index=cell_index)
            server_cells[cell_id] = ServerCell(
                args=args,
                worker_type=deployment.worker_type,
                cell_id=cell_id,
                num_nodes=deployment.nodes_per_engine,
                num_gpus_per_engine=deployment.num_gpus_per_engine,
                gpu_offset=(
                    deployment.group_gpu_offset
                    + cell_index * deployment.nodes_per_engine * deployment.num_gpus_per_engine_local
                ),
                needs_offload=deployment.needs_offload,
                model_path=deployment.model_path,
                update_weights=deployment.update_weights,
                spec_name=deployment.spec.name,
                cell_index=cell_index,
                provider=provider,
                worker_cell_control=worker_cell_control,
            )
    return server_cells


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, as a dict of cell id -> cell.

    Each RolloutServer represents one model deployed behind a single router.
    """

    server_cells: dict[str, ServerCell]
    args: Any = None
    # NOTE: this may have risk when recovering engines parallelly; may use source of truth (cells) later
    has_new_engines: bool = False
    router_ip: str | None = None
    router_port: int | None = None
    model_name: str = "default"
    update_weights: bool = True

    @property
    def api_clients(self) -> list[SGLangApiClient]:
        """One client per allocated cell, talking to its primary (node-0) engine."""
        return [cell.api_client for cell in self._allocated_cells_of()]

    def clear_has_new_engines(self):
        self.has_new_engines = False

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count for allocated node-0 engines, parallel to ``api_clients``."""
        return [cell.num_gpus_per_engine for cell in self._allocated_cells_of()]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        return [cell.gpu_offset for cell in self._allocated_cells_of()]

    async def promote_weight_synced_cells(self) -> None:
        for cell in self._allocated_cells_of():
            if not cell.is_alive:
                await cell.promote_to_alive(self._router_api_client)

    async def start_all_cells(self):
        cell_ids = [cell_id for cell_id, cell in self.server_cells.items() if not cell.is_allocated]
        await asyncio.gather(*[self.server_cells[cell_id].start(self._router_api_client) for cell_id in cell_ids])
        self.has_new_engines |= bool(cell_ids)

    async def recover(self, cell_ids: list[str] | None = None):
        """Recover dead cells, overlapping init across cells.

        Recovered cells of an updatable server stay out of the router until the
        next weight update promotes them; a frozen server's cells serve right away."""
        if cell_ids is None:
            cell_ids = list(self.server_cells)
        cell_ids = [cell_id for cell_id in cell_ids if not self.server_cells[cell_id].is_allocated]

        await asyncio.gather(*[self.server_cells[cell_id].recover_unsynced() for cell_id in cell_ids])
        if self.update_weights:
            self.has_new_engines |= bool(cell_ids)
        else:
            await self.promote_weight_synced_cells()

        logger.info(f"Recovered {len(cell_ids)} dead rollout cells")

    async def stop_cells(self, cell_ids: list[str]):
        logger.info(f"Killing server {cell_ids=}...")
        stopped_any = False
        for cell_id in sorted(set(cell_ids)):
            cell = self.server_cells[cell_id]
            stopped_any |= cell.is_allocated
            await cell.stop(self._router_api_client)
        if self.update_weights and stopped_any:
            self.has_new_engines = True

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

    def _allocated_cells_of(self, cell_ids: list[str] | None = None) -> list[ServerCell]:
        if cell_ids is None:
            cell_ids = list(self.server_cells)
        return [self.server_cells[cell_id] for cell_id in cell_ids if self.server_cells[cell_id].is_allocated]

    @property
    def _router_api_client(self) -> SGLangRouterApiClient:
        return SGLangRouterApiClient(router_url=f"http://{self.router_ip}:{self.router_port}")


def list_cell_ids(servers: dict[str, "RolloutServer"]) -> list[str]:
    return [cell_id for model_id in sorted(servers) for cell_id in servers[model_id].server_cells]
