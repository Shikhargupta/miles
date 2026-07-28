import asyncio
import dataclasses
import logging
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, SglangConfig
from miles.backends.sglang_utils.sglang_router_api_client import SGLangRouterApiClient
from miles.ray.rollout.addr_allocator import PortAllocator
from miles.ray.rollout.router_manager import start_router
from miles.ray.rollout.server_cell import ServerCell, compute_nodes_per_engine
from miles.utils import async_utils

logger = logging.getLogger(__name__)


def start_rollout_servers(args, pg) -> dict[str, "RolloutServer"]:
    """Start rollout servers: one per model, each with its own router.

    Returns a dict mapping model name -> ``RolloutServer``.
    """
    config = _resolve_sglang_config(args)

    servers: dict[str, RolloutServer] = {}
    gpu_offset = 0
    engine_offset = 0

    rollout_pg_offset = _compute_rollout_offset(args)
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    for model_idx, model_cfg in enumerate(config.models):
        model_cfg.resolve(args)

        has_pd = model_cfg.has_pd_disaggregation
        router_ip, router_port = start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))

        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port

        server_cells: dict[str, ServerCell] = {}
        port_allocator = PortAllocator()

        for group_cfg in model_cfg.server_groups:
            gpus_per_engine = group_cfg.num_gpus_per_engine
            num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
            num_engines = group_cfg.num_gpus // num_gpu_per_engine_local
            nodes_per_engine = compute_nodes_per_engine(
                num_gpus_per_engine=gpus_per_engine, num_gpus_per_node=args.num_gpus_per_node
            )

            group_abs_start = rollout_pg_offset + gpu_offset
            needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
            overrides = dict(group_cfg.overrides)
            if args.offload_rollout and not needs_offload:
                overrides.setdefault("enable_memory_saver", False)
            logger.info(
                f"Engine group '{group_cfg.worker_type}' gpu_offset={gpu_offset} "
                f"(abs={group_abs_start}): needs_offload={needs_offload}"
            )

            if group_cfg.worker_type != "placeholder":
                for cell_start in range(0, num_engines, nodes_per_engine):
                    server_cells[f"idx-{len(server_cells)}"] = ServerCell(
                        num_nodes=nodes_per_engine,
                        args=args,
                        worker_type=group_cfg.worker_type,
                        pg=pg,
                        num_gpus_per_engine=gpus_per_engine,
                        rank_offset=engine_offset + cell_start,
                        gpu_offset=gpu_offset + cell_start * num_gpu_per_engine_local,
                        sglang_overrides=overrides,
                        needs_offload=needs_offload,
                        model_path=overrides.get("model_path", args.hf_checkpoint),
                        update_weights=model_cfg.update_weights,
                    )

            engine_offset += num_engines
            gpu_offset += group_cfg.num_gpus

        srv = RolloutServer(
            server_cells=server_cells,
            args=args,
            router_ip=router_ip,
            router_port=router_port,
            model_name=model_cfg.name,
            update_weights=model_cfg.update_weights,
        )
        async_utils.run(srv.start_all_cells(port_allocator))
        servers[model_cfg.name] = srv

    args.sglang_model_routers = {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()}

    return servers


def _resolve_sglang_config(args) -> SglangConfig:
    """Build a SglangConfig from args, choosing the right source."""
    if getattr(args, "sglang_config", None) is not None:
        config = SglangConfig.from_yaml(args.sglang_config)
        expected = args.rollout_num_gpus
        actual = config.total_num_gpus
        assert actual == expected, f"sglang_config total GPUs ({actual}) != rollout_num_gpus ({expected})"
        return config

    if args.prefill_num_servers is not None:
        return SglangConfig.from_prefill_num_servers(args)

    return SglangConfig(
        models=[
            ModelConfig(
                name="default",
                server_groups=[ServerGroupConfig(worker_type="regular", num_gpus=args.rollout_num_gpus)],
            )
        ]
    )


def _compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    if getattr(args, "critic_train_only", False):
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    if getattr(args, "use_critic", False):
        offset += args.critic_num_nodes * args.critic_num_gpus_per_node
    return offset


def _compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if getattr(args, "debug_rollout_only", False):
        return 0
    if getattr(args, "critic_train_only", False):
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    if getattr(args, "use_critic", False):
        num += args.critic_num_nodes * args.critic_num_gpus_per_node
    return num


@dataclasses.dataclass
class RolloutServer:
    """A model served behind a shared router, as a list of engine cells.

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

    @property
    def nodes_per_engine(self):
        values = {cell.num_nodes for cell in self.server_cells.values()}
        if len(values) != 1:
            raise ValueError(f"Heterogeneous nodes_per_engine across cells: {values}")
        return values.pop()

    async def start_all_cells(self, port_allocator: PortAllocator):
        if self.args.debug_train_only:
            return

        cell_ids = [cell_id for cell_id, cell in self.server_cells.items() if not cell.is_allocated]
        await asyncio.gather(
            *[self.server_cells[cell_id].start(port_allocator, self._router_api_client) for cell_id in cell_ids]
        )
        self.has_new_engines |= bool(cell_ids)

    async def recover(self, cell_ids: list[str] | None = None):
        """Recover dead cells, overlapping init across cells."""
        port_allocator = PortAllocator()
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


@dataclasses.dataclass(frozen=True)
class ParsedId:
    """A cell's address across all models: which model serves it, and which of
    that model's cells it is."""

    model_id: str
    cell_id: str

    SEPARATOR = "--"

    @staticmethod
    def parse(global_id: str) -> "ParsedId":
        model_id, separator, cell_id = global_id.partition(ParsedId.SEPARATOR)
        assert separator, f"{global_id=} must be '<model_id>{ParsedId.SEPARATOR}<cell_id>'"
        return ParsedId(model_id=model_id, cell_id=cell_id)

    def format(self) -> str:
        return f"{self.model_id}{ParsedId.SEPARATOR}{self.cell_id}"


def list_global_cell_ids(servers: dict[str, RolloutServer]) -> list[str]:
    """Every cell across every model, sorted by model id then by cell order."""
    return [
        ParsedId(model_id=model_id, cell_id=cell_id).format()
        for model_id in sorted(servers)
        for cell_id in servers[model_id].server_cells
    ]
