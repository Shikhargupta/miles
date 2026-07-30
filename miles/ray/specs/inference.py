import dataclasses
import logging
import os
from typing import TYPE_CHECKING

from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils import dumper_utils

if TYPE_CHECKING:
    from miles.ray.rollout.server_cell import ServerCell

logger = logging.getLogger(__name__)


def compute_engine_env_vars(args) -> dict[str, str]:
    env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
        key: os.environ.get(key, default_val)
        for key, default_val in {
            # DeepEP/NVSHMEM's internal NCCL conflicts with our NCCL and hangs under CUDA graphs.
            "NVSHMEM_DISABLE_NCCL": "1",
            "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
            "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
            "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
            "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2": (
                "0" if args.colocate and args.rollout_num_gpus_per_engine > 1 else "1"
            ),
            "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
        }.items()
    }
    env_vars.update(dumper_utils.get_sglang_env(args))
    return env_vars


@dataclasses.dataclass(frozen=True)
class EngineGroupSetup:
    server_cells: dict[str, "ServerCell"]
    engine_offset: int
    gpu_offset: int


def setup_engine_group(
    args,
    *,
    model_cfg: ModelConfig,
    group_cfg: ServerGroupConfig,
    pg,
    cell_index_offset: int,
    engine_offset: int,
    gpu_offset: int,
    rollout_pg_offset: int,
    megatron_num_gpus: int,
) -> EngineGroupSetup:
    # Imported here because server_cell reads this module's env vars, so importing it at
    # module level would close a cycle.
    from miles.ray.rollout.rollout_server import format_cell_id
    from miles.ray.rollout.server_cell import ServerCell

    server_cells: dict[str, ServerCell] = {}

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
        assert num_engines % nodes_per_engine == 0, (
            f"group '{group_cfg.worker_type}' has {num_engines=} which is not a whole number of "
            f"{nodes_per_engine=} engines; the trailing engine would have no node to run its remaining ranks"
        )
        assert engine_offset % nodes_per_engine == 0, (
            f"group '{group_cfg.worker_type}' starts at {engine_offset=}, which is not aligned to "
            f"{nodes_per_engine=}: sglang derives each engine's node_rank from its global rank, so a "
            f"misaligned start would make the cell's primary a worker node"
        )

        for cell_start in range(0, num_engines, nodes_per_engine):
            cell_id = format_cell_id(server_id=model_cfg.name, index=cell_index_offset + len(server_cells))
            server_cells[cell_id] = ServerCell(
                num_nodes=nodes_per_engine,
                args=args,
                worker_type=group_cfg.worker_type,
                cell_id=cell_id,
                pg=pg,
                num_gpus_per_engine=gpus_per_engine,
                rank_offset=engine_offset + cell_start,
                gpu_offset=gpu_offset + cell_start * num_gpu_per_engine_local,
                sglang_overrides=overrides,
                needs_offload=needs_offload,
                model_path=overrides.get("model_path", args.hf_checkpoint),
                update_weights=model_cfg.update_weights,
            )

    return EngineGroupSetup(
        server_cells=server_cells,
        engine_offset=engine_offset + num_engines,
        gpu_offset=gpu_offset + group_cfg.num_gpus,
    )


def compute_rollout_offset(args) -> int:
    """Offset (in PG bundle slots) where rollout GPUs start."""
    if args.debug_train_only or args.debug_rollout_only or args.colocate:
        return 0
    if getattr(args, "critic_train_only", False):
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    offset = args.actor_num_nodes * args.actor_num_gpus_per_node
    if getattr(args, "use_critic", False):
        offset += args.critic_num_nodes * args.critic_num_gpus_per_node
    return offset


def compute_megatron_num_gpus(args) -> int:
    """Total number of megatron (actor + critic) GPU slots in the placement group."""
    if getattr(args, "debug_rollout_only", False):
        return 0
    if getattr(args, "critic_train_only", False):
        return args.critic_num_nodes * args.critic_num_gpus_per_node
    num = args.actor_num_nodes * args.actor_num_gpus_per_node
    if getattr(args, "use_critic", False):
        num += args.critic_num_nodes * args.critic_num_gpus_per_node
    return num


def compute_nodes_per_engine(*, num_gpus_per_engine: int, num_gpus_per_node: int) -> int:
    return max(1, num_gpus_per_engine // num_gpus_per_node)
