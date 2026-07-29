import os

from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, resolve_sglang_config
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils import dumper_utils
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

_ENGINE_SERVER_PORT = 30000
_ENGINE_NCCL_PORT = 30500
_ENGINE_INFO_BOOTSTRAP_PORT = 31000
_ENGINE_DIST_INIT_PORT = 31500
_ENGINE_DISAGGREGATION_BOOTSTRAP_PORT = 32000


def compute_inference_specs(args) -> list[ServeWorkerSpec]:
    if args.debug_train_only or args.rollout_external:
        return []

    config = resolve_sglang_config(args)
    rollout_pg_offset = _compute_rollout_pg_offset(args)
    megatron_num_gpus = _compute_megatron_num_gpus(args)

    specs: list[ServeWorkerSpec] = []
    gpu_offset = 0
    for model_cfg in config.models:
        model_cfg.resolve(args)
        for group_index, group_cfg in enumerate(model_cfg.server_groups):
            needs_offload = args.offload_rollout and rollout_pg_offset + gpu_offset < megatron_num_gpus
            spec = _spec_engine_group(
                args,
                model_cfg=model_cfg,
                group_cfg=group_cfg,
                group_index=group_index,
                needs_offload=needs_offload,
            )
            if spec is not None:
                specs.append(spec)
            gpu_offset += group_cfg.num_gpus
    return specs


def _spec_engine_group(
    args,
    *,
    model_cfg: ModelConfig,
    group_cfg: ServerGroupConfig,
    group_index: int,
    needs_offload: bool,
) -> ServeWorkerSpec | None:
    if group_cfg.worker_type == "placeholder":
        return None

    gpus_per_engine = group_cfg.num_gpus_per_engine
    gpus_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
    num_engines = group_cfg.num_gpus // gpus_per_engine_local
    nodes_per_engine = _compute_nodes_per_engine(
        num_gpus_per_engine=gpus_per_engine, num_gpus_per_node=args.num_gpus_per_node
    )
    assert num_engines % nodes_per_engine == 0, (
        f"group '{group_cfg.worker_type}' of model '{model_cfg.name}' has {num_engines=} which is not a whole "
        f"number of {nodes_per_engine=} engines"
    )

    overrides = dict(group_cfg.overrides)
    if args.offload_rollout and not needs_offload:
        overrides.setdefault("enable_memory_saver", False)

    worker_type = group_cfg.worker_type
    return ServeWorkerSpec(
        name=f"sglang-{model_cfg.name}-group{group_index}",
        port_infos=_engine_port_infos(args, worker_type=worker_type),
        env_var=lambda: _compute_engine_env_vars(args),
        scheduling=SchedulingSpec(
            num_cells=num_engines // nodes_per_engine,
            num_workers_per_cell=nodes_per_engine,
            num_gpus_per_worker=gpus_per_engine_local,
        ),
        worker_class="miles.backends.sglang_utils.sglang_engine.SGLangEngine",
        ctor_kwargs=lambda: dict(
            args=args,
            worker_type=worker_type,
            sglang_overrides=overrides,
            num_gpus_per_engine=gpus_per_engine,
        ),
    )


def _engine_port_infos(args, *, worker_type: str) -> list[PortInfo]:
    port_infos = [
        PortInfo(name="server", static_port=_ENGINE_SERVER_PORT, mode="per_worker", allow_dynamic=True),
        PortInfo(name="nccl", static_port=_ENGINE_NCCL_PORT, mode="per_worker", allow_dynamic=True),
        PortInfo(
            name="engine_info_bootstrap",
            static_port=_ENGINE_INFO_BOOTSTRAP_PORT,
            mode="per_worker",
            allow_dynamic=True,
        ),
        PortInfo(
            name="dist_init",
            static_port=_ENGINE_DIST_INIT_PORT,
            mode="master",
            allow_dynamic=True,
            num_consecutive=30 + args.sglang_dp_size,
        ),
    ]
    if worker_type == "prefill":
        port_infos.append(
            PortInfo(
                name="disaggregation_bootstrap",
                static_port=_ENGINE_DISAGGREGATION_BOOTSTRAP_PORT,
                mode="per_worker",
                allow_dynamic=True,
            )
        )
    return port_infos


def _compute_engine_env_vars(args) -> dict[str, str]:
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


def _compute_rollout_pg_offset(args) -> int:
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


def _compute_nodes_per_engine(*, num_gpus_per_engine: int, num_gpus_per_node: int) -> int:
    return max(1, num_gpus_per_engine // num_gpus_per_node)
