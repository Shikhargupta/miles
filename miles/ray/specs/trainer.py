import os

from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils.environ import default_fp8_block_scaling_fp32_scales
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

_TRAINER_MASTER_PORT = 29500

TRAINER_RAY_NUM_GPUS_PER_WORKER = 0.4
TRAINER_RAY_NUM_CPUS_PER_WORKER = 0.4
TRAINER_FT_CONCURRENCY_GROUPS = {"heartbeat_status": 1, "default": 1, "fault_injector": 1}


def compute_trainer_specs(args) -> list[ServeWorkerSpec]:
    if args.debug_rollout_only:
        return []

    specs = [spec_trainer_ranks(args, role="actor")]
    if args.use_critic:
        specs.append(spec_trainer_ranks(args, role="critic"))
    return specs


def spec_trainer_ranks(args, *, role: str) -> ServeWorkerSpec:
    if role == "actor":
        total_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    elif role == "critic":
        total_gpus = args.critic_num_nodes * args.critic_num_gpus_per_node
    else:
        raise ValueError(f"Unknown trainer role: {role}")

    num_cells = (total_gpus // compute_megatron_world_size_except_dp(args)) if args.indep_dp else 1
    assert num_cells > 0 and total_gpus % num_cells == 0, f"{total_gpus=} must split evenly into {num_cells=}"
    num_workers_per_cell = total_gpus // num_cells

    return ServeWorkerSpec(
        name=f"train-{role}",
        port_infos=[PortInfo(name="master", static_port=_TRAINER_MASTER_PORT, mode="master", allow_dynamic=True)],
        env_var=lambda: _compute_trainer_env_vars(args),
        scheduling=SchedulingSpec(
            num_cells=num_cells,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=1,
            num_cpus_per_worker=1,
        ),
        worker_class=_compute_trainer_worker_class(args),
        ctor_kwargs=lambda cell_index, worker_index: dict(
            args=args,
            world_size=num_workers_per_cell,
            rank=worker_index,
            master_addr=None,
            master_port=None,
            indep_dp_store_addr=None,
            role=role,
            cell_index=cell_index,
        ),
    )


def _compute_trainer_env_vars(args) -> dict[str, str]:
    env_vars = {
        # because sglang will always set NCCL_CUMEM_ENABLE to 0
        # we need also set it to 0 to prevent nccl error.
        "NCCL_CUMEM_ENABLE": os.environ.get("NCCL_CUMEM_ENABLE", "0"),
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": os.environ.get(
            "NVTE_FP8_BLOCK_SCALING_FP32_SCALES", default_fp8_block_scaling_fp32_scales()
        ),
        # DeepEP/NVSHMEM's internal NCCL conflicts with our NCCL and hangs under CUDA graphs.
        "NVSHMEM_DISABLE_NCCL": os.environ.get("NVSHMEM_DISABLE_NCCL", "1"),
        **{name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST},
        **args.train_env_vars,
    }

    if source_patcher_config := args.dumper_source_patcher_config_train:
        env_vars["DUMPER_SOURCE_PATCHER_CONFIG"] = source_patcher_config

    if args.offload_train and args.train_backend == "megatron":
        from torch_memory_saver.utils import get_binary_path_from_package

        dynlib_path = str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))

        env_vars["LD_PRELOAD"] = dynlib_path
        env_vars["TMS_INIT_ENABLE"] = "1"
        env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"

    return env_vars


def _compute_trainer_worker_class(args) -> str:
    if args.train_backend == "megatron":
        if args.use_fault_tolerance:
            return "miles.ray.train.ft_actor_classes.MegatronTrainRayActorFt"
        return "miles.backends.megatron_utils.actor.MegatronTrainRayActor"
    if args.use_fault_tolerance:
        return "miles.ray.train.ft_actor_classes.FSDPTrainRayActorFt"
    return "miles.backends.experimental.fsdp_utils.actor.FSDPTrainRayActor"
