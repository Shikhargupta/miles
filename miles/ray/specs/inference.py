import asyncio
import dataclasses
import functools
import logging
import os
from collections.abc import Callable
from typing import Any, Literal

from pydantic import model_validator

from miles.backends.sglang_utils.sglang_api_client import wait_server_healthy
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, resolve_sglang_config
from miles.backends.sglang_utils.sglang_engine import build_server_url, compute_engine_launch_plan, format_v6_uri
from miles.ray.utils import NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
from miles.utils import dumper_utils
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.worker_spec import (
    BaseCellSpec,
    CellAddressing,
    CommandWorkerSpec,
    PortInfo,
    RayActorOptions,
    SchedulingSpec,
    WorkerLaunchPlan,
    WorkerPlacement,
)

logger = logging.getLogger(__name__)


ENGINE_RAY_OPTIONS = RayActorOptions(num_cpus=0.2, num_gpus=0.2)


def compute_engine_env_vars(args, placement: WorkerPlacement, *, worker_type: str) -> dict[str, str]:
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
    # TODO: this is hacky. Use env var SGLANG_DG_CACHE_DIR_PER_PROCESS=1
    # to enable this isolation.
    env_vars["SGLANG_DG_CACHE_DIR"] = os.environ.get(
        "SGLANG_DG_CACHE_DIR", f"/tmp/sglang_deep_gemm/{worker_type}_rank_{placement.global_rank}"
    )
    return env_vars


ENGINE_SERVER_PORT_NAME = "server"
ENGINE_NCCL_PORT_NAME = "nccl"
ENGINE_INFO_BOOTSTRAP_PORT_NAME = "engine_info_bootstrap"
ENGINE_DIST_INIT_PORT_NAME = "dist_init"
ENGINE_DISAGGREGATION_BOOTSTRAP_PORT_NAME = "disaggregation_bootstrap"


class InferenceWorkerSpec(CommandWorkerSpec):
    worker_type: Literal["regular", "prefill", "decode"]
    sglang_overrides: dict[str, Any]
    needs_offload: bool
    model_path: str | None

    @model_validator(mode="after")
    def _reject_addressing_overrides(self) -> "InferenceWorkerSpec":
        assert not ({"host", "port"} & set(self.sglang_overrides)), (
            f"sglang_overrides must not override host/port ({self.sglang_overrides=}): the rollout process derives "
            f"each engine's url from the addr allocator, so an override would make it talk to the wrong endpoint"
        )
        return self

    @property
    def num_gpus_per_engine(self) -> int:
        num_gpus = self.scheduling.num_workers_per_cell * self.scheduling.num_gpus_per_worker
        assert num_gpus == int(num_gpus), f"{self.scheduling=} does not give a whole number of gpus per engine"
        return int(num_gpus)


class InferenceCellSpec(BaseCellSpec):
    worker: InferenceWorkerSpec


class InferenceModelSpec(FrozenStrictBaseModel):
    name: str
    update_weights: bool
    cells: list[InferenceCellSpec]

    @property
    def has_pd_disaggregation(self) -> bool:
        return any(cell.worker.worker_type in ("prefill", "decode") for cell in self.cells)


@dataclasses.dataclass(frozen=True)
class _GroupSpecs:
    cells: list[InferenceCellSpec]
    engine_offset: int
    gpu_offset: int


def compute_inference_model_specs(args) -> list[InferenceModelSpec]:
    if args.rollout_external:
        raise NotImplementedError("external rollout address allocation was removed and a new implementation is coming")

    config = resolve_sglang_config(args)

    rollout_pg_offset = compute_rollout_offset(args)
    megatron_num_gpus = compute_megatron_num_gpus(args)

    model_specs: list[InferenceModelSpec] = []
    gpu_offset = 0
    engine_offset = 0

    for model_cfg in config.models:
        model_cfg.resolve(args)

        cells: list[InferenceCellSpec] = []

        for group_index, group_cfg in enumerate(model_cfg.server_groups):
            group = _compute_specs_of_group(
                args,
                model_cfg=model_cfg,
                group_cfg=group_cfg,
                group_index=group_index,
                cell_index_offset=len(cells),
                engine_offset=engine_offset,
                gpu_offset=gpu_offset,
                rollout_pg_offset=rollout_pg_offset,
                megatron_num_gpus=megatron_num_gpus,
            )
            cells.extend(group.cells)
            engine_offset = group.engine_offset
            gpu_offset = group.gpu_offset

        model_specs.append(
            InferenceModelSpec(name=model_cfg.name, update_weights=model_cfg.update_weights, cells=cells)
        )

    return model_specs


def _compute_specs_of_group(
    args,
    *,
    model_cfg: ModelConfig,
    group_cfg: ServerGroupConfig,
    group_index: int,
    cell_index_offset: int,
    engine_offset: int,
    gpu_offset: int,
    rollout_pg_offset: int,
    megatron_num_gpus: int,
) -> _GroupSpecs:
    cells: list[InferenceCellSpec] = []

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
        assert nodes_per_engine * num_gpu_per_engine_local == gpus_per_engine, (
            f"group '{group_cfg.worker_type}' asks for {gpus_per_engine=}, which is neither within one node of "
            f"{args.num_gpus_per_node} gpus nor a whole number of nodes: its engines would be given "
            f"{nodes_per_engine * num_gpu_per_engine_local} gpus instead"
        )
        assert num_engines > 0, (
            f"group '{group_cfg.worker_type}' has {group_cfg.num_gpus=}, which is not enough for a single engine "
            f"of {gpus_per_engine} gpus"
        )
        assert num_engines % nodes_per_engine == 0, (
            f"group '{group_cfg.worker_type}' has {num_engines=} which is not a whole number of "
            f"{nodes_per_engine=} engines; the trailing engine would have no node to run its remaining ranks"
        )
        assert engine_offset % nodes_per_engine == 0, (
            f"group '{group_cfg.worker_type}' starts at {engine_offset=}, which is not aligned to "
            f"{nodes_per_engine=}: sglang derives each engine's node_rank from its global rank, so a "
            f"misaligned start would make the cell's primary a worker node"
        )

        worker = InferenceWorkerSpec(
            name=f"sglang-{model_cfg.name}-group{group_index}",
            port_infos=compute_engine_port_infos(args, worker_type=group_cfg.worker_type),
            env_var=functools.partial(compute_engine_env_vars, args, worker_type=group_cfg.worker_type),
            scheduling=SchedulingSpec(
                num_cells=num_engines // nodes_per_engine,
                num_workers_per_cell=nodes_per_engine,
                num_gpus_per_worker=num_gpu_per_engine_local,
            ),
            ray_options=ENGINE_RAY_OPTIONS,
            build_launch_plan=functools.partial(
                _build_engine_launch_plan,
                args,
                worker_type=group_cfg.worker_type,
                sglang_overrides=overrides,
                num_gpus_per_engine=gpus_per_engine,
            ),
            build_member_payloads=_build_engine_member_payloads,
            wait_cell_ready=functools.partial(_wait_engine_ready, args, sglang_overrides=overrides),
            prepare_workers=functools.partial(_prepare_engine_workers, args),
            worker_type=group_cfg.worker_type,
            sglang_overrides=overrides,
            needs_offload=needs_offload,
            model_path=overrides.get("model_path", args.hf_checkpoint),
        )

        for cell_start in range(0, num_engines, nodes_per_engine):
            cells.append(
                InferenceCellSpec(
                    worker=worker,
                    cell_id=format_cell_id(server_id=model_cfg.name, index=cell_index_offset + len(cells)),
                    rank_offset=engine_offset + cell_start,
                    gpu_offset=gpu_offset + cell_start * num_gpu_per_engine_local,
                )
            )

    return _GroupSpecs(
        cells=cells,
        engine_offset=engine_offset + num_engines,
        gpu_offset=gpu_offset + group_cfg.num_gpus,
    )


def _build_engine_launch_plan(
    args,
    placement: WorkerPlacement,
    addressing: CellAddressing,
    *,
    worker_type: str,
    sglang_overrides: dict[str, Any],
    num_gpus_per_engine: int,
) -> WorkerLaunchPlan:
    addr_and_ports = _build_engine_member_payloads(addressing)[placement.local_index]
    plan = compute_engine_launch_plan(
        args,
        rank=placement.global_rank,
        worker_type=worker_type,
        base_gpu_id=placement.base_gpu_id,
        sglang_overrides=sglang_overrides,
        num_gpus_per_engine=num_gpus_per_engine,
        addr_and_ports=addr_and_ports,
    )
    return WorkerLaunchPlan(cmd=plan.cmd, envs={})


async def _prepare_engine_workers(args, placements: list[WorkerPlacement], actor_handles: list[Any]) -> None:
    if not (env_report := args.env_report):
        return

    await asyncio.gather(
        *[
            actor._collect_env_report.remote(role="rollout", rank=placement.global_rank, partial_env_report=env_report)
            for placement, actor in zip(placements, actor_handles, strict=True)
        ]
    )


async def _wait_engine_ready(
    args,
    addressing: CellAddressing,
    is_worker_alive: Callable[[], bool],
    *,
    sglang_overrides: dict[str, Any],
) -> None:
    primary = _build_engine_member_payloads(addressing)[0]
    default_api_key = args.sglang_api_key if hasattr(args, "sglang_api_key") else None
    await wait_server_healthy(
        server_url=build_server_url(host=primary["host"], port=primary["port"]),
        api_key=sglang_overrides.get("api_key", default_api_key),
        is_process_alive=is_worker_alive,
    )


def _build_engine_member_payloads(addressing: CellAddressing) -> list[dict[str, Any]]:
    assert set(addressing.master_ports) == {ENGINE_DIST_INIT_PORT_NAME}, f"{addressing.master_ports=}"
    dist_init_addr = f"{format_v6_uri(addressing.node_ips[0])}:{addressing.master_ports[ENGINE_DIST_INIT_PORT_NAME]}"

    payloads: list[dict[str, Any]] = []
    for node_ip, ports in zip(addressing.node_ips, addressing.per_worker_ports, strict=True):
        payload = dict(
            host=format_v6_uri(node_ip),
            port=ports[ENGINE_SERVER_PORT_NAME],
            nccl_port=ports[ENGINE_NCCL_PORT_NAME],
            engine_info_bootstrap_port=ports[ENGINE_INFO_BOOTSTRAP_PORT_NAME],
            dist_init_addr=dist_init_addr,
        )
        if ENGINE_DISAGGREGATION_BOOTSTRAP_PORT_NAME in ports:
            payload["disaggregation_bootstrap_port"] = ports[ENGINE_DISAGGREGATION_BOOTSTRAP_PORT_NAME]
        payloads.append(payload)
    return payloads


def compute_engine_port_infos(args, *, worker_type: str) -> list[PortInfo]:
    port_infos = [
        PortInfo(name=ENGINE_SERVER_PORT_NAME, static_port=30000, mode="per_worker", allow_dynamic=True),
        PortInfo(name=ENGINE_NCCL_PORT_NAME, static_port=30500, mode="per_worker", allow_dynamic=True),
        PortInfo(name=ENGINE_INFO_BOOTSTRAP_PORT_NAME, static_port=31000, mode="per_worker", allow_dynamic=True),
        PortInfo(
            name=ENGINE_DIST_INIT_PORT_NAME,
            static_port=31500,
            mode="master",
            allow_dynamic=True,
            num_consecutive=30 + args.sglang_dp_size,
        ),
    ]
    if worker_type == "prefill":
        port_infos.append(
            PortInfo(
                name=ENGINE_DISAGGREGATION_BOOTSTRAP_PORT_NAME,
                static_port=32000,
                mode="per_worker",
                allow_dynamic=True,
            )
        )
    return port_infos


def format_cell_id(*, server_id: str, index: int) -> str:
    return f"{server_id}-{index}"


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
