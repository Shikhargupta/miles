from collections.abc import Awaitable, Callable
from typing import Any, Literal

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class PortInfo(FrozenStrictBaseModel):
    name: str
    static_port: int
    mode: Literal["per_worker", "master"]
    allow_dynamic: bool
    num_consecutive: int = 1


class SchedulingSpec(FrozenStrictBaseModel):
    num_cells: int
    num_workers_per_cell: int
    num_gpus_per_worker: float


class RayActorOptions(FrozenStrictBaseModel):
    num_cpus: float
    num_gpus: float


class WorkerPlacement(FrozenStrictBaseModel):
    local_index: int
    global_rank: int
    base_gpu_id: int


class CellAddressing(FrozenStrictBaseModel):
    node_ips: list[str]
    master_ports: dict[str, int]
    per_worker_ports: list[dict[str, int]]


class BaseWorkerSpec(FrozenStrictBaseModel):
    name: str
    port_infos: list[PortInfo]
    env_var: Callable[[WorkerPlacement], dict[str, str]]
    scheduling: SchedulingSpec
    ray_options: RayActorOptions
    prepare_workers: Callable[[list[WorkerPlacement], list[Any]], Awaitable[None]] | None = None


class WorkerLaunchPlan(FrozenStrictBaseModel):
    cmd: str
    envs: dict[str, str] = {}


class CommandWorkerSpec(BaseWorkerSpec):
    build_launch_plan: Callable[[WorkerPlacement, CellAddressing], WorkerLaunchPlan]
    build_member_payloads: Callable[[CellAddressing], list[dict[str, Any]]]
    wait_cell_ready: Callable[[CellAddressing, Callable[[], bool]], Awaitable[None]]


class ServeWorkerSpec(BaseWorkerSpec):
    worker_class: str
    ctor_kwargs: Callable[[WorkerPlacement], dict[str, Any]]
    build_init_payloads: Callable[[CellAddressing], list[dict[str, Any]]]


class BaseCellSpec(FrozenStrictBaseModel):
    worker: BaseWorkerSpec
    cell_id: str
    rank_offset: int
    gpu_offset: int
