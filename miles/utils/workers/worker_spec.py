from collections.abc import Callable
from typing import Any, Literal

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class PortInfo(FrozenStrictBaseModel):
    name: str
    static_port: int
    mode: Literal["per_worker", "master"]
    allow_dynamic: bool


class SchedulingSpec(FrozenStrictBaseModel):
    num_cells: int
    num_workers_per_cell: int
    num_gpus_per_worker: float


class BaseWorkerSpec(FrozenStrictBaseModel):
    name: str
    port_infos: list[PortInfo]
    env_var: Callable[[], dict[str, str]]
    scheduling: SchedulingSpec


class CommandWorkerSpec(BaseWorkerSpec):
    launch_command: str


class ServeWorkerSpec(BaseWorkerSpec):
    worker_class: str
    ctor_kwargs: Callable[[], dict[str, Any]]
