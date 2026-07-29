from dataclasses import dataclass, field

import ray

from miles.utils.workers.worker_spec import BaseWorkerSpec


@dataclass(frozen=True)
class CellLaunch:
    spec: BaseWorkerSpec
    cell_id: str
    cell_index: int
    generation: int


@dataclass
class ActorState:
    name: str
    spec: BaseWorkerSpec
    cell_id: str
    generation: int
    actor: ray.actor.ActorHandle
    node_ip: str = ""
    owned_ports: dict[str, int] = field(default_factory=dict)
    url: str | None = None
