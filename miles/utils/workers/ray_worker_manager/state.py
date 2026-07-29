from dataclasses import dataclass, field

import ray

from miles.utils.workers.worker_spec import BaseWorkerSpec


@dataclass
class CellState:
    spec: BaseWorkerSpec
    cell_id: str
    cell_index: int
    generation: int


@dataclass
class WorkerState:
    name: str
    cell: CellState
    actor: ray.actor.ActorHandle
    node_ip: str = ""
    owned_ports: dict[str, int] = field(default_factory=dict)
    url: str | None = None
