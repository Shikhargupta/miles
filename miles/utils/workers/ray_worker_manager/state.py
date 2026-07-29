from dataclasses import dataclass, field

import ray

from miles.utils.workers.worker_spec import BaseWorkerSpec


@dataclass
class _CellState:
    spec: BaseWorkerSpec
    cell_id: str
    cell_index: int
    generation: int


@dataclass
class _WorkerState:
    name: str
    cell: _CellState
    actor: ray.actor.ActorHandle
    node_ip: str = ""
    owned_ports: dict[str, int] = field(default_factory=dict)
    url: str | None = None
