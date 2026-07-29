from dataclasses import dataclass

import ray

from miles.utils.workers.naming import compute_cell_id, compute_worker_name


@dataclass
class ManagerTrainWorkerSource:
    """Fetches a trainer cell's worker actors from the RayWorkerManager.

    The first allocation attaches to the workers the manager launched at init;
    later allocations relaunch the cell (healing) before attaching."""

    manager: ray.actor.ActorHandle
    spec_name: str
    cell_index: int
    num_workers: int
    _first_allocation_done: bool = False

    @property
    def cell_id(self) -> str:
        return compute_cell_id(spec_name=self.spec_name, cell_index=self.cell_index)

    def allocate(self) -> list[ray.actor.ActorHandle]:
        if self._first_allocation_done:
            ray.get(self.manager.start_cell.remote(self.cell_id))
        self._first_allocation_done = True

        return [
            ray.get_actor(
                compute_worker_name(spec_name=self.spec_name, cell_index=self.cell_index, worker_index=worker_index)
            )
            for worker_index in range(self.num_workers)
        ]

    def release(self) -> None:
        ray.get(self.manager.stop_cell.remote(self.cell_id))
