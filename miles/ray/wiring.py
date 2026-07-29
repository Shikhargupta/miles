import ray

from miles.ray.placement_group import _create_placement_group
from miles.utils.workers.ray_worker_manager import RayWorkerManager, SpecPlacement
from miles.utils.workers.worker_spec import BaseWorkerSpec

_WORKER_MANAGER_ACTOR_NAME = "miles_ray_worker_manager"


def launch_worker_manager(worker_specs: list[BaseWorkerSpec]) -> ray.actor.ActorHandle:
    placements = _create_placements(worker_specs)
    manager = ray.remote(RayWorkerManager).options(name=_WORKER_MANAGER_ACTOR_NAME, num_cpus=1).remote()
    ray.get(manager.init.remote(worker_specs=worker_specs, placements=placements))
    return manager


def get_worker_manager() -> ray.actor.ActorHandle:
    return ray.get_actor(_WORKER_MANAGER_ACTOR_NAME)


def _create_placements(worker_specs: list[BaseWorkerSpec]) -> dict[str, SpecPlacement]:
    placements: dict[str, SpecPlacement] = {}
    for spec in worker_specs:
        scheduling = spec.scheduling
        if scheduling.num_gpus_per_worker == 0:
            continue
        assert scheduling.num_gpus_per_worker <= 1, (
            f"{spec.name=} needs one single-gpu bundle per worker, got {scheduling.num_gpus_per_worker=}"
        )

        num_workers = scheduling.num_cells * scheduling.num_workers_per_cell
        placement_group, bundle_indices, _gpu_ids = _create_placement_group(num_workers)
        placements[spec.name] = SpecPlacement(placement_group=placement_group, bundle_indices=bundle_indices)
    return placements
