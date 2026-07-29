import ray

from miles.ray.specs.inference import (
    ENGINE_RAY_NUM_CPUS_PER_WORKER,
    ENGINE_RAY_NUM_GPUS_PER_WORKER,
    InferenceDeployment,
)
from miles.ray.specs.trainer import (
    TRAINER_FT_CONCURRENCY_GROUPS,
    TRAINER_RAY_NUM_CPUS_PER_WORKER,
    TRAINER_RAY_NUM_GPUS_PER_WORKER,
)
from miles.utils.workers.ray_worker_manager import RayWorkerManager, SpecPlacement
from miles.utils.workers.worker_spec import BaseWorkerSpec, ServeWorkerSpec

_WORKER_MANAGER_ACTOR_NAME = "miles_ray_worker_manager"


def launch_worker_manager(
    worker_specs: list[BaseWorkerSpec], placements: dict[str, SpecPlacement]
) -> ray.actor.ActorHandle:
    manager = ray.remote(RayWorkerManager).options(name=_WORKER_MANAGER_ACTOR_NAME, num_cpus=1).remote()
    ray.get(manager.init.remote(worker_specs=worker_specs, placements=placements))
    return manager


def get_worker_manager() -> ray.actor.ActorHandle:
    return ray.get_actor(_WORKER_MANAGER_ACTOR_NAME)


def compute_inference_placements(*, deployments: list[InferenceDeployment], pg) -> dict[str, SpecPlacement]:
    placement_group, reordered_bundle_indices, _reordered_gpu_ids = pg

    placements: dict[str, SpecPlacement] = {}
    for deployment in deployments:
        scheduling = deployment.spec.scheduling
        bundle_indices = []
        for cell_index in range(scheduling.num_cells):
            for worker_index in range(scheduling.num_workers_per_cell):
                engine_index_in_group = cell_index * deployment.nodes_per_engine + worker_index
                gpu_index = deployment.group_gpu_offset + engine_index_in_group * deployment.num_gpus_per_engine_local
                bundle_indices.append(reordered_bundle_indices[gpu_index])
        placements[deployment.spec.name] = SpecPlacement(
            placement_group=placement_group,
            bundle_indices=bundle_indices,
            num_gpus_per_worker=ENGINE_RAY_NUM_GPUS_PER_WORKER,
            num_cpus_per_worker=ENGINE_RAY_NUM_CPUS_PER_WORKER,
        )
    return placements


def compute_trainer_placements(args, *, trainer_specs: list[ServeWorkerSpec], pgs: dict) -> dict[str, SpecPlacement]:
    placements: dict[str, SpecPlacement] = {}
    for spec in trainer_specs:
        role = spec.name.removeprefix("train-")
        placement_group, reordered_bundle_indices, _reordered_gpu_ids = pgs[role]
        num_workers = spec.scheduling.num_cells * spec.scheduling.num_workers_per_cell
        placements[spec.name] = SpecPlacement(
            placement_group=placement_group,
            bundle_indices=list(reordered_bundle_indices[:num_workers]),
            num_gpus_per_worker=TRAINER_RAY_NUM_GPUS_PER_WORKER,
            num_cpus_per_worker=TRAINER_RAY_NUM_CPUS_PER_WORKER,
            concurrency_groups=TRAINER_FT_CONCURRENCY_GROUPS if args.use_fault_tolerance else None,
        )
    return placements
