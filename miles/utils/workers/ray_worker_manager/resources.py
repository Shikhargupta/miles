from dataclasses import dataclass

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from miles.utils.workers.ray_worker_manager.placement import SpecPlacement
from miles.utils.workers.worker_spec import SchedulingSpec


@dataclass(frozen=True)
class ActorOptions:
    num_cpus: float
    num_gpus: float
    scheduling_strategy: PlacementGroupSchedulingStrategy | None


def resolve_actor_options(
    *, scheduling: SchedulingSpec, placement: SpecPlacement | None, flat_worker_index: int
) -> ActorOptions:
    scheduling_strategy: PlacementGroupSchedulingStrategy | None = None
    num_gpus = scheduling.num_gpus_per_worker
    num_cpus = scheduling.num_cpus_per_worker

    if placement is not None:
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=placement.placement_group,
            placement_group_bundle_index=placement.bundle_indices[flat_worker_index],
        )
        if placement.num_gpus_per_worker is not None:
            num_gpus = placement.num_gpus_per_worker
        if placement.num_cpus_per_worker is not None:
            num_cpus = placement.num_cpus_per_worker

    return ActorOptions(num_cpus=num_cpus, num_gpus=num_gpus, scheduling_strategy=scheduling_strategy)
