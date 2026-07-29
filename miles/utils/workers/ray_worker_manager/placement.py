from dataclasses import dataclass

from ray.util.placement_group import PlacementGroup


@dataclass(frozen=True)
class SpecPlacement:
    placement_group: PlacementGroup
    bundle_indices: list[int]
    num_gpus_per_worker: float | None = None
    num_cpus_per_worker: float | None = None
    concurrency_groups: dict[str, int] | None = None
