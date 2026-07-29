from ray.util.placement_group import PlacementGroup

from miles.utils.workers.ray_worker_manager.placement import SpecPlacement
from miles.utils.workers.ray_worker_manager.resources import resolve_actor_options
from miles.utils.workers.worker_spec import SchedulingSpec


def _make_scheduling() -> SchedulingSpec:
    return SchedulingSpec(num_cells=2, num_workers_per_cell=2, num_gpus_per_worker=1, num_cpus_per_worker=2.0)


def _make_placement_group() -> PlacementGroup:
    return PlacementGroup.empty()


class TestResolveActorOptions:
    def test_without_placement_uses_scheduling_resources(self):
        """Without a placement the spec's scheduling resources apply and no strategy is set."""
        options = resolve_actor_options(scheduling=_make_scheduling(), placement=None, flat_worker_index=0)

        assert options.num_gpus == 1
        assert options.num_cpus == 2.0
        assert options.scheduling_strategy is None

    def test_placement_picks_bundle_by_flat_index(self):
        """A placement pins the worker to the bundle addressed by its flat index."""
        placement = SpecPlacement(placement_group=_make_placement_group(), bundle_indices=[7, 8, 9, 10])

        options = resolve_actor_options(scheduling=_make_scheduling(), placement=placement, flat_worker_index=2)

        assert options.scheduling_strategy is not None
        assert options.scheduling_strategy.placement_group_bundle_index == 9
        assert options.num_gpus == 1
        assert options.num_cpus == 2.0

    def test_placement_resource_overrides_win(self):
        """Placement-level gpu/cpu overrides replace the scheduling defaults."""
        placement = SpecPlacement(
            placement_group=_make_placement_group(),
            bundle_indices=[0, 1, 2, 3],
            num_gpus_per_worker=0.2,
            num_cpus_per_worker=0.5,
        )

        options = resolve_actor_options(scheduling=_make_scheduling(), placement=placement, flat_worker_index=0)

        assert options.num_gpus == 0.2
        assert options.num_cpus == 0.5
