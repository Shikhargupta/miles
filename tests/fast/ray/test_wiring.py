import pytest
import ray
from tests.fast.ray.specs.conftest import make_args

from miles.ray.specs.inference import compute_inference_deployments
from miles.ray.wiring import compute_inference_placements, get_worker_manager, launch_worker_manager
from miles.utils.workers.worker_spec import SchedulingSpec, ServeWorkerSpec

_DUMMY_WORKER_CLASS = "tests.fast.utils.workers.manager_dummy_worker.DummyServeWorker"


@pytest.fixture(scope="module", autouse=True)
def _ray_cluster(ray_local_mode):
    yield


def _make_serve_spec() -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="wiring-dummy",
        port_infos=[],
        env_var=lambda: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=2, num_gpus_per_worker=0, num_cpus_per_worker=0.1),
        worker_class=_DUMMY_WORKER_CLASS,
        ctor_kwargs=lambda cell_index, worker_index: {"tag": "wired"},
    )


class TestLaunchWorkerManager:
    def test_launches_named_manager_with_workers(self):
        """The manager is reachable under its well-known name with all spec workers launched."""
        manager = launch_worker_manager([_make_serve_spec()], placements={})

        infos = ray.get(get_worker_manager().get_worker_infos.remote(spec_names=["wiring-dummy"]))
        assert [info.name for info in infos] == ["wiring-dummy-0-0", "wiring-dummy-0-1"]
        described = ray.get(ray.get_actor("wiring-dummy-0-0").describe.remote())
        assert described["tag"] == "wired"

        ray.kill(manager)


class TestComputeInferencePlacements:
    def test_maps_workers_onto_reordered_rollout_bundles(self):
        """Each engine worker lands on the reordered bundle of its first gpu and uses fractional resources."""
        args = make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=4, num_gpus_per_node=8)
        deployments = compute_inference_deployments(args)
        pg = (object(), [10, 11, 12, 13, 14, 15, 16, 17], list(range(8)))

        placements = compute_inference_placements(deployments=deployments, pg=pg)

        (placement,) = placements.values()
        assert placement.bundle_indices == [10, 14]
        assert placement.num_gpus_per_worker == 0.2

    def test_multi_node_engines_take_one_bundle_per_worker_node(self):
        """A 16-gpu engine on 8-gpu nodes claims the bundle heading each node's gpu block."""
        args = make_args(rollout_num_gpus=32, rollout_num_gpus_per_engine=16, num_gpus_per_node=8)
        deployments = compute_inference_deployments(args)
        pg = (object(), list(range(100, 132)), list(range(32)))

        placements = compute_inference_placements(deployments=deployments, pg=pg)

        (placement,) = placements.values()
        assert placement.bundle_indices == [100, 108, 116, 124]
