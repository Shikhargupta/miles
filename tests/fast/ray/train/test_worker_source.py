import pytest
import ray

from miles.ray.train.worker_source import ManagerTrainWorkerSource
from miles.utils.workers.ray_worker_manager.manager import RayWorkerManager
from miles.utils.workers.worker_spec import SchedulingSpec, ServeWorkerSpec

_DUMMY_WORKER_CLASS = "tests.fast.utils.workers.manager_dummy_worker.DummyServeWorker"


def _make_spec(name: str) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=name,
        port_infos=[],
        env_var=lambda: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=2, num_gpus_per_worker=0, num_cpus_per_worker=0.1),
        worker_class=_DUMMY_WORKER_CLASS,
        ctor_kwargs=lambda cell_index, worker_index: {"tag": f"{cell_index}-{worker_index}"},
    )


class TestManagerTrainWorkerSource:
    async def test_first_allocation_attaches_to_manager_launched_workers(self, ray_env):
        """The initial allocate resolves the named actors the manager already launched."""
        spec = _make_spec("wsrc-a")
        manager = ray.remote(RayWorkerManager).options(name="wsrc-a-manager", num_cpus=0.1).remote()
        await manager.init.remote(worker_specs=[spec], placements={})
        source = ManagerTrainWorkerSource(manager=manager, spec_name="wsrc-a", cell_index=0, num_workers=2)

        handles = source.allocate()

        assert [ray.get(handle.describe.remote())["tag"] for handle in handles] == ["0-0", "0-1"]
        ray.kill(manager)
        for handle in handles:
            ray.kill(handle)

    async def test_release_then_allocate_relaunches_the_cell(self, ray_env):
        """Healing goes stop_cell then start_cell, yielding fresh workers."""
        spec = _make_spec("wsrc-b")
        manager = ray.remote(RayWorkerManager).options(name="wsrc-b-manager", num_cpus=0.1).remote()
        await manager.init.remote(worker_specs=[spec], placements={})
        source = ManagerTrainWorkerSource(manager=manager, spec_name="wsrc-b", cell_index=0, num_workers=2)
        source.allocate()

        source.release()
        with pytest.raises(ValueError):
            ray.get_actor("wsrc-b-0-0")

        second = source.allocate()
        assert [ray.get(handle.describe.remote())["tag"] for handle in second] == ["0-0", "0-1"]
        infos = await manager.get_worker_infos.remote(spec_name="wsrc-b")
        assert all(info.generation == 1 for info in infos)
        ray.kill(manager)
        for handle in second:
            ray.kill(handle)
