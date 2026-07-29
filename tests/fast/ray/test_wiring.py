import os

import pytest
import ray

from miles.ray.wiring import get_worker_manager, launch_worker_manager
from miles.utils.workers.worker_spec import SchedulingSpec, ServeWorkerSpec

_DUMMY_WORKER_CLASS = "tests.fast.utils.workers.manager_dummy_worker.DummyServeWorker"


@pytest.fixture(scope="module", autouse=True)
def ray_env():
    if ray.is_initialized():
        yield
        return

    init_kwargs: dict = {"ignore_reinit_error": True}
    if "RAY_ADDRESS" not in os.environ:
        init_kwargs["address"] = "local"
        init_kwargs["num_cpus"] = 16
        init_kwargs["num_gpus"] = 0
    ray.init(**init_kwargs)
    yield
    ray.shutdown()


def _make_serve_spec() -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="wiring-dummy",
        port_infos=[],
        env_var=lambda: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=2, num_gpus_per_worker=0),
        worker_class=_DUMMY_WORKER_CLASS,
        ctor_kwargs=lambda: {"tag": "wired"},
    )


class TestLaunchWorkerManager:
    def test_launches_named_manager_with_workers(self):
        """The manager comes up under its well-known name with all spec workers launched."""
        manager = launch_worker_manager([_make_serve_spec()])

        assert get_worker_manager() is not None
        infos = ray.get(manager.get_worker_infos.remote(spec_name="wiring-dummy"))
        assert [info.name for info in infos] == ["wiring-dummy-0-0", "wiring-dummy-0-1"]
        described = ray.get(ray.get_actor("wiring-dummy-0-0").describe.remote())
        assert described["tag"] == "wired"

        ray.kill(manager)
