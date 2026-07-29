import itertools
import os
import time
from pathlib import Path

import pytest
import ray

from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

_DUMMY_WORKER_CLASS = "tests.fast.utils.workers.manager_dummy_worker.DummyServeWorker"
_WAIT_TIMEOUT_SECONDS = 30.0

_unique_counter = itertools.count()


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


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{next(_unique_counter)}"


def _make_serve_spec(name: str, **overrides) -> ServeWorkerSpec:
    kwargs = dict(
        name=name,
        port_infos=[
            PortInfo(name="http", static_port=18123, mode="per_worker", allow_dynamic=False),
            PortInfo(name="rendezvous", static_port=0, mode="master", allow_dynamic=True),
        ],
        env_var=lambda: {"MANAGER_DUMMY_ENV": "42"},
        scheduling=SchedulingSpec(num_cells=2, num_workers_per_cell=2, num_gpus_per_worker=0),
        worker_class=_DUMMY_WORKER_CLASS,
        ctor_kwargs=lambda: {"tag": "hello"},
    )
    kwargs.update(overrides)
    return ServeWorkerSpec(**kwargs)


async def _make_manager(worker_specs) -> "ray.actor.ActorHandle":
    manager = ray.remote(RayWorkerManager).options(name=_unique_name("test-manager"), num_cpus=1).remote()
    await manager.init.remote(worker_specs=worker_specs, placements={})
    return manager


def _wait_until(condition, *, message: str) -> None:
    deadline = time.monotonic() + _WAIT_TIMEOUT_SECONDS
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for: {message}"
        time.sleep(0.1)


def _actor_exists(name: str) -> bool:
    try:
        ray.get_actor(name)
    except ValueError:
        return False
    return True


class TestInit:
    async def test_launches_named_actors_with_remote_ctor_kwargs_and_envs(self):
        """Workers come up as named actors with ctor kwargs evaluated remotely and spec env vars applied."""
        spec_name = _unique_name("serve")
        manager = await _make_manager([_make_serve_spec(spec_name)])

        described = ray.get(ray.get_actor(f"{spec_name}-0-0").describe.remote())

        assert described["tag"] == "hello"
        assert described["dummy_env"] == "42"
        ray.kill(manager)

    async def test_configures_master_and_per_worker_ports(self):
        """Cell members share the master port, keep their static per-worker port, and cells differ."""
        spec_name = _unique_name("serve")
        manager = await _make_manager([_make_serve_spec(spec_name)])

        cell_0 = [ray.get(ray.get_actor(f"{spec_name}-0-{i}").describe.remote())["addr_ports"] for i in range(2)]
        cell_1 = [ray.get(ray.get_actor(f"{spec_name}-1-{i}").describe.remote())["addr_ports"] for i in range(2)]

        assert cell_0[0]["rendezvous_port"] == cell_0[1]["rendezvous_port"] >= 15000
        assert cell_0[0]["rendezvous_addr"] == cell_0[1]["rendezvous_addr"]
        assert cell_0[0]["http_port"] == cell_0[1]["http_port"] == 18123
        assert cell_1[0]["rendezvous_port"] != cell_0[0]["rendezvous_port"]
        ray.kill(manager)

    async def test_rejects_duplicate_spec_names(self):
        """Two specs sharing a name are rejected."""
        spec_name = _unique_name("serve")
        manager = ray.remote(RayWorkerManager).options(name=_unique_name("test-manager"), num_cpus=1).remote()
        with pytest.raises(ray.exceptions.RayTaskError):
            await manager.init.remote(
                worker_specs=[_make_serve_spec(spec_name), _make_serve_spec(spec_name)], placements={}
            )
        ray.kill(manager)


class TestGetWorkerInfos:
    async def test_reports_names_cells_generation_and_urls(self):
        """Worker infos expose stable names, cell ids, generation 0, and the http url."""
        spec_name = _unique_name("serve")
        manager = await _make_manager([_make_serve_spec(spec_name)])

        infos = await manager.get_worker_infos.remote(spec_name=spec_name)

        assert [info.name for info in infos] == [f"{spec_name}-{c}-{w}" for c in range(2) for w in range(2)]
        assert [info.cell_id for info in infos] == [f"{spec_name}-{c}" for c in range(2) for w in range(2)]
        assert all(info.generation == 0 for info in infos)
        assert all(info.url and info.url.endswith(":18123") for info in infos)
        ray.kill(manager)

    async def test_filters_by_spec_name(self):
        """Asking for an unknown spec name returns nothing."""
        spec_name = _unique_name("serve")
        manager = await _make_manager([_make_serve_spec(spec_name)])

        assert await manager.get_worker_infos.remote(spec_name="nonexistent") == []
        ray.kill(manager)


class TestStopStartCell:
    async def test_stop_cell_kills_actors_and_hides_them(self):
        """A stopped cell's actors are gone and its workers vanish from the listing."""
        spec_name = _unique_name("serve")
        manager = await _make_manager([_make_serve_spec(spec_name)])

        await manager.stop_cell.remote(f"{spec_name}-0")

        assert not _actor_exists(f"{spec_name}-0-0")
        infos = await manager.get_worker_infos.remote(spec_name=spec_name)
        assert [info.cell_id for info in infos] == [f"{spec_name}-1", f"{spec_name}-1"]
        ray.kill(manager)

    async def test_start_cell_relaunches_with_bumped_generation(self):
        """A restarted cell comes back as fresh actors reporting generation 1."""
        spec_name = _unique_name("serve")
        manager = await _make_manager([_make_serve_spec(spec_name)])

        await manager.stop_cell.remote(f"{spec_name}-0")
        await manager.start_cell.remote(f"{spec_name}-0")

        described = ray.get(ray.get_actor(f"{spec_name}-0-0").describe.remote())
        assert described["tag"] == "hello"
        infos = await manager.get_worker_infos.remote(spec_name=spec_name)
        generations = {info.cell_id: info.generation for info in infos}
        assert generations == {f"{spec_name}-0": 1, f"{spec_name}-1": 0}
        ray.kill(manager)

    async def test_stop_requires_alive_and_start_requires_stopped(self):
        """Stopping a stopped cell or starting an alive cell is rejected."""
        spec_name = _unique_name("serve")
        manager = await _make_manager([_make_serve_spec(spec_name)])

        with pytest.raises(ray.exceptions.RayTaskError):
            await manager.start_cell.remote(f"{spec_name}-0")
        await manager.stop_cell.remote(f"{spec_name}-0")
        with pytest.raises(ray.exceptions.RayTaskError):
            await manager.stop_cell.remote(f"{spec_name}-0")
        ray.kill(manager)


class TestCommandSpec:
    async def test_runs_rendered_command_per_worker(self, tmp_path: Path):
        """Each command worker runs the launch command with its ports and the spec envs rendered in."""
        spec_name = _unique_name("cmd")
        spec = CommandWorkerSpec(
            name=spec_name,
            port_infos=[PortInfo(name="http", static_port=19001, mode="per_worker", allow_dynamic=False)],
            env_var=lambda: {"MANAGER_DUMMY_ENV": "cmd-env"},
            scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=2, num_gpus_per_worker=0),
            launch_command=f"echo port={{http_port}} env=$MANAGER_DUMMY_ENV > {tmp_path}/out-$$.txt",
        )
        manager = await _make_manager([spec])

        _wait_until(lambda: len(list(tmp_path.glob("out-*.txt"))) == 2, message="both command outputs to appear")
        contents = {path.read_text().strip() for path in tmp_path.glob("out-*.txt")}
        assert contents == {"port=19001 env=cmd-env"}
        ray.kill(manager)
