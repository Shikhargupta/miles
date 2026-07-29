import os
import time
from pathlib import Path

import pytest
import ray

from miles.utils.workers.command_actor import CommandActor

_WAIT_TIMEOUT_SECONDS = 30.0


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


def _wait_until(condition, *, message: str) -> None:
    deadline = time.monotonic() + _WAIT_TIMEOUT_SECONDS
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for: {message}"
        time.sleep(0.1)


def _is_actor_dead(actor: "ray.actor.ActorHandle") -> bool:
    try:
        ray.get(actor._get_node_ip.remote(), timeout=1.0)
    except (ray.exceptions.RayActorError, ray.exceptions.GetTimeoutError):
        return True
    return False


class TestPortHelpers:
    def test_reports_node_ip_and_free_ports(self):
        """The port helpers answer with the node ip and a free port at or above the start."""
        actor = ray.remote(CommandActor).remote()
        node_ip = ray.get(actor._get_node_ip.remote())
        port = ray.get(actor._get_free_consecutive_ports.remote(start_port=15000, consecutive=2))

        assert isinstance(node_ip, str) and node_ip
        assert port >= 15000
        ray.kill(actor)


class TestRun:
    def test_runs_command_with_envs_and_exits_when_subprocess_ends(self, tmp_path: Path):
        """The command sees the given envs, and the actor kills itself once the subprocess exits."""
        out_file = tmp_path / "out.txt"
        actor = ray.remote(CommandActor).remote()

        actor.run.remote(cmd=f"echo value=$COMMAND_ACTOR_TEST_ENV > {out_file}", envs={"COMMAND_ACTOR_TEST_ENV": "42"})

        _wait_until(out_file.exists, message="command output file to appear")
        assert out_file.read_text().strip() == "value=42"
        _wait_until(lambda: _is_actor_dead(actor), message="command actor to exit after its subprocess")
