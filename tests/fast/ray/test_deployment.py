from __future__ import annotations

import itertools
from argparse import Namespace

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.ray import deployment
from miles.ray.specs.static_addrs import inference_controller_urls, static_router_addrs, trainer_controller_urls
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import RPC_PORT_NAME, HostAndPort, NamedHostAndPorts

pytestmark = pytest.mark.asyncio


class _AddressBookProvider(BaseWorkerProvider):
    def __init__(self, addrs_by_worker_name: dict[str, HostAndPort]) -> None:
        self._addrs_by_worker_name = addrs_by_worker_name

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        addr = self._addrs_by_worker_name[worker_name]
        return {RPC_PORT_NAME: addr, "primary": addr}

    async def invalidate_cell(self, cell_id: str) -> None:
        return None

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        raise NotImplementedError


def _inference_args(tmp_path, **overrides) -> Namespace:
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(make_sglang_config_yaml())
    return make_args(sglang_config=str(config_path), deploy_component="inference", **overrides)


def _describe(monkeypatch, *, args: Namespace, component: DeployComponent, addrs: dict[str, HostAndPort]) -> str:
    capability = FakeBackendCapability(static_provider=_AddressBookProvider(addrs))
    monkeypatch.setattr(deployment, "get_backend_capability", lambda _args: capability)
    return deployment._describe_controller_addrs(args, component=component)


def _entries_after(description: str, flag: str) -> list[str]:
    tokens = description.split()
    tail = tokens[tokens.index(flag) + 1 :]
    return list(itertools.takewhile(lambda entry: not entry.startswith("--"), tail))


class TestDescribeControllerAddrs:
    async def test_the_inference_address_it_prints_is_the_one_the_next_launch_takes(self, monkeypatch, tmp_path):
        """This string is the only place a user reads the address from, so it has to parse back unchanged."""
        description = await _describe(
            monkeypatch,
            args=_inference_args(tmp_path),
            component=DeployComponent.INFERENCE,
            addrs={
                "inference-controller-0-0": HostAndPort(host="inference-host", port=8000),
                "inference-router-0-0-0": HostAndPort(host="router-host", port=8100),
            },
        )

        entries = _entries_after(description, "--inference-controller-addrs")

        assert inference_controller_urls(Namespace(inference_controller_addrs=entries)) == ["inference-host:8000"]

    async def test_the_router_address_is_printed_beside_the_controller_it_belongs_to(self, monkeypatch, tmp_path):
        """The primary launch needs all of an inference release's addresses, and deriving them by hand is hopeless."""
        args = _inference_args(tmp_path)
        description = await _describe(
            monkeypatch,
            args=args,
            component=DeployComponent.INFERENCE,
            addrs={
                "inference-controller-0-0": HostAndPort(host="inference-host", port=8000),
                "inference-router-0-0-0": HostAndPort(host="router-host", port=8100),
            },
        )

        entries = _entries_after(description, "--inference-router-addrs")
        args.inference_router_addrs = entries

        assert static_router_addrs(args) == {"default": HostAndPort(host="router-host", port=8100)}

    async def test_the_trainer_address_it_prints_is_the_one_the_next_launch_takes(self, monkeypatch):
        """Same round trip for the trainer, whose entries carry a role prefix the parser has to accept."""
        description = await _describe(
            monkeypatch,
            args=Namespace(use_critic=False),
            component=DeployComponent.TRAINER,
            addrs={"trainer-controller-actor-0-0": HostAndPort(host="trainer-host", port=8000)},
        )

        entries = _entries_after(description, "--trainer-controller-addrs")

        assert trainer_controller_urls(Namespace(trainer_controller_addrs=entries), role="actor") == [
            "trainer-host:8000"
        ]

    async def test_a_run_with_a_critic_prints_both_of_its_controllers(self, monkeypatch):
        """A critic is its own controller, and an unnamed one leaves the next launch unable to reach it."""
        description = await _describe(
            monkeypatch,
            args=Namespace(use_critic=True),
            component=DeployComponent.TRAINER,
            addrs={
                "trainer-controller-actor-0-0": HostAndPort(host="actor-host", port=8000),
                "trainer-controller-critic-0-0": HostAndPort(host="critic-host", port=8000),
            },
        )

        entries = _entries_after(description, "--trainer-controller-addrs")
        args = Namespace(trainer_controller_addrs=entries)

        assert trainer_controller_urls(args, role="actor") == ["actor-host:8000"]
        assert trainer_controller_urls(args, role="critic") == ["critic-host:8000"]


class TestRunDeployment:
    @staticmethod
    def _run(monkeypatch, *, deploy_component: str) -> list[str]:
        taken: list[str] = []

        async def _script(_args) -> None:
            taken.append("orchestration_script")

        async def _serve(_args) -> None:
            taken.append("serve_deployed_workers")

        monkeypatch.setattr(deployment, "_serve_deployed_workers", _serve)
        deployment.run_deployment(Namespace(deploy_component=deploy_component), run_orchestration_script=_script)
        return taken

    @pytest.mark.parametrize("deploy_component", ["all", "primary"])
    def test_the_deployments_carrying_the_script_run_it(self, monkeypatch, deploy_component):
        """These are the only two launches with a training verdict, so the run happens here or nowhere."""
        assert self._run(monkeypatch, deploy_component=deploy_component) == ["orchestration_script"]

    @pytest.mark.parametrize("deploy_component", ["trainer", "inference"])
    def test_the_deployments_without_the_script_only_serve_their_workers(self, monkeypatch, deploy_component):
        """Running the script here would start a second run against the same workers."""
        assert self._run(monkeypatch, deploy_component=deploy_component) == ["serve_deployed_workers"]
