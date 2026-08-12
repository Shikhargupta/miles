from __future__ import annotations

import itertools

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher import entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.entrypoint import describe_reachable_addrs
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name, static_worker_host

RUN_ID = "260101-000000-000"
NAMESPACE = "rl"
RPC_PORT = 8000
ROUTER_PORT = 8000

_SPLIT_COMPONENTS = [DeployComponent.PRIMARY, DeployComponent.TRAINER, DeployComponent.INFERENCE]


def _args(tmp_path, *, component: DeployComponent):
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(make_sglang_config_yaml())
    return make_args(
        sglang_config=str(config_path),
        rollout_num_gpus=8,
        use_session_server=False,
        use_critic=False,
        sglang_router_port=None,
        deploy_component=component.value,
    )


def _release(component: DeployComponent) -> str:
    return RunNames.release(run_id=RUN_ID, deploy_component=component)


def _object_names(tmp_path, *, component: DeployComponent) -> set[str]:
    release = _release(component)
    return {component_name(release, spec.name) for spec in compute_specs(_args(tmp_path, component=component))}


class TestThreeReleasesOfOneRun:
    def test_no_two_releases_name_the_same_object(self, tmp_path):
        """One name shared between two releases is one launch quietly upgrading another launch's workload."""
        names_by_component = {
            component: _object_names(tmp_path, component=component) for component in _SPLIT_COMPONENTS
        }

        for first, second in itertools.combinations(_SPLIT_COMPONENTS, 2):
            assert not names_by_component[first] & names_by_component[second]

    def test_every_worker_of_the_run_is_deployed_by_exactly_one_release(self, tmp_path):
        """The three components partition the run, so a spec no release carries never comes up at all."""
        split = [
            spec.name
            for component in _SPLIT_COMPONENTS
            for spec in compute_specs(_args(tmp_path, component=component))
        ]

        assert sorted(split) == sorted(
            spec.name for spec in compute_specs(_args(tmp_path, component=DeployComponent.ALL))
        )

    def test_the_trainer_launch_prints_the_address_its_own_release_answers_on(self, tmp_path):
        """This string is pasted straight into the primary launch, so a derived name would be a dead address."""
        args = _args(tmp_path, component=DeployComponent.TRAINER)
        release = _release(DeployComponent.TRAINER)

        printed = describe_reachable_addrs(args, specs=compute_specs(args), release=release)

        host = static_worker_host(release, "trainer-controller-actor", 0)
        assert printed == f"--trainer-controller-addrs actor={host}:{RPC_PORT}"

    def test_the_inference_launch_prints_its_router_beside_its_controller(self, tmp_path):
        """The primary needs both, and a router taken from another release samples the wrong engines."""
        args = _args(tmp_path, component=DeployComponent.INFERENCE)
        release = _release(DeployComponent.INFERENCE)

        printed = describe_reachable_addrs(args, specs=compute_specs(args), release=release)

        controller = static_worker_host(release, "inference-controller", 0)
        router = static_worker_host(release, "inference-router-0", 0)
        assert printed == (
            f"--inference-controller-addrs {controller}:{RPC_PORT} "
            f"--inference-router-addrs default={router}:{ROUTER_PORT}"
        )

    def test_the_object_store_master_the_other_releases_name_is_the_primary_releases_own(self):
        """The trainer and inference launches type this address by hand, so one place has to compute it."""
        primary = _release(DeployComponent.PRIMARY)

        master = MooncakeInfo.master_service_host(primary, NAMESPACE)

        assert master == f"{component_name(primary, 'mooncake-master')}.{NAMESPACE}.svc.cluster.local"
        assert master != MooncakeInfo.master_service_host(_release(DeployComponent.TRAINER), NAMESPACE)


def test_a_run_id_ending_in_a_component_name_is_refused() -> None:
    """Its unsplit release would carry the very name another run's split launch installs its own release under."""
    with pytest.raises(AssertionError, match="ends in a component name"):
        entrypoint.execute_train(
            request=None, config=ExecuteTrainConfig(run_id=f"{RUN_ID}-trainer", namespace=NAMESPACE)
        )
