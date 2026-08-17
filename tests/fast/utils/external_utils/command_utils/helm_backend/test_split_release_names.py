from __future__ import annotations

import itertools
import json
import shlex

import pytest
from tests.fast.ray.rollout.conftest import make_args_with_sglang_config

from miles.ray.specs.entrypoint import compute_specs
from miles.ray.specs.inference import compute_registration_reporter_id
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher import entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo, MooncakePlan
from miles.utils.external_utils.command_utils.helm_backend.naming import (
    _HELM_RELEASE_NAME_MAX,
    RUN_ID_MAX_LENGTH,
    RunNames,
)
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name

RUN_ID = "260101-000000-000"
NAMESPACE = "rl"

_SPLIT_COMPONENTS = ["primary", "trainer", "inference"]


def _args(tmp_path, *, component: str, instance: str | None = None, **overrides):
    return make_args_with_sglang_config(
        tmp_path,
        rollout_num_gpus=8,
        use_session_server=False,
        use_critic=False,
        sglang_router_port=None,
        deploy_component=component,
        deploy_instance=instance,
        **overrides,
    )


def _release(component: str, instance: str | None = None) -> str:
    return RunNames.release(run_id=RUN_ID, deploy_component=DeployComponent(component), deploy_instance=instance)


def _object_names(tmp_path, *, component: str, instance: str | None = None, **overrides) -> set[str]:
    release = _release(component, instance)
    specs = compute_specs(_args(tmp_path, component=component, instance=instance, **overrides))
    return {component_name(release, spec.name) for spec in specs}


class TestTwoReleasesOfOneRun:
    def test_no_two_releases_name_the_same_object(self, tmp_path):
        """One name shared between two releases is one launch quietly upgrading another launch's workload."""
        names_by_component = {
            component: _object_names(tmp_path, component=component) for component in _SPLIT_COMPONENTS
        }

        for first, second in itertools.combinations(_SPLIT_COMPONENTS, 2):
            assert not names_by_component[first] & names_by_component[second]

    def test_every_worker_of_the_run_is_deployed_by_exactly_one_release(self, tmp_path):
        """The two components partition the run, so a spec no release carries never comes up at all."""
        split = [
            spec.name
            for component in _SPLIT_COMPONENTS
            for spec in compute_specs(_args(tmp_path, component=component))
        ]

        assert sorted(split) == sorted(spec.name for spec in compute_specs(_args(tmp_path, component="all")))

    def test_the_store_flags_it_prints_carry_every_init_kwarg_the_run_was_launched_with(self):
        """The other launch pastes this line, and a dropped kwarg leaves the deployments on different protocols."""
        primary = _release("primary")
        plan = MooncakePlan(init_kwargs={"master_server_address": "0.0.0.0:50051", "protocol": "tcp"}, port=50051)

        printed = entrypoint._describe_shared_object_store(plan, release=primary, namespace=NAMESPACE)

        tokens = shlex.split(printed)
        init_kwargs = json.loads(tokens[tokens.index("--mooncake-store-init-kwargs") + 1])
        assert init_kwargs == {
            "master_server_address": f"{MooncakeInfo.master_service_host(primary, NAMESPACE)}:50051",
            "protocol": "tcp",
        }

    def test_the_object_store_master_the_trainer_release_names_is_the_primary_releases_own(self):
        """The trainer launch types this address by hand, so one place has to compute it."""
        primary = _release("primary")

        master = MooncakeInfo.master_service_host(primary, NAMESPACE)

        assert master == f"{component_name(primary, 'mooncake-master')}.{NAMESPACE}.svc.cluster.local"
        assert master != MooncakeInfo.master_service_host(_release("trainer"), NAMESPACE)


class TestReleasesOfInstancesAndEngineGroups:
    def test_a_trainer_role_installs_a_release_named_after_that_role(self, tmp_path):
        """Two roles installed under one name would be one launch quietly upgrading the other's ranks."""
        assert _release("trainer", "actor") == f"miles-run-{RUN_ID}-trainer-actor"
        assert _release("trainer", "critic") != _release("trainer", "actor")

    def test_an_engine_group_installs_a_release_of_its_own(self, tmp_path):
        """Its pool ids are namespaced by this name, so it is what keeps two engine groups apart."""
        assert _release("inference") == f"miles-run-{RUN_ID}-inference"

    def test_a_named_engine_group_installs_a_release_named_after_it(self, tmp_path):
        """Two engine groups under one release would be one launch quietly upgrading the other's engines."""
        assert _release("inference", "dc1") == f"miles-run-{RUN_ID}-inference-dc1"
        assert _release("inference", "dc2") != _release("inference", "dc1")

    def test_two_named_engine_groups_report_under_reporter_ids_of_their_own(self, tmp_path):
        """One reporter id carries one whole membership, and the second would replace the first's."""
        first = compute_registration_reporter_id(
            _args(
                tmp_path,
                component="inference",
                instance="dc1",
                inference_controller_addr="controller:8000",
            )
        )
        second = compute_registration_reporter_id(
            _args(
                tmp_path,
                component="inference",
                instance="dc2",
                inference_controller_addr="controller:8000",
            )
        )

        assert first != second

    def test_no_engine_group_names_an_object_of_the_run_it_registers_into(self, tmp_path):
        """It runs the same engine pools as the primary deployment, under names that must not collide."""
        engines = _object_names(
            tmp_path,
            component="inference",
            inference_controller_addr="controller:8000",
        )

        assert not engines & _object_names(tmp_path, component="primary")


@pytest.mark.parametrize(
    ("component", "instance"),
    [(DeployComponent.ALL, None), (DeployComponent.TRAINER, "actor"), (DeployComponent.INFERENCE, "dc1")],
)
def test_the_launch_config_carries_the_component_and_the_instance_apart(component, instance) -> None:
    """The two are orthogonal, so the config holds them as two fields rather than one joined string."""
    config = ExecuteTrainConfig(
        run_id=RUN_ID, namespace=NAMESPACE, deploy_component=component, deploy_instance=instance
    )

    assert (config.deploy_component, config.deploy_instance) == (component, instance)


@pytest.mark.parametrize("suffix", ["trainer", "trainer-actor", "inference-dc1"])
def test_a_run_id_carrying_a_component_name_is_refused(suffix: str) -> None:
    """Its unsplit release would carry the very name another run's split launch installs its own release under."""
    with pytest.raises(AssertionError, match="component name"):
        entrypoint.execute_train(
            request=None, config=ExecuteTrainConfig(run_id=f"{RUN_ID}-{suffix}", namespace=NAMESPACE)
        )


class TestTheRunIdLeavesRoomForTheComponentSuffix:
    def test_the_longest_accepted_run_id_names_a_legal_release_for_every_component(self):
        """A run id that only fits unsplit is a trap: the split launch of it fails inside helm."""
        run_id = "a" * RUN_ID_MAX_LENGTH

        for component in DeployComponent:
            release = RunNames.release(run_id=run_id, deploy_component=component)
            assert len(release) <= _HELM_RELEASE_NAME_MAX

    def test_a_longer_run_id_is_refused_where_the_release_is_named(self):
        """helm would refuse the install itself, long after the launch computed every object name from it."""
        with pytest.raises(AssertionError, match=str(_HELM_RELEASE_NAME_MAX)):
            RunNames.release(run_id="a" * (RUN_ID_MAX_LENGTH + 1))
