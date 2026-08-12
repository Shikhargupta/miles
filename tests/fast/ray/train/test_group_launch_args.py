from __future__ import annotations

from argparse import Namespace

import pytest
from tests.fast.fixtures.controller_fixtures import make_trainer_controller

from miles.ray.train.group import _adopt_launch_level_args
from miles.utils.workers.types import DeploymentIdentity


def _args(**overrides) -> Namespace:
    defaults = dict(
        deploy_component="trainer",
        trainer_controller_addrs=None,
        inference_controller_addrs=None,
        inference_router_addrs=None,
        api_server_port=0,
        num_rollout=3,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestAdoptLaunchLevelArgs:
    def test_the_flags_naming_this_launch_survive_the_scripts_arguments(self):
        """A trainer pod is not the primary release, and believing it is makes it compute the wrong spec table."""
        adopted = _adopt_launch_level_args(
            _args(
                deploy_component="primary",
                inference_controller_addrs=["10.0.0.1:8000"],
                trainer_controller_addrs=["10.0.0.2:8000"],
                api_server_port=8080,
            ),
            launch_args=_args(),
        )

        assert adopted.deploy_component == "trainer"
        assert adopted.inference_controller_addrs is None
        assert adopted.trainer_controller_addrs is None
        assert adopted.api_server_port == 0

    def test_everything_else_comes_from_the_orchestration_script(self):
        """The script is still the single source of the run's arguments; only the launch flags are the pod's own."""
        adopted = _adopt_launch_level_args(_args(deploy_component="primary", num_rollout=17), launch_args=_args())

        assert adopted.num_rollout == 17

    def test_the_arguments_handed_in_are_left_alone(self):
        """The caller's namespace is the script's own, and rewriting it would change what the script sees."""
        handed_in = _args(deploy_component="primary")

        _adopt_launch_level_args(handed_in, launch_args=_args())

        assert handed_in.deploy_component == "primary"

    @pytest.mark.parametrize("deploy_component", ["trainer", "inference"])
    def test_arguments_of_a_launch_that_carries_no_script_are_refused(self, deploy_component):
        """Only the launch with the orchestration script drives a trainer, so nothing else may hand it arguments."""
        with pytest.raises(AssertionError, match="carries no orchestration script"):
            _adopt_launch_level_args(_args(deploy_component=deploy_component), launch_args=_args())


class TestDeploymentIdentity:
    async def test_the_identity_names_the_launch_this_controller_was_started_by(self):
        """A trainer answers for the deployment that launched it, not for the arguments a script hands it later."""
        controller = make_trainer_controller(launch_args=_args(run_uuid="run-1", deploy_component="trainer"))

        assert await controller.get_deployment_identity() == DeploymentIdentity(
            run_uuid="run-1", deploy_component="trainer", router_addrs={}
        )

    async def test_a_controller_built_without_launch_arguments_has_no_launch_fields(self):
        """A test that never said which launch it belongs to must fail loudly instead of being handed an invention."""
        controller = make_trainer_controller()

        with pytest.raises(AttributeError):
            await controller.get_deployment_identity()
