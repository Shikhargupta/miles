from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from miles.ray.placement_group import create_rollout_components
from miles.ray.rollout.inference_controller import RouterInfo


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        pin_rollout_manager_to_head=False,
        num_rollout=None,
        num_epoch=2,
        check_weight_update_equal=False,
        check_weight_update_skip_list=[],
        offload_rollout=False,
        sglang_router_ip=None,
        sglang_router_port=None,
        sglang_model_routers=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class _FakeActorClass:
    def __init__(self, handle: MagicMock) -> None:
        self._handle = handle
        self.remote_calls: list[tuple] = []
        self.arg_snapshots: list[Namespace] = []

    def options(self, **_kwargs):
        return self

    def remote(self, *args):
        self.remote_calls.append(args)
        self.arg_snapshots.append(deepcopy(args[0]) if args and isinstance(args[0], Namespace) else None)
        return self._handle


@pytest.fixture
def fake_actors():
    controller_handle = MagicMock(name="inference_controller")
    controller_handle.get_router_info.remote.return_value = "router-info-ref"
    executor_handle = MagicMock(name="rollout_executor")

    controller_cls = _FakeActorClass(controller_handle)
    executor_cls = _FakeActorClass(executor_handle)

    router_info = RouterInfo(router_ip="10.0.0.1", router_port=4321, model_routers={"actor": ("10.0.0.1", 4321)})

    def fake_ray_get(ref):
        if ref == "router-info-ref":
            return router_info
        return 5

    with patch("miles.ray.placement_group.InferenceController", controller_cls), patch(
        "miles.ray.placement_group.RolloutExecutor", executor_cls
    ), patch("miles.ray.placement_group.ray.get", side_effect=fake_ray_get):
        yield Namespace(
            controller_cls=controller_cls,
            executor_cls=executor_cls,
            controller_handle=controller_handle,
            executor_handle=executor_handle,
            router_info=router_info,
        )


class TestCreateRolloutComponents:
    def test_publishes_router_info_into_args_before_creating_executor(self, fake_actors):
        """The router address must already be in args when the executor actor is constructed."""
        args = _make_args(num_rollout=1)

        create_rollout_components(args, pg=MagicMock())

        (executor_args_at_construction,) = fake_actors.executor_cls.arg_snapshots
        assert executor_args_at_construction.sglang_router_ip == "10.0.0.1"
        assert executor_args_at_construction.sglang_router_port == 4321
        assert executor_args_at_construction.sglang_model_routers == {"actor": ("10.0.0.1", 4321)}

    def test_controller_is_created_before_the_router_address_is_known(self, fake_actors):
        """The controller starts the router itself, so it cannot be handed the address."""
        args = _make_args(num_rollout=1)

        create_rollout_components(args, pg=MagicMock())

        (controller_args_at_construction,) = fake_actors.controller_cls.arg_snapshots
        assert controller_args_at_construction.sglang_router_ip is None

    def test_passes_controller_handle_to_executor(self, fake_actors):
        """The executor calls back into the controller, so it is built with that handle."""
        args = _make_args(num_rollout=1)

        components = create_rollout_components(args, pg=MagicMock())

        (executor_ctor_args,) = fake_actors.executor_cls.remote_calls
        assert executor_ctor_args[1] is fake_actors.controller_handle
        assert components.inference_controller is fake_actors.controller_handle
        assert components.rollout_executor is fake_actors.executor_handle

    def test_result_unpacks_in_controller_executor_epoch_order(self, fake_actors):
        """Callers unpack positionally, so a swapped handle pair must be caught here."""
        args = _make_args(num_rollout=None, num_epoch=1)

        inference_controller, rollout_executor, num_rollout_per_epoch = create_rollout_components(args, pg=MagicMock())

        assert inference_controller is fake_actors.controller_handle
        assert rollout_executor is fake_actors.executor_handle
        assert num_rollout_per_epoch == 5

    def test_num_rollout_derived_from_executor_epoch_length(self, fake_actors):
        """num_rollout comes from the dataset, which the executor owns."""
        args = _make_args(num_rollout=None, num_epoch=2)

        components = create_rollout_components(args, pg=MagicMock())

        fake_actors.executor_handle.get_num_rollout_per_epoch.remote.assert_called_once()
        assert components.num_rollout_per_epoch == 5
        assert args.num_rollout == 10

    def test_num_rollout_left_alone_when_explicitly_set(self, fake_actors):
        """An explicit --num-rollout skips asking the executor for the epoch length."""
        args = _make_args(num_rollout=3)

        components = create_rollout_components(args, pg=MagicMock())

        fake_actors.executor_handle.get_num_rollout_per_epoch.remote.assert_not_called()
        assert components.num_rollout_per_epoch is None
        assert args.num_rollout == 3

    def test_weight_check_and_offload_go_to_the_controller(self, fake_actors):
        """Engine-side startup steps go to the controller, never to the executor."""
        args = _make_args(num_rollout=1, check_weight_update_equal=True, offload_rollout=True)

        create_rollout_components(args, pg=MagicMock())

        actions = [call.kwargs["action"] for call in fake_actors.controller_handle.check_weights.remote.call_args_list]
        assert actions == ["snapshot", "reset_tensors"]
        fake_actors.controller_handle.offload.remote.assert_called_once()
        fake_actors.executor_handle.check_weights.remote.assert_not_called()
        fake_actors.executor_handle.offload.remote.assert_not_called()
