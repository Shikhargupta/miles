from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.fast.fixtures.provider_fixtures import FakeProviderFactory

from miles.ray.rollout.components import create_rollout_components

pytestmark = pytest.mark.asyncio


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        pin_rollout_manager_to_head=False,
        num_rollout=None,
        num_epoch=2,
        sglang_router_ip=None,
        sglang_router_port=None,
        cluster_backend="ray",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.fixture
def fake_components():
    controller = MagicMock(name="inference_controller")

    async def build_controller(args, *, providers):
        args.sglang_router_ip = "10.0.0.1"
        args.sglang_router_port = 4321
        return controller

    controller_cls = MagicMock(name="InferenceController")
    controller_cls.create = AsyncMock(side_effect=build_controller)

    executor_handle = MagicMock(name="rollout_executor")
    executor_handle.get_num_rollout_per_epoch = AsyncMock(return_value=5)
    arg_snapshots: list[Namespace] = []

    def build_handle(args, *, providers) -> MagicMock:
        arg_snapshots.append(deepcopy(args))
        return executor_handle

    with patch("miles.ray.rollout.components.InferenceController", controller_cls), patch(
        "miles.ray.rollout.components.create_rollout_executor_handle", build_handle
    ):
        yield Namespace(controller=controller, executor_handle=executor_handle, arg_snapshots=arg_snapshots)


class TestCreateRolloutComponents:
    async def test_executor_is_built_after_the_router_address_is_known(self, fake_components):
        """Starting the engines fills the router address into args, which the executor is built from."""
        args = _make_args(num_rollout=1)

        await create_rollout_components(args, providers=FakeProviderFactory())

        (executor_args,) = fake_components.arg_snapshots
        assert executor_args.sglang_router_ip == "10.0.0.1"
        assert executor_args.sglang_router_port == 4321

    async def test_returns_a_plain_controller_and_a_worker_handle(self, fake_components):
        """The controller stays in the driver; the executor is only ever reached through its handle."""
        args = _make_args(num_rollout=1)

        controller, executor, _ = await create_rollout_components(args, providers=FakeProviderFactory())

        assert controller is fake_components.controller
        assert executor is fake_components.executor_handle

    async def test_num_rollout_derived_from_executor_epoch_length(self, fake_components):
        """num_rollout comes from the dataset, which the executor owns."""
        args = _make_args(num_rollout=None, num_epoch=2)

        _, _, num_rollout_per_epoch = await create_rollout_components(args, providers=FakeProviderFactory())

        fake_components.executor_handle.get_num_rollout_per_epoch.assert_awaited_once_with()
        assert num_rollout_per_epoch == 5
        assert args.num_rollout == 10

    async def test_num_rollout_left_alone_when_explicitly_set(self, fake_components):
        """An explicit --num-rollout skips asking the executor for the epoch length."""
        args = _make_args(num_rollout=3)

        _, _, num_rollout_per_epoch = await create_rollout_components(args, providers=FakeProviderFactory())

        fake_components.executor_handle.get_num_rollout_per_epoch.assert_not_awaited()
        assert num_rollout_per_epoch is None
        assert args.num_rollout == 3
