"""Factory contract for the role-separated rollout construction
(codex-rollout-fullparameter-design-0810 §4.3/§4.8/§8.2): the factory unpacks
(rollout_manager, num_rollout_per_epoch), returns two DISTINCT role objects
sharing one legacy handle, the bundle disposes exactly once, and
future-shaped fakes can replace the factory without changing driver call
sites."""

from types import SimpleNamespace

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu")

import asyncio

import miles.ray.rollout.components as components_module
from miles.ray.rollout.components import InferenceEndpoint, RolloutComponents, create_rollout_components


class Remote:
    def __init__(self, log, name, value=None):
        self._log, self._name, self._value = log, name, value

    async def remote(self, *args):
        self._log.append((self._name, args))
        return self._value


def make_fake_manager(log):
    return SimpleNamespace(
        get_router_address=Remote(log, "get_router_address", ("10.0.0.7", 30001)),
        generate=Remote(log, "generate", {"batch": 1}),
        dispose=Remote(log, "dispose"),
    )


def build(monkeypatch, log):
    manager = make_fake_manager(log)
    monkeypatch.setattr(
        "miles.ray.placement_group.create_rollout_manager", lambda args, pg: (manager, 7), raising=True
    )
    components = create_rollout_components(SimpleNamespace(), pg=None)
    return components, manager


def test_factory_builds_two_role_views_over_one_legacy_handle(monkeypatch):
    log: list = []
    components, manager = build(monkeypatch, log)

    assert components.num_rollout_per_epoch == 7
    assert components.inference_controller is not components.rollout_executor
    # Both roles wrap the SAME combined actor today.
    assert components.inference_controller.manager is manager
    assert components.rollout_executor._manager is manager

    endpoint = asyncio.run(components.inference_controller.get_inference_endpoint())
    assert endpoint == InferenceEndpoint(host="10.0.0.7", port=30001)
    assert endpoint.base_url == "http://10.0.0.7:30001"

    assert asyncio.run(components.rollout_executor.generate(3)) == {"batch": 1}
    assert ("generate", (3,)) in log


def test_bundle_disposes_the_shared_actor_exactly_once(monkeypatch):
    log: list = []
    components, _ = build(monkeypatch, log)
    asyncio.run(components.dispose())
    asyncio.run(components.dispose())  # second call must be a no-op
    assert [name for name, _ in log].count("dispose") == 1


def test_future_shaped_fakes_satisfy_the_bundle_without_the_factory():
    """A split-world construction (separate controller/executor objects) fits
    the same bundle: driver call sites depend only on the role surface."""

    class FakeController:
        async def get_inference_endpoint(self):
            return InferenceEndpoint(host="h", port=1)

    class FakeExecutor:
        async def generate(self, rollout_id):
            return rollout_id

    class FakeLifecycle:
        def __init__(self):
            self.disposed = 0

        async def dispose_once(self):
            self.disposed += 1

    lifecycle = FakeLifecycle()
    components = RolloutComponents(
        inference_controller=FakeController(),
        rollout_executor=FakeExecutor(),
        lifecycle=lifecycle,
        num_rollout_per_epoch=None,
    )
    assert asyncio.run(components.rollout_executor.generate(5)) == 5
    asyncio.run(components.dispose())
    assert lifecycle.disposed == 1


def test_module_never_imports_ray_directly():
    # The construction seam isolates Ray invocation shapes behind adapters.
    import inspect

    source = inspect.getsource(components_module)
    assert "import ray" not in source
