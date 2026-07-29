import pytest

from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle


class _FakeActorMethod:
    def __init__(self, result) -> None:
        self._result = result
        self.calls: list[dict] = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)

        async def resolve():
            return self._result

        return resolve()


class _FakeActor:
    def __init__(self) -> None:
        self.do_work = _FakeActorMethod(result="work-done")


class TestRayWorkerHandle:
    def test_is_a_worker_handle(self):
        """The ray handle satisfies the shared handle contract."""
        assert isinstance(RayWorkerHandle(_FakeActor()), BaseWorkerHandle)

    async def test_forwards_method_calls_with_kwargs(self):
        """A method call awaits the underlying actor method with the given kwargs."""
        actor = _FakeActor()
        handle = RayWorkerHandle(actor)

        result = await handle.do_work(x=1, y="z")

        assert result == "work-done"
        assert actor.do_work.calls == [{"x": 1, "y": "z"}]

    def test_rejects_private_attribute_forwarding(self):
        """Underscore attributes are not forwarded to the actor."""
        with pytest.raises(AttributeError):
            _ = RayWorkerHandle(_FakeActor())._nonexistent

    async def test_wait_ready_is_a_noop(self):
        """wait_ready returns immediately because a resolvable named actor is already initialized."""
        assert await RayWorkerHandle(_FakeActor()).wait_ready(timeout=1.0) is None
