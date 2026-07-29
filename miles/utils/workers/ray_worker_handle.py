from collections.abc import Awaitable, Callable
from typing import Any

import ray

from miles.utils.workers.worker_handle import BaseWorkerHandle


class RayWorkerHandle(BaseWorkerHandle):
    def __init__(self, actor: ray.actor.ActorHandle) -> None:
        self._actor = actor

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        if name.startswith("_"):
            raise AttributeError(name)
        method = getattr(self._actor, name)

        async def call(**kwargs: Any) -> Any:
            return await method.remote(**kwargs)

        return call

    async def wait_ready(self, *, timeout: float) -> None:
        pass
