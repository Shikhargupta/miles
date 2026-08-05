from __future__ import annotations

import abc
from typing import Any


class RolloutExecutorApi(abc.ABC):
    @abc.abstractmethod
    def dispose(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def get(self, rollout_id: int) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    async def eval(self, rollout_id: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, rollout_id: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, rollout_id: int | None = None) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_rollout_per_epoch(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def set_train_parallel_config(self, config: dict[str, Any]) -> None:
        raise NotImplementedError
