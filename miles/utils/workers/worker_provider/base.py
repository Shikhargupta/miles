import abc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

ReconcileFn = Callable[[str, "CellInfo | None"], Awaitable[None]]
StopWatchFn = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class CellMember:
    handle: Any
    payload: dict
    placement: Any


@dataclass(frozen=True)
class CellInfo:
    cell_id: str
    members: list[CellMember]


class BaseWorkerProvider(abc.ABC):
    """Read-only view of the workers some infrastructure layer runs.

    Consumers observe cells through this; commanding a cell to start or stop
    goes to the infrastructure layer directly, never through here.
    """

    @abc.abstractmethod
    async def list_cells(self) -> dict[str, CellInfo]: ...

    @abc.abstractmethod
    async def watch_cells(self, reconcile: ReconcileFn, seen: dict[str, "CellInfo"] | None = None) -> StopWatchFn:
        """Call ``reconcile`` on every change; ``seen`` is the snapshot the caller already handled."""
