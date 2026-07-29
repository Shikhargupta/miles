# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Upsert:
    key: str
    obj: Any


@dataclass(frozen=True)
class Delete:
    key: str
    last_obj: Any


@dataclass(frozen=True)
class SyncStart:
    pass


@dataclass(frozen=True)
class SyncDone:
    pass


SourceEvent = Upsert | Delete | SyncStart | SyncDone

SourceWatchFn = Callable[[], AsyncGenerator[SourceEvent, None]]
