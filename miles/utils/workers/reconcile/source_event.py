# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any


ObjectKey = str
ParentKey = str


@dataclass(frozen=True)
class Upsert:
    key: ObjectKey
    obj: Any


@dataclass(frozen=True)
class Delete:
    key: ObjectKey
    last_obj: Any


@dataclass(frozen=True)
class Replace:
    objects: dict[ObjectKey, Any]


SourceEvent = Upsert | Delete | Replace

SourceWatchFn = Callable[[], AsyncGenerator[SourceEvent, None]]
