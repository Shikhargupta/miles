# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.workers.reconcile.source_event import Delete, SourceEvent, SyncDone, SyncStart, Upsert

logger = logging.getLogger(__name__)

KeyMapFn = Callable[[Any], str]


@dataclass(frozen=True)
class StoreUpdate:
    affected: set[str]
    synced: bool = False


class ObjectStore:
    def __init__(self, *, key_map: KeyMapFn | None) -> None:
        self._key_map = key_map
        self._objects: dict[str, Any] = {}
        self._parents: dict[str, str] = {}
        self._open_segment: dict[str, Any] | None = None

    def get_by_parent(self, parent_key: str) -> list[Any]:
        return [self._objects[key] for key in sorted(self._parents) if self._parents[key] == parent_key]

    def parent_keys(self) -> set[str]:
        return set(self._parents.values())

    def __contains__(self, key: str) -> bool:
        return key in self._objects

    def reset_segment(self) -> None:
        self._open_segment = None

    def handle_event(self, event: SourceEvent) -> StoreUpdate:
        if isinstance(event, SyncStart):
            if self._open_segment is not None:
                raise RuntimeError("SyncStart while a LIST segment is still open")
            self._open_segment = {}
            return StoreUpdate(affected=set())
        if isinstance(event, Upsert):
            if self._open_segment is not None:
                self._open_segment[event.key] = event.obj
                return StoreUpdate(affected=set())
            parent = self._parent_key_or_none(key=event.key, obj=event.obj)
            if parent is None:
                return StoreUpdate(affected=self._apply_delete(key=event.key, last_obj=None))
            return StoreUpdate(affected=self._apply_upsert(key=event.key, obj=event.obj, parent=parent))
        if isinstance(event, Delete):
            if self._open_segment is not None:
                self._open_segment.pop(event.key, None)
                return StoreUpdate(affected=set())
            return StoreUpdate(affected=self._apply_delete(key=event.key, last_obj=event.last_obj))
        if isinstance(event, SyncDone):
            if self._open_segment is None:
                raise RuntimeError("SyncDone must terminate a LIST opened by SyncStart")
            affected = self._replace(self._open_segment)
            self._open_segment = None
            return StoreUpdate(affected=affected, synced=True)
        raise AssertionError(f"Unknown source event {event=}")

    def _replace(self, listed: dict[str, Any]) -> set[str]:
        parents = {key: self._parent_key_or_none(key=key, obj=obj) for key, obj in listed.items()}
        mapped = {key: obj for key, obj in listed.items() if parents[key] is not None}

        affected: set[str] = set()
        for key, obj in mapped.items():
            affected |= self._apply_upsert(key=key, obj=obj, parent=parents[key])
        for key in [key for key in self._objects if key not in mapped]:
            affected |= self._apply_delete(key=key, last_obj=None)
        return affected

    def _apply_upsert(self, *, key: str, obj: Any, parent: str) -> set[str]:
        new_parent = parent
        old_parent = self._parents.get(key)

        self._objects[key] = obj
        self._parents[key] = new_parent

        return {new_parent} if old_parent is None else {new_parent, old_parent}

    def _apply_delete(self, *, key: str, last_obj: Any) -> set[str]:
        self._objects.pop(key, None)
        parent = self._parents.pop(key, None)
        if parent is None:
            if last_obj is None:
                logger.warning(f"ObjectStore dropping a delete it cannot attribute to a parent {key=}")
                return set()
            parent = self._parent_key_or_none(key=key, obj=last_obj)
            if parent is None:
                return set()
        return {parent}

    def _parent_key_or_none(self, *, key: str, obj: Any) -> str | None:
        try:
            return key if self._key_map is None else self._key_map(obj)
        except Exception:
            logger.error(f"ObjectStore dropping an object whose key_map failed {key=}", exc_info=True)
            return None
