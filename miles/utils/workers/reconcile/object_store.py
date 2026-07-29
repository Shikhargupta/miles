# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.workers.reconcile.source_event import (
    Delete,
    ObjectKey,
    ParentKey,
    SourceEvent,
    SyncDone,
    SyncStart,
    Upsert,
)

logger = logging.getLogger(__name__)

KeyMapFn = Callable[[Any], ParentKey]


@dataclass(frozen=True)
class StoreUpdate:
    affected: set[ParentKey]


@dataclass(frozen=True)
class _CachedObject:
    obj: Any
    parent: ParentKey


class ObjectStore:
    def __init__(self, *, key_map: KeyMapFn | None) -> None:
        self._key_map = key_map
        self._cache: dict[ObjectKey, _CachedObject] = {}
        self._open_segment: dict[ObjectKey, Any] | None = None
        self._handler_of_event: dict[type[SourceEvent], Callable[[Any], StoreUpdate]] = {
            SyncStart: self._handle_sync_start,
            Upsert: self._handle_upsert,
            Delete: self._handle_delete,
            SyncDone: self._handle_sync_done,
        }

    def get_by_parent(self, parent_key: ParentKey) -> list[Any]:
        return [self._cache[key].obj for key in sorted(self._cache) if self._cache[key].parent == parent_key]

    def parent_keys(self) -> set[ParentKey]:
        return {entry.parent for entry in self._cache.values()}

    def __contains__(self, key: ObjectKey) -> bool:
        return key in self._cache

    def reset_segment(self) -> None:
        self._open_segment = None

    def handle_event(self, event: SourceEvent) -> StoreUpdate:
        handler = self._handler_of_event.get(type(event))
        assert handler is not None, f"Unknown source event {event=}"
        return handler(event)

    def _handle_sync_start(self, event: SyncStart) -> StoreUpdate:
        if self._open_segment is not None:
            raise RuntimeError("SyncStart while a LIST segment is still open")
        self._open_segment = {}
        return StoreUpdate(affected=set())

    def _handle_upsert(self, event: Upsert) -> StoreUpdate:
        if self._open_segment is not None:
            self._open_segment[event.key] = event.obj
            return StoreUpdate(affected=set())
        parent = self._parent_key_or_none(key=event.key, obj=event.obj)
        if parent is None:
            return self._apply_delete(key=event.key, last_obj=None)
        return self._apply_upsert(key=event.key, obj=event.obj, parent=parent)

    def _handle_delete(self, event: Delete) -> StoreUpdate:
        if self._open_segment is not None:
            self._open_segment.pop(event.key, None)
            return StoreUpdate(affected=set())
        return self._apply_delete(key=event.key, last_obj=event.last_obj)

    def _handle_sync_done(self, event: SyncDone) -> StoreUpdate:
        if self._open_segment is None:
            raise RuntimeError("SyncDone must terminate a LIST opened by SyncStart")
        update = self._replace(self._open_segment)
        self._open_segment = None
        return update

    def _replace(self, listed: dict[ObjectKey, Any]) -> StoreUpdate:
        parents = {key: self._parent_key_or_none(key=key, obj=obj) for key, obj in listed.items()}
        mapped = {key: obj for key, obj in listed.items() if parents[key] is not None}

        affected: set[ParentKey] = set()
        for key, obj in mapped.items():
            affected |= self._apply_upsert(key=key, obj=obj, parent=parents[key]).affected
        for key in [key for key in self._cache if key not in mapped]:
            affected |= self._apply_delete(key=key, last_obj=None).affected
        return StoreUpdate(affected=affected)

    def _apply_upsert(self, *, key: ObjectKey, obj: Any, parent: ParentKey) -> StoreUpdate:
        previous = self._cache.get(key)
        self._cache[key] = _CachedObject(obj=obj, parent=parent)
        return StoreUpdate(affected={parent} if previous is None else {parent, previous.parent})

    def _apply_delete(self, *, key: ObjectKey, last_obj: Any) -> StoreUpdate:
        removed = self._cache.pop(key, None)
        parent = removed.parent if removed is not None else None
        if parent is None:
            if last_obj is None:
                logger.warning(f"ObjectStore dropping a delete it cannot attribute to a parent {key=}")
                return StoreUpdate(affected=set())
            parent = self._parent_key_or_none(key=key, obj=last_obj)
            if parent is None:
                return StoreUpdate(affected=set())
        return StoreUpdate(affected={parent})

    def _parent_key_or_none(self, *, key: ObjectKey, obj: Any) -> ParentKey | None:
        try:
            return key if self._key_map is None else self._key_map(obj)
        except Exception:
            logger.error(f"ObjectStore dropping an object whose key_map failed {key=}", exc_info=True)
            return None
