# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.workers.reconcile.source_event import (
    DeleteEvent,
    ObjectKey,
    ParentKey,
    ReplaceEvent,
    SourceEvent,
    UpsertEvent,
)

logger = logging.getLogger(__name__)

KeyMapFn = Callable[[Any], ParentKey]


@dataclass(frozen=True)
class StoreUpdate:
    affected_parents: set[ParentKey]


@dataclass(frozen=True)
class _CachedObject:
    obj: Any
    parent: ParentKey


class ObjectStore:
    def __init__(self, *, key_map: KeyMapFn | None) -> None:
        self._key_map = key_map
        self._cache: dict[ObjectKey, _CachedObject] = {}
        self._handler_of_event: dict[type[SourceEvent], Callable[[Any], StoreUpdate]] = {
            ReplaceEvent: self._handle_replace,
            UpsertEvent: self._handle_upsert,
            DeleteEvent: self._handle_delete,
        }

    def get_by_parent(self, parent_key: ParentKey) -> list[Any]:
        return [self._cache[key].obj for key in sorted(self._cache) if self._cache[key].parent == parent_key]

    def parent_keys(self) -> set[ParentKey]:
        return {entry.parent for entry in self._cache.values()}

    def __contains__(self, key: ObjectKey) -> bool:
        return key in self._cache

    def handle_event(self, event: SourceEvent) -> StoreUpdate:
        handler = self._handler_of_event.get(type(event))
        assert handler is not None, f"Unknown source event {event=}"
        return handler(event)

    def _handle_replace(self, event: ReplaceEvent) -> StoreUpdate:
        parents = {key: self._parent_key_or_none(key=key, obj=obj) for key, obj in event.objects.items()}
        mapped = {key: obj for key, obj in event.objects.items() if parents[key] is not None}

        affected_parents: set[ParentKey] = set()
        for key, obj in mapped.items():
            affected_parents |= self._apply_upsert(key=key, obj=obj, parent=parents[key]).affected_parents
        for key in [key for key in self._cache if key not in mapped]:
            affected_parents |= self._apply_delete(key=key, last_obj=None).affected_parents
        return StoreUpdate(affected_parents=affected_parents)

    def _handle_upsert(self, event: UpsertEvent) -> StoreUpdate:
        parent = self._parent_key_or_none(key=event.key, obj=event.obj)
        if parent is None:
            return self._apply_delete(key=event.key, last_obj=None)
        return self._apply_upsert(key=event.key, obj=event.obj, parent=parent)

    def _handle_delete(self, event: DeleteEvent) -> StoreUpdate:
        return self._apply_delete(key=event.key, last_obj=event.last_obj)

    def _apply_upsert(self, *, key: ObjectKey, obj: Any, parent: ParentKey) -> StoreUpdate:
        previous = self._cache.get(key)
        self._cache[key] = _CachedObject(obj=obj, parent=parent)
        return StoreUpdate(affected_parents={parent} if previous is None else {parent, previous.parent})

    def _apply_delete(self, *, key: ObjectKey, last_obj: Any) -> StoreUpdate:
        removed = self._cache.pop(key, None)
        parent = removed.parent if removed is not None else None
        if parent is None:
            if last_obj is None:
                logger.warning(f"ObjectStore dropping a delete it cannot attribute to a parent {key=}")
                return StoreUpdate(affected_parents=set())
            parent = self._parent_key_or_none(key=key, obj=last_obj)
            if parent is None:
                return StoreUpdate(affected_parents=set())
        return StoreUpdate(affected_parents={parent})

    def _parent_key_or_none(self, *, key: ObjectKey, obj: Any) -> ParentKey | None:
        try:
            return key if self._key_map is None else self._key_map(obj)
        except Exception:
            logger.error(f"ObjectStore dropping an object whose key_map failed {key=}", exc_info=True)
            return None
