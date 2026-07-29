# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from miles.utils.test_utils.clock import Clock, RealClock
from miles.utils.workers.reconcile.k8s_api import (
    EVENT_TYPE_ADDED,
    EVENT_TYPE_BOOKMARK,
    EVENT_TYPE_DELETED,
    EVENT_TYPE_ERROR,
    EVENT_TYPE_MODIFIED,
    KubernetesPodApi,
    PodWatchEvent,
)
from miles.utils.workers.reconcile.source_event import Delete, SourceEvent, SyncDone, SyncStart, Upsert

logger = logging.getLogger(__name__)

_CURSOR_INVALID_STATUS_CODES = (410, 504)
_CURSOR_INVALID_REASONS = ("Expired", "ResourceVersionTooLarge")


class KubernetesReflector:
    def __init__(
        self,
        *,
        kube_client: KubernetesPodApi,
        namespace: str,
        label_selector: str,
        watch_timeout_seconds: int = 300,
        retry_delay: float = 1.0,
        clock: Clock | None = None,
    ) -> None:
        assert retry_delay > 0, f"{retry_delay=} must be positive"
        assert watch_timeout_seconds > 0, f"{watch_timeout_seconds=} must be positive"

        self._kube_client = kube_client
        self._namespace = namespace
        self._label_selector = label_selector
        self._watch_timeout_seconds = watch_timeout_seconds
        self._retry_delay = retry_delay
        self._clock = clock or RealClock()

    async def watch(self) -> AsyncGenerator[SourceEvent, None]:
        cursor = _WatchCursor()
        while True:
            try:
                async for event in self._watch_once(cursor):
                    yield event
                await self._clock.sleep(self._retry_delay)
            except asyncio.CancelledError:
                raise
            except Exception as exception:
                if _is_cursor_invalid(exception):
                    logger.warning(f"KubernetesReflector cursor is no longer usable, relisting {cursor=}")
                    cursor.resource_version = None
                else:
                    logger.error("KubernetesReflector stream failed, retrying", exc_info=True)
                await self._clock.sleep(self._retry_delay)

    async def _watch_once(self, cursor: _WatchCursor) -> AsyncGenerator[SourceEvent, None]:
        if cursor.resource_version is None:
            page = await self._kube_client.list_pods(namespace=self._namespace, label_selector=self._label_selector)
            upserts = [Upsert(key=_pod_key(pod), obj=pod) for pod in page.pods]
            yield SyncStart()
            for upsert in upserts:
                yield upsert
            yield SyncDone()
            cursor.resource_version = page.resource_version

        async with aclosing(
            self._kube_client.stream_pods(
                namespace=self._namespace,
                label_selector=self._label_selector,
                resource_version=cursor.resource_version,
                timeout_seconds=self._watch_timeout_seconds,
            )
        ) as stream:
            async for raw_event in stream:
                if raw_event.type == EVENT_TYPE_ERROR:
                    if not _is_cursor_invalid(raw_event.obj):
                        raise RuntimeError(f"KubernetesReflector received error event {raw_event=}")
                    logger.warning(f"KubernetesReflector received a cursor error event, relisting {raw_event=}")
                    cursor.resource_version = None
                    return

                event = _to_source_event(raw_event)
                cursor.resource_version = _resource_version_of(raw_event.obj) or cursor.resource_version
                if event is not None:
                    yield event


@dataclass
class _WatchCursor:
    resource_version: str | None = None


def _to_source_event(raw_event: PodWatchEvent) -> SourceEvent | None:
    if raw_event.type in (EVENT_TYPE_ADDED, EVENT_TYPE_MODIFIED, EVENT_TYPE_DELETED):
        key = _pod_key_or_none(raw_event.obj)
        if key is None:
            return None
        if raw_event.type == EVENT_TYPE_DELETED:
            return Delete(key=key, last_obj=raw_event.obj)
        return Upsert(key=key, obj=raw_event.obj)
    if raw_event.type != EVENT_TYPE_BOOKMARK:
        logger.warning(f"KubernetesReflector ignoring unknown event {raw_event.type=}")
    return None


def _pod_key_or_none(obj: Any) -> str | None:
    try:
        return _pod_key(obj)
    except Exception:
        logger.error(f"KubernetesReflector skipping a watch event whose key cannot be read {obj=}", exc_info=True)
        return None


def _pod_key(pod: Any) -> str:
    return pod.metadata.name


def _resource_version_of(obj: Any) -> str | None:
    if isinstance(obj, dict):
        metadata = obj.get("metadata")
        if not isinstance(metadata, dict):
            return None
        return metadata.get("resourceVersion") or metadata.get("resource_version")
    metadata = getattr(obj, "metadata", None)
    if metadata is None:
        return None
    return getattr(metadata, "resource_version", None)


def _is_cursor_invalid(candidate: Any) -> bool:
    if isinstance(candidate, dict):
        return (
            candidate.get("code") in _CURSOR_INVALID_STATUS_CODES or candidate.get("reason") in _CURSOR_INVALID_REASONS
        )
    status = getattr(candidate, "status", None)
    if status in _CURSOR_INVALID_STATUS_CODES or status in {str(code) for code in _CURSOR_INVALID_STATUS_CODES}:
        return True
    return getattr(candidate, "code", None) in _CURSOR_INVALID_STATUS_CODES
