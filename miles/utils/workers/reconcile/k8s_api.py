# doc-dev: docs/developer/reconcile-loop.md
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol

from miles.utils.pydantic_utils import FrozenStrictBaseModel

logger = logging.getLogger(__name__)

EVENT_TYPE_ADDED = "ADDED"
EVENT_TYPE_MODIFIED = "MODIFIED"
EVENT_TYPE_DELETED = "DELETED"
EVENT_TYPE_BOOKMARK = "BOOKMARK"
EVENT_TYPE_ERROR = "ERROR"


class PodListPage(FrozenStrictBaseModel):
    pods: list[Any]
    resource_version: str


class PodWatchEvent(FrozenStrictBaseModel):
    type: str
    obj: Any


class KubernetesPodApi(Protocol):
    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage: ...

    def stream_pods(
        self, *, namespace: str, label_selector: str, resource_version: str, timeout_seconds: int
    ) -> AsyncIterator[PodWatchEvent]: ...


class KubernetesAsyncioPodApi:
    def __init__(self, *, core_v1_api: Any) -> None:
        self._core_v1_api = core_v1_api

    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage:
        pod_list = await self._core_v1_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
        return PodListPage(pods=list(pod_list.items), resource_version=pod_list.metadata.resource_version)

    async def stream_pods(
        self, *, namespace: str, label_selector: str, resource_version: str, timeout_seconds: int
    ) -> AsyncIterator[PodWatchEvent]:
        from kubernetes_asyncio import watch as kubernetes_watch

        watcher = kubernetes_watch.Watch()
        try:
            async for event in watcher.stream(
                self._core_v1_api.list_namespaced_pod,
                namespace=namespace,
                label_selector=label_selector,
                resource_version=resource_version,
                timeout_seconds=timeout_seconds,
                allow_watch_bookmarks=True,
            ):
                yield PodWatchEvent(type=event["type"], obj=event["object"])
        finally:
            await close_quietly(watcher.close())


async def close_quietly(closing: Any) -> None:
    try:
        await closing
    except Exception:
        logger.error("failed to close a Kubernetes watch stream", exc_info=True)
