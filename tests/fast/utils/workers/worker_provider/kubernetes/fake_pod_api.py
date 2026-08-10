from __future__ import annotations

from typing import Any

_INSTALLED: list[Any] = []
CLOSE_CALLS: list[Any] = []


def install(api: object) -> object:
    _INSTALLED.append(api)
    return api


def installed() -> Any:
    assert _INSTALLED, "no fake pod api was installed, so this test would talk to a real cluster"
    from miles.utils.workers.worker_provider.kubernetes.core.provider import _KubernetesClient

    api = _INSTALLED[-1]

    async def close() -> None:
        CLOSE_CALLS.append(api)

    return _KubernetesClient(api=api, close=close)


def reset() -> None:
    _INSTALLED.clear()
    CLOSE_CALLS.clear()
