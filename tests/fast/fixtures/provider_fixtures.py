from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.factory import ProviderFactory


class FakeProviderFactory(ProviderFactory):
    def __init__(
        self,
        *,
        cells_provider: Any = None,
        static_provider: Any = None,
        cell_operations: Any = None,
    ) -> None:
        self.cells_provider = cells_provider
        self.static_provider = static_provider
        self.operations = cell_operations
        self.requested_spec_names: list[list[str]] = []
        self.requested_worker_names: list[str] = []

    def cells(self, *, spec_names: Sequence[str]) -> BaseWorkerProvider:
        self.requested_spec_names.append(list(spec_names))
        assert self.cells_provider is not None, "this factory was built without a cells provider"
        return self.cells_provider

    def static(self, *, worker_name: str) -> BaseWorkerProvider:
        self.requested_worker_names.append(worker_name)
        assert self.static_provider is not None, "this factory was built without a static provider"
        return self.static_provider

    def cell_operations(self) -> Any:
        assert self.operations is not None, "this factory was built without cell operations"
        return self.operations
