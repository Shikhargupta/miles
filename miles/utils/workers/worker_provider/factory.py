from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miles.utils.ft_utils.api_server.cell_operations import BaseCellOperations
    from miles.utils.workers.worker_provider.base import BaseWorkerProvider
    from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider


class ProviderFactory(abc.ABC):
    @abc.abstractmethod
    def cells(self, *, spec_names: Sequence[str]) -> BaseWorkerProvider: ...

    @abc.abstractmethod
    def static(self, *, worker_name: str) -> BaseWorkerProvider: ...

    @abc.abstractmethod
    def cell_operations(self) -> BaseCellOperations: ...


class KubernetesProviderFactory(ProviderFactory):
    def __init__(
        self,
        *,
        cells_provider: BaseWorkerProvider,
        cells_spec_names: Sequence[str],
        static_provider: SimpleWorkerProvider,
        cell_operations: BaseCellOperations,
    ) -> None:
        self._cells_provider = cells_provider
        self._cells_spec_names = list(cells_spec_names)
        self._static_provider = static_provider
        self._cell_operations = cell_operations

    def cells(self, *, spec_names: Sequence[str]) -> BaseWorkerProvider:
        unwatched = [name for name in spec_names if name not in self._cells_spec_names]
        assert not unwatched, (
            f"{unwatched} are not watched by this run's provider, which observes {self._cells_spec_names}; "
            f"their cells would never be reported"
        )
        return self._cells_provider

    def static(self, *, worker_name: str) -> BaseWorkerProvider:
        assert self._static_provider.knows_worker(
            worker_name
        ), f"worker {worker_name} is not in this run's static address book, so no address can be predicted for it"
        return self._static_provider

    def cell_operations(self) -> BaseCellOperations:
        return self._cell_operations


class RayProviderFactory(ProviderFactory):
    def __init__(self, *, worker_manager_handle: Any) -> None:
        self._worker_manager_handle = worker_manager_handle

    def cells(self, *, spec_names: Sequence[str]) -> BaseWorkerProvider:
        return self._create_provider()

    def static(self, *, worker_name: str) -> BaseWorkerProvider:
        return self._create_provider()

    def cell_operations(self) -> BaseCellOperations:
        from miles.utils.ft_utils.api_server.cell_operations import RayCellOperations

        return RayCellOperations(self._worker_manager_handle)

    def _create_provider(self) -> BaseWorkerProvider:
        from miles.utils.workers.worker_provider.ray import RayWorkerProvider

        return RayWorkerProvider(worker_manager_handle=self._worker_manager_handle)


class DeferredProviderFactory(ProviderFactory):
    def __init__(self, *, create: Callable[[], ProviderFactory]) -> None:
        self._create = create
        self._inner: ProviderFactory | None = None

    def cells(self, *, spec_names: Sequence[str]) -> BaseWorkerProvider:
        return self._resolve().cells(spec_names=spec_names)

    def static(self, *, worker_name: str) -> BaseWorkerProvider:
        return self._resolve().static(worker_name=worker_name)

    def cell_operations(self) -> BaseCellOperations:
        return self._resolve().cell_operations()

    def _resolve(self) -> ProviderFactory:
        if self._inner is None:
            self._inner = self._create()
        return self._inner
