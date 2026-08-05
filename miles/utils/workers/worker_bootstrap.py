from __future__ import annotations

from typing import Any

from miles.utils.function_registry import load_function
from miles.utils.workers.worker_spec import WorkerCtorContext, WorkerLaunchContext

CTOR_KWARGS_FN = "miles.ray.specs.bootstrap.compute_ctor_kwargs"


def bootstrapped_worker_class(worker_class: type) -> type:
    class BootstrappedWorker(worker_class):
        def __init__(
            self,
            *,
            spec_name: str,
            worker_argv: list[str],
            cell_index: int,
            worker_in_cell_index: int,
            gpu_ids: list[int],
        ) -> None:
            super().__init__(
                **_compute_worker_ctor_kwargs(
                    spec_name=spec_name,
                    worker_argv=worker_argv,
                    cell_index=cell_index,
                    worker_in_cell_index=worker_in_cell_index,
                    gpu_ids=gpu_ids,
                )
            )

    BootstrappedWorker.__name__ = worker_class.__name__
    BootstrappedWorker.__qualname__ = worker_class.__qualname__
    BootstrappedWorker.__module__ = worker_class.__module__
    return BootstrappedWorker


def worker_bootstrap_kwargs(*, spec_name: str, worker_argv: list[str], context: WorkerLaunchContext) -> dict[str, Any]:
    return dict(
        spec_name=spec_name,
        worker_argv=list(worker_argv),
        cell_index=context.cell_index,
        worker_in_cell_index=context.worker_in_cell_index,
        gpu_ids=list(context.gpu_ids),
    )


def _compute_worker_ctor_kwargs(
    *,
    spec_name: str,
    worker_argv: list[str],
    cell_index: int,
    worker_in_cell_index: int,
    gpu_ids: list[int],
) -> dict[str, Any]:
    from miles.ray.wiring import create_worker_provider_factory

    context = WorkerCtorContext(
        cell_index=cell_index,
        worker_in_cell_index=worker_in_cell_index,
        gpu_ids=gpu_ids,
        providers=create_worker_provider_factory(worker_argv=worker_argv),
    )
    return load_function(CTOR_KWARGS_FN)(spec_name=spec_name, worker_argv=worker_argv, context=context)
