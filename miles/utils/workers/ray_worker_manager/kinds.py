import abc
import asyncio
from collections.abc import Callable
from typing import Any

import ray

from miles.utils.misc import NodeProbeMixin, load_function
from miles.utils.workers.command_actor import CommandActor
from miles.utils.workers.ray_worker_manager.addressing import WorkerAddressing
from miles.utils.workers.ray_worker_manager.placement import SpecPlacement
from miles.utils.workers.ray_worker_manager.resources import ActorOptions
from miles.utils.workers.ray_worker_manager.state import CellState, WorkerState
from miles.utils.workers.worker_spec import BaseWorkerSpec, CommandWorkerSpec, ServeWorkerSpec


class WorkerKind(abc.ABC):
    @abc.abstractmethod
    def create_actor(
        self,
        *,
        cell: CellState,
        worker_index: int,
        name: str,
        env_vars: dict[str, str],
        options: ActorOptions,
        placement: SpecPlacement | None,
    ) -> ray.actor.ActorHandle: ...

    @abc.abstractmethod
    async def activate_workers(
        self,
        *,
        workers: list[WorkerState],
        addressings: dict[str, WorkerAddressing],
        env_vars: dict[str, str],
    ) -> None: ...


class ServeWorkerKind(WorkerKind):
    def __init__(self) -> None:
        self._actor_classes: dict[str, Any] = {}

    def create_actor(
        self,
        *,
        cell: CellState,
        worker_index: int,
        name: str,
        env_vars: dict[str, str],
        options: ActorOptions,
        placement: SpecPlacement | None,
    ) -> ray.actor.ActorHandle:
        spec = cell.spec
        assert isinstance(spec, ServeWorkerSpec)
        return (
            self._actor_class(spec=spec, placement=placement)
            .options(
                name=name,
                num_cpus=options.num_cpus,
                num_gpus=options.num_gpus,
                max_restarts=0,
                scheduling_strategy=options.scheduling_strategy,
                runtime_env={"env_vars": env_vars},
            )
            .remote(ctor_kwargs_fn=spec.ctor_kwargs, cell_index=cell.cell_index, worker_index=worker_index)
        )

    async def activate_workers(
        self,
        *,
        workers: list[WorkerState],
        addressings: dict[str, WorkerAddressing],
        env_vars: dict[str, str],
    ) -> None:
        spec = workers[0].cell.spec
        if not spec.port_infos:
            return
        await asyncio.gather(
            *[
                worker.actor.configure_addrs_and_ports.remote(**addressings[worker.name].addr_port_kwargs)
                for worker in workers
            ]
        )

    def _actor_class(self, *, spec: ServeWorkerSpec, placement: SpecPlacement | None) -> Any:
        if spec.name not in self._actor_classes:
            wrapped_cls = _make_wrapped_worker_cls(load_function(spec.worker_class))
            remote_kwargs: dict[str, Any] = {}
            if placement is not None and placement.concurrency_groups is not None:
                remote_kwargs["concurrency_groups"] = placement.concurrency_groups
            self._actor_classes[spec.name] = (
                ray.remote(**remote_kwargs)(wrapped_cls) if remote_kwargs else ray.remote(wrapped_cls)
            )
        return self._actor_classes[spec.name]


class CommandWorkerKind(WorkerKind):
    def create_actor(
        self,
        *,
        cell: CellState,
        worker_index: int,
        name: str,
        env_vars: dict[str, str],
        options: ActorOptions,
        placement: SpecPlacement | None,
    ) -> ray.actor.ActorHandle:
        return (
            ray.remote(CommandActor)
            .options(
                name=name,
                num_cpus=options.num_cpus,
                num_gpus=options.num_gpus,
                max_restarts=0,
                scheduling_strategy=options.scheduling_strategy,
            )
            .remote()
        )

    async def activate_workers(
        self,
        *,
        workers: list[WorkerState],
        addressings: dict[str, WorkerAddressing],
        env_vars: dict[str, str],
    ) -> None:
        spec = workers[0].cell.spec
        assert isinstance(spec, CommandWorkerSpec)
        for worker in workers:
            command = spec.launch_command.format(**addressings[worker.name].addr_port_kwargs)
            worker.actor.run.remote(cmd=command, envs=env_vars)


def make_worker_kinds() -> dict[type[BaseWorkerSpec], WorkerKind]:
    return {ServeWorkerSpec: ServeWorkerKind(), CommandWorkerSpec: CommandWorkerKind()}


def _make_wrapped_worker_cls(worker_cls: type) -> type:
    class _WrappedWorker(worker_cls, NodeProbeMixin):
        def __init__(
            self, *, ctor_kwargs_fn: Callable[[int, int], dict[str, Any]], cell_index: int, worker_index: int
        ) -> None:
            super().__init__(**ctor_kwargs_fn(cell_index, worker_index))

    _WrappedWorker.__name__ = worker_cls.__name__
    _WrappedWorker.__qualname__ = worker_cls.__qualname__
    return _WrappedWorker
