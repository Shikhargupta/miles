from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from tests.fast.utils.workers.fake_ray import FakeRayCluster

from miles.utils.function_registry import function_registry
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_bootstrap import CTOR_KWARGS_FN, bootstrapped_worker_class, worker_bootstrap_kwargs
from miles.utils.workers.worker_spec import (
    PortInfo,
    SchedulingSpec,
    ServeWorkerSpec,
    WorkerCtorContext,
    WorkerLaunchContext,
)

pytestmark = pytest.mark.asyncio

WORKER_ARGV = ["--cluster-backend", "ray", "--rollout-num-gpus", "8"]
SPEC_NAME = "trainer-actor"


class DemoWorker:
    def __init__(self, *, rank: int, role: str) -> None:
        self.rank = rank
        self.role = role


_WORKER_CLASS_PATH = f"{DemoWorker.__module__}.{DemoWorker.__qualname__}"


@dataclass
class _CtorKwargsProbe:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, *, spec_name: str, worker_argv: list[str], context: WorkerCtorContext) -> dict[str, Any]:
        self.calls.append(dict(spec_name=spec_name, worker_argv=worker_argv, context=context, pid=os.getpid()))
        return dict(rank=context.worker_in_cell_index, role="actor")


@pytest.fixture
def ctor_kwargs_probe():
    probe = _CtorKwargsProbe()
    with function_registry.temporary(CTOR_KWARGS_FN, probe):
        yield probe


def _make_spec(*, num_workers_per_cell: int = 1, ctor_kwargs=None) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name=SPEC_NAME,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master", allow_dynamic=True)],
        env_var=lambda _ctx: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=0,
        ),
        worker_class=_WORKER_CLASS_PATH,
        ctor_kwargs=ctor_kwargs if ctor_kwargs is not None else (lambda _ctx: {}),
    )


def _launch_context(*, worker_in_cell_index: int = 0) -> WorkerLaunchContext:
    return WorkerLaunchContext(cell_index=0, worker_in_cell_index=worker_in_cell_index, gpu_ids=[])


class TestTheManagerShipsAnIdentityRatherThanCtorKwargs:
    async def test_a_poisoned_spec_still_launches_because_nothing_evaluates_it_here(
        self, fake_ray_cluster: FakeRayCluster, ctor_kwargs_probe: _CtorKwargsProbe
    ):
        """A provider cannot be pickled to the actor, so the manager must never build one."""

        def explode(_ctx) -> dict:
            raise AssertionError("the manager evaluated the spec's ctor kwargs")

        manager = RayWorkerManager(worker_argv=WORKER_ARGV)
        await manager.init([_make_spec(ctor_kwargs=explode)], {})

        assert len(fake_ray_cluster.handles) == 1
        assert ctor_kwargs_probe.calls == []

    async def test_the_shipped_kwargs_are_only_the_run_and_the_rank(
        self, fake_ray_cluster: FakeRayCluster, ctor_kwargs_probe: _CtorKwargsProbe
    ):
        """Everything in this dict crosses a process boundary, so it may hold nothing live."""
        manager = RayWorkerManager(worker_argv=WORKER_ARGV)
        await manager.init([_make_spec()], {})

        assert fake_ray_cluster.ctor_kwargs == [
            dict(
                spec_name=SPEC_NAME,
                worker_argv=WORKER_ARGV,
                cell_index=0,
                worker_in_cell_index=0,
                gpu_ids=[],
            )
        ]

    async def test_the_actor_class_is_the_spec_s_worker_class_with_a_bootstrapping_constructor(
        self, fake_ray_cluster: FakeRayCluster, ctor_kwargs_probe: _CtorKwargsProbe
    ):
        """Callers reach the worker's own methods on the handle, so the actor has to be that class."""
        manager = RayWorkerManager(worker_argv=WORKER_ARGV)
        await manager.init([_make_spec()], {})

        actor_class = fake_ray_cluster.handles[0].actor_class
        assert issubclass(actor_class, DemoWorker)
        assert actor_class is not DemoWorker

    async def test_the_worker_is_built_only_when_the_actor_process_constructs_it(
        self, fake_ray_cluster: FakeRayCluster, ctor_kwargs_probe: _CtorKwargsProbe
    ):
        """This is the whole point: ctor kwargs are computed where the worker lives, not where it is launched."""
        manager = RayWorkerManager(worker_argv=WORKER_ARGV)
        await manager.init([_make_spec(num_workers_per_cell=2)], {})
        assert ctor_kwargs_probe.calls == []

        workers = [
            handle.actor_class(**kwargs)
            for handle, kwargs in zip(fake_ray_cluster.handles, fake_ray_cluster.ctor_kwargs)
        ]

        assert [call["pid"] for call in ctor_kwargs_probe.calls] == [os.getpid(), os.getpid()]
        assert [worker.rank for worker in workers] == [0, 1]


class TestTheBootstrappedClass:
    async def test_hands_the_spec_name_and_the_runs_argv_to_the_bootstrap(self, ctor_kwargs_probe: _CtorKwargsProbe):
        """The actor rebuilds the run's specs from these two, and can rebuild nothing else without them."""
        actor_class = bootstrapped_worker_class(DemoWorker)

        actor_class(**worker_bootstrap_kwargs(spec_name=SPEC_NAME, worker_argv=WORKER_ARGV, context=_launch_context()))

        assert ctor_kwargs_probe.calls[0]["spec_name"] == SPEC_NAME
        assert ctor_kwargs_probe.calls[0]["worker_argv"] == WORKER_ARGV

    async def test_builds_the_context_with_a_provider_factory_of_its_own_process(
        self, ctor_kwargs_probe: _CtorKwargsProbe
    ):
        """A spec that asks for its engines is answered by the backend this process sees, not the launcher's."""
        actor_class = bootstrapped_worker_class(DemoWorker)

        actor_class(**worker_bootstrap_kwargs(spec_name=SPEC_NAME, worker_argv=WORKER_ARGV, context=_launch_context()))

        assert ctor_kwargs_probe.calls[0]["context"].providers is not None

    async def test_passes_the_computed_keywords_to_the_wrapped_constructor(self, ctor_kwargs_probe: _CtorKwargsProbe):
        """The worker class is keyword-only, exactly as it is when a pod builds it in serve_inner."""
        actor_class = bootstrapped_worker_class(DemoWorker)

        worker = actor_class(
            **worker_bootstrap_kwargs(
                spec_name=SPEC_NAME, worker_argv=WORKER_ARGV, context=_launch_context(worker_in_cell_index=2)
            )
        )

        assert (worker.rank, worker.role) == (2, "actor")

    async def test_keeps_the_name_of_the_class_it_wraps(self, ctor_kwargs_probe: _CtorKwargsProbe):
        """Ray names actors and their errors after the class, and 'BootstrappedWorker' would name them all alike."""
        assert bootstrapped_worker_class(DemoWorker).__name__ == DemoWorker.__name__
        assert bootstrapped_worker_class(DemoWorker).__module__ == DemoWorker.__module__
