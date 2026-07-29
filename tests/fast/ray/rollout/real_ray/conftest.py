"""Fixtures that drive ``MockSGLangEngine`` actors through a real ``RayWorkerManager``."""

from __future__ import annotations

import dataclasses
import itertools
import time

import pytest
import ray

from miles.ray.rollout.rollout_server import build_server_cells
from miles.ray.specs.inference import InferenceDeployment, compute_inference_deployments
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.ray_worker_manager.manager import RayWorkerManager
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

_MOCK_ENGINE_CLASS = "miles.utils.test_utils.mock_sglang_engine.MockSGLangEngine"
_PROVIDER_POLL_INTERVAL_SECONDS = 0.5
_ACTORS_GONE_TIMEOUT_SECONDS = 30.0

_unique_counter = itertools.count()


def make_mock_deployments(args) -> list[InferenceDeployment]:
    """Inference deployments whose engines are gpu-less MockSGLangEngine actors."""
    deployments = []
    for deployment in compute_inference_deployments(args):
        spec = deployment.spec.model_copy(
            update={
                "worker_class": _MOCK_ENGINE_CLASS,
                "scheduling": deployment.spec.scheduling.model_copy(
                    update={"num_gpus_per_worker": 0, "num_cpus_per_worker": 0.1}
                ),
            }
        )
        deployments.append(dataclasses.replace(deployment, spec=spec))
    return deployments


@dataclasses.dataclass
class ManagerHarness:
    manager: ray.actor.ActorHandle
    deployments: list[InferenceDeployment]

    @property
    def provider(self) -> RayWorkerProvider:
        return RayWorkerProvider(
            manager=self.manager,
            spec_names=[deployment.spec.name for deployment in self.deployments],
            poll_interval_seconds=_PROVIDER_POLL_INTERVAL_SECONDS,
        )

    @property
    def worker_cell_control(self) -> RayWorkerHandle:
        return RayWorkerHandle(self.manager)

    def build_cells(self, args):
        return build_server_cells(
            args,
            deployments=self.deployments,
            provider=self.provider,
            worker_cell_control=self.worker_cell_control,
        )

    def kill_all(self) -> None:
        spec_names = [deployment.spec.name for deployment in self.deployments]
        worker_names = [info.name for info in ray.get(self.manager.get_worker_infos.remote(spec_names=spec_names))]
        for name in worker_names:
            try:
                ray.kill(ray.get_actor(name), no_restart=True)
            except ValueError:
                pass
        ray.kill(self.manager)
        _wait_names_gone(worker_names)


@pytest.fixture
def manager_harness_factory(ray_local_mode):
    """Yields ``async make(args) -> ManagerHarness``; all managers and their
    named worker actors are torn down (and confirmed gone) after the test."""
    harnesses: list[ManagerHarness] = []

    async def _make(args) -> ManagerHarness:
        deployments = make_mock_deployments(args)
        manager = (
            ray.remote(RayWorkerManager)
            .options(name=f"test-worker-manager-{next(_unique_counter)}", num_cpus=0.1)
            .remote()
        )
        await manager.init.remote(worker_specs=[deployment.spec for deployment in deployments], placements={})
        harness = ManagerHarness(manager=manager, deployments=deployments)
        harnesses.append(harness)
        return harness

    yield _make

    for harness in harnesses:
        harness.kill_all()


def _wait_names_gone(names: list[str]) -> None:
    deadline = time.monotonic() + _ACTORS_GONE_TIMEOUT_SECONDS
    remaining = set(names)
    while remaining:
        remaining = {name for name in remaining if _actor_exists(name)}
        if not remaining:
            return
        assert time.monotonic() < deadline, f"actors {sorted(remaining)} still resolvable during test teardown"
        time.sleep(0.1)


def _actor_exists(name: str) -> bool:
    try:
        ray.get_actor(name)
    except ValueError:
        return False
    return True
