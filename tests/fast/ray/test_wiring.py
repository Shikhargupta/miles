from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from miles.ray import wiring
from miles.utils.workers.types import ClusterBackend


class TestLaunchWorkerManager:
    def test_a_ray_run_launches_the_ray_worker_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Ray path is the one every existing run takes, and it must be untouched."""
        launched: list[Any] = []
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", lambda args: launched.append(args))

        args = SimpleNamespace(cluster_backend=ClusterBackend.RAY.value)
        wiring.launch_worker_manager(args)

        assert launched == [args]

    def test_a_kubernetes_run_launches_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under Kubernetes the pods already exist, so launching actors would double the run."""
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", _refuse_ray)

        assert wiring.launch_worker_manager(SimpleNamespace(cluster_backend=ClusterBackend.KUBERNETES.value)) is None


def _refuse_ray(args: Any) -> None:
    raise AssertionError("the Kubernetes path must not launch Ray workers")
