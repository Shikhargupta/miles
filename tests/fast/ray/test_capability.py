from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from miles.ray import capability as ray_capability
from miles.utils.workers.backend_capability import factory
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.types import ClusterBackend


@dataclass
class _Stub:
    capability: object
    specs_computed_from: list[Any] = field(default_factory=list)


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _Stub:
    stub = _Stub(capability=object())
    monkeypatch.setattr(ray_capability, "compute_specs", lambda args: stub.specs_computed_from.append(args) or [])
    monkeypatch.setattr(factory, "compute_helm_backend_capability", lambda **kwargs: stub.capability)
    return stub


class TestGetBackendCapability:
    def test_a_ray_run_is_answered_from_the_worker_manager(self, monkeypatch: pytest.MonkeyPatch, stub: _Stub) -> None:
        """The manager was launched by the driver's own first line; the capability only looks it up."""
        monkeypatch.setattr(factory.RayWorkerManager, "get_handle", staticmethod(lambda: object()))

        args = SimpleNamespace(cluster_backend=ClusterBackend.RAY.value)

        assert isinstance(ray_capability.get_backend_capability(args), RayBackendCapability)

    def test_a_kubernetes_run_is_answered_by_observing_the_namespace(self, stub: _Stub) -> None:
        """Under Kubernetes nothing was launched, so the capability is what observes the pods instead."""
        args = SimpleNamespace(cluster_backend=ClusterBackend.KUBERNETES.value)

        assert ray_capability.get_backend_capability(args) is stub.capability
        assert stub.specs_computed_from == [args]

    def test_the_import_graph_has_no_cycle_through_the_driver(self) -> None:
        """placement_group asks for the capability, so defining it beside wiring would make either import fail."""
        import miles.ray.placement_group  # noqa: F401
        import miles.ray.wiring  # noqa: F401

        assert "get_backend_capability" not in vars(miles.ray.wiring)
