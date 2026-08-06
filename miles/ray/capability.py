from __future__ import annotations

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.workers.backend_capability import factory
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.types import ClusterBackend


def get_backend_capability(args) -> BackendCapability:
    return factory.get_backend_capability(
        specs=compute_specs(args), cluster_backend=ClusterBackend(args.cluster_backend)
    )
