from __future__ import annotations

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.ray_worker_manager import RayWorkerManager


def get_backend_capability(args) -> BackendCapability:
    # TODO: after k8s native mode is created, answer with the kubernetes capability in that mode
    return RayBackendCapability(worker_manager_handle=RayWorkerManager.get_handle())
