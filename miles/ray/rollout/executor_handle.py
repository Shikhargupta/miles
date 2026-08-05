from __future__ import annotations

from miles.ray.specs.rollout import rollout_executor_cell_id, rollout_executor_worker_name
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_provider.factory import ProviderFactory


def create_rollout_executor_handle(args, *, providers: ProviderFactory) -> BaseWorkerHandle:
    if ClusterBackend(args.cluster_backend) is ClusterBackend.KUBERNETES:
        worker_name = rollout_executor_worker_name()
        provider = providers.static(worker_name=worker_name)
        return provider.get_handle(worker_name, cell_id=rollout_executor_cell_id())

    from miles.ray.rollout.ray_executor import create_ray_rollout_executor_handle

    return create_ray_rollout_executor_handle(args)
