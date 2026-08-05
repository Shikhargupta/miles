from __future__ import annotations

import ray

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils.ray_utils import compute_ray_pin_head_options
from miles.utils.workers.ray_worker_handle import RayWorkerHandle


def create_ray_rollout_executor_handle(args) -> RayWorkerHandle:
    pin_options = compute_ray_pin_head_options() if args.pin_rollout_manager_to_head else {}
    actor_handle = ray.remote(RolloutExecutor).options(num_cpus=1, num_gpus=0, **pin_options).remote(args=args)
    return RayWorkerHandle(actor_handle)
