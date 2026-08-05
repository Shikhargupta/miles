import ray

from miles.utils.ray_utils import compute_ray_pin_head_options
from miles.utils.workers.worker_bootstrap import bootstrapped_worker_class, worker_bootstrap_kwargs
from miles.utils.workers.worker_spec import WorkerLaunchContext


def create_head_worker_actor(
    *,
    worker_cls: type,
    env_vars: dict[str, str],
    num_cpus: float,
    spec_name: str,
    worker_argv: list[str],
    context: WorkerLaunchContext,
) -> ray.actor.ActorHandle:
    """A gpu-less worker runs on the head node, where its ports stay forwardable."""
    actor_cls = ray.remote(bootstrapped_worker_class(worker_cls))
    return actor_cls.options(
        num_cpus=num_cpus,
        num_gpus=0,
        runtime_env={
            "env_vars": env_vars,
        },
        **compute_ray_pin_head_options(),
    ).remote(**worker_bootstrap_kwargs(spec_name=spec_name, worker_argv=worker_argv, context=context))
