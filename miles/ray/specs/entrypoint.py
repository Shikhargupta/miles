from miles.ray.specs.inference import compute_inference_specs
from miles.ray.specs.rollout import spec_rollout_executor
from miles.ray.specs.router import compute_router_specs
from miles.ray.specs.session import compute_session_server_specs
from miles.ray.specs.trainer import compute_trainer_specs
from miles.utils.workers.worker_spec import BaseWorkerSpec


def compute_specs(args) -> list[BaseWorkerSpec]:
    specs: list[BaseWorkerSpec] = [
        *compute_trainer_specs(args),
        *compute_inference_specs(args),
        *compute_router_specs(args),
        *compute_session_server_specs(args),
        spec_rollout_executor(args),
    ]

    names = [spec.name for spec in specs]
    assert len(names) == len(set(names)), f"worker spec names must be unique, got {names}"
    return specs
