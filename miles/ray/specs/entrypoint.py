from miles.ray.specs import inference, rollout, train
from miles.utils.workers.worker_spec import BaseWorkerSpec


def compute_specs(args) -> list[BaseWorkerSpec]:
    return [
        *inference.specs_router(args),
        inference.spec_session_server(args),
        *inference.specs_inference_engine(args),
        rollout.spec_rollout_executor(args),
        *train.specs_trainer(args),
    ]
