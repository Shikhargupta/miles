import dataclasses
import logging

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.router_manager import start_model_routers, start_session_server
from miles.ray.specs.inference import compute_inference_model_specs
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class InferenceWiring:
    inference_controller: InferenceController
    worker_manager: RayWorkerManager | None


async def create_inference_controller(args, pg) -> InferenceWiring:
    if args.debug_train_only:
        controller = await InferenceController.create(args, model_specs=[], provider=None)
        return InferenceWiring(inference_controller=controller, worker_manager=None)

    worker_manager = RayWorkerManager(pg=pg)
    model_specs = compute_inference_model_specs(args)
    await start_model_routers(args, worker_manager, model_specs)
    dashboard_hooks.register_router(args)
    await start_session_server(args, worker_manager)
    await worker_manager.register_cells([cell for model_spec in model_specs for cell in model_spec.cells])
    provider = RayWorkerProvider(worker_manager=worker_manager)
    controller = await InferenceController.create(args, model_specs=model_specs, provider=provider)
    return InferenceWiring(inference_controller=controller, worker_manager=worker_manager)
