import logging

from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.rollout.rollout_server import start_rollout_servers
from miles.ray.rollout.router_manager import start_session_server
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_provider.ray import RayWorkerProvider

logger = logging.getLogger(__name__)


async def create_inference_controller(args, pg) -> InferenceController:
    if args.debug_train_only:
        return await InferenceController.create(args, servers={}, provider=None)

    worker_manager = RayWorkerManager(pg=pg)
    servers = await start_rollout_servers(args, worker_manager)
    dashboard_hooks.register_router(args)
    await start_session_server(args, worker_manager)
    provider = RayWorkerProvider(worker_manager=worker_manager)
    return await InferenceController.create(args, servers=servers, provider=provider)
