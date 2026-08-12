import asyncio
import logging
from collections.abc import Awaitable, Callable

from miles.ray.specs.addressing import AddressedWorker, compute_addressed_workers, format_addressed_workers
from miles.ray.specs.entrypoint import compute_specs
from miles.ray.wiring import get_backend_capability, launch_worker_manager
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.logging_utils import configure_logger
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)


def run_deployment(args, *, run_orchestration_script: Callable[[object], Awaitable[None]]) -> None:
    if DeployComponent(args.deploy_component).deploys_orchestration_script():
        asyncio.run(run_orchestration_script(args))
        return

    asyncio.run(_serve_deployed_workers(args))


async def _serve_deployed_workers(args) -> None:
    configure_logger(args, source=SimpleProcessIdentity(component="main"))
    component = DeployComponent(args.deploy_component)

    _worker_manager = launch_worker_manager(args)
    logger.info(
        f"Deployed the {component.value} workers of this run: "
        f"{[spec.name for spec in compute_specs(args)]}. "
        f"{await _describe_controller_addrs(args, component=component)}"
    )
    logger.info(
        "This deployment carries no orchestration script, so it has no training to finish and stays up until it is "
        "uninstalled"
    )

    await asyncio.Event().wait()


async def _describe_controller_addrs(args, *, component: DeployComponent) -> str:
    capability = get_backend_capability(args)
    workers = compute_addressed_workers(args, component=component)
    addrs = await asyncio.gather(*[_addr_of(capability, worker=worker) for worker in workers])
    return f"Reach it with {format_addressed_workers(list(zip(workers, addrs, strict=True)))}"


async def _addr_of(capability: BackendCapability, *, worker: AddressedWorker) -> HostAndPort:
    addrs = await capability.static_worker_provider(pool_id=worker.pool_id).get_addrs(worker.worker_name)
    return addrs[worker.port_name]
