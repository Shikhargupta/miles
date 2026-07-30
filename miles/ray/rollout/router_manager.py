import logging

from miles.ray.specs.router import compute_router_cell_spec
from miles.ray.specs.session import compute_session_server_cell_specs, resolve_session_server_ports
from miles.utils.http_utils import is_port_available
from miles.utils.workers.ray_worker_manager import RayWorkerManager

logger = logging.getLogger(__name__)


async def start_model_routers(args, worker_manager: RayWorkerManager, model_specs) -> None:
    """Bring up one router per model and publish their addresses on args."""
    routers: dict[str, tuple[str, int]] = {}
    for model_idx, model_spec in enumerate(model_specs):
        router_ip, router_port = await start_router(
            args,
            worker_manager,
            model_name=model_spec.name,
            has_pd_disaggregation=model_spec.has_pd_disaggregation,
            force_new=(model_idx > 0),
        )
        if model_idx == 0:
            args.sglang_router_ip = router_ip
            args.sglang_router_port = router_port
        routers[model_spec.name] = (router_ip, router_port)
    args.sglang_model_routers = routers


async def start_router(
    args,
    worker_manager: RayWorkerManager,
    *,
    model_name: str,
    has_pd_disaggregation: bool = False,
    force_new: bool = False,
) -> tuple[str, int]:
    """Start sgl router or miles router as a managed cell and return (router_ip, router_port).

    If ``args.sglang_router_ip`` is already set and ``force_new`` is False,
    skip launching and return the existing values.
    """
    if not force_new and args.sglang_router_ip is not None:
        return args.sglang_router_ip, args.sglang_router_port

    static_port = None if force_new else args.sglang_router_port
    if static_port is not None and not is_port_available(static_port):
        raise RuntimeError(
            f"Port {static_port} is already in use — a stale router process may still be running. "
            f"Run 'pkill -9 python' to kill it, then retry."
        )

    spec = compute_router_cell_spec(
        args,
        cell_id=f"router-{model_name}",
        has_pd_disaggregation=has_pd_disaggregation,
        static_port=static_port,
    )
    await worker_manager.register_cells([spec])

    (worker,) = worker_manager.cell_workers(spec.cell_id)
    logger.info(f"Router launched at {worker.payload['host']}:{worker.payload['port']}")
    return worker.payload["host"], worker.payload["port"]


async def start_session_server(args, worker_manager: RayWorkerManager) -> None:
    """Start the standalone session servers as managed cells when ``--use-session-server`` is set.

    One independent single-process server per resolved port; the rollout side
    picks one per session and its URL carries the affinity from then on.
    Always started standalone regardless of whether ``--use-miles-router`` is
    active.
    """
    if not getattr(args, "use_session_server", False):
        return

    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        raise ValueError("--use-session-server requires --hf-checkpoint to be set.")

    static_ports = resolve_session_server_ports(getattr(args, "session_server_port", None))
    for port in static_ports or []:
        if not is_port_available(port):
            raise RuntimeError(
                f"Port {port} is already in use — a stale session server may still be running. "
                f"Run 'pkill -9 python' to kill it, then retry."
            )

    specs = compute_session_server_cell_specs(args)
    await worker_manager.register_cells(specs)

    payloads = []
    for spec in specs:
        (worker,) = worker_manager.cell_workers(spec.cell_id)
        payloads.append(worker.payload)

    # The canonical driver-side values; rollout code picks from these.
    if getattr(args, "session_server_ip", None) is None:
        args.session_server_ip = payloads[0]["host"]
    args.session_server_ports = [payload["port"] for payload in payloads]
    # The per-port map OpenAIEndpointTracer.create reads instance ids from,
    # replacing the per-session /health probe.
    args.session_server_instance_ids = {payload["port"]: payload["instance_id"] for payload in payloads}

    logger.info(
        f"Session servers launched at {args.session_server_ip}, "
        f"ports {args.session_server_ports} ({len(payloads)} instances)"
    )
