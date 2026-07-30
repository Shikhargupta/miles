import functools
import shlex
import sys
import uuid
from collections.abc import Callable
from typing import Any

from miles.rollout.session.config import compute_session_server_config
from miles.utils.http_utils import _wrap_ipv6, wait_tcp_ready
from miles.utils.workers.argv_utils import config_to_argv
from miles.utils.workers.worker_spec import (
    BaseCellSpec,
    CellAddressing,
    CommandWorkerSpec,
    PortInfo,
    RayActorOptions,
    SchedulingSpec,
    WorkerLaunchPlan,
    WorkerPlacement,
)

SESSION_SERVER_PORT_NAME = "port"

SESSION_SERVER_RAY_OPTIONS = RayActorOptions(num_cpus=0.2, num_gpus=0)


def compute_session_server_cell_specs(args) -> list[BaseCellSpec]:
    static_ports = resolve_session_server_ports(getattr(args, "session_server_port", None))
    ports: list[int | None] = [None] if static_ports is None else list(static_ports)

    specs = []
    for index, static_port in enumerate(ports):
        cell_id = f"session-server-{index}"
        instance_id = uuid.uuid4().hex
        worker = CommandWorkerSpec(
            name=cell_id,
            port_infos=[
                PortInfo(
                    name=SESSION_SERVER_PORT_NAME,
                    static_port=static_port if static_port is not None else 5000,
                    mode="per_worker",
                    allow_dynamic=static_port is None,
                )
            ],
            env_var=_no_env_vars,
            scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
            ray_options=SESSION_SERVER_RAY_OPTIONS,
            build_launch_plan=functools.partial(_build_session_server_launch_plan, args, instance_id=instance_id),
            build_member_payloads=functools.partial(_build_session_server_member_payloads, instance_id=instance_id),
            wait_cell_ready=_wait_session_server_ready,
        )
        specs.append(BaseCellSpec(worker=worker, cell_id=cell_id, rank_offset=0, gpu_offset=0))
    return specs


def resolve_session_server_ports(raw: list[int] | None) -> list[int] | None:
    """Resolve the ``--session-server-port`` value into the static ports to serve on.

    None: one dynamically allocated port. One value: a single server on that port.
    Two values: the half-open range [start, end), one server per port.
    """
    if raw is None:
        return None
    if len(raw) == 1:
        return raw
    if len(raw) == 2:
        start, end = raw
        if start >= end:
            raise ValueError(f"--session-server-port range [{start}, {end}) is empty.")
        return list(range(start, end))
    raise ValueError(f"--session-server-port takes one port or a start/end range, got {len(raw)} values: {raw}")


def _build_session_server_launch_plan(
    args, placement: WorkerPlacement, addressing: CellAddressing, *, instance_id: str
) -> WorkerLaunchPlan:
    (payload,) = _build_session_server_member_payloads(addressing, instance_id=instance_id)
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    config = compute_session_server_config(
        args, host=payload["host"], port=payload["port"], instance_id=instance_id, backend_url=router_url
    )
    argv = [sys.executable, "-m", "miles.rollout.session.server", *config_to_argv(config)]
    return WorkerLaunchPlan(cmd=shlex.join(argv))


def _build_session_server_member_payloads(addressing: CellAddressing, *, instance_id: str) -> list[dict[str, Any]]:
    return [
        {
            "host": _wrap_ipv6(addressing.node_ips[0]),
            "port": addressing.per_worker_ports[0][SESSION_SERVER_PORT_NAME],
            "instance_id": instance_id,
        }
    ]


async def _wait_session_server_ready(addressing: CellAddressing, is_worker_alive: Callable[[], bool]) -> None:
    port = addressing.per_worker_ports[0][SESSION_SERVER_PORT_NAME]
    await wait_tcp_ready(_wrap_ipv6(addressing.node_ips[0]), port, is_alive=is_worker_alive)


def _no_env_vars(placement: WorkerPlacement) -> dict[str, str]:
    return {}
