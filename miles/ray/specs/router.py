import functools
import shlex
import sys
from collections.abc import Callable
from typing import Any

from miles.backends.sglang_utils.router_args_utils import compute_sglang_router_args, router_args_to_argv
from miles.router.config import compute_miles_router_config
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

ROUTER_PORT_NAME = "port"
ROUTER_PROMETHEUS_PORT_NAME = "prometheus_port"

ROUTER_RAY_OPTIONS = RayActorOptions(num_cpus=0.2, num_gpus=0)


def compute_router_cell_spec(
    args, *, cell_id: str, has_pd_disaggregation: bool, static_port: int | None
) -> BaseCellSpec:
    if args.use_miles_router:
        assert not has_pd_disaggregation, "miles router does not support PD disaggregation."

    worker = CommandWorkerSpec(
        name=cell_id,
        port_infos=_compute_router_port_infos(args, static_port=static_port),
        env_var=_no_env_vars,
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        ray_options=ROUTER_RAY_OPTIONS,
        build_launch_plan=functools.partial(
            _build_router_launch_plan, args, has_pd_disaggregation=has_pd_disaggregation
        ),
        build_member_payloads=_build_router_member_payloads,
        wait_cell_ready=_wait_router_ready,
    )
    return BaseCellSpec(worker=worker, cell_id=cell_id, rank_offset=0, gpu_offset=0)


def _compute_router_port_infos(args, *, static_port: int | None) -> list[PortInfo]:
    port_infos = [
        PortInfo(
            name=ROUTER_PORT_NAME,
            static_port=static_port if static_port is not None else 3000,
            mode="per_worker",
            allow_dynamic=static_port is None,
        )
    ]
    if not args.use_miles_router:
        port_infos.append(
            PortInfo(name=ROUTER_PROMETHEUS_PORT_NAME, static_port=4000, mode="per_worker", allow_dynamic=True)
        )
    return port_infos


def _build_router_launch_plan(
    args, placement: WorkerPlacement, addressing: CellAddressing, *, has_pd_disaggregation: bool
) -> WorkerLaunchPlan:
    host = _wrap_ipv6(addressing.node_ips[0])
    ports = addressing.per_worker_ports[0]

    if args.use_miles_router:
        router_config = compute_miles_router_config(args, host=host, port=ports[ROUTER_PORT_NAME])
        argv = [sys.executable, "-m", "miles.router.router", *config_to_argv(router_config)]
    else:
        router_args = compute_sglang_router_args(
            args,
            host=host,
            port=ports[ROUTER_PORT_NAME],
            prometheus_port=ports[ROUTER_PROMETHEUS_PORT_NAME],
            has_pd_disaggregation=has_pd_disaggregation,
        )
        argv = [sys.executable, "-m", "sglang_router.launch_router", *router_args_to_argv(router_args)]

    return WorkerLaunchPlan(cmd=shlex.join(argv))


def _build_router_member_payloads(addressing: CellAddressing) -> list[dict[str, Any]]:
    return [{"host": _wrap_ipv6(addressing.node_ips[0]), "port": addressing.per_worker_ports[0][ROUTER_PORT_NAME]}]


async def _wait_router_ready(addressing: CellAddressing, is_worker_alive: Callable[[], bool]) -> None:
    (payload,) = _build_router_member_payloads(addressing)
    await wait_tcp_ready(payload["host"], payload["port"], is_alive=is_worker_alive)


def _no_env_vars(placement: WorkerPlacement) -> dict[str, str]:
    return {}
