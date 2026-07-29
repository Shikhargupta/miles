import shlex

from miles.backends.sglang_utils.router_args_utils import compute_sglang_router_args, router_args_to_argv
from miles.backends.sglang_utils.sglang_config import resolve_sglang_config
from miles.router.config import compute_miles_router_config
from miles.utils.workers.argv_utils import config_to_argv
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec

_ROUTER_PORT = 30080
_ROUTER_PROMETHEUS_PORT = 30081
_ROUTER_BIND_HOST = "0.0.0.0"


def compute_router_specs(args) -> list[CommandWorkerSpec]:
    if args.debug_train_only:
        return []

    config = resolve_sglang_config(args)
    return [
        _spec_router(args, model_name=model_cfg.name, has_pd_disaggregation=model_cfg.has_pd_disaggregation)
        for model_cfg in config.models
    ]


def _spec_router(args, *, model_name: str, has_pd_disaggregation: bool) -> CommandWorkerSpec:
    port_infos = [
        PortInfo(
            name="router",
            static_port=args.sglang_router_port or _ROUTER_PORT,
            mode="per_worker",
            allow_dynamic=True,
        )
    ]
    if not args.use_miles_router:
        port_infos.append(
            PortInfo(name="prometheus", static_port=_ROUTER_PROMETHEUS_PORT, mode="per_worker", allow_dynamic=True)
        )

    return CommandWorkerSpec(
        name=f"router-{model_name}",
        port_infos=port_infos,
        env_var=lambda: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0, num_cpus_per_worker=1),
        launch_command=_compute_router_launch_command(args, has_pd_disaggregation=has_pd_disaggregation),
    )


def _compute_router_launch_command(args, *, has_pd_disaggregation: bool) -> str:
    static_port = args.sglang_router_port or _ROUTER_PORT

    if args.use_miles_router:
        assert not has_pd_disaggregation, "miles router does not support PD disaggregation."
        router_config = compute_miles_router_config(args, host=_ROUTER_BIND_HOST, port=static_port)
        return shlex.join(["python", "-m", "miles.router.router", *config_to_argv(router_config)])

    router_args = compute_sglang_router_args(
        args,
        host=_ROUTER_BIND_HOST,
        port=static_port,
        prometheus_port=_ROUTER_PROMETHEUS_PORT,
        has_pd_disaggregation=has_pd_disaggregation,
    )
    return shlex.join(["python", "-m", "sglang_router.launch_router", *router_args_to_argv(router_args)])
