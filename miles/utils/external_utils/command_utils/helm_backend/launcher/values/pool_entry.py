from __future__ import annotations

import shlex
import sys

from miles.utils.external_utils.colocate_pairing.config import PairingLayout
from miles.utils.external_utils.command_utils.base_backend import TRAINER_ROLE
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (
    PoolEntry,
    PortEntry,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    SECTION_OF_CATEGORY,
    TRAINER_ENGINES_SECTION,
    LaunchPlan,
)
from miles.utils.workers import env_vars as worker_env_vars
from miles.utils.workers.worker_provider.kubernetes.helm import env
from miles.utils.workers.worker_spec import (
    BaseWorkerSpec,
    CommandWorkerSpec,
    HostAndPort,
    LaunchCommandContext,
    NamedHostAndPorts,
    ServeWorkerSpec,
)

_WORKER_INDEX_PLACEHOLDER = "$(LWS_WORKER_INDEX)"
_LEADER_ADDRESS_PLACEHOLDER = "$(LWS_LEADER_ADDRESS)"
_BASE_GPU_ID_PLACEHOLDER = f"$({worker_env_vars.BASE_GPU_ID_ENV_VAR})"

_BIND_HOST = "0.0.0.0"

_SERVE_MODULE = "miles.utils.workers.serving.serve"
_SUPERVISOR_MODULE = "miles.utils.workers.process_supervisor"
_SPECS_FN = "miles.ray.specs.entrypoint.compute_specs_from_argv"

_RENDERED_CELL_INDEX = 0
_WORKER_INDEX_SENTINEL = 987654321
_BASE_GPU_ID_SENTINEL = 987654322


def build_entry(
    spec: BaseWorkerSpec,
    plan: LaunchPlan,
    addresses: dict[str, dict[str, NamedHostAndPorts]],
    pairing_layout: PairingLayout | None = None,
) -> PoolEntry:
    assert spec.scheduling.num_cells > 0, (
        f"Spec '{spec.name}' asks for {spec.scheduling.num_cells} cells; a spec a run has turned off is dropped "
        f"before conversion, because a values entry always renders at least one pod"
    )
    shares_its_node = _shares_its_node(pairing_layout)
    context = _launch_context(
        spec,
        addresses=addresses,
        cell_index=_RENDERED_CELL_INDEX,
        worker_in_cell_index=_WORKER_INDEX_SENTINEL,
        shares_its_node=shares_its_node,
    )
    pods_per_cell = spec.scheduling.pods_per_cell()
    gpus_per_pod = spec.scheduling.gpus_per_pod()

    return PoolEntry(
        name=spec.name,
        object_name=naming.component_name(plan.release, spec.name),
        pool_id=spec.name,
        command=_with_prepare_cmd(_command_of_spec(spec, context, plan=plan), spec, plan=plan),
        ports=[PortEntry(name=_port_name(port.name), port=port.static_port) for port in spec.port_infos],
        env=_command_env_of_spec(spec, context, addresses=addresses, shares_its_node=shares_its_node) or None,
        meta=_meta_of_spec(spec) or None,
        service_account_name=(
            naming.component_name(plan.release, naming.ORCHESTRATOR_COMPONENT)
            if spec.needs_platform_read_permission
            else None
        ),
        replicas=spec.scheduling.num_cells,
        size=pods_per_cell if pods_per_cell > 1 else None,
        resources={"limits": {"nvidia.com/gpu": gpus_per_pod}} if gpus_per_pod else None,
        restart_at=plan.rendered_restart_at(spec.name),
    )


def _command_env_of_spec(
    spec: BaseWorkerSpec,
    context: LaunchCommandContext,
    *,
    addresses: dict[str, dict[str, NamedHostAndPorts]],
    shares_its_node: bool,
) -> dict[str, str]:
    if isinstance(spec, ServeWorkerSpec):
        return {}

    first = dict(spec.env_var(context))
    second = dict(
        spec.env_var(
            _launch_context(
                spec,
                addresses=addresses,
                cell_index=1,
                worker_in_cell_index=1,
                shares_its_node=shares_its_node,
            )
        )
    )
    assert first == second, (
        f"Spec '{spec.name}' builds its environment out of the cell and worker it is given, but a values entry "
        f"describes a whole pool and is rendered before any of them exists; serve the spec so its pod can "
        f"compute the environment itself, or drop the dependency"
    )
    return first


def _with_prepare_cmd(command: list[str], spec: BaseWorkerSpec, plan: LaunchPlan) -> list[str]:
    if SECTION_OF_CATEGORY[spec.category] != TRAINER_ENGINES_SECTION:
        return command
    prepare = plan.prepare_cmd.get(TRAINER_ROLE)
    if not prepare:
        return command

    assert spec.scheduling.gpus_per_pod() >= spec.scheduling.num_gpus_per_node, (
        f"A prepare command runs once per pod of '{spec.name}', but that pool takes "
        f"{spec.scheduling.gpus_per_pod()} of a node's {spec.scheduling.num_gpus_per_node} gpus, so two of its "
        f"pods can land on one node and run the command against the same node-local path at the same time; "
        f"give the pool whole nodes, or serialize the command yourself with flock"
    )
    return ["bash", "-c", f"{prepare} && exec {shlex.join(command)}"]


def _meta_of_spec(spec: BaseWorkerSpec) -> dict[str, str]:
    gpus_per_pod = spec.scheduling.gpus_per_pod()
    if not gpus_per_pod:
        return {}
    return {env.DEFAULT_LABEL_KEYS.gpu_ids_meta: ",".join(str(gpu_id) for gpu_id in range(gpus_per_pod))}


def _launch_context(
    spec: BaseWorkerSpec,
    addresses: dict[str, dict[str, NamedHostAndPorts]],
    *,
    cell_index: int,
    worker_in_cell_index: int,
    shares_its_node: bool = False,
) -> LaunchCommandContext:
    self_addrs = {
        port.name: HostAndPort(
            host=_LEADER_ADDRESS_PLACEHOLDER if port.mode == "master" else _BIND_HOST,
            port=port.static_port,
        )
        for port in spec.port_infos
    }
    return LaunchCommandContext(
        cell_index=cell_index,
        worker_in_cell_index=worker_in_cell_index,
        gpu_ids=_rendered_gpu_ids(spec, shares_its_node=shares_its_node),
        self_addrs=self_addrs,
        spec_addrs={pool_id: list(cells.values()) for pool_id, cells in addresses.items()},
    )


def _rendered_gpu_ids(spec: BaseWorkerSpec, *, shares_its_node: bool) -> list[int]:
    gpus_per_pod = max(1, spec.scheduling.gpus_per_pod())
    if shares_its_node:
        return [_BASE_GPU_ID_SENTINEL] * gpus_per_pod
    return list(range(gpus_per_pod))


def _command_of_spec(spec: BaseWorkerSpec, context: LaunchCommandContext, plan: LaunchPlan) -> list[str]:
    match spec:
        case CommandWorkerSpec():
            return _with_base_gpu_id(_with_worker_index(shlex.split(spec.launch_command(context)), spec), spec)
        case ServeWorkerSpec():
            return _serve_command(spec, plan)
        case _:
            raise AssertionError(f"{spec.name} is neither launched by a command nor served over rpc: {spec}")


def _serve_command(spec: ServeWorkerSpec, plan: LaunchPlan) -> list[str]:
    workers_per_pod = spec.scheduling.workers_per_pod()
    serve = [
        sys.executable,
        "-m",
        _SERVE_MODULE,
        "--specs",
        _SPECS_FN,
        "--pool-id",
        spec.name,
        "--",
    ] + plan.worker_argv
    if workers_per_pod == 1:
        return serve
    return [sys.executable, "-m", _SUPERVISOR_MODULE, "--num-subprocesses", str(workers_per_pod), "--"] + serve


def _shares_its_node(pairing_layout: PairingLayout | None) -> bool:
    if pairing_layout is None:
        return False
    return pairing_layout.num_gpus_per_inference_pod < pairing_layout.num_gpus_per_node


def _with_base_gpu_id(argv: list[str], spec: BaseWorkerSpec) -> list[str]:
    sentinel = str(_BASE_GPU_ID_SENTINEL)
    _assert_sentinel_is_a_whole_token(argv, sentinel=sentinel, spec=spec, built_out_of="base gpu id")
    return [_BASE_GPU_ID_PLACEHOLDER if argument == sentinel else argument for argument in argv]


def _with_worker_index(argv: list[str], spec: BaseWorkerSpec) -> list[str]:
    sentinel = str(_WORKER_INDEX_SENTINEL)
    _assert_sentinel_is_a_whole_token(argv, sentinel=sentinel, spec=spec, built_out_of="pod index")
    return [_WORKER_INDEX_PLACEHOLDER if argument == sentinel else argument for argument in argv]


def _assert_sentinel_is_a_whole_token(
    argv: list[str], *, sentinel: str, spec: BaseWorkerSpec, built_out_of: str
) -> None:
    embedded = [argument for argument in argv if sentinel in argument and argument != sentinel]
    assert not embedded, (
        f"Spec '{spec.name}' builds {embedded} out of its {built_out_of}; the value is substituted a whole "
        f"argument at a time, so it has to reach the command unchanged"
    )


def _port_name(name: str) -> str:
    return name.replace("_", "-")[:15]
