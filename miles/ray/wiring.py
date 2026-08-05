from __future__ import annotations

import os
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from miles.utils.ft_utils.api_server.cell_operations import K8sCellOperations
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.naming import compute_cell_id, compute_worker_name, static_worker_host
from miles.utils.workers.reconcile.k8s_api import KubernetesAsyncioPodApi
from miles.utils.workers.types import DEFAULT_GPUS_PER_NODE, ClusterBackend
from miles.utils.workers.worker_provider.base import CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.factory import (
    DeferredProviderFactory,
    KubernetesProviderFactory,
    ProviderFactory,
    RayProviderFactory,
)
from miles.utils.workers.worker_provider.k8s import K8sWorkerProvider
from miles.utils.workers.worker_provider.k8s_labels import CellLabelKeys
from miles.utils.workers.worker_provider.shared import SharedK8sWorkerProvider
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    BaseWorkerSpec,
    HostAndPort,
    NamedHostAndPorts,
    ServeWorkerSpec,
    SpecMetaFn,
)

NAMESPACE_ENV_VAR = "MILES_K8S_NAMESPACE"
RELEASE_ENV_VAR = "MILES_K8S_RELEASE"
NAMESPACE_FILE = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
INSTANCE_LABEL = "app.kubernetes.io/instance"

LABEL_KEY_ENV_VARS = {
    "fleet": "MILES_K8S_FLEET_LABEL",
    "cell_index": "MILES_K8S_CELL_INDEX_LABEL",
    "worker_index": "MILES_K8S_WORKER_INDEX_LABEL",
    "spec_name": "MILES_K8S_SPEC_NAME_LABEL",
    "cell_size": "MILES_K8S_CELL_SIZE_LABEL",
    "meta_annotation_prefix": "MILES_K8S_META_ANNOTATION_PREFIX",
}


def create_provider_factory(args) -> ProviderFactory:
    if ClusterBackend(args.cluster_backend) is ClusterBackend.KUBERNETES:
        return _install_kubernetes_workers_from_args(args)
    return RayProviderFactory(worker_manager_handle=_launch_ray_worker_manager(args))


def create_worker_provider_factory(*, worker_argv: list[str]) -> ProviderFactory:
    return DeferredProviderFactory(create=lambda: _attach_provider_factory_from_argv(worker_argv))


def driver_worker_argv() -> list[str]:
    return list(sys.argv[1:])


def install_kubernetes_workers(
    *,
    specs: list[BaseWorkerSpec],
    namespace: str,
    release: str,
    kube_client_factory: Callable[[], Any],
    delete_pods: Callable[[list[str]], Awaitable[None]],
    num_gpus_per_node: int = DEFAULT_GPUS_PER_NODE,
    label_keys: CellLabelKeys | None = None,
) -> KubernetesProviderFactory:
    watched_spec_names = fleet_spec_names(specs=specs)
    provider = SharedK8sWorkerProvider(
        inner=K8sWorkerProvider(
            namespace=namespace,
            label_selector=f"{INSTANCE_LABEL}={release}",
            static_addrs=static_worker_addrs(specs=specs, release=release),
            worker_ports=fleet_worker_ports(specs=specs),
            worker_classes=fleet_worker_classes(specs=specs),
            spec_metas=fleet_spec_metas(specs=specs),
            ranks_per_pod=fleet_ranks_per_pod(specs=specs, num_gpus_per_node=num_gpus_per_node),
            kube_client_factory=kube_client_factory,
            label_keys=label_keys,
        ),
        spec_names=watched_spec_names,
    )

    return KubernetesProviderFactory(
        cells_provider=provider,
        cells_spec_names=watched_spec_names,
        static_provider=static_worker_provider(specs=specs, release=release),
        cell_operations=WatchingK8sCellOperations(
            provider=provider,
            spec_names=watched_spec_names,
            delete_pods=delete_pods,
        ),
    )


class WatchingK8sCellOperations:
    def __init__(
        self,
        *,
        provider: SharedK8sWorkerProvider,
        spec_names: list[str],
        delete_pods: Callable[[list[str]], Awaitable[None]],
    ) -> None:
        self._provider = provider
        self._spec_names = spec_names
        self._operations = K8sCellOperations(provider=provider, delete_pods=delete_pods)
        self._stop_watch: StopWatchFn | None = None

    async def cell_infos(self, spec_names: list[str]) -> dict[str, CellInfo]:
        await self._ensure_watching()
        return await self._operations.cell_infos(spec_names)

    async def suspend(self, cell_id: str) -> None:
        await self._ensure_watching()
        await self._operations.suspend(cell_id)

    async def resume(self, cell_id: str) -> None:
        await self._ensure_watching()
        await self._operations.resume(cell_id)

    async def inject_fault(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        await self._ensure_watching()
        await self._operations.inject_fault(cell_id, mode=mode, sub_index=sub_index)

    async def _ensure_watching(self) -> None:
        if self._stop_watch is None:
            self._stop_watch = await self._provider.watch_cells(_ignore_cell, spec_names=self._spec_names)


def static_worker_addrs(*, specs: list[BaseWorkerSpec], release: str) -> dict[str, NamedHostAndPorts]:
    addrs: dict[str, NamedHostAndPorts] = {}
    for spec in specs:
        if is_fleet_spec(spec):
            continue
        for cell_index in range(spec.scheduling.num_cells):
            host = static_worker_host(release, spec.name, cell_index)
            for worker_in_cell_index in range(spec.scheduling.num_workers_per_cell):
                worker_name = compute_worker_name(
                    spec_name=spec.name, cell_index=cell_index, worker_in_cell_index=worker_in_cell_index
                )
                addrs[worker_name] = {
                    port.name: HostAndPort(host=host, port=port.static_port) for port in spec.port_infos
                }
    return addrs


def static_worker_provider(*, specs: list[BaseWorkerSpec], release: str) -> SimpleWorkerProvider:
    cells: dict[str, list[str]] = {}
    spec_names: dict[str, str] = {}
    for spec in specs:
        if is_fleet_spec(spec):
            continue
        for cell_index in range(spec.scheduling.num_cells):
            cell_id = compute_cell_id(spec_name=spec.name, cell_index=cell_index)
            cells[cell_id] = [
                compute_worker_name(spec_name=spec.name, cell_index=cell_index, worker_in_cell_index=index)
                for index in range(spec.scheduling.num_workers_per_cell)
            ]
            spec_names[cell_id] = spec.name

    return SimpleWorkerProvider(
        addrs=static_worker_addrs(specs=specs, release=release),
        cells=cells,
        spec_names=spec_names,
        worker_classes=static_worker_classes(specs=specs),
    )


def static_worker_classes(*, specs: list[BaseWorkerSpec]) -> dict[str, str]:
    return {
        spec.name: spec.caller_facing_class
        for spec in specs
        if not is_fleet_spec(spec) and isinstance(spec, ServeWorkerSpec)
    }


def fleet_worker_ports(*, specs: list[BaseWorkerSpec]) -> dict[str, dict[str, int]]:
    return {
        spec.name: {port.name: port.static_port for port in spec.port_infos} for spec in specs if is_fleet_spec(spec)
    }


def fleet_ranks_per_pod(*, specs: list[BaseWorkerSpec], num_gpus_per_node: int) -> dict[str, int]:
    return {
        spec.name: _ranks_per_pod_of(spec, num_gpus_per_node=num_gpus_per_node)
        for spec in specs
        if is_fleet_spec(spec)
    }


def fleet_worker_classes(*, specs: list[BaseWorkerSpec]) -> dict[str, str]:
    return {
        spec.name: spec.caller_facing_class
        for spec in specs
        if is_fleet_spec(spec) and isinstance(spec, ServeWorkerSpec)
    }


def fleet_spec_metas(*, specs: list[BaseWorkerSpec]) -> dict[str, SpecMetaFn]:
    return {spec.name: spec.meta for spec in specs if is_fleet_spec(spec) and spec.meta is not None}


def fleet_spec_names(*, specs: list[BaseWorkerSpec]) -> list[str]:
    return [spec.name for spec in specs if is_fleet_spec(spec)]


def is_fleet_spec(spec: BaseWorkerSpec) -> bool:
    return spec.scheduling.num_workers_per_cell * spec.scheduling.num_gpu_slots_per_worker > 0


def current_release() -> str:
    release = os.environ.get(RELEASE_ENV_VAR, "")
    assert release, (
        f"the orchestrator cannot tell which release created its workers, so it cannot select their pods: "
        f"set {RELEASE_ENV_VAR}"
    )
    return release


def current_label_keys() -> CellLabelKeys:
    overrides = {
        field: value for field, env_var in LABEL_KEY_ENV_VARS.items() if (value := os.environ.get(env_var, ""))
    }
    return CellLabelKeys(**overrides)


def current_namespace() -> str:
    if namespace := os.environ.get(NAMESPACE_ENV_VAR, ""):
        return namespace
    assert NAMESPACE_FILE.exists(), (
        f"the driver runs outside a pod, so it cannot tell which namespace holds its workers: "
        f"set {NAMESPACE_ENV_VAR}"
    )
    namespace = NAMESPACE_FILE.read_text().strip()
    assert namespace, f"{NAMESPACE_FILE} is empty, so no namespace can be observed"
    return namespace


def _ranks_per_pod_of(spec: BaseWorkerSpec, *, num_gpus_per_node: int) -> int:
    if not isinstance(spec, ServeWorkerSpec):
        return 1

    ranks_per_pod = min(spec.scheduling.num_workers_per_cell, num_gpus_per_node)
    _assert_rank_ports_are_free(spec, ranks_per_pod=ranks_per_pod)
    return ranks_per_pod


def _assert_rank_ports_are_free(spec: ServeWorkerSpec, *, ranks_per_pod: int) -> None:
    rpc_port = next(port.static_port for port in spec.port_infos if port.name == RPC_PORT_NAME)
    for port in spec.port_infos:
        if port.name == RPC_PORT_NAME:
            continue
        assert rpc_port + ranks_per_pod <= port.static_port or port.static_port + port.num_consecutive <= rpc_port, (
            f"spec '{spec.name}' serves {ranks_per_pod} ranks per pod from {RPC_PORT_NAME} port {rpc_port} "
            f"upwards, which reaches into the {port.num_consecutive} port(s) '{port.name}' claims from "
            f"{port.static_port}"
        )


def _attach_provider_factory_from_argv(worker_argv: list[str]) -> ProviderFactory:
    from miles.utils.arguments import parse_args_from_argv

    args = parse_args_from_argv(worker_argv)
    if ClusterBackend(args.cluster_backend) is ClusterBackend.KUBERNETES:
        return _install_kubernetes_workers_from_args(args)

    from miles.utils.workers.ray_worker_manager import RayWorkerManager

    return RayProviderFactory(worker_manager_handle=RayWorkerManager.get_handle())


def _install_kubernetes_workers_from_args(args) -> KubernetesProviderFactory:
    from miles.ray.specs.entrypoint import compute_specs

    namespace = current_namespace()
    return install_kubernetes_workers(
        specs=compute_specs(args),
        namespace=namespace,
        release=current_release(),
        kube_client_factory=_create_kube_client,
        delete_pods=_pod_deleter(namespace=namespace),
        num_gpus_per_node=args.num_gpus_per_node,
        label_keys=current_label_keys(),
    )


def _launch_ray_worker_manager(args):
    from miles.ray.placement_group import create_placement_groups
    from miles.ray.specs.entrypoint import compute_specs
    from miles.utils.workers.ray_worker_manager import RayWorkerManager

    specs = compute_specs(args)
    # TODO: pass in specs instead of args
    pgs = create_placement_groups(args)
    return RayWorkerManager.launch(specs, pgs, worker_argv=driver_worker_argv())


def _create_kube_client() -> KubernetesAsyncioPodApi:
    from kubernetes_asyncio import client as kubernetes_client
    from kubernetes_asyncio import config as kubernetes_config

    kubernetes_config.load_incluster_config()
    return KubernetesAsyncioPodApi(core_v1_api=kubernetes_client.CoreV1Api(kubernetes_client.ApiClient()))


def _pod_deleter(*, namespace: str) -> Callable[[list[str]], Awaitable[None]]:
    async def delete_pods(pod_names: list[str]) -> None:
        from kubernetes_asyncio import client as kubernetes_client
        from kubernetes_asyncio import config as kubernetes_config

        kubernetes_config.load_incluster_config()
        async with kubernetes_client.ApiClient() as api_client:
            core_v1_api = kubernetes_client.CoreV1Api(api_client)
            for pod_name in pod_names:
                await core_v1_api.delete_namespaced_pod(name=pod_name, namespace=namespace)

    return delete_pods


async def _ignore_cell(cell_id: str, info: CellInfo | None) -> None:
    return None
