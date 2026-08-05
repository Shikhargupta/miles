from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.function_registry import load_function
from miles.utils.workers.naming import compute_worker_name, parse_cell_id
from miles.utils.workers.reconcile.k8s_reflector import KubernetesReflector
from miles.utils.workers.reconcile.loop import ReconcileLoop
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_provider.k8s_labels import (
    LWS_WORKER_INDEX_LABEL,
    CellLabelKeys,
    ObservedWorker,
    cell_members_hash,
    observe_pod,
    read_meta,
)
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    HostAndPort,
    NamedHostAndPorts,
    SpecMetaFn,
    WorkerMetaContext,
)

logger = logging.getLogger(__name__)

_NOT_A_WORKER_PREFIX = "__not-a-worker__/"


@dataclass(frozen=True)
class _RankedWorker:
    pod: ObservedWorker
    name: str
    rank_in_pod: int
    gpu_ids: list[int]


def _has_every_pod(pods: list[ObservedWorker]) -> bool:
    expected = max(pod.cell_size for pod in pods)
    return len(pods) >= expected if expected else True


def _worker_index_of(pod) -> int:
    labels = pod.metadata.labels or {}
    return int(labels.get(LWS_WORKER_INDEX_LABEL, 0))


class K8sWorkerProvider(BaseWorkerProvider):
    def __init__(
        self,
        *,
        namespace: str,
        label_selector: str,
        static_addrs: dict[str, NamedHostAndPorts],
        worker_ports: dict[str, dict[str, int]],
        kube_client_factory: Callable[[], object],
        worker_classes: dict[str, str] | None = None,
        spec_metas: dict[str, SpecMetaFn] | None = None,
        ranks_per_pod: dict[str, int] | None = None,
        label_keys: CellLabelKeys | None = None,
        resync_period: float | None = 60.0,
    ) -> None:
        self._namespace = namespace
        self._label_selector = label_selector
        self._static_addrs = static_addrs
        self._ports_by_spec_name = worker_ports
        self._ranks_per_pod_by_spec_name = ranks_per_pod or {}
        self._worker_class_paths = worker_classes or {}
        self._spec_metas = spec_metas or {}
        self._worker_classes: dict[str, type] = {}
        self._kube_client_factory = kube_client_factory
        self._label_keys = label_keys or CellLabelKeys()
        self._resync_period = resync_period
        self._loop: ReconcileLoop | None = None
        self._reported: set[str] = set()

    async def get_addr(self, worker_name: str) -> HostAndPort:
        return (await self.get_addrs(worker_name=worker_name))["primary"]

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        if (addrs := self._static_addrs.get(worker_name)) is not None:
            return addrs

        worker = self._find_worker(worker_name)
        assert worker is not None, f"worker {worker_name} is neither a static worker nor an observed pod"
        return self._addrs_of(worker)

    async def watch_cells(self, reconcile: ReconcileFn, *, spec_names: list[str]) -> StopWatchFn:
        wanted = set(spec_names)
        reflector = KubernetesReflector(
            kube_client=self._kube_client_factory(),
            namespace=self._namespace,
            label_selector=self._label_selector,
        )
        loop = ReconcileLoop(
            source=reflector.watch,
            reconcile=lambda cell_id: self._reconcile_cell(cell_id, reconcile=reconcile, wanted=wanted),
            key_map=self._cell_id_of_pod,
            resync_period=self._resync_period,
        )
        try:
            await loop.start()
        except BaseException:
            await loop.stop()
            raise
        self._loop = loop
        return loop.stop

    def get_worker_infos(self, *, cell_id: str) -> list[WorkerInfo]:
        return [
            WorkerInfo(
                name=worker.name,
                generation=worker.pod.restart_count,
                self_addrs=self._addrs_of(worker),
                gpu_ids=list(worker.gpu_ids),
                handle=self._rpc_handle_of(worker),
            )
            for worker in self._ranked_workers_of(cell_id)
        ]

    def get_worker_infos_of(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [self.get_worker_infos(cell_id=cell_id) for cell_id in cell_ids]

    def cell_info(self, cell_id: str) -> CellInfo | None:
        pods = self._pods_of(cell_id)
        if not pods:
            return None

        spec_name = pods[0].spec_name
        meta: dict[str, Any] = self._spec_meta_of(spec_name, cell_id=cell_id)
        meta.update(self._pod_meta_of(cell_id))

        return CellInfo(
            cell_id=cell_id,
            spec_name=spec_name,
            alive=all(pod.ready for pod in pods) and _has_every_pod(pods),
            worker_names=[worker.name for worker in self._fanned_workers_of(cell_id)],
            workers_hash=cell_members_hash(pods),
            meta=meta,
        )

    def pod_names(self, cell_id: str) -> list[str]:
        return [pod.name for pod in self._pods_of(cell_id)]

    def cell_ids(self) -> list[str]:
        return sorted(key for key in self._loop_or_fail().parent_keys() if not key.startswith(_NOT_A_WORKER_PREFIX))

    def _reconcile_cell(self, cell_id: str, reconcile: ReconcileFn, wanted: set[str]) -> Awaitable[None]:
        pods = self._loop_or_fail().get_by_parent(cell_id)
        for pod in pods:
            if (observed := observe_pod(pod, self._label_keys)) is not None:
                if observed.spec_name not in wanted:
                    return self._report_gone(cell_id, reconcile=reconcile)
                break

        info = self.cell_info(cell_id)
        if info is None or not info.alive:
            return self._report_gone(cell_id, reconcile=reconcile)

        return self._report_alive(cell_id, info, reconcile=reconcile)

    async def _report_gone(self, cell_id: str, *, reconcile: ReconcileFn) -> None:
        if cell_id not in self._reported:
            return
        await reconcile(cell_id, None)
        self._reported.discard(cell_id)

    async def _report_alive(self, cell_id: str, info: CellInfo, *, reconcile: ReconcileFn) -> None:
        await reconcile(cell_id, info)
        self._reported.add(cell_id)

    def _spec_meta_of(self, spec_name: str, *, cell_id: str) -> dict[str, Any]:
        compute_meta = self._spec_metas.get(spec_name)
        if compute_meta is None:
            return {}
        return dict(compute_meta(WorkerMetaContext(cell_index=parse_cell_id(cell_id).cell_index)))

    def _pod_meta_of(self, cell_id: str) -> dict[str, str]:
        merged: dict[str, str] = {}
        for pod in sorted(self._loop_or_fail().get_by_parent(cell_id), key=_worker_index_of):
            for key, value in read_meta(pod, self._label_keys).items():
                assert merged.get(key, value) == value, (
                    f"cell {cell_id} annotates {key!r} as both {merged[key]!r} and {value!r}; a cell reports one "
                    f"value for a key, so which pod wins would be whatever order the store hands them back in"
                )
                merged[key] = value
        return merged

    def _cell_id_of_pod(self, pod) -> str:
        observed = observe_pod(pod, self._label_keys)
        return observed.cell_id if observed is not None else f"{_NOT_A_WORKER_PREFIX}{pod.metadata.name}"

    def _find_worker(self, worker_name: str) -> _RankedWorker | None:
        for cell_id in self.cell_ids():
            for worker in self._fanned_workers_of(cell_id):
                if worker.name == worker_name:
                    return worker
        return None

    def _loop_or_fail(self) -> ReconcileLoop:
        assert self._loop is not None, "watch_cells must be running before the cells can be read"
        return self._loop

    def _pods_of(self, cell_id: str) -> list[ObservedWorker]:
        pods = [
            observed
            for pod in self._loop_or_fail().get_by_parent(cell_id)
            if (observed := observe_pod(pod, self._label_keys)) is not None
        ]
        return sorted(pods, key=lambda pod: pod.worker_index)

    def _ranked_workers_of(self, cell_id: str) -> list[_RankedWorker]:
        pods = self._pods_of(cell_id)
        assert pods, f"cell {cell_id} has no observed worker pods, so it cannot be driven"
        indices = [pod.worker_index for pod in pods]
        assert indices == list(range(len(pods))), f"cell {cell_id} is missing pods: observed {indices}"
        return [worker for pod in pods for worker in self._ranks_of(pod)]

    def _fanned_workers_of(self, cell_id: str) -> list[_RankedWorker]:
        return [worker for pod in self._pods_of(cell_id) for worker in self._ranks_of(pod)]

    def _ranks_of(self, pod: ObservedWorker) -> list[_RankedWorker]:
        ranks_per_pod = self._ranks_per_pod_by_spec_name.get(pod.spec_name, 1)
        assert len(pod.gpu_ids) % ranks_per_pod == 0, (
            f"pod {pod.name} was annotated with {len(pod.gpu_ids)} gpus for the {ranks_per_pod} ranks it serves, "
            f"so no rank owns an equal share of them"
        )
        gpus_per_rank = len(pod.gpu_ids) // ranks_per_pod
        cell_index = parse_cell_id(pod.cell_id).cell_index
        return [
            _RankedWorker(
                pod=pod,
                name=compute_worker_name(
                    spec_name=pod.spec_name,
                    cell_index=cell_index,
                    worker_in_cell_index=pod.worker_index * ranks_per_pod + rank_in_pod,
                ),
                rank_in_pod=rank_in_pod,
                gpu_ids=list(pod.gpu_ids[rank_in_pod * gpus_per_rank : (rank_in_pod + 1) * gpus_per_rank]),
            )
            for rank_in_pod in range(ranks_per_pod)
        ]

    def _addrs_of(self, worker: _RankedWorker) -> NamedHostAndPorts:
        host = self._host_of(worker.pod)
        ports = self._ports_by_spec_name.get(worker.pod.spec_name)
        assert ports, f"spec {worker.pod.spec_name} declares no ports, so {worker.name} has no address"
        return {
            name: HostAndPort(host=host, port=port + (worker.rank_in_pod if name == RPC_PORT_NAME else 0))
            for name, port in ports.items()
        }

    def _host_of(self, observed: ObservedWorker) -> str:
        if observed.pod_ip:
            return observed.pod_ip
        assert (
            observed.subdomain
        ), f"worker {observed.name} has neither a pod ip nor a headless service to be addressed through"
        return f"{observed.name}.{observed.subdomain}.{self._namespace}.svc"

    def _rpc_handle_of(self, worker: _RankedWorker) -> BaseWorkerHandle:
        addrs = self._addrs_of(worker)
        assert (
            RPC_PORT_NAME in addrs
        ), f"spec {worker.pod.spec_name} has no {RPC_PORT_NAME!r} port to be called through"
        return RpcWorkerHandle(
            self._worker_class_of(worker.pod.spec_name),
            server_url=addrs[RPC_PORT_NAME].addr,
            require_stable_boot_uuid=True,
        )

    def _worker_class_of(self, spec_name: str) -> type:
        if spec_name not in self._worker_classes:
            path = self._worker_class_paths.get(spec_name)
            assert path is not None, (
                f"spec {spec_name} has no worker class, so its rpc methods are unknown; "
                f"known specs are {sorted(self._worker_class_paths)}"
            )
            self._worker_classes[spec_name] = load_function(path)
        return self._worker_classes[spec_name]
