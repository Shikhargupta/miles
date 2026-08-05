from __future__ import annotations

from miles.utils.function_registry import load_function
from miles.utils.workers.naming import parse_worker_name
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, ReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import RPC_PORT_NAME, HostAndPort, NamedHostAndPorts


class SimpleWorkerProvider(BaseWorkerProvider):
    def __init__(
        self,
        *,
        addrs: dict[str, NamedHostAndPorts],
        cells: dict[str, list[str]],
        spec_names: dict[str, str],
        worker_classes: dict[str, str] | None = None,
    ) -> None:
        self._addrs = addrs
        self._cells = cells
        self._spec_names = spec_names
        self._worker_class_paths = worker_classes or {}
        self._worker_classes: dict[str, type] = {}

    def knows_worker(self, worker_name: str) -> bool:
        return worker_name in self._addrs

    async def get_addr(self, worker_name: str) -> HostAndPort:
        return (await self.get_addrs(worker_name=worker_name))["primary"]

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        addrs = self._addrs.get(worker_name)
        assert addrs is not None, f"worker {worker_name} is not in the address book: {sorted(self._addrs)}"
        return addrs

    async def watch_cells(self, reconcile: ReconcileFn, *, spec_names: list[str]) -> StopWatchFn:
        for cell_id in sorted(self._cells):
            info = self.cell_info(cell_id)
            if info is not None and info.spec_name in spec_names:
                await reconcile(cell_id, info)
        return _never_changes

    def cell_ids(self) -> list[str]:
        return sorted(self._cells)

    def cell_info(self, cell_id: str) -> CellInfo | None:
        worker_names = self._cells.get(cell_id)
        if worker_names is None:
            return None

        spec_name = self._spec_names.get(cell_id)
        assert spec_name is not None, f"cell {cell_id} is in the address book without a spec name"
        return CellInfo(
            cell_id=cell_id,
            spec_name=spec_name,
            alive=True,
            worker_names=list(worker_names),
            workers_hash=f"static-{cell_id}",
            meta={},
        )

    def get_worker_infos(self, *, cell_id: str) -> list[WorkerInfo]:
        worker_names = self._cells.get(cell_id)
        assert worker_names, f"cell {cell_id} is not in the address book: {sorted(self._cells)}"
        return [
            WorkerInfo(
                name=worker_name,
                generation=0,
                self_addrs=self._addrs[worker_name],
                gpu_ids=[],
                handle=self.get_handle(worker_name, cell_id=cell_id),
            )
            for worker_name in worker_names
        ]

    def get_worker_infos_of(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        return [self.get_worker_infos(cell_id=cell_id) for cell_id in cell_ids]

    def get_handle(self, worker_name: str, *, cell_id: str | None = None) -> BaseWorkerHandle:
        addrs = self._addrs.get(worker_name)
        assert addrs is not None, f"worker {worker_name} is not in the address book: {sorted(self._addrs)}"
        assert RPC_PORT_NAME in addrs, f"worker {worker_name} has no {RPC_PORT_NAME!r} port to be called through"
        return RpcWorkerHandle(
            self._worker_class_of(worker_name),
            server_url=addrs[RPC_PORT_NAME].addr,
            require_stable_boot_uuid=True,
        )

    def _worker_class_of(self, worker_name: str) -> type:
        spec_name = parse_worker_name(worker_name)[0]
        if spec_name not in self._worker_classes:
            path = self._worker_class_paths.get(spec_name)
            assert path is not None, (
                f"spec {spec_name} has no worker class, so its rpc methods are unknown; "
                f"known specs are {sorted(self._worker_class_paths)}"
            )
            self._worker_classes[spec_name] = load_function(path)
        return self._worker_classes[spec_name]


async def _never_changes() -> None:
    return None
