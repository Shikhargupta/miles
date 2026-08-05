from __future__ import annotations

import asyncio
import contextlib
import importlib
import sys
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from tests.fast.utils.workers.worker_provider.test_k8s_labels import make_pod

from miles.ray.rollout.rollout_executor_api import RolloutExecutorApi
from miles.utils.workers.reconcile.k8s_api import PodListPage

NAMESPACE = "rl"
RELEASE = "miles-run-260805"
SPEC_NAME = "trainer-actor"
CELL_ID = "trainer-actor-0"

REFUSED_MODULES = (
    "ray",
    "miles.utils.workers.ray_worker_manager",
    "miles.utils.workers.ray_worker_handle",
    "miles.ray.placement_group",
    "miles.ray.train_actor",
)


class FakeTrainWorker:
    def __init__(self) -> None:
        self.configured: list[tuple[str, int]] = []

    def configure_master_addr_and_port(self, master_addr: str, master_port: int) -> int:
        self.configured.append((master_addr, master_port))
        return master_port

    def kill_self(self) -> None:
        return None


class FakeRolloutExecutor(RolloutExecutorApi):
    def __init__(self) -> None:
        self.loaded: list[int | None] = []
        self.train_parallel_config: dict[str, Any] | None = None

    def dispose(self) -> None:
        return None

    async def get(self, rollout_id: int) -> dict[str, Any]:
        return {"sample_indices": [rollout_id], "data_ref": f"ref-{rollout_id}"}

    async def eval(self, rollout_id: int) -> None:
        return None

    def save(self, rollout_id: int) -> None:
        return None

    def load(self, rollout_id: int | None = None) -> None:
        self.loaded.append(rollout_id)

    def get_num_rollout_per_epoch(self) -> int:
        return 7

    def set_train_parallel_config(self, config: dict[str, Any]) -> None:
        self.train_parallel_config = config


class FakePodApi:
    def __init__(self, pods: list[Any]) -> None:
        self.pods = pods

    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage:
        return PodListPage(pods=list(self.pods), resource_version="1")

    async def stream_pods(self, *, namespace, label_selector, resource_version, timeout_seconds):
        await asyncio.sleep(3600)
        yield None


class _RefusingFinder:
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname in REFUSED_MODULES or fullname.startswith("ray."):
            raise AssertionError(f"the kubernetes assembly reached for {fullname}, which a namespace has no use for")
        return None


@contextlib.contextmanager
def kubernetes_only_imports():
    saved = dict(sys.modules)
    for name in list(sys.modules):
        if name == "ray" or name.startswith("ray.") or name.startswith("miles."):
            del sys.modules[name]

    finder = _RefusingFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)
        sys.modules.clear()
        sys.modules.update(saved)


class _PerHostTransport(httpx.AsyncBaseTransport):
    def __init__(self, apps_by_host: dict[str, Any]) -> None:
        self._transports = {host: httpx.ASGITransport(app=app) for host, app in apps_by_host.items()}
        self.hosts_called: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.hosts_called.append(host)
        transport = self._transports.get(host)
        if transport is None:
            raise httpx.ConnectError(f"nothing listens on {host}", request=request)
        return await transport.handle_async_request(request)


def trainer_spec(worker_spec_module, *, num_workers_per_cell: int):
    return worker_spec_module.ServeWorkerSpec(
        name=SPEC_NAME,
        port_infos=[worker_spec_module.PortInfo(name="master", static_port=9000, mode="master")],
        env_var=lambda context: {},
        scheduling=worker_spec_module.SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=1,
            num_gpu_slots_per_worker=1,
        ),
        worker_class=f"{__name__}.FakeTrainWorker",
        ctor_kwargs=lambda context: {},
    )


def cell_pods(count: int):
    return [
        make_pod(
            name=f"{SPEC_NAME}-0-{index}",
            fleet=SPEC_NAME,
            cell_index="0",
            worker_index=str(index),
            pod_ip=f"10.0.0.{index + 1}",
        )
        for index in range(count)
    ]


def rollout_executor_args():
    return SimpleNamespace(cluster_backend="kubernetes", pin_rollout_manager_to_head=False)


def install(*, pods: list[Any], deleted: list[list[str]], ranks_per_pod: int = 1):
    wiring = importlib.import_module("miles.ray.wiring")
    worker_spec = importlib.import_module("miles.utils.workers.worker_spec")
    specs_rollout = importlib.import_module("miles.ray.specs.rollout")

    async def delete_pods(pod_names: list[str]) -> None:
        deleted.append(list(pod_names))

    factory = wiring.install_kubernetes_workers(
        specs=[
            trainer_spec(worker_spec, num_workers_per_cell=len(pods) * ranks_per_pod),
            specs_rollout.spec_rollout_executor(rollout_executor_args()),
        ],
        namespace=NAMESPACE,
        release=RELEASE,
        kube_client_factory=lambda: FakePodApi(pods),
        delete_pods=delete_pods,
        num_gpus_per_node=ranks_per_pod,
    )
    return factory


def installed_cells_provider(factory):
    return factory.cells(spec_names=[SPEC_NAME])


class TestKubernetesDriverAssembly:
    def test_the_installed_factory_answers_every_component_of_the_process(self):
        """A run announces its backend once, and every later component has to see that answer."""
        with kubernetes_only_imports():
            shared = importlib.import_module("miles.utils.workers.worker_provider.shared")
            factory = install(pods=cell_pods(2), deleted=[])

            provider = installed_cells_provider(factory)

            assert provider is installed_cells_provider(factory)
            assert isinstance(provider, shared.SharedK8sWorkerProvider)

    def test_refuses_to_hand_out_a_provider_for_a_fleet_it_does_not_watch(self):
        """A provider that silently watches nothing would leave those cells unhealed forever."""
        with kubernetes_only_imports():
            factory = install(pods=cell_pods(2), deleted=[])

            with pytest.raises(AssertionError, match="not watched"):
                factory.cells(spec_names=["some-other-fleet"])

    def test_observing_a_cell_yields_rank_ordered_workers_with_handles(self):
        """This is what a trainer cell is built from, so the order and the handles are the whole product."""
        with kubernetes_only_imports():
            rpc_handle = importlib.import_module("miles.utils.workers.rpc.client.handle")
            provider = installed_cells_provider(install(pods=cell_pods(3), deleted=[]))

            async def scenario():
                await provider.start()
                try:
                    return provider.get_worker_infos(cell_id=CELL_ID)
                finally:
                    await provider.stop()

            infos = asyncio.run(scenario())

            assert [info.name for info in infos] == [f"{SPEC_NAME}-0-{index}" for index in range(3)]
            assert [info.self_addrs["rpc"].host for info in infos] == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
            assert all(isinstance(info.handle, rpc_handle.RpcWorkerHandle) for info in infos)

    def test_one_pod_serving_several_ranks_yields_one_worker_per_rank(self):
        """A trainer pod supervises one process per gpu, and a cell that saw one of them would hang the collective."""
        with kubernetes_only_imports():
            provider = installed_cells_provider(install(pods=cell_pods(1), deleted=[], ranks_per_pod=2))

            async def scenario():
                await provider.start()
                try:
                    return provider.get_worker_infos(cell_id=CELL_ID)
                finally:
                    await provider.stop()

            infos = asyncio.run(scenario())

            assert [info.name for info in infos] == [f"{SPEC_NAME}-0-0", f"{SPEC_NAME}-0-1"]
            assert [(info.self_addrs["rpc"].host, info.self_addrs["rpc"].port) for info in infos] == [
                ("10.0.0.1", 8000),
                ("10.0.0.1", 8001),
            ]
            assert [info.handle._transport._server_url for info in infos] == [
                "http://10.0.0.1:8000",
                "http://10.0.0.1:8001",
            ]
            assert [info.self_addrs["master"].port for info in infos] == [9000, 9000]

    def test_a_trainer_cell_builds_and_drives_its_ranks_over_rpc(self, monkeypatch: pytest.MonkeyPatch):
        """The point of the whole assembly: training under Kubernetes with no ray anywhere in the process."""
        with kubernetes_only_imports():
            cell_module = importlib.import_module("miles.ray.train.cell")
            http_utils = importlib.import_module("miles.utils.http_utils")
            rpc_app = importlib.import_module("miles.utils.workers.rpc.server.app")
            provider = installed_cells_provider(install(pods=cell_pods(2), deleted=[]))

            workers = [FakeTrainWorker(), FakeTrainWorker()]
            apps = {f"10.0.0.{index + 1}": rpc_app.create_rpc_app(worker) for index, worker in enumerate(workers)}
            transport = _PerHostTransport(apps)

            async def scenario():
                async with httpx.AsyncClient(transport=transport) as client:
                    monkeypatch.setattr(
                        http_utils.GeneralHttpClientProvider, "client", classmethod(lambda cls: client)
                    )
                    for app in apps.values():
                        await app.router.lifespan_context(app).__aenter__()
                    await provider.start()
                    try:
                        cell = cell_module.TrainerCell(
                            args=SimpleNamespace(),
                            role="actor",
                            with_ref=False,
                            cell_id=CELL_ID,
                            cell_index=0,
                            workers_hash="hash-1",
                            health_checker=SimpleNamespace(start=lambda: None, status=None),
                            provider=provider,
                        )
                        return cell, await cell.execute(
                            "configure_master_addr_and_port", master_addr="10.0.0.1", master_port=9000
                        )
                    finally:
                        await provider.stop()

            cell, results = asyncio.run(scenario())

            assert results == [9000, 9000]
            assert [worker.configured for worker in workers] == [[("10.0.0.1", 9000)], [("10.0.0.1", 9000)]]
            assert set(transport.hosts_called) == {"10.0.0.1", "10.0.0.2"}

    def test_healing_a_cell_deletes_its_pods_without_a_ray_worker_manager(self):
        """Under Kubernetes a cell heals because its workload recreates the pods somebody deleted."""
        with kubernetes_only_imports():
            deleted: list[list[str]] = []
            factory = install(pods=cell_pods(2), deleted=deleted)

            operations = factory.cell_operations()
            asyncio.run(operations.suspend(CELL_ID))

            assert deleted == [[f"{SPEC_NAME}-0-0", f"{SPEC_NAME}-0-1"]]

    def test_the_rollout_executor_answers_over_rpc_with_no_actor_behind_it(self, monkeypatch: pytest.MonkeyPatch):
        """Under Kubernetes the executor is a pod in the address book, not an actor the driver creates."""
        with kubernetes_only_imports():
            executor_handle = importlib.import_module("miles.ray.rollout.executor_handle")
            specs_rollout = importlib.import_module("miles.ray.specs.rollout")
            wiring = importlib.import_module("miles.ray.wiring")
            http_utils = importlib.import_module("miles.utils.http_utils")
            rpc_app = importlib.import_module("miles.utils.workers.rpc.server.app")
            rpc_handle = importlib.import_module("miles.utils.workers.rpc.client.handle")
            factory = install(pods=cell_pods(2), deleted=[])

            executor = FakeRolloutExecutor()
            host = wiring.static_worker_host(RELEASE, specs_rollout.ROLLOUT_EXECUTOR_SPEC_NAME)
            app = rpc_app.create_rpc_app(executor)
            transport = _PerHostTransport({host: app})

            async def scenario():
                async with httpx.AsyncClient(transport=transport) as client:
                    monkeypatch.setattr(
                        http_utils.GeneralHttpClientProvider, "client", classmethod(lambda cls: client)
                    )
                    await app.router.lifespan_context(app).__aenter__()
                    handle = executor_handle.create_rollout_executor_handle(rollout_executor_args(), providers=factory)
                    await handle.set_train_parallel_config(config={"dp_size": 4})
                    await handle.load(rollout_id=11)
                    return handle, await handle.get(rollout_id=3)

            handle, rollout_data = asyncio.run(scenario())

            assert isinstance(handle, rpc_handle.RpcWorkerHandle)
            assert rollout_data == {"sample_indices": [3], "data_ref": "ref-3"}
            assert executor.train_parallel_config == {"dp_size": 4}
            assert executor.loaded == [11]
            assert set(transport.hosts_called) == {host}

    def test_the_rollout_components_assemble_around_that_handle(self, monkeypatch: pytest.MonkeyPatch):
        """create_rollout_components is the driver's door into rollout, so it too must open without ray."""
        with kubernetes_only_imports():
            components = importlib.import_module("miles.ray.rollout.components")
            specs_rollout = importlib.import_module("miles.ray.specs.rollout")
            wiring = importlib.import_module("miles.ray.wiring")
            http_utils = importlib.import_module("miles.utils.http_utils")
            rpc_app = importlib.import_module("miles.utils.workers.rpc.server.app")
            factory = install(pods=cell_pods(2), deleted=[])

            controller = object()
            host = wiring.static_worker_host(RELEASE, specs_rollout.ROLLOUT_EXECUTOR_SPEC_NAME)
            app = rpc_app.create_rpc_app(FakeRolloutExecutor())
            transport = _PerHostTransport({host: app})
            args = SimpleNamespace(
                cluster_backend="kubernetes", pin_rollout_manager_to_head=False, num_rollout=None, num_epoch=3
            )

            async def create_controller(_args, *, providers) -> object:
                return controller

            monkeypatch.setattr(components, "InferenceController", SimpleNamespace(create=create_controller))

            async def scenario():
                async with httpx.AsyncClient(transport=transport) as client:
                    monkeypatch.setattr(
                        http_utils.GeneralHttpClientProvider, "client", classmethod(lambda cls: client)
                    )
                    await app.router.lifespan_context(app).__aenter__()
                    return await components.create_rollout_components(args, providers=factory)

            result = asyncio.run(scenario())

            assert result.inference_controller is controller
            assert result.num_rollout_per_epoch == 7
            assert args.num_rollout == 21


class TestNoRayInTheAssembly:
    @pytest.mark.parametrize("module", REFUSED_MODULES)
    def test_the_guard_would_notice_an_import_of(self, module: str):
        """A guard that lets ray through would make every test above prove nothing."""
        with kubernetes_only_imports(), pytest.raises(AssertionError, match=module.replace(".", r"\.")):
            importlib.import_module(module)
