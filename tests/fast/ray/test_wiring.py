from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.utils.workers.worker_provider.test_k8s_labels import make_pod

from miles.ray import wiring
from miles.utils.workers.reconcile.k8s_api import PodListPage
from miles.utils.workers.types import ClusterBackend
from miles.utils.workers.worker_provider.factory import RayProviderFactory
from miles.utils.workers.worker_provider.shared import SharedK8sWorkerProvider
from miles.utils.workers.worker_provider.simple import SimpleWorkerProvider
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

RELEASE = "miles-run-c0ffee"
NAMESPACE = "team-a"
RAY_WORKER_MANAGER_MODULE = "miles.utils.workers.ray_worker_manager"


class FakePodApi:
    def __init__(self, pods: list[Any]) -> None:
        self.pods = pods

    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage:
        return PodListPage(pods=list(self.pods), resource_version="1")

    async def stream_pods(self, *, namespace, label_selector, resource_version, timeout_seconds):
        await asyncio.sleep(3600)
        yield None


def _router_spec() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        launch_command=lambda context: "python -m router",
    )


def _engine_spec() -> CommandWorkerSpec:
    return CommandWorkerSpec(
        name="engine",
        port_infos=[PortInfo(name="primary", static_port=8000), PortInfo(name="nccl", static_port=10000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(
            num_cells=2, num_workers_per_cell=1, num_gpus_per_worker=1, num_gpu_slots_per_worker=8
        ),
        launch_command=lambda context: "python -m engine",
    )


def _trainer_spec(*, num_workers_per_cell: int, port_infos: list[PortInfo] | None = None) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="trainer-actor",
        port_infos=port_infos or [PortInfo(name="master", static_port=9000, mode="master")],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=num_workers_per_cell,
            num_gpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
        ),
        worker_class="miles.fake.TrainWorker",
        ctor_kwargs=lambda context: {},
        meta=lambda context: dict(role="actor", cell_index=context.cell_index),
    )


def _install(deleted: list[list[str]], pods: list[Any] | None = None):
    api = FakePodApi(pods or [])

    async def delete_pods(pod_names: list[str]) -> None:
        deleted.append(list(pod_names))

    return wiring.install_kubernetes_workers(
        specs=[_router_spec(), _engine_spec()],
        namespace=NAMESPACE,
        release=RELEASE,
        kube_client_factory=lambda: api,
        delete_pods=delete_pods,
    )


class TestCreateProviderFactory:
    def test_a_ray_run_still_launches_the_ray_worker_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Ray path is the one every existing run takes, and it must be untouched."""
        sentinel = object()
        launched: list[Any] = []
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", lambda args: launched.append(args) or sentinel)

        args = SimpleNamespace(cluster_backend=ClusterBackend.RAY.value)
        factory = wiring.create_provider_factory(args)

        assert launched == [args]
        assert isinstance(factory, RayProviderFactory)

    def test_a_kubernetes_run_installs_a_provider_rather_than_launching_workers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under Kubernetes the pods already exist, so launching actors would double the run."""
        installed: list[Any] = []
        sentinel = object()
        monkeypatch.setattr(
            wiring, "_install_kubernetes_workers_from_args", lambda args: installed.append(args) or sentinel
        )
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", _refuse_ray)

        args = SimpleNamespace(cluster_backend=ClusterBackend.KUBERNETES.value)
        assert wiring.create_provider_factory(args) is sentinel
        assert installed == [args]

    def test_a_worker_process_builds_its_factory_only_when_something_asks_for_a_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every served worker builds this context, and most specs never look at it."""
        attached: list[list[str]] = []

        def _attach(worker_argv: list[str]):
            attached.append(worker_argv)
            return _install(deleted=[])

        monkeypatch.setattr(wiring, "_attach_provider_factory_from_argv", _attach)

        factory = wiring.create_worker_provider_factory(worker_argv=["--rollout-num-gpus", "8"])
        assert attached == []

        factory.cells(spec_names=["engine"])
        factory.cells(spec_names=["engine"])

        assert attached == [["--rollout-num-gpus", "8"]]


class TestKubernetesAssembly:
    def test_components_of_the_process_then_see_a_kubernetes_provider(self) -> None:
        """The whole point of the assembly: the factory must stop answering with Ray."""
        factory = _install(deleted=[])

        assert isinstance(factory.cells(spec_names=["engine"]), SharedK8sWorkerProvider)

    def test_a_static_worker_resolves_to_the_address_the_chart_gives_it(self) -> None:
        """A router has no cell to observe, so its address is predicted from the release name."""
        factory = _install(deleted=[])

        provider = factory.static(worker_name="inference-router-0-0-0")
        addr = asyncio.run(provider.get_addr("inference-router-0-0-0"))

        assert addr.host == f"{RELEASE}-inference-router-0-0.{RELEASE}-inference-router-0"
        assert addr.port == 8000

    def test_refuses_a_static_worker_the_run_never_deployed(self) -> None:
        """Answering with an invented address would send the caller at nothing at all."""
        factory = _install(deleted=[])

        with pytest.raises(AssertionError, match="static address book"):
            factory.static(worker_name="inference-router-9-0-0")

    def test_every_component_shares_one_observation_of_the_namespace(self) -> None:
        """A second instance would open a second watch of the same pods and cache them twice."""
        factory = _install(deleted=[])

        assert factory.cells(spec_names=["engine"]) is factory.cells(spec_names=["engine"])

    def test_suspending_a_cell_deletes_its_pods(self) -> None:
        """Kubernetes has no suspend: a cell heals because its workload recreates deleted pods."""
        deleted: list[list[str]] = []
        factory = _install(deleted=deleted, pods=[make_pod(name="engine-0-0", fleet="engine", cell_index="0")])
        operations = factory.cell_operations()

        asyncio.run(operations.suspend("engine-0"))

        assert deleted == [["engine-0-0"]]

    def test_listing_cells_starts_the_watch_it_needs(self) -> None:
        """The api server asks for cells without knowing that observation has to be started first."""
        factory = _install(deleted=[], pods=[make_pod(name="engine-0-0", fleet="engine", cell_index="0")])
        operations = factory.cell_operations()

        infos = asyncio.run(operations.cell_infos(["engine"]))

        assert list(infos) == ["engine-0"]

    def test_never_reaches_for_the_ray_worker_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A namespace has no Ray cluster, so touching the manager would fail the run there."""
        monkeypatch.setitem(sys.modules, RAY_WORKER_MANAGER_MODULE, None)
        deleted: list[list[str]] = []
        factory = _install(deleted=deleted, pods=[make_pod(name="engine-0-0", fleet="engine", cell_index="0")])

        operations = factory.cell_operations()
        asyncio.run(operations.suspend("engine-0"))

        assert isinstance(factory.cells(spec_names=["engine"]), SharedK8sWorkerProvider)
        assert deleted == [["engine-0-0"]]


class TestSpecDerivedValues:
    def test_only_gpu_bearing_specs_are_watched_as_cells(self) -> None:
        """A router is one pod behind a service, so it is addressed rather than healed."""
        assert wiring.fleet_spec_names(specs=[_router_spec(), _engine_spec()]) == ["engine"]

    def test_the_ports_of_an_observed_worker_come_from_its_spec(self) -> None:
        """A pod has a network namespace of its own, so every rank publishes the spec's ports."""
        assert wiring.fleet_worker_ports(specs=[_router_spec(), _engine_spec()]) == {
            "engine": {"primary": 8000, "nccl": 10000}
        }

    def test_the_specs_that_compute_meta_are_the_fleet_specs_that_declare_it(self) -> None:
        """A cell's meta is evaluated per observation, so only the fleets that have one may be listed."""
        specs = [_router_spec(), _engine_spec(), _trainer_spec(num_workers_per_cell=8)]

        assert list(wiring.fleet_spec_metas(specs=specs)) == ["trainer-actor"]

    def test_the_address_book_covers_every_static_worker(self) -> None:
        """A spec with several cells still has one address per cell, and all of them are needed."""
        addrs = wiring.static_worker_addrs(specs=[_router_spec(), _engine_spec()], release=RELEASE)

        assert list(addrs) == ["inference-router-0-0-0"]


class TestRanksPerPod:
    def test_an_engine_pod_runs_one_command_and_therefore_holds_one_rank(self) -> None:
        """An engine pod is a single server process spanning its gpus, whatever its spec counts as a worker."""
        assert wiring.fleet_ranks_per_pod(specs=[_engine_spec()], num_gpus_per_node=8) == {"engine": 1}

    def test_ignores_a_spec_that_is_addressed_rather_than_observed(self) -> None:
        """A router has no cell and therefore no pod to fan out."""
        assert wiring.fleet_ranks_per_pod(specs=[_router_spec()], num_gpus_per_node=8) == {}

    def test_refuses_a_spec_whose_rank_ports_would_reach_into_another_port(self) -> None:
        """Rank n binds the rpc port plus n, so a neighbour close above it would be taken from under it."""
        spec = _trainer_spec(
            num_workers_per_cell=8,
            port_infos=[
                PortInfo(name="rpc", static_port=8000, allow_dynamic=True),
                PortInfo(name="dist_init", static_port=8004, num_consecutive=30),
            ],
        )

        with pytest.raises(AssertionError, match="reaches into"):
            wiring.fleet_ranks_per_pod(specs=[spec], num_gpus_per_node=8)


class TestNamespaceDiscovery:
    def test_prefers_the_namespace_the_environment_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A driver run outside a pod has no service account file to read."""
        monkeypatch.setenv(wiring.NAMESPACE_ENV_VAR, "team-b")

        assert wiring.current_namespace() == "team-b"

    def test_reads_the_namespace_of_its_own_pod(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """In a pod nobody passes the namespace in, but the service account always says it."""
        monkeypatch.delenv(wiring.NAMESPACE_ENV_VAR, raising=False)
        namespace_file = tmp_path / "namespace"
        namespace_file.write_text("team-c\n")
        monkeypatch.setattr(wiring, "NAMESPACE_FILE", namespace_file)

        assert wiring.current_namespace() == "team-c"

    def test_refuses_to_guess(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Guessing 'default' would watch someone else's pods and heal them."""
        monkeypatch.delenv(wiring.NAMESPACE_ENV_VAR, raising=False)
        monkeypatch.setattr(wiring, "NAMESPACE_FILE", tmp_path / "missing")

        with pytest.raises(AssertionError, match=wiring.NAMESPACE_ENV_VAR):
            wiring.current_namespace()


class TestReleaseDiscovery:
    def test_reads_the_release_the_chart_told_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The orchestrator cannot recompute the release name, because its run uuid is its own."""
        monkeypatch.setenv(wiring.RELEASE_ENV_VAR, "miles-run-260805")

        assert wiring.current_release() == "miles-run-260805"

    def test_refuses_to_guess_the_release(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A guessed release selects no pods at all, and the run would wait forever for its cells."""
        monkeypatch.delenv(wiring.RELEASE_ENV_VAR, raising=False)

        with pytest.raises(AssertionError, match=wiring.RELEASE_ENV_VAR):
            wiring.current_release()


class TestLabelKeyDiscovery:
    def test_defaults_to_the_labels_a_leader_worker_set_writes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The bundled charts deploy LeaderWorkerSets, so no override is the common case."""
        for env_var in wiring.LABEL_KEY_ENV_VARS.values():
            monkeypatch.delenv(env_var, raising=False)

        assert wiring.current_label_keys().fleet == "leaderworkerset.sigs.k8s.io/name"

    def test_lets_a_platform_name_its_own_labels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A platform that already labels its pods says which key means what instead of relabelling."""
        monkeypatch.setenv(wiring.LABEL_KEY_ENV_VARS["fleet"], "platform.example/group")

        assert wiring.current_label_keys().fleet == "platform.example/group"


class TestImportDirection:
    def test_reaching_the_kubernetes_assembly_never_imports_ray(self) -> None:
        """Importing this module must not pull in Ray, which a namespace-only image may not even have."""
        tree = ast.parse(Path(wiring.__file__).read_text(), filename=wiring.__file__)
        imported = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0
        }

        assert [module for module in imported if "ray" in module.split(".")] == []
        assert RAY_WORKER_MANAGER_MODULE not in imported


def _refuse_ray(args: Any) -> None:
    raise AssertionError("the Kubernetes path must not launch Ray workers")


class TestStaticProvider:
    def test_serves_the_statically_addressed_workers_of_a_run(self) -> None:
        """A router is one pod behind a service, so it is addressed rather than observed."""
        provider = wiring.static_worker_provider(specs=[_router_spec(), _engine_spec()], release=RELEASE)

        assert provider.cell_ids() == ["inference-router-0-0"]
        assert asyncio.run(provider.get_addr("inference-router-0-0-0")).port == 8000

    def test_the_static_scope_answers_with_it_rather_than_with_the_watcher(self) -> None:
        """Statically addressed components need no watch, and a watch would never report them anyway."""
        factory = _install(deleted=[])

        static = factory.static(worker_name="inference-router-0-0-0")

        assert isinstance(static, SimpleWorkerProvider)
        assert static is not factory.cells(spec_names=["engine"])
