from typing import Any

from miles.utils.workers.ray_worker_manager.addressing import compute_worker_addressings
from miles.utils.workers.ray_worker_manager.state import CellState, WorkerState
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec


def _make_spec(port_infos: list[PortInfo]) -> ServeWorkerSpec:
    return ServeWorkerSpec(
        name="spec",
        port_infos=port_infos,
        env_var=lambda: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=2, num_gpus_per_worker=0, num_cpus_per_worker=1),
        worker_class="unused.Worker",
        ctor_kwargs=lambda cell_index, worker_index: {},
    )


def _make_workers(spec: ServeWorkerSpec, *, ports_by_worker: list[dict[str, int]]) -> list[WorkerState]:
    cell = CellState(spec=spec, cell_id="spec-0", cell_index=0, generation=0)
    fake_actor: Any = object()
    return [
        WorkerState(name=f"spec-0-{i}", cell=cell, actor=fake_actor, node_ip=f"10.0.0.{i}", owned_ports=ports)
        for i, ports in enumerate(ports_by_worker)
    ]


class TestComputeWorkerAddressings:
    def test_master_port_is_shared_and_per_worker_port_is_own(self):
        """Every worker addresses the master's port at the master's ip but keeps its own per-worker port."""
        spec = _make_spec(
            [
                PortInfo(name="rdzv", static_port=0, mode="master", allow_dynamic=True),
                PortInfo(name="http", static_port=0, mode="per_worker", allow_dynamic=True),
            ]
        )
        workers = _make_workers(spec, ports_by_worker=[{"rdzv": 15000, "http": 15001}, {"http": 15002}])

        addressings = compute_worker_addressings(spec=spec, workers=workers)

        assert addressings["spec-0-1"].addr_port_kwargs == {
            "rdzv_addr": "10.0.0.0",
            "rdzv_port": 15000,
            "http_addr": "10.0.0.1",
            "http_port": 15002,
        }

    def test_url_comes_from_the_url_scheme_port(self):
        """The worker url is rendered from the port declaring a url scheme."""
        spec = _make_spec([PortInfo(name="http", static_port=0, mode="per_worker", allow_dynamic=True, url_scheme="http")])
        workers = _make_workers(spec, ports_by_worker=[{"http": 15000}, {"http": 15001}])

        addressings = compute_worker_addressings(spec=spec, workers=workers)

        assert addressings["spec-0-0"].url == "http://10.0.0.0:15000"
        assert addressings["spec-0-1"].url == "http://10.0.0.1:15001"

    def test_url_is_none_without_a_url_scheme(self):
        """Workers get no url when no port declares a url scheme."""
        spec = _make_spec([PortInfo(name="http", static_port=0, mode="per_worker", allow_dynamic=True)])
        workers = _make_workers(spec, ports_by_worker=[{"http": 15000}])

        addressings = compute_worker_addressings(spec=spec, workers=workers)

        assert addressings["spec-0-0"].url is None
