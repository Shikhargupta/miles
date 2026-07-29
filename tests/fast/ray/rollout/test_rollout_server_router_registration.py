from __future__ import annotations

from unittest.mock import patch

from tests.fast.ray.rollout.conftest import FakeWorkerHandle, make_args

from miles.ray.rollout.cell_state import AddrInfo
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell
from miles.ray.specs.inference import _compute_nodes_per_engine


class _RecordingRouterApiClient:
    def __init__(self, events: list[tuple[str, dict]]):
        self._events = events

    async def add_worker(self, **kwargs):
        self._events.append(("add_worker", kwargs))

    async def remove_worker(self, **kwargs):
        self._events.append(("remove_worker", kwargs))


def _build_server(
    *,
    events: list[tuple[str, dict]],
    num_engines: int = 1,
    num_gpus_per_engine: int = 1,
    worker_type: str = "regular",
    router_ip: str = "10.0.0.9",
    router_port: int = 9000,
    bootstrap_port: int | None = None,
    use_miles_router: bool = False,
) -> RolloutServer:
    args = make_args(num_gpus_per_node=8, use_miles_router=use_miles_router)
    nodes_per_engine = _compute_nodes_per_engine(num_gpus_per_engine=num_gpus_per_engine, num_gpus_per_node=8)
    cells = []
    for cell_start in range(0, num_engines, nodes_per_engine):
        cell = ServerCell(
            args=args,
            cell_id=f"cell-{len(cells)}",
            num_nodes=nodes_per_engine,
            num_gpus_per_engine=num_gpus_per_engine,
            worker_type=worker_type,
        )
        cell._mark_allocated_uninitialized([FakeWorkerHandle() for _ in range(nodes_per_engine)])
        cell._mark_addressing(
            [
                AddrInfo(
                    server_url=f"http://10.0.0.{cell_start + i + 1}:3000{cell_start + i}",
                    bootstrap_port=bootstrap_port,
                )
                for i in range(nodes_per_engine)
            ]
        )
        cell._mark_alive()
        cells.append(cell)

    srv = RolloutServer(
        server_cells={cell.cell_id: cell for cell in cells},
        args=args,
        router_ip=router_ip,
        router_port=router_port,
    )
    srv._recording_router_client = _RecordingRouterApiClient(events)
    return srv


def _with_recording_client(srv: RolloutServer):
    return patch.object(RolloutServer, "_router_api_client", property(lambda self: self._recording_router_client))


async def test_registration_publishes_the_url_the_engine_actually_serves():
    """The router must be told the url derived from the manager's port allocation."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events)

    with _with_recording_client(srv):
        await srv.server_cells["cell-0"].register(srv._router_api_client)

    assert events == [
        (
            "add_worker",
            {
                "worker_url": "http://10.0.0.1:30000",
                "worker_type": "regular",
                "use_legacy_api": False,
                "bootstrap_port": None,
            },
        )
    ]


async def test_registration_passes_the_bootstrap_port_of_a_prefill_worker():
    """PD disaggregation needs the decode side to dial this port."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events, worker_type="prefill", bootstrap_port=8998)

    with _with_recording_client(srv):
        await srv.server_cells["cell-0"].register(srv._router_api_client)

    assert events[0][1]["worker_type"] == "prefill"
    assert events[0][1]["bootstrap_port"] == 8998


async def test_registration_addresses_only_the_primary_engine_of_a_multi_node_cell():
    """Only the primary (node-0) engine serves the router-visible endpoint."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events, num_engines=2, num_gpus_per_engine=16)

    with _with_recording_client(srv):
        await srv.server_cells["cell-0"].register(srv._router_api_client)

    assert [kwargs["worker_url"] for _name, kwargs in events] == ["http://10.0.0.1:30000"]


async def test_use_miles_router_reaches_both_router_calls():
    """--use-miles-router pins the legacy query-string API on register and unregister alike."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events, use_miles_router=True)

    with _with_recording_client(srv):
        await srv.server_cells["cell-0"].register(srv._router_api_client)
        await srv.server_cells["cell-0"].unregister(srv._router_api_client)

    assert [kwargs.get("use_legacy_api") for name, kwargs in events if name in ("add_worker", "remove_worker")] == [
        True,
        True,
    ]
