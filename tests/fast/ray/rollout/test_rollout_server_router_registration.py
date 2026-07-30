from __future__ import annotations

from unittest.mock import patch

from tests.fast.ray.rollout.conftest import adopt_cell_workers, make_args, make_cell_spec

from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell
from miles.ray.specs.inference import compute_nodes_per_engine
from miles.utils.workers.ray_worker_manager import RayWorkerManager
from miles.utils.workers.worker_provider.base import CellInfo, CellMember


class _RecordingRouterApiClient:
    def __init__(self, events: list[tuple[str, dict]], remove_worker_effect=None):
        self._events = events
        self._remove_worker_effect = remove_worker_effect

    async def add_worker(self, **kwargs):
        self._events.append(("add_worker", kwargs))

    async def remove_worker(self, **kwargs):
        self._events.append(("remove_worker", kwargs))
        if self._remove_worker_effect is not None:
            await self._remove_worker_effect()


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
    remove_worker_effect=None,
) -> RolloutServer:
    args = make_args(num_gpus_per_node=8, use_miles_router=use_miles_router)
    nodes_per_engine = compute_nodes_per_engine(num_gpus_per_engine=num_gpus_per_engine, num_gpus_per_node=8)
    worker_manager = RayWorkerManager(pg=None)
    cells = []
    for cell_start in range(0, num_engines, nodes_per_engine):
        cell = ServerCell(
            args=args,
            spec=make_cell_spec(
                args=args,
                cell_id=f"cell-{len(cells)}",
                num_nodes=nodes_per_engine,
                num_gpus_per_engine=num_gpus_per_engine,
                worker_type=worker_type,
                rank_offset=cell_start,
            ),
        )
        adopt_cell_workers(
            worker_manager,
            cell_id=cell.cell_id,
            payloads=[
                {
                    "host": f"10.0.0.{cell_start + local_index + 1}",
                    "port": int(f"3000{cell_start + local_index}"),
                    "disaggregation_bootstrap_port": bootstrap_port,
                }
                for local_index in range(nodes_per_engine)
            ],
        )
        workers = worker_manager.cell_workers(cell.cell_id)
        cell.attach(
            CellInfo(
                cell_id=cell.cell_id,
                members=[
                    CellMember(handle=worker.actor, payload=worker.payload, placement=worker.placement)
                    for worker in workers
                ],
            )
        )
        cell.mark_alive()
        cells.append(cell)

    srv = RolloutServer(
        server_cells={cell.cell_id: cell for cell in cells},
        args=args,
        router_ip=router_ip,
        router_port=router_port,
    )
    srv._recording_router_client = _RecordingRouterApiClient(events, remove_worker_effect=remove_worker_effect)
    srv._test_worker_manager = worker_manager
    return srv


def _with_recording_client(srv: RolloutServer):
    return patch.object(RolloutServer, "router_api_client", property(lambda self: self._recording_router_client))


async def test_registration_publishes_the_url_the_engine_actually_serves():
    """The router must be told the url the rollout process derived from the allocator."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events)

    with _with_recording_client(srv):
        await srv.server_cells["cell-0"].register(srv.router_api_client)

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
        await srv.server_cells["cell-0"].register(srv.router_api_client)

    assert events[0][1]["worker_type"] == "prefill"
    assert events[0][1]["bootstrap_port"] == 8998


async def test_registration_addresses_only_the_primary_engine_of_a_multi_node_cell():
    """Only the primary (node-0) engine serves the router-visible endpoint."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events, num_engines=2, num_gpus_per_engine=16)

    with _with_recording_client(srv):
        await srv.server_cells["cell-0"].register(srv.router_api_client)

    assert [kwargs["worker_url"] for _name, kwargs in events] == ["http://10.0.0.1:30000"]


async def test_use_miles_router_reaches_both_router_calls():
    """--use-miles-router pins the legacy query-string API on register and unregister alike."""
    events: list[tuple[str, dict]] = []
    srv = _build_server(events=events, use_miles_router=True)

    with _with_recording_client(srv):
        await srv.server_cells["cell-0"].register(srv.router_api_client)
        await srv.server_cells["cell-0"].unregister(srv.router_api_client)

    assert [kwargs["use_legacy_api"] for _name, kwargs in events] == [True, True]
