from __future__ import annotations

import asyncio

from miles.utils.workers.ray_worker_manager import ActorState, RayWorkerManager, _CellInfo, _CellRecord
from miles.utils.workers.worker_provider.ray import RayWorkerProvider
from miles.utils.workers.worker_spec import WorkerPlacement


def _manager_with(cells: dict[str, list[dict]]) -> RayWorkerManager:
    manager = RayWorkerManager(pg=None)
    for cell_id, payloads in cells.items():
        manager._infos[cell_id] = _CellInfo(
            record=_CellRecord(
                workers=[
                    ActorState(
                        actor=object(),
                        payload=payload,
                        placement=WorkerPlacement(local_index=index, global_rank=index, base_gpu_id=index),
                    )
                    for index, payload in enumerate(payloads)
                ],
                ready=True,
            )
        )
    return manager


async def _collect(provider: RayWorkerProvider, *, until: int, timeout: float = 2.0) -> list[tuple]:
    events: list[tuple] = []

    async def _reconcile(cell_id, info):
        events.append((cell_id, info))

    stop = await provider.watch_cells(_reconcile)
    deadline = asyncio.get_running_loop().time() + timeout
    while len(events) < until and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.005)
    await stop()
    return events


class TestListCells:
    async def test_reports_one_entry_per_live_cell(self):
        """The provider is the consumer's only view of which cells exist."""
        manager = _manager_with({"cell-0": [{"host": "10.0.0.1", "port": 30000}], "cell-1": [{"host": "10.0.0.2"}]})
        cells = await RayWorkerProvider(worker_manager=manager).list_cells()
        assert sorted(cells) == ["cell-0", "cell-1"]

    async def test_carries_every_member_payload_of_a_multi_node_cell(self):
        """A consumer addresses the cell through its members, so none may be dropped."""
        manager = _manager_with({"cell-0": [{"host": "10.0.0.1"}, {"host": "10.0.0.2"}]})
        cells = await RayWorkerProvider(worker_manager=manager).list_cells()
        assert [member.payload["host"] for member in cells["cell-0"].members] == ["10.0.0.1", "10.0.0.2"]

    async def test_carries_the_worker_handles_next_to_their_payloads(self):
        """Consumers still call their workers directly, so the handles travel with the urls."""
        manager = _manager_with({"cell-0": [{"host": "10.0.0.1"}, {"host": "10.0.0.2"}]})
        cells = await RayWorkerProvider(worker_manager=manager).list_cells()
        expected = [worker.actor for worker in manager.cell_workers("cell-0")]
        assert [member.handle for member in cells["cell-0"].members] == expected

    async def test_replaced_workers_report_different_payloads(self):
        """The payloads are how a consumer notices a restart handed it different workers."""
        before = await RayWorkerProvider(worker_manager=_manager_with({"c": [{"host": "10.0.0.1"}]})).list_cells()
        after = await RayWorkerProvider(worker_manager=_manager_with({"c": [{"host": "10.0.0.9"}]})).list_cells()
        assert [m.payload for m in before["c"].members] != [m.payload for m in after["c"].members]

    async def test_unchanged_workers_report_equal_payloads(self):
        """Stable payloads keep the reconcile loop from replacing healthy cells."""
        first = await RayWorkerProvider(worker_manager=_manager_with({"c": [{"host": "10.0.0.1"}]})).list_cells()
        second = await RayWorkerProvider(worker_manager=_manager_with({"c": [{"host": "10.0.0.1"}]})).list_cells()
        assert [m.payload for m in first["c"].members] == [m.payload for m in second["c"].members]


class TestWatchCells:
    async def test_a_pre_seeded_cell_is_not_reported_again(self):
        """The caller that already reconciled a snapshot must not be handed it twice."""
        manager = _manager_with({"cell-0": [{"host": "10.0.0.1"}]})
        provider = RayWorkerProvider(worker_manager=manager, poll_interval_seconds=0.01)

        events: list[tuple] = []

        async def _reconcile(cell_id, info):
            events.append((cell_id, info))

        stop = await provider.watch_cells(_reconcile, seen=await provider.list_cells())
        await asyncio.sleep(0.1)
        await stop()

        assert events == []

    async def test_reports_the_cells_that_already_exist(self):
        """A consumer attaching late must still learn about running cells."""
        manager = _manager_with({"cell-0": [{"host": "10.0.0.1"}]})
        provider = RayWorkerProvider(worker_manager=manager, poll_interval_seconds=0.01)
        events = await _collect(provider, until=1)
        assert events[0][0] == "cell-0"
        assert events[0][1] is not None

    async def test_reports_none_when_a_cell_vanishes(self):
        """A vanished cell must reach the consumer so it can drop its bookkeeping."""
        manager = _manager_with({"cell-0": [{"host": "10.0.0.1"}]})
        provider = RayWorkerProvider(worker_manager=manager, poll_interval_seconds=0.01)

        events: list[tuple] = []

        async def _reconcile(cell_id, info):
            events.append((cell_id, info))
            if info is not None:
                manager._infos.clear()

        stop = await provider.watch_cells(_reconcile)
        deadline = asyncio.get_running_loop().time() + 2.0
        while len(events) < 2 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.005)
        await stop()

        assert events[1] == ("cell-0", None)

    async def test_an_unchanged_cell_is_reported_once(self):
        """Re-reporting a stable cell would make the consumer re-attach it every poll."""
        manager = _manager_with({"cell-0": [{"host": "10.0.0.1"}]})
        provider = RayWorkerProvider(worker_manager=manager, poll_interval_seconds=0.01)

        events: list[tuple] = []

        async def _reconcile(cell_id, info):
            events.append((cell_id, info))

        stop = await provider.watch_cells(_reconcile)
        await asyncio.sleep(0.2)
        await stop()

        assert len(events) == 1

    async def test_a_failing_poll_does_not_end_the_watch(self):
        """A transient infrastructure error must not silently stop the reconcile loop."""
        manager = _manager_with({"cell-0": [{"host": "10.0.0.1"}]})
        provider = RayWorkerProvider(worker_manager=manager, poll_interval_seconds=0.01)

        events: list[tuple] = []
        calls = {"n": 0}

        async def _reconcile(cell_id, info):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("consumer blew up")
            events.append((cell_id, info))

        stop = await provider.watch_cells(_reconcile)
        deadline = asyncio.get_running_loop().time() + 2.0
        while not events and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.005)
        await stop()

        assert events
