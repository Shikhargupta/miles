from __future__ import annotations

import asyncio

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations.serialized import SerializedCellOperations


class _RecordingOperations:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def cell_infos(self, *, pool_ids: list[str]) -> dict:
        self.calls.append(("cell_infos", ",".join(pool_ids)))
        return {}

    async def suspend(self, *, cell_id: str) -> None:
        self.calls.append(("suspend", cell_id))

    async def resume(self, *, cell_id: str) -> None:
        self.calls.append(("resume", cell_id))

    async def inject_fault(self, *, cell_id: str, mode: FailureMode, sub_index: int) -> None:
        self.calls.append(("inject_fault", cell_id))


class _BusyGate:
    """Stands in for the controller: it answers only once whoever holds its lock is done."""

    def __init__(self) -> None:
        self.released = asyncio.Event()
        self.stopped: list[str] = []

    async def stop_cell_between_weight_updates(self, cell_id: str) -> None:
        await self.released.wait()
        self.stopped.append(cell_id)


async def test_a_suspend_waits_for_the_gate_instead_of_reaching_the_inner_operations():
    """A suspend arriving mid weight update must not reach the worker manager until the update ends."""
    inner, gate = _RecordingOperations(), _BusyGate()
    operations = SerializedCellOperations(inner, gate=gate)

    suspending = asyncio.create_task(operations.suspend(cell_id="engine-0-2"))
    await asyncio.sleep(0)
    assert gate.stopped == []
    assert inner.calls == []

    gate.released.set()
    await suspending
    assert gate.stopped == ["engine-0-2"]
    assert inner.calls == []


async def test_every_other_operation_goes_straight_through():
    """Only a suspend takes a rank out of a live collective, so nothing else pays for the lock."""
    inner, gate = _RecordingOperations(), _BusyGate()
    operations = SerializedCellOperations(inner, gate=gate)

    await operations.cell_infos(pool_ids=["engine-0"])
    await operations.resume(cell_id="engine-0-2")
    await operations.inject_fault(cell_id="engine-0-2", mode=FailureMode.SIGKILL, sub_index=0)

    assert [name for name, _ in inner.calls] == ["cell_infos", "resume", "inject_fault"]
    assert gate.stopped == []
