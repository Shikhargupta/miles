from types import SimpleNamespace

import pytest
import ray
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_alive_cell, make_cell

from miles.ray.train.group import TrainerController

pytestmark = pytest.mark.asyncio


def _make_group(cells: list) -> TrainerController:
    group = object.__new__(TrainerController)
    group._cells_by_id = {cell.cell_id: cell for cell in cells}
    group.args = SimpleNamespace()
    return group


def _reconcile_calls_of(cell) -> list:
    return [
        [call for call in ray.get(handle.get_calls.remote()) if call[0] == "reconcile_adapters"]
        for handle in get_raw_actor_handles(cell)
    ]


class TestReconcileAdapters:
    async def test_every_worker_is_asked_to_reconcile(self):
        """Each rank owns its adapter registry, so all of them must be reached."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        group = _make_group([cell])

        await group.reconcile_adapters()

        assert _reconcile_calls_of(cell) == [[("reconcile_adapters", (), {})]] * 2

    async def test_a_failing_worker_propagates_instead_of_being_swallowed(self):
        """A stale adapter set would corrupt routing, so the failure must reach the caller."""
        cell = make_cell(0)
        ray.get(get_raw_actor_handles(cell)[0].set_fail_methods.remote(["reconcile_adapters"]))
        group = _make_group([cell])

        with pytest.raises(Exception, match="Injected failure"):
            await group.reconcile_adapters()
