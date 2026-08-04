import pytest
import ray
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import make_cell

pytestmark = pytest.mark.asyncio


class TestCellKillAndRestart:
    async def test_killing_a_failed_cell_reaches_the_workers_directly(self):
        """Waiting for an external controller would leave the other cells hanging in NCCL."""
        cell = make_cell(2)
        handles = cell._get_actor_handles()

        await cell._kill_workers_and_confirm_dead()

        assert train_conftest.fake_worker_manager.stopped_cell_ids == []
        for handle in handles:
            with pytest.raises(ray.exceptions.RayActorError):
                ray.get(handle.get_calls.remote())

    async def test_a_replacement_cell_picks_up_the_fresh_actor_handles(self):
        """Reusing the dead handles would make every later call fail."""
        cell = make_cell(0)
        old_handles = cell._get_actor_handles()
        await cell._kill_workers_and_confirm_dead()

        train_conftest.fake_worker_manager._stop_cells([cell.cell_id])
        replacement = make_cell(0)

        assert replacement._get_actor_handles() != old_handles

    async def test_killing_twice_is_harmless(self):
        """Healing may tear down an already dead cell, which must not raise."""
        cell = make_cell(0)
        await cell._kill_workers_and_confirm_dead()

        await cell._kill_workers_and_confirm_dead()
